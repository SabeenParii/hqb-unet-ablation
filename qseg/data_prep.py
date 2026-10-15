"""Data preparation: de-duplication, Otsu proxy masks, patient-level split.

Changes relative to the original `data_cleaning_splitting.ipynb`:
  * Exact duplicate files (e.g. "… (1).jpg" copies) are detected by content hash and
    removed, so each image is used once.
  * Patient IDs are SORTED before splitting. The original split used the order returned
    by os.listdir(), which differs between machines, so seed 42 did not guarantee the
    same split elsewhere.
  * Images are zero-padded to a square before resizing, so the breast is not squashed
    (the originals are 2816 x 3584 and 824 x 578). Padding is added on the side away from
    the chest wall, so the breast stays attached to the image border.
  * Images and masks are saved WITHOUT augmentation. Augmentation is now applied online
    during training (a new random transform every epoch) instead of once, at save time.
  * No mean/SD normalisation is baked into the saved files; models read [0, 1] intensities.
  * A manifest.csv records every file, its patient, view, laterality, split and mask
    statistics, so the exact split can be reported and reproduced.

Requires only numpy, opencv-python, pandas and scikit-learn (no TensorFlow).
"""
import hashlib
import os
import re

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config

FILENAME_RE = re.compile(r"^(?P<year>\d{4})_(?P<patient>BC\d+)_\s*(?P<view>CC|MLO)_(?P<side>[LR])")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ----------------------------------------------------------------------------- files
def parse_filename(filename):
    """'2013_BC999901_ CC_R.jpg' -> dict(year, patient_id, view, laterality)."""
    stem = os.path.splitext(filename)[0]
    m = FILENAME_RE.match(stem)
    if not m:
        raise ValueError(f"Unrecognised file name: {filename}")
    return dict(year=int(m["year"]), patient_id=m["patient"], view=m["view"], laterality=m["side"])


def canonical_id(filename):
    """File id without duplicate suffixes such as ' (1)'."""
    stem = os.path.splitext(filename)[0]
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    return re.sub(r"\s+", "", stem)  # '2013_BC999901_ CC_R' -> '2013_BC999901_CC_R'


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def list_unique_images(raw_dir):
    """Return (kept, removed) DataFrames. Duplicates = identical content OR same canonical id."""
    names = sorted(f for f in os.listdir(raw_dir) if f.lower().endswith(IMAGE_EXT))
    rows = []
    for n in names:
        rows.append(dict(original_filename=n, file_id=canonical_id(n),
                         md5=file_md5(os.path.join(raw_dir, n)), **parse_filename(n)))
    df = pd.DataFrame(rows)
    # prefer the file without a " (n)" suffix when both exist
    df["has_suffix"] = df["original_filename"].str.contains(r"\(\d+\)")
    df = df.sort_values(["file_id", "has_suffix", "original_filename"]).reset_index(drop=True)
    dup = df.duplicated("md5", keep="first") | df.duplicated("file_id", keep="first")
    removed = df[dup].copy()
    kept = df[~dup].drop(columns="has_suffix").reset_index(drop=True)
    return kept, removed.drop(columns="has_suffix")


# ----------------------------------------------------------------------------- masks
def generate_breast_mask(gray, kernel_size=20):
    """Otsu threshold -> morphological opening + closing -> largest external contour, filled.

    This is the same proxy-labelling rule as the original notebook.
    Returns uint8 mask (0/255) and the Otsu threshold value.
    """
    thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=-1)
    return mask, float(thr), binary


def pad_to_square(arr, side="center"):
    """Zero-pad to a square. side='left' puts the padding on the left, 'right' on the right."""
    h, w = arr.shape[:2]
    s = max(h, w)
    top = (s - h) // 2
    left = {"center": (s - w) // 2, "left": s - w, "right": 0}[side]
    out = np.zeros((s, s), dtype=arr.dtype)
    out[top:top + h, left:left + w] = arr
    return out


def chest_wall_side(mask):
    """'right' if the breast is attached to the right image border, else 'left'."""
    h, w = mask.shape
    fg = mask > 0
    return "right" if fg[:, w // 2:].sum() >= fg[:, :w // 2].sum() else "left"


def resize_pair(gray, mask, size, pad=True):
    if pad:
        # pad on the side AWAY from the chest wall, so the breast stays attached to the image border
        side = "left" if chest_wall_side(mask) == "right" else "right"
        gray, mask = pad_to_square(gray, side), pad_to_square(mask, side)
    img = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)       # anti-aliased downsampling
    msk = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)    # keeps the mask binary
    return img, (msk > 127).astype(np.uint8) * 255


# ----------------------------------------------------------------------------- split
def patient_split(patient_ids, cfg: Config):
    ids = sorted(set(patient_ids))                      # deterministic order
    train, rest = train_test_split(ids, test_size=cfg.val_ratio + cfg.test_ratio,
                                   random_state=cfg.split_seed)
    val, test = train_test_split(sorted(rest), test_size=cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio),
                                 random_state=cfg.split_seed)
    return {**{p: "train" for p in train}, **{p: "val" for p in val}, **{p: "test" for p in test}}


# ----------------------------------------------------------------------------- main
def prepare_dataset(cfg: Config, verbose=True, skip_if_done=True):
    manifest_path = os.path.join(cfg.work_dir, "data", "manifest.csv")
    if skip_if_done and os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)
        if len(manifest) and all(os.path.exists(os.path.join(cfg.work_dir, "data", "masks", f"{f}.png"))
                                 for f in manifest["file_id"]):
            if verbose:
                print("[skip] data already prepared:", manifest_path)
                print(split_summary(manifest).to_string())
            rem = os.path.join(cfg.work_dir, "data", "removed_duplicates.csv")
            return manifest, (pd.read_csv(rem) if os.path.exists(rem) else pd.DataFrame())
    out_img = os.path.join(cfg.work_dir, "data", "images")
    out_msk = os.path.join(cfg.work_dir, "data", "masks")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_msk, exist_ok=True)

    kept, removed = list_unique_images(cfg.raw_image_dir)
    if verbose:
        print(f"Files found: {len(kept) + len(removed)} | duplicates removed: {len(removed)} | unique: {len(kept)}")

    split_of = patient_split(kept["patient_id"], cfg)
    records = []
    for i, r in kept.iterrows():
        gray = cv2.imread(os.path.join(cfg.raw_image_dir, r["original_filename"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise IOError(f"Cannot read {r["original_filename"]}")
        mask, thr, _ = generate_breast_mask(gray, cfg.otsu_kernel)
        img_s, msk_s = resize_pair(gray, mask, cfg.image_size, cfg.pad_to_square)
        cv2.imwrite(os.path.join(out_img, f"{r["file_id"]}.png"), img_s)
        cv2.imwrite(os.path.join(out_msk, f"{r["file_id"]}.png"), msk_s)
        records.append(dict(file_id=r["file_id"], original_filename=r["original_filename"], patient_id=r["patient_id"],
                            year=r["year"], view=r["view"], laterality=r["laterality"], split=split_of[r["patient_id"]],
                            orig_height=gray.shape[0], orig_width=gray.shape[1], otsu_threshold=thr,
                            mask_fraction_full_res=float((mask > 0).mean()),
                            mask_fraction_256=float((msk_s > 0).mean()), md5=r["md5"]))
        if verbose and (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(kept)}")

    manifest = pd.DataFrame(records)
    manifest.to_csv(os.path.join(cfg.work_dir, "data", "manifest.csv"), index=False)
    removed.to_csv(os.path.join(cfg.work_dir, "data", "removed_duplicates.csv"), index=False)
    cfg.to_json(os.path.join(cfg.work_dir, "data", "prep_config.json"))
    if verbose:
        print(split_summary(manifest).to_string())
    return manifest, removed


def split_summary(manifest):
    g = manifest.groupby("split")
    s = pd.DataFrame({"patients": g["patient_id"].nunique(), "images": g.size(),
                      "mean_mask_fraction": g["mask_fraction_256"].mean().round(3)})
    s = s.reindex(["train", "val", "test"])
    s.loc["total"] = [manifest["patient_id"].nunique(), len(manifest), round(manifest["mask_fraction_256"].mean(), 3)]
    return s


def qc_figure(cfg: Config, file_ids, path):
    """Original image | raw Otsu output | final proxy mask (the one used for training)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest = pd.read_csv(os.path.join(cfg.work_dir, "data", "manifest.csv")).set_index("file_id")
    fig, axes = plt.subplots(len(file_ids), 3, figsize=(9, 3.2 * len(file_ids)), squeeze=False)
    for row, fid in zip(axes, file_ids):
        gray = cv2.imread(os.path.join(cfg.raw_image_dir, manifest.loc[fid, "original_filename"]), cv2.IMREAD_GRAYSCALE)
        mask, _, binary = generate_breast_mask(gray, cfg.otsu_kernel)
        for ax, im, title in zip(row, (gray, binary, mask),
                                 ("Mammogram", "Otsu threshold output", "Final proxy mask")):
            ax.imshow(im, cmap="gray")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
