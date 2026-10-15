import os, shutil
import cv2, numpy as np
from qseg.config import Config
from qseg.data_prep import canonical_id, parse_filename, list_unique_images, patient_split, pad_to_square


def test_filename_parsing():
    assert canonical_id("2013_BC999902_ CC_R (1).jpg") == "2013_BC999902_CC_R"
    assert parse_filename("2014_BC999903_ MLO_L.jpg") == dict(year=2014, patient_id="BC999903", view="MLO", laterality="L")


def test_duplicates_removed(tmp_path):
    img = (np.random.default_rng(0).random((20, 20)) * 255).astype(np.uint8)
    for n in ["2013_BC000001_ CC_L.jpg", "2013_BC000001_ CC_L (1).jpg", "2013_BC000001_ CC_R.jpg"]:
        cv2.imwrite(str(tmp_path / n), img if "CC_L" in n else img.T)
    kept, removed = list_unique_images(str(tmp_path))
    assert len(kept) == 2 and len(removed) == 1
    assert removed.iloc[0]["original_filename"].endswith("(1).jpg")


def test_split_is_order_independent():
    cfg = Config()
    ids = [f"BC{i:06d}" for i in range(133)]
    a = patient_split(ids, cfg)
    b = patient_split(list(reversed(ids)), cfg)
    assert a == b
    counts = {s: sum(v == s for v in a.values()) for s in ("train", "val", "test")}
    assert counts == {"train": 93, "val": 20, "test": 20}


def test_pad_to_square_keeps_content():
    a = np.ones((4, 2), np.uint8)
    p = pad_to_square(a)
    assert p.shape == (4, 4) and p.sum() == 8


def test_padding_goes_away_from_chest_wall():
    from qseg.data_prep import resize_pair
    g = np.zeros((40, 20), np.uint8); m = np.zeros((40, 20), np.uint8)
    g[:, 12:] = 200; m[:, 12:] = 255          # breast attached to the right border
    img, msk = resize_pair(g, m, 40)
    assert msk[:, -1].any() and not msk[:, 0].any()


def test_manifest_view_column(tmp_path):
    """Regression test: manifest 'view' must be CC/MLO (r.view used to return pandas' Series.view)."""
    import pandas as pd
    from qseg.data_prep import prepare_dataset
    raw = tmp_path / "raw"; raw.mkdir()
    expected = {}
    for i in range(7):
        view = "CC" if i % 2 else "MLO"
        g = np.zeros((60, 40), np.uint8); g[5:55, 15:] = 180 + i
        cv2.imwrite(str(raw / f"2013_BC00000{i}_ {view}_L.jpg"), g)
        expected[f"2013_BC00000{i}_{view}_L"] = view
    cfg = Config(raw_image_dir=str(raw), work_dir=str(tmp_path / "work"), image_size=32)
    manifest, _ = prepare_dataset(cfg, verbose=False)
    assert set(manifest["view"]) <= {"CC", "MLO"}
    assert dict(zip(manifest["file_id"], manifest["view"])) == expected
    assert set(manifest["laterality"]) == {"L"}
