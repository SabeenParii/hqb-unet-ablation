"""tf.data input pipeline shared by ALL models.

* Grayscale, 1 channel, intensities in [0, 1] for every model (the original Attention
  U-Net used 3 replicated channels and a different batch size).
* Online augmentation on the training split only, re-drawn every epoch, seeded:
  horizontal flip (p = 0.5), rotation in [-10, 10] degrees, brightness and contrast
  factors in [0.8, 1.2]. Geometric transforms are applied identically to image and mask.
"""
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import Config


def load_manifest(cfg: Config):
    return pd.read_csv(os.path.join(cfg.work_dir, "data", "manifest.csv"))


def _paths(cfg, manifest, split):
    sub = manifest[manifest["split"] == split].sort_values("file_id")
    img = [os.path.join(cfg.work_dir, "data", "images", f"{f}.png") for f in sub["file_id"]]
    msk = [os.path.join(cfg.work_dir, "data", "masks", f"{f}.png") for f in sub["file_id"]]
    return sub.reset_index(drop=True), img, msk


def _load(img_path, msk_path):
    img = tf.image.decode_png(tf.io.read_file(img_path), channels=1)
    msk = tf.image.decode_png(tf.io.read_file(msk_path), channels=1)
    img = tf.cast(img, tf.float32) / 255.0
    msk = tf.cast(msk > 127, tf.float32)
    return img, msk


def _rotation_transform(angle, h, w):
    """Projective transform (output -> input coordinates) for a rotation about the centre."""
    c, s = tf.cos(angle), tf.sin(angle)
    wf, hf = tf.cast(w - 1, tf.float32), tf.cast(h - 1, tf.float32)
    x_off = (wf - (c * wf - s * hf)) / 2.0
    y_off = (hf - (s * wf + c * hf)) / 2.0
    return tf.reshape(tf.stack([c, -s, x_off, s, c, y_off, 0.0, 0.0]), [1, 8])


def _rotate(x, angle, interpolation):
    h, w = tf.shape(x)[0], tf.shape(x)[1]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=x[tf.newaxis], transforms=_rotation_transform(angle, h, w),
        output_shape=tf.stack([h, w]), fill_value=0.0, interpolation=interpolation, fill_mode="CONSTANT")
    return out[0]


def make_augment_fn(cfg: Config):
    max_rad = cfg.aug_rotation_deg * math.pi / 180.0

    def augment(img, msk, seed):
        # seed: int64 tensor of shape [2]; derive independent sub-seeds
        seeds = tf.random.experimental.stateless_split(seed, num=4)
        flip = tf.random.stateless_uniform([], seeds[0]) < cfg.aug_flip_prob
        img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
        msk = tf.cond(flip, lambda: tf.image.flip_left_right(msk), lambda: msk)
        angle = tf.random.stateless_uniform([], seeds[1], -max_rad, max_rad)
        img = _rotate(img, angle, "BILINEAR")
        msk = _rotate(msk, angle, "NEAREST")
        b = tf.random.stateless_uniform([], seeds[2], 1 - cfg.aug_brightness, 1 + cfg.aug_brightness)
        img = img * b
        img = tf.image.stateless_random_contrast(img, 1 - cfg.aug_contrast, 1 + cfg.aug_contrast, seeds[3])
        return tf.clip_by_value(img, 0.0, 1.0), msk

    return augment


def make_dataset(cfg: Config, split, seed=0, manifest=None):
    """Returns (dataset, dataframe_of_files_in_order)."""
    manifest = load_manifest(cfg) if manifest is None else manifest
    df, img_paths, msk_paths = _paths(cfg, manifest, split)
    ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
    if split == "train":
        ds = ds.shuffle(len(img_paths), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        # a fresh pair of stateless seeds for every element, every epoch (reproducible given `seed`)
        try:  # TF >= 2.12: new random stream every epoch, still reproducible from `seed`
            seeds = tf.data.Dataset.random(seed=seed, rerandomize_each_iteration=True).batch(2)
        except TypeError:
            seeds = tf.data.Dataset.random(seed=seed).batch(2)
        ds = tf.data.Dataset.zip((ds, seeds))
        augment = make_augment_fn(cfg)
        ds = ds.map(lambda pair, s: augment(pair[0], pair[1], s), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, df


def load_arrays(cfg: Config, split, manifest=None):
    """Whole split as numpy arrays (N, H, W, 1) - used for evaluation."""
    manifest = load_manifest(cfg) if manifest is None else manifest
    df, img_paths, msk_paths = _paths(cfg, manifest, split)
    imgs, msks = [], []
    for ip, mp in zip(img_paths, msk_paths):
        i, m = _load(ip, mp)
        imgs.append(i.numpy())
        msks.append(m.numpy())
    return np.stack(imgs), np.stack(msks), df
