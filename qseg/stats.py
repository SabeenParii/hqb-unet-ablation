"""Aggregate runs and compare models (numpy / scipy / pandas only).

* Mean +/- SD over seeds for every metric.
* Paired comparison on per-image Dice (each image's Dice averaged over seeds) with the
  Wilcoxon signed-rank test and Holm correction for multiple comparisons.
"""
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .config import MODELS

KEY_METRICS = ["dice", "iou", "mcc", "accuracy", "precision", "recall", "specificity",
               "dice_per_image_mean", "roc_auc", "pr_auc", "ece", "best_epoch", "epochs_run",
               "total_params", "size_mb", "inference_ms_per_image", "train_seconds"]


def load_runs(work_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(work_dir, "runs", "*", "summary.json"))):
        with open(p) as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)


def summary_table(runs, metrics=KEY_METRICS, decimals=4):
    g = runs.groupby("model")
    out = {}
    for m in metrics:
        if m in runs:
            mean, sd = g[m].mean(), g[m].std(ddof=1)
            out[m] = [f"{a:.{decimals}f} ± {b:.{decimals}f}" if np.isfinite(b) else f"{a:.{decimals}f}"
                      for a, b in zip(mean, sd)]
    table = pd.DataFrame(out, index=mean.index)
    order = [k for k in MODELS if k in table.index]
    table = table.loc[order]
    table.insert(0, "n_runs", g.size().loc[order])
    table.index = [MODELS[k] for k in order]
    return table


def per_image_dice(work_dir):
    frames = []
    for p in glob.glob(os.path.join(work_dir, "runs", "*", "per_image.csv")):
        name, seed = os.path.basename(os.path.dirname(p)).rsplit("_seed", 1)
        d = pd.read_csv(p)
        d["model"], d["seed"] = name, int(seed)
        frames.append(d)
    df = pd.concat(frames)
    return df.groupby(["model", "file_id"])["dice"].mean().unstack(0)  # rows: images, cols: models


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_tests(dice_wide, pairs=None):
    models = [m for m in MODELS if m in dice_wide.columns]
    pairs = pairs or list(itertools.combinations(models, 2))
    rows = []
    for a, b in pairs:
        d = dice_wide[[a, b]].dropna()
        diff = d[a] - d[b]
        stat, p = wilcoxon(d[a], d[b]) if np.any(diff != 0) else (np.nan, 1.0)
        rows.append(dict(model_a=MODELS[a], model_b=MODELS[b], n_images=len(d),
                         median_diff=float(np.median(diff)), mean_diff=float(diff.mean()),
                         wilcoxon_stat=stat, p_value=p))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"])
    return out
