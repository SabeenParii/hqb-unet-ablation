"""Evaluation on the full test set (numpy only).

The original notebooks reported Keras metrics averaged over mini-batches, with Dice/IoU
computed on soft probabilities but precision/recall on thresholded predictions. That is
why the reported Dice did not equal the F1 of the reported precision and recall.

Here every overlap metric comes from ONE pixel-level confusion matrix over the whole test
set (threshold 0.5), so Dice == F1 and IoU == Dice / (2 - Dice) by construction.
Per-image Dice/IoU are reported as mean +/- SD, and 95% CIs come from resampling PATIENTS.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def confusion(y_true, y_pred):
    t = np.asarray(y_true).astype(bool).ravel()
    p = np.asarray(y_pred).astype(bool).ravel()
    tp = int(np.sum(t & p))
    tn = int(np.sum(~t & ~p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    return tp, tn, fp, fn


def metrics_from_confusion(tp, tn, fp, fn):
    tp, tn, fp, fn = map(float, (tp, tn, fp, fn))
    div = lambda a, b: a / b if b > 0 else float("nan")
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(
        accuracy=div(tp + tn, tp + tn + fp + fn),
        precision=div(tp, tp + fp),
        recall=div(tp, tp + fn),              # = sensitivity
        specificity=div(tn, tn + fp),
        dice=div(2 * tp, 2 * tp + fp + fn),   # = F1
        iou=div(tp, tp + fp + fn),
        mcc=div(tp * tn - fp * fn, denom),
    )


def per_image_overlap(y_true, y_pred):
    """Dice and IoU per image; an image with empty truth and empty prediction scores 1."""
    rows = []
    for t, p in zip(y_true, y_pred):
        tp, tn, fp, fn = confusion(t, p)
        if tp + fp + fn == 0:
            rows.append((1.0, 1.0))
        else:
            rows.append((2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn)))
    return np.array(rows)


def expected_calibration_error(y_true, prob, n_bins=15):
    t = np.asarray(y_true).ravel().astype(float)
    p = np.asarray(prob).ravel().astype(float)
    conf = np.where(p >= 0.5, p, 1 - p)              # confidence of the predicted class
    correct = ((p >= 0.5) == (t > 0.5)).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def pixel_aucs(y_true, prob, max_pixels=2_000_000, seed=0):
    t = np.asarray(y_true).ravel()
    p = np.asarray(prob).ravel()
    if t.size > max_pixels:
        sel = np.random.default_rng(seed).choice(t.size, max_pixels, replace=False)
        t, p = t[sel], p[sel]
    if t.min() == t.max():
        return float("nan"), float("nan")
    return float(roc_auc_score(t, p)), float(average_precision_score(t, p))


def patient_bootstrap(y_true, y_pred, patient_ids, n_boot=2000, seed=0):
    """95% CIs for the global metrics by resampling patients with replacement."""
    rng = np.random.default_rng(seed)
    pids = np.asarray(patient_ids)
    uniq = np.unique(pids)
    # confusion matrix per patient, then sum -> fast resampling
    per_patient = np.array([np.sum([confusion(y_true[i], y_pred[i]) for i in np.where(pids == u)[0]], axis=0)
                            for u in uniq])
    samples = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        samples.append(metrics_from_confusion(*per_patient[pick].sum(axis=0)))
    df = pd.DataFrame(samples)
    return {k: (float(np.nanpercentile(df[k], 2.5)), float(np.nanpercentile(df[k], 97.5))) for k in df.columns}


def evaluate_predictions(y_true, prob, patient_ids, threshold=0.5, n_boot=2000, auc_max_pixels=2_000_000,
                         ece_bins=15, seed=0):
    """Returns (summary dict, per-image DataFrame)."""
    y_true = (np.asarray(y_true) > 0.5).astype(np.uint8)
    prob = np.asarray(prob, dtype=np.float32)
    y_pred = (prob >= threshold).astype(np.uint8)
    tp, tn, fp, fn = confusion(y_true, y_pred)
    summary = metrics_from_confusion(tp, tn, fp, fn)
    summary.update(tp=tp, tn=tn, fp=fp, fn=fn)
    per_img = per_image_overlap(y_true, y_pred)
    summary.update(dice_per_image_mean=float(per_img[:, 0].mean()), dice_per_image_sd=float(per_img[:, 0].std(ddof=1)),
                   iou_per_image_mean=float(per_img[:, 1].mean()), iou_per_image_sd=float(per_img[:, 1].std(ddof=1)))
    summary["roc_auc"], summary["pr_auc"] = pixel_aucs(y_true, prob, auc_max_pixels, seed)
    summary["ece"] = expected_calibration_error(y_true, prob, ece_bins)
    if n_boot:
        for k, (lo, hi) in patient_bootstrap(y_true, y_pred, patient_ids, n_boot, seed).items():
            summary[f"{k}_ci_low"], summary[f"{k}_ci_high"] = lo, hi
    per_image = pd.DataFrame({"patient_id": patient_ids, "dice": per_img[:, 0], "iou": per_img[:, 1]})
    return summary, per_image
