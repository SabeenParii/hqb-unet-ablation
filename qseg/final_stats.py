"""Statistical analysis reported in the manuscript (numpy / scipy / pandas only).

Works on the saved run outputs (results/runs/<model>_seed<k>/per_image.csv); no model is
re-run. Implements
* the hierarchical bootstrap over test patients AND seeds (primary analysis),
* the patient-only bootstrap (seeds fixed),
* per-seed differences,
* 90% CIs, the smallest equivalence bound and TOST at +/-0.005,
* the two-sided Wilcoxon signed-rank test on per-patient mean Dice (n = 20) with Holm
  correction, and the image-level version (n = 84) for comparison.

Defaults (10,000 replicates, seed 0 for the hierarchical bootstrap and per-model CIs,
seed 1 for the patient-only bootstrap) reproduce the values in the manuscript.

Command line:  python -m qseg.final_stats /path/to/results [out.txt]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

MODELS = ["unet", "attention_unet", "hybrid_quantum", "hybrid_classical", "hybrid_none"]
LABELS = dict(unet="C-UNet", attention_unet="Att-UNet", hybrid_quantum="HQB-UNet",
              hybrid_classical="HQB-cls", hybrid_none="HQB-free")
PAIRS = [("unet", "attention_unet"), ("unet", "hybrid_quantum"), ("attention_unet", "hybrid_quantum"),
         ("hybrid_quantum", "hybrid_classical"), ("hybrid_quantum", "hybrid_none"),
         ("hybrid_none", "hybrid_classical")]
SEEDS = (0, 1, 2)


def load_per_image(results_dir, models=MODELS, seeds=SEEDS):
    """Per-image Dice as {model: array (n_seeds, n_images)}, plus file ids and patient ids.

    Images are sorted by file_id, and every run must contain the same test images."""
    files = pats = None
    dice = {}
    for m in models:
        rows = []
        for s in seeds:
            p = os.path.join(results_dir, "runs", f"{m}_seed{s}", "per_image.csv")
            d = pd.read_csv(p).sort_values("file_id")
            if files is None:
                files, pats = d.file_id.to_numpy(), d.patient_id.to_numpy()
            if not (d.file_id.to_numpy() == files).all():
                raise ValueError(f"{p}: test images differ from the first run")
            rows.append(d.dice.to_numpy())
        dice[m] = np.stack(rows)
    return dice, files, pats


def patient_index(patient_ids):
    """(unique patient ids, list of image-index arrays, one per patient)."""
    upat = np.unique(patient_ids)
    return upat, [np.where(np.asarray(patient_ids) == u)[0] for u in upat]


def hierarchical_bootstrap(a, b=None, pidx=None, n_boot=10000, seed=0, resample_seeds=True):
    """Bootstrap distribution of mean(a) - mean(b) (or of mean(a) if b is None).

    a, b: arrays (n_seeds, n_images). Each replicate resamples the patients with replacement
    and, if resample_seeds, the seeds with replacement; the same resampled patients and
    seeds are used for a and b. The statistic is the mean over all images of the resampled
    patients and seeds."""
    rng = np.random.default_rng(seed)
    n_seeds = a.shape[0]
    res = np.empty(n_boot)
    for t in range(n_boot):
        pp = rng.integers(0, len(pidx), len(pidx))
        ss = rng.integers(0, n_seeds, n_seeds) if resample_seeds else np.arange(n_seeds)
        idx = np.concatenate([pidx[i] for i in pp])
        va = a[np.ix_(ss, idx)].mean()
        res[t] = va - (b[np.ix_(ss, idx)].mean() if b is not None else 0.0)
    return res


def model_cis(dice, pidx, n_boot=10000, seed=0):
    """Mean per-image Dice of every model with its hierarchical-bootstrap 95% CI."""
    rows = []
    for m, a in dice.items():
        r = hierarchical_bootstrap(a, None, pidx, n_boot, seed)
        rows.append(dict(model=LABELS.get(m, m), per_image_dice=a.mean(),
                         ci95_low=np.percentile(r, 2.5), ci95_high=np.percentile(r, 97.5)))
    return pd.DataFrame(rows)


def compare_pairs(dice, pidx, pairs=PAIRS, n_boot=10000, seed=0, patient_only_seed=1, margin=0.005):
    """Differences between models: hierarchical 95%/90% CIs, smallest equivalence bound,
    TOST at +/-margin (equivalent if the 90% CI lies inside the margin), patient-only 95% CI,
    and per-seed differences."""
    rows = []
    for a, b in pairs:
        r = hierarchical_bootstrap(dice[a], dice[b], pidx, n_boot, seed)
        r2 = hierarchical_bootstrap(dice[a], dice[b], pidx, n_boot, patient_only_seed, resample_seeds=False)
        lo90, hi90 = np.percentile(r, [5, 95])
        per_seed = (dice[a] - dice[b]).mean(axis=1)
        rows.append(dict(
            pair=f"{LABELS.get(a, a)} - {LABELS.get(b, b)}", mean_diff=(dice[a] - dice[b]).mean(),
            ci95_low=np.percentile(r, 2.5), ci95_high=np.percentile(r, 97.5), ci90_low=lo90, ci90_high=hi90,
            smallest_equivalence_bound=max(abs(lo90), abs(hi90)),
            tost_equivalent=bool(lo90 > -margin and hi90 < margin),
            patient_only_ci95_low=np.percentile(r2, 2.5), patient_only_ci95_high=np.percentile(r2, 97.5),
            **{f"diff_seed{k}": v for k, v in enumerate(per_seed)}))
    return pd.DataFrame(rows)


def holm(pvals):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def _wilcoxon_p(x, y):
    """Two-sided Wilcoxon signed-rank p-value; 1.0 if all paired differences are zero."""
    x, y = np.asarray(x), np.asarray(y)
    return 1.0 if np.all(x == y) else float(wilcoxon(x, y).pvalue)


def wilcoxon_tests(dice, pidx, pairs=PAIRS):
    """Two-sided Wilcoxon signed-rank tests with Holm correction over the pairs:
    patient level (per-patient mean Dice over seeds and images, n = number of patients) and
    image level (per-image Dice averaged over seeds, n = number of images)."""
    rows = []
    for a, b in pairs:
        da = np.array([dice[a][:, ix].mean() for ix in pidx])
        db = np.array([dice[b][:, ix].mean() for ix in pidx])
        rows.append(dict(pair=f"{LABELS.get(a, a)} - {LABELS.get(b, b)}", n_patients=len(pidx),
                         patients_first_better=int((da > db).sum()),
                         p_patient=_wilcoxon_p(da, db),
                         n_images=dice[a].shape[1],
                         p_image=_wilcoxon_p(dice[a].mean(0), dice[b].mean(0))))
    out = pd.DataFrame(rows)
    out["p_patient_holm"] = holm(out.p_patient)
    out["p_image_holm"] = holm(out.p_image)
    return out


def run_all(results_dir, n_boot=10000, seed=0):
    """All analyses as a dict of DataFrames."""
    dice, files, pats = load_per_image(results_dir)
    _, pidx = patient_index(pats)
    return dict(models=model_cis(dice, pidx, n_boot, seed),
                pairs=compare_pairs(dice, pidx, n_boot=n_boot, seed=seed),
                wilcoxon=wilcoxon_tests(dice, pidx))


if __name__ == "__main__":
    res = run_all(sys.argv[1])
    text = "\n\n".join(f"## {k}\n{v.to_string(index=False)}" for k, v in res.items())
    print(text)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            f.write(text + "\n")
