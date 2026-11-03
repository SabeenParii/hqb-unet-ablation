# Experiment E3: results (training on radiologist breast masks)

Outputs of notebook `P1_31_e3_analysis.ipynb` (run 2026-11-03 16:51 UTC, commit 96115fe) for the 100 E3 runs
(HQB-UNet `hq`, HQB-cls `hc`, HQB-trig `ht`, HQB-free `hf`; five folds of VinDr-Mammo by examination x seeds 0-4).
Protocol: `../../protocols/PROTOCOL_e3_vindr_prespecified.txt` (written 2026-10-30 06:47:56 UTC, before the first run).

| File | Content |
|------|---------|
| `contrasts.csv` | HQB-UNet minus each control: estimate, 95% and 90% CI, bootstrap P, Holm-adjusted P, resolved, equivalence within +/-0.01. Sets: `vindr_test` (primary, out-of-fold, 596 images, 573 examinations), `mini_mias` (322 images, 161 women), `hospital` (84 test images, 20 patients; reader 1 and rule-based masks), `hospital_reader2` (20 images). `region`: `target` = breast without pectoral muscle |
| `levels.csv` | per-variant mean per-image Dice with CIs; code-replacement drops (`drop_const`, `drop_swapavg`) on VinDr-Mammo |
| `shots.csv` | finite-shot readout of the HQB-UNet (100, 1,000, 10,000 shots per expectation value) |
| `seed_sensitivity.csv` | t interval over seed-level differences, seed-only resampling, leave one seed out |
| `by_fold_seed.csv` | primary contrasts in every fold x seed cell, per seed and per fold (`n` = image-seed pairs) |
| `resumed_sensitivity.csv` | primary contrasts without the nine fold-seed cells that contain a run resumed after a session interruption |
| `run_metadata_corrected.csv` | one row per run: epochs, best epoch, final learning rate, minutes, epochs at which the run was resumed, TF version; `corrected` marks the two runs whose epoch counts were corrected |
| `e3v_run_corrections.csv`, `README_corrections.txt` | the corrections and the audit of all resumed runs, recorded during training before any analysis |

Bootstrap: crossed hierarchical bootstrap over clusters and seeds, 10,000 replicates, `numpy.random.default_rng(0)`.
Hospital values are aggregates only. Per-image scores are not included: VinDr-Mammo scores are keyed to identifiers of a
credentialed-access dataset, and hospital scores fall under the permission for aggregate results only.
