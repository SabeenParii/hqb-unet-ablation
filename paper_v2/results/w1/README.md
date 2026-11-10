# Experiment W1: results (faster-learning four-to-four mapping)

W1 retrained HQB-UNet (`hq`), HQB-cls (`hc`) and HQB-trig (`ht`) in the E1 setting (hospital split, skip connections,
seeds 0-4; 15 runs) with every weight of the four-to-four mapping stored as v = w/100 and used as w = 100 v, so the
mapping learns with an effective learning rate 100 times that of the rest of the network. All three models start from
the initial weights of the HQB-UNet model of the same seed in every shared layer (checked before training).
Protocol: `../../protocols/PROTOCOL_w1_prespecified.txt` (written 2026-11-06 17:56:50 UTC, before the first run).
Training: notebooks `P1_32_w1_A.ipynb` and `P1_32_w1_B.ipynb` (two Colab T4 sessions sharing the runs through lock files).
Analysis: notebook `P1_33_w1_analysis.ipynb` (run 2026-11-07 06:40 UTC).

| File | Content |
|------|---------|
| `w1_results.csv` | All W1 estimates: primary family (W1 HQB-UNet minus W1 HQB-cls and minus W1 HQB-trig, test Dice, Holm over the two, `p_holm`); secondary contrasts on DMID; each W1 model minus its E1 counterpart; E1 reference contrasts; code-replacement drops (`drop_const`, `drop_swapavg`); model levels; per seed and leaving one seed out; mini-MIAS (descriptive). `verdict`: `difference` when the 95% CI excludes zero, `equivalent` when it lies within +/-0.01 Dice |
| `w1_weight_change.csv` | change of the mapping and of its neighbouring layers (`pre_dense`, `post_dense`) from initialization to the tested weights, in effective units (circuit angles wrapped to (-pi, pi]); epochs, minutes and Dice of each run |
| `runs_w1_summary.csv` | one row per run: epochs, best epoch, final learning rate, minutes, parameters, validation, test, DMID and mini-MIAS Dice, code-replacement drops, check of the code switch, Colab session, largest change of a mapping weight |

Bootstrap: crossed hierarchical bootstrap over clusters (patients; DMID: images; mini-MIAS: women) and seeds, 10,000
replicates, `numpy.random.default_rng(0)` for every estimate. On mini-MIAS, `dice_expert_asis` is the verified
orientation of the expert masks; `dice_expert_transposed` is shown as a check only.
Hospital values are aggregates only; per-image hospital scores are not included.
