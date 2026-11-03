# Experiments and analyses of the revised manuscript

This folder holds every notebook and script behind the results of the revised manuscript
*"A quantum circuit in a hybrid U-Net adds no measurable benefit over a parameter-matched classical layer in mammogram
segmentation"*. The models, data preparation, circuit simulation and metrics are the `qseg` package at the top of this
repository; the notebooks load the same package from a copy in Google Drive (`QSEG_FINAL/code`).

No images, masks, model weights, manual outlines or patient identifiers are included. The hospital mammograms cannot be
shared (see the main README); the public external sets are DMID (https://doi.org/10.6084/m9.figshare.24522883) and
mini-MIAS with the expert-corrected masks (https://doi.org/10.5281/zenodo.10149914).

## Prespecified protocols (`protocols/`)

Each protocol was written to Google Drive with a UTC time stamp by the first cell of the training notebook or script,
before any of its runs was trained.

| File | Written (UTC) | Content |
|------|---------------|---------|
| `PROTOCOL_m3_five_seeds.txt` | 2026-10-18 06:54:13 | five seeds, three checkpoint policies (`scripts/p1_m3_robustness.py`) |
| `PROTOCOL_v2_E1_E2.txt` | 2026-10-24 06:25:30 | E1 (with skip connections) and E2 (without), primary contrasts |
| `PROTOCOL_v3_trig_control.txt` | 2026-10-25 15:27:35 | trigonometric control HQB-trig, E2 seeds 3–4, Holm family |
| `PROTOCOL_e3_vindr_prespecified.txt` | 2026-10-30 06:47:56 | E3: four variants trained on radiologist breast masks of VinDr-Mammo (five folds x five seeds), equivalence margin ±0.01 |

## Notebooks (`notebooks/`, Google Colab; outputs removed)

Run in this order. Paths at the top of each notebook refer to Google Drive (`MyDrive/QSEG_FINAL` read only,
`MyDrive/P1_v2` for outputs).

| Notebook | Runtime | Purpose |
|----------|---------|---------|
| `P1_16b_external_sets.ipynb` | CPU | DMID external set (rule-based masks) and mini-MIAS set |
| `P1_16c_mias_only.ipynb` | CPU | mini-MIAS with the expert breast + pectoral masks |
| `P1_17c_retrain_A.ipynb`, `_B` | T4 GPU | E1/E2 training and evaluation (24 runs); the two sessions share the work through lock files |
| `21_p1_consolidate.ipynb` | CPU | per-image results of the E1/E2 runs |
| `P1_17d_v3_A.ipynb`, `_B` | T4 GPU | HQB-trig (E1, E2) and E2 seeds 3–4 (16 runs) |
| `P1_24_consolidate.ipynb` | CPU | per-image results of all runs (`p1all_*.csv.gz`) |
| `P1_23_annot_score.ipynb` | CPU | rule and every run scored against the reader-1 outlines (84 test images) |
| `P1_25_reader2.ipynb` | CPU | reader-2 outlines (20 images): masks, inter-reader agreement, scores. The list of the 20 images holds coded patient identifiers and is read from a private file in Drive (`annot_test/reader2_selection.csv`) |
| `P1_27c_e3_vindr.ipynb` | CPU | E3 data: VinDr-Mammo images with the radiologist breast masks (596 images, 573 examinations), five folds by examination; requires credentialed PhysioNet access |
| `P1_27d_uef_mask_check.ipynb` | CPU | check that the radiologist masks exclude the pectoral muscle (target region) |
| `P1_27b_e3_data_mias.ipynb` | CPU | mini-MIAS inputs and expert masks for E3 |
| `P1_28r_e3_vindr_A.ipynb` … `_D` | T4 GPU | E3 training and evaluation (100 runs; resumable, sessions share the work through lock files). `P1_28v_*` are the same notebooks without resumption |
| `P1_27e_e3_reader2.ipynb` | CPU | reader-2 masks at model resolution and post-hoc reader-2 scores of all E3 runs from the stored probability maps |
| `P1_30_rule_vs_final.ipynb` | CPU | labelling rule against the final mini-MIAS expert masks |
| `P1_31_e3_analysis.ipynb` | CPU | prespecified E3 analysis; outputs in `results/e3/` |

## Scripts (`scripts/`, Google Colab)

| Script | Purpose |
|--------|---------|
| `p1_retrain.py` | retraining of the nine original hybrid runs (for the code-replacement test) |
| `p1_severing.py`, `p1_severing_v3.py` | code replacement (bottleneck severing): constant and swapped four-value code |
| `p1_padding_side_check.py` | image-only check of the padded side |
| `p1_m2_positive_control.py` | positive controls for the code-replacement test (skip connections gated by the code; the two lowest-resolution skip connections removed) |
| `p1_m3_robustness.py`, `p1_m3_collect.py`, `m3_stats.py` | five-seed analysis with three checkpoint policies |
| `p1_m3m4_analysis.py` | inference-only analyses of the earlier revision |

## Statistics, tables and Figure 4 (`analysis/`, local Python)

Run from this folder with the consolidated CSV files in `data/` (`p1all_{test,dmid,mias,runs,history}.csv.gz` from
`P1_24`; `data/v2/` for the E1/E2-only files from notebook 21; `data/annot/reader1/` and `data/annot/reader2/` from
`P1_23` and `P1_25`). These files contain coded patient identifiers and are therefore not published.

| Script | Output |
|--------|--------|
| `analysis_p1v2.py`, `extra_p1v2.py`, `fig4.py` | E1/E2 analysis (crossed hierarchical bootstrap over patients/images and seeds, 10,000 replicates, `rng(0)`, Holm) |
| `analysis_p1v3.py` | all contrasts of the manuscript (`contrasts_v3.csv`, `levels_v3.csv`, `shots_v3.csv`); reproduces the E1/E2 contrasts of `analysis_p1v2.py` exactly |
| `readers.py` | scores against reader 1 and reader 2, inter-reader agreement (`readers_stats.csv`) |
| `supp_tables_v3.py` | values of Supplementary Tables S11–S15 |
| `fig4_v3.py` | Figure 4 |

```bash
cd paper_v2/analysis
python analysis_p1v3.py && python readers.py && python supp_tables_v3.py && python fig4_v3.py
```

Environment: Python 3.11–3.13, NumPy 2, pandas 2, Matplotlib 3. Training notebooks: TensorFlow/Keras 2.20 on Google
Colab with an NVIDIA Tesla T4 GPU.
