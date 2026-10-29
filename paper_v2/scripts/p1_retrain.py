"""
# Paper 1 · Retrain the nine hybrid runs (backup for the severing test)

The original weights of the hybrid runs were permanently deleted from Drive on 28 Sep 2026.
This script retrains HQB-UNet, HQB-cls and HQB-free with seeds 0, 1, 2 using the SAME code
(QSEG_FINAL/code), the SAME processed data and split, and the SAME settings (Config defaults:
50 epochs max, early stopping patience 10 on val_loss, Adam 1e-4, batch 16, tf circuit backend).

READ ONLY for QSEG_FINAL. The processed data are copied to the Colab VM (/content/p1w/data);
all outputs (weights, history, per-image results) go to MyDrive/P1_retrain/runs/.
RESUMABLE: finished runs (summary.json present) are skipped, so after a disconnect just run
the same cell again. Runtime on T4: ~20-25 min per run, ~3-3.5 h for all nine.
"""
from google.colab import drive
drive.mount('/content/drive')
import os, sys, json, shutil
import pandas as pd

ROOT     = '/content/drive/MyDrive/QSEG_FINAL'
CODE_DIR = f'{ROOT}/code'
SRC_DATA = f'{ROOT}/results/data'
SRC_RUNS = f'{ROOT}/results/runs'
OUT      = '/content/drive/MyDrive/P1_retrain'
LOCAL    = '/content/p1w'
MODELS   = ['hybrid_quantum', 'hybrid_classical', 'hybrid_none']

os.makedirs(f'{OUT}/runs', exist_ok=True)
os.makedirs(LOCAL, exist_ok=True)
if not os.path.exists(f'{LOCAL}/data/manifest.csv'):
    print('copying processed data to the Colab VM ...')
    shutil.copytree(SRC_DATA, f'{LOCAL}/data', dirs_exist_ok=True)
if not os.path.islink(f'{LOCAL}/runs'):
    os.symlink(f'{OUT}/runs', f'{LOCAL}/runs')          # run outputs are written straight to Drive
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import tensorflow as tf
from qseg.config import Config
from qseg.train import run_all

cfg = Config(work_dir=LOCAL)
print('GPU:', tf.config.list_physical_devices('GPU'))
print('settings:', dict(max_epochs=cfg.max_epochs, patience=cfg.early_stop_patience, lr=cfg.learning_rate,
                        batch=cfg.batch_size, seeds=cfg.seeds, backend=cfg.quantum_backend))
assert cfg.quantum_backend == 'tf' and tuple(cfg.seeds) == (0, 1, 2) and cfg.max_epochs == 50

run_all(cfg, model_names=MODELS)
for f in ('train_config.json', 'all_runs.csv'):
    if os.path.exists(f'{LOCAL}/{f}'):
        shutil.copy(f'{LOCAL}/{f}', f'{OUT}/{f}')

# Comparison with the original runs (from their saved summary.json)
rows = []
for m in MODELS:
    for s in (0, 1, 2):
        o = json.load(open(f'{SRC_RUNS}/{m}_seed{s}/summary.json'))
        n = json.load(open(f'{OUT}/runs/{m}_seed{s}/summary.json'))
        rows.append(dict(model=m, seed=s,
                         dice_img_orig=o['dice_per_image_mean'], dice_img_new=n['dice_per_image_mean'],
                         best_epoch_orig=o['best_epoch'], best_epoch_new=n['best_epoch'],
                         epochs_orig=o['epochs_run'], epochs_new=n['epochs_run']))
cmp = pd.DataFrame(rows)
cmp['diff'] = cmp.dice_img_new - cmp.dice_img_orig
cmp.to_csv(f'{OUT}/retrain_vs_original.csv', index=False)
pd.set_option('display.width', 200)
print(cmp.round(4).to_string(index=False))
print('mean per model:'); print(cmp.groupby('model')[['dice_img_orig', 'dice_img_new', 'diff']].mean().round(4))
print('ALL NINE RUNS DONE ->', OUT)
