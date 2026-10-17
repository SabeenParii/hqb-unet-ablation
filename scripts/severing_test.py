"""
# Bottleneck-severing test (inference only, no training) - Section II-H and Table 9 of the manuscript

Question: does the decoder of the hybrid models use the 4-value bottleneck code at all?
For every hybrid run (HQB-UNet, HQB-cls, HQB-free; seeds 0, 1, 2) the 4-value code z
(the input of post_dense) is replaced at test time by
  * const : the mean code over the TRAINING images (same z for every test image)
  * swap  : the code of a randomly chosen TRAINING image (fixed assignment, seed 0)
and the per-image Dice on the 84 test images is compared with the normal prediction.
Drop = Dice(normal) - Dice(severed). A clear positive drop means the code is used.

READ ONLY for QSEG_FINAL (code, weights, processed data). Raw images (530-images) are not
read. Weights are read from MyDrive/P1_severing/weights/<run>/ (backup copies) if present, else from
QSEG_FINAL/results/runs/<run>/. Outputs (numbers only, no images) go to MyDrive/P1_severing/.
Paths are those of the reported Google Colab runs (ROOT = the folder holding code/ and results/);
edit ROOT and OUT_DIR to run elsewhere. Needs the trained weights (results/runs/<run>/best.weights.h5),
which are not distributed with this repository.
Runtime: T4 GPU ~10-15 min; CPU ~45-60 min.
"""
# 1 · Drive, paths, imports
from google.colab import drive
drive.mount('/content/drive')
import os, sys, json
import numpy as np, pandas as pd

ROOT     = '/content/drive/MyDrive/QSEG_FINAL'
CODE_DIR = f'{ROOT}/code'
WORK_DIR = f'{ROOT}/results'                       # contains data/ and runs/ (read only)
OUT_DIR  = '/content/drive/MyDrive/P1_severing'
W_DIR    = f'{OUT_DIR}/weights'          # weights re-uploaded from the laptop backup (QSEG_FINAL copies were removed)
assert os.path.exists(f'{CODE_DIR}/qseg/models.py'), 'qseg package not found in QSEG_FINAL/code'
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
assert not OUT_DIR.startswith(ROOT)

import tensorflow as tf
from qseg.config import Config
from qseg.models import build_model
from qseg.datasets import load_manifest, load_arrays
from qseg.evaluation import per_image_overlap

cfg = Config(work_dir=WORK_DIR)
MODELS = ['hybrid_quantum', 'hybrid_classical', 'hybrid_none']
LABEL = {'hybrid_quantum': 'HQB-UNet', 'hybrid_classical': 'HQB-cls', 'hybrid_none': 'HQB-free'}
CODE_LAYER = {'hybrid_quantum': 'quantum_circuit', 'hybrid_classical': 'classical_replacement',
              'hybrid_none': 'pre_dense'}
SEEDS = (0, 1, 2)

# 2 · Data (processed 256x256 images and proxy masks, read only)
manifest = load_manifest(cfg)
x_tr, _, _ = load_arrays(cfg, 'train', manifest)
x_te, y_te, df_te = load_arrays(cfg, 'test', manifest)
print('train images', len(x_tr), '| test images', len(x_te), '| test patients', df_te.patient_id.nunique())
rng = np.random.default_rng(0)
swap_idx = rng.integers(0, len(x_tr), len(x_te))   # same assignment for every run


def dice_of(prob):
    y_true = (y_te > 0.5).astype(np.uint8)
    y_pred = (np.asarray(prob) >= cfg.threshold).astype(np.uint8)
    return per_image_overlap(y_true, y_pred)[:, 0]


def split_model(model, code_layer):
    """Encoder part: image -> (skip1..skip4, code). Decoder part: called layer by layer."""
    skips = [model.get_layer(f'enc{i}_drop').output for i in (1, 2, 3, 4)]
    enc = tf.keras.Model(model.input, skips + [model.get_layer(code_layer).output])

    def dec(skip_list, z):
        x = model.get_layer('post_dense')(z)
        x = model.get_layer('q_expand')(x)
        x = model.get_layer('q_reshape')(x)
        for i, s in enumerate(reversed(skip_list)):
            x = model.get_layer(f'up{i + 1}')(x)
            x = model.get_layer(f'concat{i + 1}')([x, s])
            for k in (1, 2):
                x = model.get_layer(f'dec{i + 1}_conv{k}')(x)
                x = model.get_layer(f'dec{i + 1}_bn{k}')(x, training=False)
                x = model.get_layer(f'dec{i + 1}_relu{k}')(x)
            x = model.get_layer(f'dec{i + 1}_drop')(x, training=False)
        return model.get_layer('output_mask')(x)
    return enc, dec


def severed_predictions(enc, dec, z_const, z_swap, bs=8):
    """Batch-wise: encoder once, then decoder with the normal, constant and swapped code."""
    p_norm, p_const, p_swap, z_all = [], [], [], []
    for a in range(0, len(x_te), bs):
        *sk, z = enc(tf.constant(x_te[a:a + bs]), training=False)
        n = z.shape[0]
        p_norm.append(dec(sk, z).numpy())
        p_const.append(dec(sk, tf.constant(np.repeat(z_const[None], n, 0), tf.float32)).numpy())
        p_swap.append(dec(sk, tf.constant(z_swap[a:a + n], tf.float32)).numpy())
        z_all.append(z.numpy())
    return (np.concatenate(p_norm), np.concatenate(p_const), np.concatenate(p_swap),
            np.concatenate(z_all))


# 3 · Severing for every hybrid run
rows, checks = [], []
for m in MODELS:
    for s in SEEDS:
        run = f'{WORK_DIR}/runs/{m}_seed{s}'
        tf.keras.backend.clear_session()
        model = build_model(m, cfg)
        wpath = f'{W_DIR}/{m}_seed{s}/best.weights.h5'
        if not os.path.exists(wpath):
            wpath = f'{run}/best.weights.h5'
        assert os.path.exists(wpath), f'weights not found for {m} seed {s}'
        print('weights:', wpath, round(os.path.getsize(wpath) / 1e6), 'MB')
        model.load_weights(wpath)
        enc, dec = split_model(model, CODE_LAYER[m])
        code_only = tf.keras.Model(model.input, model.get_layer(CODE_LAYER[m]).output)
        z_tr = code_only.predict(x_tr, batch_size=8, verbose=0)          # codes of the training images
        p_full = model.predict(x_te, batch_size=8, verbose=0)
        p_norm, p_const, p_swap, z_te = severed_predictions(enc, dec, z_tr.mean(0), z_tr[swap_idx])
        d_norm, d_const, d_swap = dice_of(p_norm), dice_of(p_const), dice_of(p_swap)
        # checks: the split model must reproduce the full model and the saved per-image Dice
        saved = pd.read_csv(f'{run}/per_image.csv').set_index('file_id').loc[df_te.file_id, 'dice'].to_numpy()
        chk = dict(model=m, seed=s, max_prob_diff=float(np.abs(p_full - p_norm).max()),
                   max_dice_diff_vs_saved=float(np.abs(d_norm - saved).max()),
                   code_sd_test=float(z_te.std(0).mean()), code_sd_train=float(z_tr.std(0).mean()))
        checks.append(chk); print(chk)
        assert chk['max_prob_diff'] < 1e-4, 'split model does not reproduce the full model'
        assert chk['max_dice_diff_vs_saved'] < 5e-3, 'normal Dice does not match the saved per_image.csv'
        for i, f in enumerate(df_te.file_id):
            rows.append(dict(model=m, seed=s, file_id=f, patient_id=df_te.patient_id[i],
                             dice_normal=d_norm[i], dice_const=d_const[i], dice_swap=d_swap[i]))
        print(f'{LABEL[m]} seed {s}: Dice normal {d_norm.mean():.4f} | const {d_const.mean():.4f} '
              f'| swap {d_swap.mean():.4f}')

per = pd.DataFrame(rows)
per.to_csv(f'{OUT_DIR}/severing_per_image.csv', index=False)
pd.DataFrame(checks).to_csv(f'{OUT_DIR}/severing_checks.csv', index=False)

# 4 · Hierarchical bootstrap over test patients and seeds (same scheme as the paper, 10,000 replicates)
upat = np.unique(df_te.patient_id)
pidx = [np.where(df_te.patient_id.to_numpy() == u)[0] for u in upat]


def hboot(a, n_boot=10000, seed=0):
    r = np.random.default_rng(seed); out = np.empty(n_boot)
    for t in range(n_boot):
        pp = r.integers(0, len(pidx), len(pidx)); ss = r.integers(0, a.shape[0], a.shape[0])
        out[t] = a[np.ix_(ss, np.concatenate([pidx[i] for i in pp]))].mean()
    return out


res = []
for m in MODELS:
    for mode in ('const', 'swap'):
        a = np.stack([(per[(per.model == m) & (per.seed == s)].set_index('file_id').loc[df_te.file_id]
                       .eval(f'dice_normal - dice_{mode}')).to_numpy() for s in SEEDS])
        b = hboot(a)
        p = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))
        res.append(dict(model=LABEL[m], mode=mode, drop=a.mean(), ci95_low=np.percentile(b, 2.5),
                        ci95_high=np.percentile(b, 97.5), p_boot=p,
                        **{f'drop_seed{s}': a[k].mean() for k, s in enumerate(SEEDS)}))
res = pd.DataFrame(res)
order = np.argsort(res.p_boot.to_numpy()); n = len(res); adj = np.empty(n); run_max = 0
for rank, i in enumerate(order):
    run_max = max(run_max, min(1.0, (n - rank) * res.p_boot[i])); adj[i] = run_max
res['p_holm'] = adj
res.to_csv(f'{OUT_DIR}/severing_summary.csv', index=False)
pd.set_option('display.width', 200)
print(res.round(4).to_string(index=False))
print('saved to', OUT_DIR)
