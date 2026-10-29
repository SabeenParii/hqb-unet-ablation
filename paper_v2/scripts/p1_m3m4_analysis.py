"""
Paper 1 · Reviewer analyses M3 + M4 (inference / arithmetic only, NO training)

M4  Paired comparison of every model with the no-learning rule on the same 84 test images:
    per-image Dice(model) - Dice(rule), hierarchical bootstrap over test patients (crossed with seeds;
    the rule is deterministic), 10,000 replicates. Rule = proxy-labelling rule applied to the 256x256
    test inputs with a 3x3 and a 5x5 structuring element. Self-check: the 3x3 rule must reproduce the
    reported per-image Dice 0.968.
M3  Distribution of the severing effects for the nine hybrid runs: per-image |dDice|, mean absolute
    probability change, fraction of pixels whose thresholded label flips, patient-level drops, and the
    spread of the swap result over 5 different random swap assignments.

READ ONLY for QSEG_FINAL. Outputs: small CSV files (numbers only, no images) in MyDrive/P1_severing/m3m4.
Runtime on a T4 GPU: about 15 min.
"""
from google.colab import drive
drive.mount('/content/drive')
import os, sys
import numpy as np, pandas as pd, cv2

ROOT = '/content/drive/MyDrive/QSEG_FINAL'
WORK = f'{ROOT}/results'
OUT = '/content/drive/MyDrive/P1_severing/m3m4'
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f'{ROOT}/code')
import tensorflow as tf
from qseg.config import Config
from qseg.models import build_model
from qseg.datasets import load_manifest, load_arrays
from qseg.evaluation import per_image_overlap
from qseg.data_prep import generate_breast_mask

cfg = Config(work_dir=WORK)
man = load_manifest(cfg)
x_tr, _, _ = load_arrays(cfg, 'train', man)
x_te, y_te, df_te = load_arrays(cfg, 'test', man)
YT = (y_te > 0.5).astype(np.uint8)
pid = df_te.patient_id.to_numpy()
upat = np.unique(pid)
pidx = [np.where(pid == u)[0] for u in upat]
print('test images', len(x_te), '| patients', len(upat))


def dice_of(pred_bin):
    return per_image_overlap(YT, pred_bin.astype(np.uint8))[:, 0]


def hboot(a, n_boot=10000, seed=0):
    """a: (n_seeds, n_images). Crossed resampling: patients and seeds, jointly for all images."""
    r = np.random.default_rng(seed); out = np.empty(n_boot)
    for t in range(n_boot):
        pp = r.integers(0, len(pidx), len(pidx)); ss = r.integers(0, a.shape[0], a.shape[0])
        out[t] = a[np.ix_(ss, np.concatenate([pidx[i] for i in pp]))].mean()
    return out


def summ(a, label):
    b = hboot(a)
    p = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())); p = max(p, 1 / len(b))
    return dict(comparison=label, mean_diff=a.mean(), ci95_low=np.percentile(b, 2.5),
                ci95_high=np.percentile(b, 97.5), p_boot=p,
                **{f'seed{s}': a[s].mean() for s in range(a.shape[0])})


# ------------------------------------------------------------------ M4: rule baseline, paired
rule = {}
for k in (3, 5):
    pred = np.stack([(generate_breast_mask((x[..., 0] * 255).round().astype(np.uint8), k)[0] > 0)
                     for x in x_te])
    rule[k] = dice_of(pred)
    print(f'rule {k}x{k}: per-image Dice {rule[k].mean():.4f} ± {rule[k].std(ddof=1):.4f}')
assert abs(rule[3].mean() - 0.968) < 0.002, 'rule 3x3 does not reproduce the reported 0.968 - STOP and report'

MODELS = {'unet': 'C-UNet', 'attention_unet': 'Att-UNet', 'hybrid_quantum': 'HQB-UNet',
          'hybrid_classical': 'HQB-cls', 'hybrid_none': 'HQB-free'}
rows = []
for m, lab in MODELS.items():
    d = np.stack([pd.read_csv(f'{WORK}/runs/{m}_seed{s}/per_image.csv').set_index('file_id')
                  .loc[df_te.file_id, 'dice'].to_numpy() for s in (0, 1, 2)])
    for k in (3, 5):
        rows.append(summ(d - rule[k][None, :], f'{lab} - rule {k}x{k}'))
m4 = pd.DataFrame(rows)
pd.DataFrame({'file_id': df_te.file_id, 'patient_id': pid, 'rule3': rule[3], 'rule5': rule[5]}) \
    .to_csv(f'{OUT}/rule_per_image.csv', index=False)
m4.to_csv(f'{OUT}/m4_model_minus_rule.csv', index=False)
pd.set_option('display.width', 220)
print('\nM4  model - rule (paired, crossed patient x seed bootstrap)')
print(m4.round(4).to_string(index=False))

# ------------------------------------------------------------------ M3: severing effect distribution
CODE = {'hybrid_quantum': 'quantum_circuit', 'hybrid_classical': 'classical_replacement', 'hybrid_none': 'pre_dense'}


def split_model(model, code_layer):
    skips = [model.get_layer(f'enc{i}_drop').output for i in (1, 2, 3, 4)]
    enc = tf.keras.Model(model.input, skips + [model.get_layer(code_layer).output])

    def dec(sk, z):
        x = model.get_layer('post_dense')(z)
        x = model.get_layer('q_expand')(x)
        x = model.get_layer('q_reshape')(x)
        for i, s in enumerate(reversed(sk)):
            x = model.get_layer(f'up{i + 1}')(x)
            x = model.get_layer(f'concat{i + 1}')([x, s])
            for k in (1, 2):
                x = model.get_layer(f'dec{i + 1}_conv{k}')(x)
                x = model.get_layer(f'dec{i + 1}_bn{k}')(x, training=False)
                x = model.get_layer(f'dec{i + 1}_relu{k}')(x)
            x = model.get_layer(f'dec{i + 1}_drop')(x, training=False)
        return model.get_layer('output_mask')(x)
    return enc, dec


per, runs = [], []
for m, cl in CODE.items():
    for s in (0, 1, 2):
        tf.keras.backend.clear_session()
        model = build_model(m, cfg)
        model.load_weights(f'{WORK}/runs/{m}_seed{s}/best.weights.h5')
        enc, dec = split_model(model, cl)
        z_tr = tf.keras.Model(model.input, model.get_layer(cl).output).predict(x_tr, batch_size=8, verbose=0)
        swaps = [np.random.default_rng(k).integers(0, len(x_tr), len(x_te)) for k in range(5)]  # k=0 = paper
        P = {'normal': [], 'const': [], **{f'swap{k}': [] for k in range(5)}}
        for a in range(0, len(x_te), 8):
            *sk, z = enc(tf.constant(x_te[a:a + 8]), training=False); n = z.shape[0]
            P['normal'].append(dec(sk, z).numpy())
            P['const'].append(dec(sk, tf.constant(np.repeat(z_tr.mean(0)[None], n, 0), tf.float32)).numpy())
            for k in range(5):
                P[f'swap{k}'].append(dec(sk, tf.constant(z_tr[swaps[k][a:a + n]], tf.float32)).numpy())
        P = {k: np.concatenate(v)[..., 0] for k, v in P.items()}
        bn = P['normal'] >= cfg.threshold
        d_norm = dice_of(bn)
        saved = pd.read_csv(f'{WORK}/runs/{m}_seed{s}/per_image.csv').set_index('file_id').loc[df_te.file_id, 'dice'].to_numpy()
        assert np.abs(d_norm - saved).max() < 5e-3, 'normal Dice does not match saved per_image.csv - STOP'
        for mode in ['const'] + [f'swap{k}' for k in range(5)]:
            bs = P[mode] >= cfg.threshold
            d_sev = dice_of(bs)
            for i in range(len(x_te)):
                per.append(dict(model=MODELS[m], seed=s, mode=mode, file_id=df_te.file_id[i], patient_id=pid[i],
                                dice_normal=d_norm[i], dice_sev=d_sev[i], drop=d_norm[i] - d_sev[i],
                                mean_abs_dprob=float(np.abs(P['normal'][i] - P[mode][i]).mean()),
                                frac_pixels_flipped=float((bn[i] != bs[i]).mean())))
        print(f'{MODELS[m]} seed {s} done')

per = pd.DataFrame(per)
per.to_csv(f'{OUT}/m3_per_image.csv', index=False)

per['absdrop'] = per.drop.abs()
main = per[per['mode'].isin(['const', 'swap0'])].replace({'mode': {'swap0': 'swap'}})
g = main.groupby(['model', 'mode'])
m3 = pd.DataFrame({
    'mean_drop': g.drop.mean(), 'mean_abs_drop': g.absdrop.mean(), 'median_abs_drop': g.absdrop.median(),
    'p95_abs_drop': g.absdrop.quantile(0.95), 'max_abs_drop': g.absdrop.max(), 'min_drop': g.drop.min(),
    'max_drop': g.drop.max(), 'frac_images_absdrop_gt_0.01': g.absdrop.apply(lambda v: (v > 0.01).mean()),
    'frac_images_absdrop_gt_0.05': g.absdrop.apply(lambda v: (v > 0.05).mean()),
    'mean_abs_dprob': g.mean_abs_dprob.mean(), 'mean_frac_pixels_flipped': g.frac_pixels_flipped.mean(),
    'max_frac_pixels_flipped': g.frac_pixels_flipped.max()}).reset_index()
pat = main.groupby(['model', 'mode', 'patient_id']).drop.mean().groupby(['model', 'mode']).agg(['min', 'max'])
m3 = m3.merge(pat.rename(columns={'min': 'patient_mean_drop_min', 'max': 'patient_mean_drop_max'}).reset_index(),
              on=['model', 'mode'])
sw = per[per['mode'].str.startswith('swap')].groupby(['model', 'mode']).drop.mean().groupby('model') \
    .agg(['min', 'max', 'std']).rename(columns=lambda c: f'swap5_mean_drop_{c}').reset_index()
m3.to_csv(f'{OUT}/m3_summary.csv', index=False)
sw.to_csv(f'{OUT}/m3_swap_realisations.csv', index=False)
print('\nM3  distribution of severing effects (all 3 seeds pooled, 84 images each)')
print(m3.round(4).to_string(index=False))
print('\nM3  mean drop over 5 different swap assignments')
print(sw.round(4).to_string(index=False))
print('\nSaved to', OUT)
