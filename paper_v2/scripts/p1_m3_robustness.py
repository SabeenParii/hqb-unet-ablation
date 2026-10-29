"""
Paper 1 · M3 robustness analysis (training-policy and seed sensitivity of the circuit contrasts)

PRE-SPECIFIED on 18 October 2026, before any of these runs was trained or any new test result was seen.
  Models : hybrid_quantum (HQB-UNet), hybrid_classical (HQB-cls), hybrid_none (HQB-free)
  Seeds  : 0, 1, 2 (re-runs) and 3, 4 (new); all 15 runs are reported, none is discarded.
  Training: shared settings of the paper (Adam 1e-4, ReduceLROnPlateau x0.5 / patience 4 / min 1e-7, batch 16,
            BCE + soft Dice, fp32, same augmentation and seed handling), 50 epochs WITHOUT early stopping.
  From the single trajectory of each run, three checkpoints are evaluated on the 84 test images:
    A   'es'    : the weights the original rule would have tested (early stopping on val_loss, patience 10,
                  best weights restored) - reproduces the paper's policy
    B1  'final' : epoch-50 weights (fixed training budget)                        <- PRIMARY for M3
    B2  'min20' : lowest val_loss among epochs 20-50 (minimum training period, then validation selection)
  Primary estimand: mean per-image Dice of HQB-UNet - HQB-cls and HQB-UNet - HQB-free under B1,
  5 seeds x 20 test patients, crossed hierarchical bootstrap (10,000 replicates). A and B2 are secondary.

READ ONLY for QSEG_FINAL (the processed 256x256 data are copied to the Colab VM for speed).
Outputs (numbers only, no images, no weights) go to MyDrive/P1_m3/runs/<model>_seed<k>/.
The script is resumable: finished runs are skipped. Expected time on a T4: about 25-35 min per run.
"""
from google.colab import drive
drive.mount('/content/drive')
import os, sys, json, time, shutil, random
import numpy as np, pandas as pd

ROOT = '/content/drive/MyDrive/QSEG_FINAL'
OUT = '/content/drive/MyDrive/P1_m3'
LOCAL = '/content/qseg_local'          # local copy of results/data (read-only source stays untouched)
TMP = '/content/m3_tmp'                # weights live here only during one run
MODELS = ['hybrid_quantum', 'hybrid_classical', 'hybrid_none']
SEEDS = [0, 1, 2, 3, 4]
MIN_EPOCH = 20
os.makedirs(OUT + '/runs', exist_ok=True)
os.makedirs(TMP, exist_ok=True)

# ---- protocol record (written once, never overwritten)
proto = OUT + '/PROTOCOL_prespecified.txt'
if not os.path.exists(proto):
    with open(proto, 'w') as f:
        f.write(__doc__ + '\nwritten (UTC): ' + time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()) + '\n')

# ---- local copy of the processed data
if not os.path.exists(LOCAL + '/data/manifest.csv'):
    shutil.copytree(ROOT + '/results/data', LOCAL + '/data',
                    ignore=shutil.ignore_patterns('*.npy', '*.h5'))
print('local data ready:', len(os.listdir(LOCAL + '/data/images')), 'images')

sys.path.insert(0, ROOT + '/code')
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
assert gpus, 'No GPU - switch the runtime to T4 before running'
from qseg.config import Config
from qseg.models import build_model
from qseg.datasets import load_manifest, make_dataset, load_arrays
from qseg.evaluation import evaluate_predictions
from qseg.train import bce_dice_loss, soft_dice

cfg = Config(work_dir=LOCAL)
man = load_manifest(cfg)
x_te, y_te, df_te = load_arrays(cfg, 'test', man)
print('test images', len(x_te))


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)


class Track(tf.keras.callbacks.Callback):
    """Saves the checkpoints of policies A (simulated early stopping) and B2 (best after MIN_EPOCH)."""
    def __init__(self, d):
        super().__init__(); self.d = d
        self.best = np.inf; self.wait = 0; self.es_stop = None; self.es_epoch = None
        self.best_min = np.inf; self.min_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        v = float(logs.get('val_loss', np.nan)); e = epoch + 1
        if not np.isfinite(v):
            return
        if self.es_stop is None:                      # identical rule to keras EarlyStopping(patience=10)
            if v < self.best:
                self.best, self.es_epoch, self.wait = v, e, 0
                self.model.save_weights(self.d + '/es.weights.h5')
            else:
                self.wait += 1
                if self.wait >= cfg.early_stop_patience:
                    self.es_stop = e
        if e >= MIN_EPOCH and v < self.best_min:
            self.best_min, self.min_epoch = v, e
            self.model.save_weights(self.d + '/min20.weights.h5')


def evaluate(model, tag, out_dir, seed):
    prob = model.predict(x_te, batch_size=cfg.batch_size, verbose=0)
    summ, per = evaluate_predictions(y_te, prob, df_te['patient_id'].to_numpy(), cfg.threshold, 0,
                                     cfg.auc_max_pixels, cfg.ece_bins, seed)
    per.insert(0, 'file_id', df_te['file_id'].to_numpy())
    per.to_csv(f'{out_dir}/{tag}_per_image.csv', index=False)
    return {k: summ[k] for k in ('dice_per_image_mean', 'dice', 'iou', 'mcc', 'precision', 'recall', 'ece')}


t_all = time.time()
for seed in SEEDS:
    for m in MODELS:
        out_dir = f'{OUT}/runs/{m}_seed{seed}'
        if os.path.exists(out_dir + '/run.json'):
            print('[skip]', m, seed); continue
        os.makedirs(out_dir, exist_ok=True)
        d = f'{TMP}/{m}_{seed}'; os.makedirs(d, exist_ok=True)
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy('float32')
        set_seeds(seed)
        tr, _ = make_dataset(cfg, 'train', seed, man)
        va, _ = make_dataset(cfg, 'val', seed, man)
        model = build_model(m, cfg)
        model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate), loss=bce_dice_loss,
                      metrics=[soft_dice], jit_compile=False)
        track = Track(d)
        cbs = [tf.keras.callbacks.TerminateOnNaN(), track,
               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=cfg.lr_factor,
                                                    patience=cfg.lr_patience, min_lr=cfg.min_lr),
               tf.keras.callbacks.CSVLogger(out_dir + '/history.csv')]
        t0 = time.time()
        h = model.fit(tr, validation_data=va, epochs=cfg.max_epochs, callbacks=cbs, verbose=2)
        secs = time.time() - t0
        n_ep = len(h.history['loss'])
        res = dict(model=m, seed=seed, epochs_run=n_ep, train_seconds=round(secs, 1),
                   es_best_epoch=track.es_epoch, es_stop_epoch=track.es_stop, min20_epoch=track.min_epoch)
        res['final'] = evaluate(model, 'final', out_dir, seed)            # B1: epoch-50 weights
        model.load_weights(d + '/es.weights.h5');    res['es'] = evaluate(model, 'es', out_dir, seed)
        model.load_weights(d + '/min20.weights.h5'); res['min20'] = evaluate(model, 'min20', out_dir, seed)
        with open(out_dir + '/run.json', 'w') as f:
            json.dump(res, f, indent=2)
        shutil.rmtree(d, ignore_errors=True)
        print(f"DONE {m} seed {seed}: {n_ep} epochs, {secs/60:.1f} min | per-image Dice  "
              f"final {res['final']['dice_per_image_mean']:.4f}  es {res['es']['dice_per_image_mean']:.4f} "
              f"(epoch {track.es_epoch}, stop {track.es_stop})  min20 {res['min20']['dice_per_image_mean']:.4f} "
              f"(epoch {track.min_epoch}) | total {(time.time()-t_all)/3600:.2f} h")

# ---- compact overview of all finished runs (the full statistics are done offline)
rows = []
for seed in SEEDS:
    for m in MODELS:
        p = f'{OUT}/runs/{m}_seed{seed}/run.json'
        if os.path.exists(p):
            r = json.load(open(p))
            rows.append(dict(model=m, seed=seed, es_epoch=r['es_best_epoch'], min20_epoch=r['min20_epoch'],
                             **{f'dice_{k}': r[k]['dice_per_image_mean'] for k in ('es', 'final', 'min20')}))
ov = pd.DataFrame(rows); ov.to_csv(OUT + '/overview.csv', index=False)
print(ov.round(4).to_string(index=False))
print('\nFinished runs:', len(ov), 'of', len(SEEDS) * len(MODELS), '| results in', OUT)
