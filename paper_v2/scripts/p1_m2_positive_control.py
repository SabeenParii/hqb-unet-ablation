"""
Paper 1 · M2 positive control for the bottleneck-severing diagnostic

PRE-SPECIFIED on 18 October 2026, before any of these runs was trained or any new result was seen.
Goal: show that the severing diagnostic detects dependence on the four-value code when dependence is present.
Base: the HQB-UNet (same encoder, circuit, expansion, decoder, training settings and early-stopping policy as
the paper). The code is the four-value circuit output (input of the 4->512 dense layer), as in the paper.
Two candidate positive controls that force information through the code:
  C1 'gated'   : every skip connection is multiplied channel-wise by sigmoid(Dense(z)), z = the code
  C2 'deepcut' : the two deepest skip connections (32x32 and 64x64 levels) are removed
Selection (validation set only, test set not used): train C1 and C2 with seed 0. A candidate qualifies if
  (i) validation per-image Dice >= 0.94  (learns the target adequately) and
  (ii) validation mean drop >= 0.02 under the five-assignment swap average (measurable dependence).
The qualifying candidate with the larger validation swap drop is selected; if none qualifies, this is reported
as a failed positive control and no further designs are searched. The selected candidate is trained with seeds
1 and 2, and the identical severing procedure of the paper (const = mean training code; swap = code of a
training image drawn with replacement, assignments A1-A5 with numpy seeds 0-4) is applied to the test set for
all three seeds. All runs are reported.

READ ONLY for QSEG_FINAL. Outputs (numbers only) go to MyDrive/P1_m2/. Resumable per run.
Expected time on a T4: about 2-2.5 h in total.
"""
from google.colab import drive
drive.mount('/content/drive')
import os, sys, json, time, shutil, random
import numpy as np, pandas as pd

ROOT = '/content/drive/MyDrive/QSEG_FINAL'
OUT = '/content/drive/MyDrive/P1_m2'
LOCAL = '/content/qseg_local'
os.makedirs(OUT + '/runs', exist_ok=True)
proto = OUT + '/PROTOCOL_prespecified.txt'
if not os.path.exists(proto):
    with open(proto, 'w') as f:
        f.write(__doc__ + '\nwritten (UTC): ' + time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()) + '\n')
if not os.path.exists(LOCAL + '/data/manifest.csv'):
    shutil.copytree(ROOT + '/results/data', LOCAL + '/data', ignore=shutil.ignore_patterns('*.npy', '*.h5'))

sys.path.insert(0, ROOT + '/code')
import tensorflow as tf
from tensorflow.keras import layers, models
assert tf.config.list_physical_devices('GPU'), 'No GPU - switch the runtime to T4'
from qseg.config import Config
from qseg.models import conv_block, output_layer, TFCircuitLayer
from qseg.datasets import load_manifest, make_dataset, load_arrays
from qseg.evaluation import per_image_overlap
from qseg.train import bce_dice_loss, soft_dice

cfg = Config(work_dir=LOCAL)
man = load_manifest(cfg)
X = {s: load_arrays(cfg, s, man) for s in ('train', 'val', 'test')}
print({s: len(v[0]) for s, v in X.items()})


class CodeSwitch(layers.Layer):
    """Identity during training; at severing time replaces the code by a supplied batch of codes."""
    def build(self, input_shape):
        self.on = self.add_weight(shape=(), initializer='zeros', trainable=False, name='on')
        self.buf = self.add_weight(shape=(cfg.batch_size, 4), initializer='zeros', trainable=False, name='buf')

    def call(self, z):
        n = tf.shape(z)[0]
        return z * (1.0 - self.on) + self.buf[:n] * self.on


def build_pc(kind, filters=(32, 64, 128, 256, 512), dropouts=(0.1, 0.1, 0.2, 0.2, 0.3)):
    inp = layers.Input((256, 256, 1), name='input_image')
    skips, x = [], inp
    for i, (f, d) in enumerate(zip(filters[:-1], dropouts[:-1])):
        x = conv_block(x, f, d, name=f'enc{i + 1}'); skips.append(x)
        x = layers.MaxPooling2D(2, name=f'pool{i + 1}')(x)
    x = conv_block(x, filters[-1], dropouts[-1], name='bottleneck')
    z = layers.Dense(4, activation='tanh', name='pre_dense')(layers.GlobalAveragePooling2D(name='gap')(x))
    z = TFCircuitLayer(4, 2, name='quantum_circuit')(z)
    z = CodeSwitch(name='code_switch')(z)
    q = layers.Dense(512, activation='relu', name='post_dense')(z)
    x = layers.Reshape((16, 16, filters[-1]), name='q_reshape')(layers.Dense(16 * 16 * filters[-1], activation='relu', name='q_expand')(q))
    for i, (f, s, d) in enumerate(zip(reversed(filters[:-1]), reversed(skips), reversed(dropouts[:-1]))):
        x = layers.UpSampling2D(2, name=f'up{i + 1}')(x)
        if kind == 'gated':
            g = layers.Reshape((1, 1, f), name=f'gate{i + 1}_r')(layers.Dense(f, activation='sigmoid', name=f'gate{i + 1}')(z))
            s = layers.Multiply(name=f'gate{i + 1}_mul')([s, g])
            x = layers.Concatenate(name=f'concat{i + 1}')([x, s])
        elif kind == 'deepcut' and i < 2:
            pass                                   # deepest two skips removed
        else:
            x = layers.Concatenate(name=f'concat{i + 1}')([x, s])
        x = conv_block(x, f, d, name=f'dec{i + 1}')
    return models.Model(inp, output_layer(x), name=f'PC_{kind}')


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed); random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)


def predict(model, x, codes=None):
    sw = model.get_layer('code_switch'); out = []
    for a in range(0, len(x), cfg.batch_size):
        xb = x[a:a + cfg.batch_size]
        if codes is None:
            sw.on.assign(0.0)
        else:
            buf = np.zeros((cfg.batch_size, 4), np.float32); buf[:len(xb)] = codes[a:a + len(xb)]
            sw.buf.assign(buf); sw.on.assign(1.0)
        out.append(model(xb, training=False).numpy()[..., 0])
    sw.on.assign(0.0)
    return np.concatenate(out)


def severing(model, split):
    x, y, df = X[split]; Y = (y[..., 0] > 0.5).astype(np.uint8)
    enc = tf.keras.Model(model.input, model.get_layer('quantum_circuit').output)
    z_tr = enc.predict(X['train'][0], batch_size=cfg.batch_size, verbose=0)
    dice = lambda p: per_image_overlap(Y, (p >= cfg.threshold).astype(np.uint8))[:, 0]
    pn = predict(model, x); d0 = dice(pn)
    res = pd.DataFrame({'file_id': df.file_id, 'patient_id': df.patient_id, 'dice_normal': d0})
    res['drop_const'] = d0 - dice(predict(model, x, np.repeat(z_tr.mean(0)[None], len(x), 0)))
    for k in range(5):
        idx = np.random.default_rng(k).integers(0, len(z_tr), len(x))
        res[f'drop_swap{k}'] = d0 - dice(predict(model, x, z_tr[idx]))
    res['drop_swapavg'] = res[[f'drop_swap{k}' for k in range(5)]].mean(1)
    res['code_sd'] = float(enc.predict(x, batch_size=cfg.batch_size, verbose=0).std(0).mean())
    return res


def train_run(kind, seed):
    out = f'{OUT}/runs/{kind}_seed{seed}'
    if os.path.exists(out + '/run.json'):
        print('[skip]', kind, seed); return json.load(open(out + '/run.json'))
    os.makedirs(out, exist_ok=True)
    tf.keras.backend.clear_session(); set_seeds(seed)
    tr, _ = make_dataset(cfg, 'train', seed, man); va, _ = make_dataset(cfg, 'val', seed, man)
    model = build_pc(kind)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate), loss=bce_dice_loss, metrics=[soft_dice])
    ck = f'/content/pc_{kind}_{seed}.weights.h5'
    cbs = [tf.keras.callbacks.TerminateOnNaN(),
           tf.keras.callbacks.ModelCheckpoint(ck, monitor='val_loss', save_best_only=True, save_weights_only=True),
           tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg.early_stop_patience, restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr),
           tf.keras.callbacks.CSVLogger(out + '/history.csv')]
    t0 = time.time()
    h = model.fit(tr, validation_data=va, epochs=cfg.max_epochs, callbacks=cbs, verbose=2)
    model.load_weights(ck)
    vl = np.array(h.history['val_loss'])
    val = severing(model, 'val'); val.to_csv(out + '/val_severing.csv', index=False)
    r = dict(kind=kind, seed=seed, epochs_run=len(vl), best_epoch=int(np.nanargmin(vl)) + 1,
             minutes=round((time.time() - t0) / 60, 1), params=int(model.count_params()),
             val_dice=float(val.dice_normal.mean()), val_drop_const=float(val.drop_const.mean()),
             val_drop_swapavg=float(val.drop_swapavg.mean()), val_code_sd=float(val.code_sd.iloc[0]))
    shutil.copy(ck, f'/content/keep_{kind}_{seed}.weights.h5')     # kept on the VM only (not on Drive)
    json.dump(r, open(out + '/run.json', 'w'), indent=2)
    print('DONE', r)
    return r


def test_severing(kind, seed):
    out = f'{OUT}/runs/{kind}_seed{seed}'
    if os.path.exists(out + '/test_severing.csv'):
        return
    w = f'/content/keep_{kind}_{seed}.weights.h5'
    if not os.path.exists(w):
        raise SystemExit(f'Weights of {kind} seed {seed} were lost with the session: delete {out}/run.json and rerun the script.')
    tf.keras.backend.clear_session(); model = build_pc(kind); model(X['test'][0][:1])
    model.load_weights(w)
    te = severing(model, 'test'); te.to_csv(out + '/test_severing.csv', index=False)
    print(f'TEST {kind} seed {seed}: Dice {te.dice_normal.mean():.4f} | drop const {te.drop_const.mean():.4f} | '
          f'swap avg {te.drop_swapavg.mean():.4f} | swaps ' + ' '.join(f'{te[f"drop_swap{k}"].mean():.4f}' for k in range(5)))


# ---- stage 1: candidates, seed 0, validation-based selection
cand = {k: train_run(k, 0) for k in ('gated', 'deepcut')}
ok = {k: r for k, r in cand.items() if r['val_dice'] >= 0.94 and r['val_drop_swapavg'] >= 0.02}
sel_path = OUT + '/selection.json'
if os.path.exists(sel_path):
    sel = json.load(open(sel_path))['selected']
else:
    sel = max(ok, key=lambda k: ok[k]['val_drop_swapavg']) if ok else None
    json.dump(dict(selected=sel, candidates=cand), open(sel_path, 'w'), indent=2)
print('\nSELECTED:', sel, '\n', pd.DataFrame(cand).T.round(4).to_string())

# ---- stage 2: two more seeds and test-set severing of the selected control
if sel is None:
    print('No candidate met the pre-specified criteria: report as a failed positive control.')
else:
    for s in (1, 2):
        train_run(sel, s)
    for s in (0, 1, 2):
        test_severing(sel, s)
    rows = []
    for s in (0, 1, 2):
        te = pd.read_csv(f'{OUT}/runs/{sel}_seed{s}/test_severing.csv')
        rows.append(dict(seed=s, dice=te.dice_normal.mean(), drop_const=te.drop_const.mean(),
                         drop_swapavg=te.drop_swapavg.mean(), abs_change=te.drop_swapavg.abs().mean()))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
print('results in', OUT)
