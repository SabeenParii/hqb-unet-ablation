"""Train and evaluate one model with one seed, using settings shared by all models.

Fixes relative to the original notebooks:
  * Same optimiser, learning rate, schedule, loss, batch size, input channels, epochs
    and callbacks for every model (the originals differed in all of these).
  * Early stopping monitors val_loss (the Classical U-Net monitored val_dice, which was
    NaN at epoch 1, so epoch-1 weights were restored and later epochs were discarded).
  * TerminateOnNaN + float32 training (mixed precision off by default).
  * The best checkpoint (lowest val_loss) is always reloaded before testing, and the
    epoch it came from is recorded, so the manuscript can state exactly which weights
    were evaluated.
  * Every model is evaluated on the same held-out test set with the same metric code.
  * Seeds are set for Python, NumPy and TensorFlow.
"""
import json
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import Config, MODELS
from .datasets import make_dataset, load_arrays, load_manifest
from .evaluation import evaluate_predictions
from .models import build_model, parameter_breakdown


# ----------------------------------------------------------------------------- loss / monitor metric
def soft_dice(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = (1, 2, 3)
    inter = tf.reduce_sum(y_true * y_pred, axis=axes)
    denom = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
    return tf.reduce_mean((2.0 * inter + smooth) / (denom + smooth))  # mean per-image soft Dice


def bce_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + (1.0 - soft_dice(y_true, y_pred))


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def environment_info(cfg):
    """Software versions and quantum backend, stored in summary.json of every new run."""
    import platform
    import scipy
    info = dict(quantum_backend=getattr(cfg, "quantum_backend", None), python_version=platform.python_version(),
                numpy_version=np.__version__, pandas_version=pd.__version__, scipy_version=scipy.__version__)
    try:
        import pennylane as qml
        info["pennylane_version"] = qml.__version__
    except ImportError:
        info["pennylane_version"] = None
    return info


# ----------------------------------------------------------------------------- run
def run_experiment(cfg: Config, model_name: str, seed: int, verbose=1):
    assert model_name in MODELS, model_name
    out_dir = os.path.join(cfg.work_dir, "runs", f"{model_name}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "summary.json")):
        print(f"[skip] {out_dir} already finished")
        with open(os.path.join(out_dir, "summary.json")) as f:
            return json.load(f)

    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy("mixed_float16" if cfg.mixed_precision else "float32")
    set_seeds(seed)
    manifest = load_manifest(cfg)
    train_ds, _ = make_dataset(cfg, "train", seed, manifest)
    val_ds, _ = make_dataset(cfg, "val", seed, manifest)

    model = build_model(model_name, cfg)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate), loss=bce_dice_loss, metrics=[soft_dice],
                  jit_compile=False)
    ckpt = os.path.join(out_dir, "best.weights.h5")
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", mode="min", save_best_only=True,
                                           save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=cfg.early_stop_patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=cfg.lr_factor, patience=cfg.lr_patience,
                                             min_lr=cfg.min_lr),
        tf.keras.callbacks.CSVLogger(os.path.join(out_dir, "history.csv")),
    ]
    t0 = time.perf_counter()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=cfg.max_epochs, callbacks=callbacks, verbose=verbose)
    train_seconds = time.perf_counter() - t0

    h = pd.DataFrame(hist.history)
    val_loss = h["val_loss"].to_numpy()
    if not np.isfinite(val_loss).any():
        raise RuntimeError(f"{model_name} seed {seed}: validation loss was never finite")
    best_epoch = int(np.nanargmin(val_loss)) + 1  # 1-based
    model.load_weights(ckpt)                      # always test the best-val_loss weights

    # ---- test-set evaluation (same code for every model)
    x_test, y_test, df_test = load_arrays(cfg, "test", manifest)
    model.predict(x_test[:cfg.batch_size], batch_size=cfg.batch_size, verbose=0)  # warm-up
    t1 = time.perf_counter()
    prob = model.predict(x_test, batch_size=cfg.batch_size, verbose=0)
    infer_ms = (time.perf_counter() - t1) / len(x_test) * 1000
    summary, per_image = evaluate_predictions(
        y_test, prob, df_test["patient_id"].to_numpy(), cfg.threshold, cfg.n_bootstrap,
        cfg.auc_max_pixels, cfg.ece_bins, seed)

    summary.update(model=model_name, model_label=MODELS[model_name], seed=seed,
                   epochs_run=int(len(h)), best_epoch=best_epoch,
                   best_val_loss=float(val_loss[best_epoch - 1]),
                   train_loss_at_best=float(h["loss"].iloc[best_epoch - 1]),
                   train_seconds=round(train_seconds, 1), inference_ms_per_image=round(infer_ms, 2),
                   gpu=str([d.name for d in tf.config.list_physical_devices("GPU")]),
                   tf_version=tf.__version__, **environment_info(cfg), **parameter_breakdown(model))
    per_image.insert(0, "file_id", df_test["file_id"].to_numpy())
    per_image.to_csv(os.path.join(out_dir, "per_image.csv"), index=False)
    np.save(os.path.join(out_dir, "test_prob.npy"), prob.astype(np.float16))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"{MODELS[model_name]} | seed {seed} | best epoch {best_epoch}/{len(h)} | "
              f"Dice {summary['dice']:.4f} | IoU {summary['iou']:.4f} | MCC {summary['mcc']:.4f}")
    return summary


def run_all(cfg: Config, model_names=None, seeds=None):
    cfg.to_json(os.path.join(cfg.work_dir, "train_config.json"))
    results = []
    for name in (model_names or list(MODELS)):
        for s in (seeds or cfg.seeds):
            results.append(run_experiment(cfg, name, s))
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(cfg.work_dir, "all_runs.csv"), index=False)
    return df
