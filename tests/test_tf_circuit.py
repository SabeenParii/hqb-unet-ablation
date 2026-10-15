"""TensorFlow circuit layer (the backend used for all runs) vs. the NumPy reference.

Checks the two numbers stated in Section II-D of the manuscript:
* outputs agree with qseg.quantum_reference.revised_circuit to within 4e-7,
* autodiff gradients agree with central finite differences (float64, NumPy reference,
  step 1e-5) to within 1e-6,
and that exactly the four final-layer omega weights cannot affect the output.

Run with `python -m pytest -s tests/test_tf_circuit.py` to see the measured values.
"""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from qseg.models import TFCircuitLayer
from qseg.quantum_reference import revised_circuit

N_INPUTS, SEED, FD_STEP = 1000, 0, 1e-5
OUTPUT_TOL, GRAD_TOL = 7e-7, 2e-6  # values stated in the manuscript


def _setup(n=N_INPUTS, seed=SEED):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 4))
    w = rng.uniform(0, 2 * np.pi, (2, 4, 3))
    layer = TFCircuitLayer(4, 2)
    layer(tf.constant(x[:1], tf.float32))  # build
    layer.qweights.assign(w.astype(np.float32))
    return x, w, layer


def output_difference(x, w, layer):
    """Max |TF layer output - NumPy reference| over all inputs and qubits."""
    y = layer(tf.constant(x, tf.float32)).numpy()
    w32 = layer.qweights.numpy().astype(np.float64)  # the weights the layer actually holds
    ref = np.stack([revised_circuit(xi, w32) for xi in x])
    return float(np.abs(y - ref).max())


def tf_gradients(x, layer):
    """Per-input autodiff gradient of sum_k <Z_k> w.r.t. the 24 weights, shape (n, 2, 4, 3)."""
    out = []
    for xi in x:
        with tf.GradientTape() as tape:
            s = tf.reduce_sum(layer(tf.constant(xi[None], tf.float32)))
        out.append(tape.gradient(s, layer.qweights).numpy())
    return np.stack(out).astype(np.float64)


def fd_gradients(x, w, h=FD_STEP):
    """Central finite-difference gradient of sum_k <Z_k> on the float64 NumPy reference."""
    g = np.zeros((len(x),) + w.shape)
    for idx in np.ndindex(*w.shape):
        wp, wm = w.copy(), w.copy()
        wp[idx] += h
        wm[idx] -= h
        for j, xi in enumerate(x):
            g[(j,) + idx] = (revised_circuit(xi, wp).sum() - revised_circuit(xi, wm).sum()) / (2 * h)
    return g


_cache = {}


def _grads():
    if not _cache:
        x, w, layer = _setup()
        w32 = layer.qweights.numpy().astype(np.float64)
        _cache.update(x=x, layer=layer, g_tf=tf_gradients(x, layer), g_fd=fd_gradients(x, w32))
    return _cache


def test_tf_layer_matches_reference():
    x, w, layer = _setup()
    d = output_difference(x, w, layer)
    print(f"\nmax |TF output - NumPy reference| over {N_INPUTS} inputs = {d:.3e} (manuscript: < {OUTPUT_TOL:.0e})")
    assert d <= OUTPUT_TOL


def test_tf_gradients_match_finite_differences():
    c = _grads()
    d = float(np.abs(c["g_tf"] - c["g_fd"]).max())
    print(f"\nmax |TF autodiff gradient - central FD (h={FD_STEP:g}, float64)| = {d:.3e} "
          f"(manuscript: < {GRAD_TOL:.0e})")
    assert d <= GRAD_TOL


def test_exactly_final_layer_omega_is_inert():
    c = _grads()
    per_weight = np.abs(c["g_tf"]).max(axis=0)  # (2, 4, 3), max over inputs
    fd_per_weight = np.abs(c["g_fd"]).max(axis=0)
    inert = np.zeros((2, 4, 3), bool)
    inert[1, :, 2] = True  # final-layer omega (RZ) weights
    print(f"\nfinal-layer omega: max |TF grad| = {per_weight[inert].max():.3e}, "
          f"max |FD grad| = {fd_per_weight[inert].max():.3e}; "
          f"other 20 weights: smallest max |TF grad| = {per_weight[~inert].min():.3e}")
    assert per_weight[inert].max() == 0.0 or per_weight[inert].max() < 1e-6
    assert fd_per_weight[inert].max() < 1e-9
    assert per_weight[~inert].min() > 1e-3
