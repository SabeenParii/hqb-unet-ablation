"""Exact state-vector reference implementation of the two bottleneck circuits (numpy only).

ORIGINAL circuit (hybrid_model.ipynb):
    RY(x_i) on each qubit; then per layer: RZ(w_li) on each qubit, CNOT ring; measure <Z_i>.
    RZ is diagonal in the computational basis and CNOTs only permute basis states, so the
    measured <Z_i> values do NOT depend on the trainable weights w. The 8 "trainable"
    quantum parameters therefore received zero gradient and had no effect.

REVISED circuit (qseg/models.py):
    RY(pi/2 * x_i) on each qubit; then per layer: Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)
    on each qubit, CNOT ring; measure <Z_i>. The RY component of Rot changes measurement
    probabilities, so every layer's weights now affect the output.
"""
import numpy as np


def _ry(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)


def _rz(t):
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]], dtype=complex)


def _rot(phi, theta, omega):  # PennyLane convention: Rot = RZ(omega) RY(theta) RZ(phi)
    return _rz(omega) @ _ry(theta) @ _rz(phi)


def _apply_1q(psi, u, wire, n):
    psi = psi.reshape([2] * n)
    psi = np.moveaxis(np.tensordot(u, psi, axes=([1], [wire])), 0, wire)
    return psi.reshape(-1)


def _apply_cnot(psi, control, target, n):
    psi = psi.reshape([2] * n).copy()
    idx = [slice(None)] * n
    idx[control] = 1
    sub = psi[tuple(idx)]
    axis = target if target < control else target - 1
    psi[tuple(idx)] = np.flip(sub, axis=axis)
    return psi.reshape(-1)


def _ring(psi, n):
    for i in range(n - 1):
        psi = _apply_cnot(psi, i, i + 1, n)
    if n > 1:
        psi = _apply_cnot(psi, n - 1, 0, n)
    return psi


def _expval_z(psi, n):
    p = np.abs(psi.reshape([2] * n)) ** 2
    out = []
    for i in range(n):
        marg = p.sum(axis=tuple(j for j in range(n) if j != i))
        out.append(marg[0] - marg[1])
    return np.array(out)


def original_circuit(x, weights):
    """x: (n,), weights: (L, n)"""
    n = len(x)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1
    for i in range(n):
        psi = _apply_1q(psi, _ry(x[i]), i, n)
    for layer in weights:
        for i in range(n):
            psi = _apply_1q(psi, _rz(layer[i]), i, n)
        psi = _ring(psi, n)
    return _expval_z(psi, n)


def revised_circuit(x, weights):
    """x: (n,) in [-1, 1] (tanh output), weights: (L, n, 3)"""
    n = len(x)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1
    for i in range(n):
        psi = _apply_1q(psi, _ry(np.pi / 2 * x[i]), i, n)
    for layer in weights:
        for i in range(n):
            psi = _apply_1q(psi, _rot(*layer[i]), i, n)
        psi = _ring(psi, n)
    return _expval_z(psi, n)


def weight_sensitivity(circuit, weight_shape, n_inputs=4, trials=20, seed=0):
    """Largest change in the circuit output when only the weights are re-drawn."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        x = rng.uniform(-1, 1, n_inputs)
        outs = np.array([circuit(x, rng.normal(0, np.pi, weight_shape)) for _ in range(5)])
        worst = max(worst, float(np.max(np.ptp(outs, axis=0))))
    return worst
