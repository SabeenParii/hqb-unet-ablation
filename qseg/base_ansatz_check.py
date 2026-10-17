"""Jacobian-rank check of the base-design ansatz and of our bottleneck circuit (numpy only).

Base design: Lim & Kim, "Hybrid quantum-classical UNet for image segmentation: Breaking the
parameter scaling barrier", IEEE Access 14, 33934-33947, 2026, doi:10.1109/ACCESS.2026.3669206.

Both circuits belong to the same family, simulated here exactly (noise-free state vector) for any
number of qubits n and layers L:

    RY(scale * x_i) on each qubit;
    L x [ RZ(c) RY(b) RZ(a) on each qubit (operator product: RZ(a) applied first, RZ(c) last),
          CNOT(i, i+1) for i = 0..n-2, then CNOT(n-1, 0) ];
    outputs <Z_i> on every qubit.

Weights have shape (L, n, 3); w[l, i] = (a, b, c) in the order the gates are applied.

* Ours (HQB-UNet, qseg/models.py): n = 4, L = 2, scale = pi/2,
  Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi), i.e. (a, b, c) = (phi, theta, omega).
* Lim & Kim, eqs. (3)-(8): n = 8, L = 4, scale = 1 (no pi/2 factor),
  Rot(theta, phi, lambda) = RZ(phi) RY(theta) RZ(lambda), i.e. (a, b, c) = (lambda, theta, phi).

The check stacks the Jacobians d<Z_i>/dw (central finite differences, as for our circuit in the
manuscript) over random inputs and parameter settings and reports the rank and the identically
zero columns. The last-applied RZ of the final layer acts directly before the Z measurement and
commutes with it, so those n columns are zero in both circuits.

Run: python -m qseg.base_ansatz_check
"""
import numpy as np

FD_STEP = 1e-6
RANK_TOL = 1e-6
ZERO_TOL = 1e-8

# name, n qubits, L layers, encoding scale, names of (a, b, c), n inputs, n parameter settings, seed
CIRCUITS = {
    "ours": dict(label="Ours (HQB-UNet)", n=4, layers=2, scale=np.pi / 2,
                 angles=("phi", "theta", "omega"), n_inputs=64, n_settings=3, seed=0),
    "lim": dict(label="Lim & Kim (2026)", n=8, layers=4, scale=1.0,
                angles=("lambda", "theta", "phi"), n_inputs=24, n_settings=2, seed=0),
}


def _ry(t):
    """Batched RY(t), shape (B, 2, 2)."""
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(complex)


def _rz(t):
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]], dtype=complex)


def _ry1(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)


def _apply_1q(psi, u, wire):
    """psi: (B, 2, ..., 2); u: (2, 2) or (B, 2, 2)."""
    ax = wire + 1
    if u.ndim == 2:
        out = np.tensordot(u, psi, axes=([1], [ax]))  # (2, B, ...)
        return np.moveaxis(out, 0, ax)
    psi = np.moveaxis(psi, ax, -1)
    out = np.einsum("bij,b...j->b...i", u, psi)
    return np.moveaxis(out, -1, ax)


def _apply_cnot(psi, control, target):
    psi = psi.copy()
    idx = [slice(None)] * psi.ndim
    idx[control + 1] = 1
    sub = psi[tuple(idx)]
    axis = (target if target < control else target - 1) + 1
    psi[tuple(idx)] = np.flip(sub, axis=axis)
    return psi


def circuit(x, w, scale):
    """Exact <Z_i> of the ansatz family.

    x: (B, n) or (n,) inputs; w: (L, n, 3) weights in applied order (a, b, c); scale: encoding factor.
    Returns (B, n) (or (n,) for a single input). Wire 0 is the most significant bit.
    """
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    x = np.atleast_2d(x)
    B, n = x.shape
    psi = np.zeros((B,) + (2,) * n, dtype=complex)
    psi[(slice(None),) + (0,) * n] = 1.0
    for i in range(n):
        psi = _apply_1q(psi, _ry(scale * x[:, i]), i)
    for layer in np.asarray(w, dtype=float):
        for i in range(n):
            a, b, c = layer[i]
            psi = _apply_1q(psi, _rz(c) @ _ry1(b) @ _rz(a), i)
        for i in range(n - 1):
            psi = _apply_cnot(psi, i, i + 1)
        if n > 1:
            psi = _apply_cnot(psi, n - 1, 0)
    p = np.abs(psi) ** 2
    out = np.stack([p.take(0, axis=i + 1).reshape(B, -1).sum(1) - p.take(1, axis=i + 1).reshape(B, -1).sum(1)
                    for i in range(n)], axis=1)
    return out[0] if single else out


def jacobian(x, w, scale, method="fd", h=FD_STEP):
    """Jacobian d<Z_i>/dw for each input, shape (B, n, w.size) (columns in w.reshape(-1) order).

    method="fd": central finite differences with step h (the procedure used in the manuscript);
    method="shift": exact parameter-shift rule (each angle is a single Pauli rotation).
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    w = np.asarray(w, dtype=float)
    flat = w.reshape(-1)
    d, den = (h, 2 * h) if method == "fd" else (np.pi / 2, 2.0)
    J = np.zeros((x.shape[0], x.shape[1], flat.size))
    for k in range(flat.size):
        e = np.zeros_like(flat)
        e[k] = d
        J[:, :, k] = (circuit(x, (flat + e).reshape(w.shape), scale)
                      - circuit(x, (flat - e).reshape(w.shape), scale)) / den
    return J


def stacked_jacobian(n, layers, scale, n_inputs, n_settings, seed=0, method="fd"):
    """Jacobians stacked over n_settings random weight draws (uniform in [0, 2pi)) and n_inputs
    random inputs each (uniform in [-1, 1]); shape (n_settings * n_inputs * n, 3 * n * layers)."""
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(n_settings):
        w = rng.uniform(0, 2 * np.pi, (layers, n, 3))
        x = rng.uniform(-1, 1, (n_inputs, n))
        blocks.append(jacobian(x, w, scale, method).reshape(-1, w.size))
    return np.vstack(blocks)


def param_names(n, layers, angles):
    """Labels of the flattened weight vector, e.g. 'L4 q0 RZ(phi)'."""
    gates = ("RZ", "RY", "RZ")
    return [f"L{l + 1} q{i} {gates[j]}({angles[j]})" for l in range(layers) for i in range(n) for j in range(3)]


def analyse(key, n_inputs=None, n_settings=None, method="fd"):
    """Returns dict with n_params, rank, zero (indices of identically zero columns) and names."""
    c = CIRCUITS[key]
    J = stacked_jacobian(c["n"], c["layers"], c["scale"], n_inputs or c["n_inputs"],
                         n_settings or c["n_settings"], c["seed"], method)
    zero = [k for k in range(J.shape[1]) if np.abs(J[:, k]).max() < ZERO_TOL]
    return dict(n_params=J.shape[1], rank=int(np.linalg.matrix_rank(J, tol=RANK_TOL)), zero=zero,
                names=param_names(c["n"], c["layers"], c["angles"]), rows=J.shape[0])


def final_layer_last_rz(n, layers):
    """Flattened indices of the last-applied RZ of every qubit in the final layer."""
    return [(layers - 1) * n * 3 + i * 3 + 2 for i in range(n)]


def main():
    for key, c in CIRCUITS.items():
        r = analyse(key)
        expected = final_layer_last_rz(c["n"], c["layers"])
        print(f"{c['label']}: {c['n']} qubits, {c['layers']} layers, encoding RY({'pi/2*' if key == 'ours' else ''}x)")
        print(f"  inputs {c['n_inputs']} x parameter settings {c['n_settings']} -> Jacobian {r['rows']} x {r['n_params']}")
        print(f"  parameters {r['n_params']}, rank {r['rank']}, zero columns {len(r['zero'])}"
              f" (= final-layer last-applied RZ: {r['zero'] == expected})")
        print("  zero: " + ", ".join(r["names"][k] for k in r["zero"]))


if __name__ == "__main__":
    main()
