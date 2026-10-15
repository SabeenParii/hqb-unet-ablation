"""Backend-agnostic exact state-vector simulation of the revised 4-qubit circuit.

Default backend (config.quantum_backend = "tf"), used for all reported runs. The same function runs
with a NumPy adapter (tested against qseg/quantum_reference.py) and a TensorFlow adapter
(used inside the Keras model, fully differentiable).

Circuit: RY(pi/2 * x_i) on each qubit; L x [Rot(phi, theta, omega) on each qubit + CNOT ring]; <Z_i>.
Wire 0 is the most significant bit (PennyLane convention).
"""
import numpy as np


def _cnot_perm(n, control, target):
    perm = np.arange(2 ** n)
    for k in range(2 ** n):
        if (k >> (n - 1 - control)) & 1:
            perm[k] = k ^ (1 << (n - 1 - target))
    return perm


def _ring_perms(n):
    pairs = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if n > 1 else [])
    return [_cnot_perm(n, c, t) for c, t in pairs]


_LETTERS = "acdefghi"


def circuit_expvals(x, w, ops, n=4):
    """x: (B, n) real in [-1, 1]; w: (L, n, 3) real. Returns (B, n) expectation values <Z_i>."""
    theta = x * (np.pi / 2.0)
    qubits = [ops.stack([ops.cos(theta[:, i] / 2), ops.sin(theta[:, i] / 2)], axis=1) for i in range(n)]
    idx = _LETTERS[:n]
    psi = ops.einsum(",".join("b" + c for c in idx) + "->b" + idx, *qubits)  # product state (B, 2, ..., 2)
    psi = ops.to_complex(psi)
    perms = _ring_perms(n)
    L = ops.shape(w)[0]
    for l in range(L):
        for i in range(n):
            phi, th, om = w[l, i, 0], w[l, i, 1], w[l, i, 2]
            c, s = ops.cos(th / 2), ops.sin(th / 2)
            a, b = (phi + om) / 2, (phi - om) / 2
            u = ops.stack([
                ops.stack([ops.complex(ops.cos(a) * c, -ops.sin(a) * c), ops.complex(-ops.cos(b) * s, -ops.sin(b) * s)]),
                ops.stack([ops.complex(ops.cos(b) * s, -ops.sin(b) * s), ops.complex(ops.cos(a) * c, ops.sin(a) * c)]),
            ])
            out = "b" + idx[:i] + "z" + idx[i + 1:]
            psi = ops.einsum("z" + idx[i] + ",b" + idx + "->" + out, u, psi)
            psi = ops.reshape(psi, [-1] + [2] * n)
        flat = ops.reshape(psi, [-1, 2 ** n])
        for p in perms:
            flat = ops.gather(flat, p)
        psi = ops.reshape(flat, [-1] + [2] * n)
    prob = ops.abs2(psi)
    z = ops.const([1.0, -1.0])
    outs = []
    for i in range(n):
        outs.append(ops.einsum("b" + idx + "," + idx[i] + "->b", prob, z))
    return ops.stack(outs, axis=1)


class NumpyOps:
    stack = staticmethod(lambda xs, axis=0: np.stack(xs, axis=axis))
    cos, sin = staticmethod(np.cos), staticmethod(np.sin)
    einsum = staticmethod(np.einsum)
    to_complex = staticmethod(lambda a: a.astype(complex))
    complex = staticmethod(lambda re, im: re + 1j * im)
    reshape = staticmethod(lambda a, s: np.reshape(a, s))
    gather = staticmethod(lambda a, p: a[:, p])
    abs2 = staticmethod(lambda a: np.abs(a) ** 2)
    const = staticmethod(lambda v: np.array(v))
    shape = staticmethod(lambda a: np.shape(a))


def tf_ops():
    import tensorflow as tf

    class TFOps:
        stack = staticmethod(lambda xs, axis=0: tf.stack(xs, axis=axis))
        cos, sin = staticmethod(tf.cos), staticmethod(tf.sin)
        einsum = staticmethod(tf.einsum)
        to_complex = staticmethod(lambda a: tf.cast(a, tf.complex64))
        complex = staticmethod(lambda re, im: tf.complex(re, im))
        reshape = staticmethod(lambda a, s: tf.reshape(a, s))
        gather = staticmethod(lambda a, p: tf.gather(a, p, axis=1))
        abs2 = staticmethod(lambda a: tf.math.real(a * tf.math.conj(a)))
        const = staticmethod(lambda v: tf.constant(v, tf.float32))
        shape = staticmethod(lambda a: a.shape)

    return TFOps
