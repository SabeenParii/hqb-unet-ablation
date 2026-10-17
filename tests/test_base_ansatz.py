"""Jacobian rank of the base-design ansatz (Lim & Kim 2026) and of our circuit (Discussion IV-B, Table 2).

Lim & Kim: 96 parameters, rank 88, the 8 zero columns are the final-layer RZ(phi) (last applied).
Ours: 24 parameters, rank 20, the 4 zero columns are the final-layer RZ(omega).
"""
import numpy as np

from qseg.base_ansatz_check import analyse, circuit, final_layer_last_rz, jacobian
from qseg.quantum_reference import revised_circuit


def test_lim_ansatz_rank_88_of_96():
    r = analyse("lim", n_inputs=16, n_settings=1)
    assert r["n_params"] == 96 and r["rank"] == 88
    assert r["zero"] == final_layer_last_rz(8, 4)
    assert all(r["names"][k].startswith("L4 ") and r["names"][k].endswith("RZ(phi)") for k in r["zero"])


def test_our_ansatz_rank_20_of_24():
    r = analyse("ours", n_inputs=16, n_settings=2)
    assert r["n_params"] == 24 and r["rank"] == 20
    assert r["zero"] == final_layer_last_rz(4, 2)
    assert all(r["names"][k].startswith("L2 ") and r["names"][k].endswith("RZ(omega)") for k in r["zero"])


def test_generic_simulator_matches_reference():
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, (6, 4)); w = rng.uniform(0, 2 * np.pi, (2, 4, 3))
    ref = np.stack([revised_circuit(xi, w) for xi in x])
    assert np.abs(circuit(x, w, np.pi / 2) - ref).max() < 1e-10


def test_finite_differences_match_parameter_shift():
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, (3, 8)); w = rng.uniform(0, 2 * np.pi, (4, 8, 3))
    assert np.abs(jacobian(x, w, 1.0, "fd") - jacobian(x, w, 1.0, "shift")).max() < 1e-8
