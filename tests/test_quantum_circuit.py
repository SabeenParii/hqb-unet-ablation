"""The original circuit ignores its weights; the revised one does not."""
from qseg.quantum_reference import original_circuit, revised_circuit, weight_sensitivity


def test_original_circuit_weights_have_no_effect():
    assert weight_sensitivity(original_circuit, (2, 4)) < 1e-12


def test_revised_circuit_weights_change_output():
    assert weight_sensitivity(revised_circuit, (2, 4, 3)) > 0.1


def test_backend_agnostic_simulator_matches_reference():
    import numpy as np
    from qseg.quantum_sim import circuit_expvals, NumpyOps
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (6, 4)); w = rng.uniform(0, 2 * np.pi, (2, 4, 3))
    ref = np.stack([revised_circuit(xi, w) for xi in x])
    assert np.abs(circuit_expvals(x, w, NumpyOps) - ref).max() < 1e-10
