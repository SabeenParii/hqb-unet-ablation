"""summary.json of future runs records the software environment and the quantum backend."""
import pytest

pytest.importorskip("tensorflow")


def test_environment_info_fields():
    from qseg.config import Config
    from qseg.train import environment_info
    info = environment_info(Config())
    assert info["quantum_backend"] == "tf"
    for k in ("python_version", "numpy_version", "pandas_version", "scipy_version", "pennylane_version"):
        assert k in info
