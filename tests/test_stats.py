import numpy as np, pandas as pd
from qseg.stats import holm, paired_tests


def test_holm():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_paired_tests_runs():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"unet": rng.random(40) * 0.1 + 0.9, "attention_unet": rng.random(40) * 0.1 + 0.8})
    out = paired_tests(df)
    assert out.loc[0, "p_value"] < 0.001


# ---- qseg.final_stats (hierarchical bootstrap, TOST, patient-level Wilcoxon)
def _fake_results(tmp_path, shift):
    import os
    rng = np.random.default_rng(0)
    base = rng.random(12) * 0.05 + 0.9
    for m in ["unet", "attention_unet", "hybrid_quantum", "hybrid_classical", "hybrid_none"]:
        for s in (0, 1, 2):
            d = os.path.join(tmp_path, "runs", f"{m}_seed{s}")
            os.makedirs(d, exist_ok=True)
            pd.DataFrame({"file_id": [f"f{i:02d}" for i in range(12)],
                          "patient_id": [f"P{i // 2}" for i in range(12)],
                          "dice": base + (shift if m == "unet" else 0.0) + 0.001 * s}).to_csv(
                os.path.join(d, "per_image.csv"), index=False)
    return str(tmp_path)


def test_final_stats_constant_shift(tmp_path):
    from qseg.final_stats import run_all
    res = run_all(_fake_results(tmp_path, 0.01), n_boot=200)
    row = res["pairs"].iloc[0]  # C-UNet - Att-UNet
    assert abs(row.mean_diff - 0.01) < 1e-12
    assert abs(row.ci95_low - 0.01) < 1e-12 and abs(row.ci95_high - 0.01) < 1e-12
    assert not row.tost_equivalent
    same = res["pairs"].iloc[3]  # HQB-UNet - HQB-cls: identical inputs
    assert same.ci90_low == same.ci90_high == 0 and same.tost_equivalent
    assert res["wilcoxon"].iloc[0].patients_first_better == 6


def test_final_stats_bootstrap_is_seeded(tmp_path):
    from qseg.final_stats import load_per_image, patient_index, hierarchical_bootstrap
    dice, _, pats = load_per_image(_fake_results(tmp_path, 0.0))
    _, pidx = patient_index(pats)
    a = hierarchical_bootstrap(dice["unet"], None, pidx, 100, seed=0)
    b = hierarchical_bootstrap(dice["unet"], None, pidx, 100, seed=0)
    assert np.array_equal(a, b)
