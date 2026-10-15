import numpy as np
from qseg.evaluation import evaluate_predictions, metrics_from_confusion, expected_calibration_error


def test_dice_equals_f1_and_iou_relation():
    rng = np.random.default_rng(1)
    y = (rng.random((10, 32, 32, 1)) < 0.3).astype(np.uint8)
    prob = np.clip(y * 0.7 + rng.random(y.shape) * 0.5, 0, 1)
    s, per = evaluate_predictions(y, prob, np.repeat(np.arange(5), 2), n_boot=200)
    f1 = 2 * s["precision"] * s["recall"] / (s["precision"] + s["recall"])
    assert abs(s["dice"] - f1) < 1e-12
    assert abs(s["iou"] - s["dice"] / (2 - s["dice"])) < 1e-12
    assert s["dice_ci_low"] <= s["dice"] <= s["dice_ci_high"]
    assert len(per) == 10


def test_perfect_prediction():
    m = metrics_from_confusion(50, 50, 0, 0)
    assert m["dice"] == m["iou"] == m["mcc"] == m["accuracy"] == 1.0


def test_ece_perfectly_calibrated_is_zero():
    y = np.array([1, 1, 0, 0])
    p = np.array([1.0, 1.0, 0.0, 0.0])
    assert expected_calibration_error(y, p) == 0.0
