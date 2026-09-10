import numpy as np
import pytest

from app.ml.data.evaluate import full_report, mae, nasa_score, rmse, warning_f1


def test_rmse_and_mae_known_values():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]
    # errors: 2, -2, 3 -> squared: 4, 4, 9 -> mean 5.667 -> sqrt ~2.38
    assert rmse(y_true, y_pred) == pytest.approx(2.3805, abs=1e-3)
    assert mae(y_true, y_pred) == pytest.approx((2 + 2 + 3) / 3, abs=1e-6)


def test_nasa_score_zero_for_perfect_prediction():
    y_true = [10, 20, 30]
    assert nasa_score(y_true, y_true) == pytest.approx(0.0, abs=1e-9)


def test_nasa_score_penalizes_late_predictions_more_than_early():
    """Over-predicting remaining life (predicting failure is farther away
    than it really is) is the dangerous failure mode for maintenance --
    the scoring function must penalize it more than the symmetric
    under-prediction, per the official C-MAPSS/PHM08 definition."""
    y_true = [50]
    late_pred = [60]  # d = +10 (predicted more life than there is)
    early_pred = [40]  # d = -10 (predicted less life than there is)
    assert nasa_score(y_true, late_pred) > nasa_score(y_true, early_pred)


def test_warning_f1_perfect_classification():
    y_true = [5, 40, 10, 50]  # <=30 -> [True, False, True, False]
    y_pred = [8, 35, 12, 45]  # same classification outcome
    assert warning_f1(y_true, y_pred, threshold=30) == pytest.approx(1.0)


def test_warning_f1_handles_no_positive_case():
    y_true = [100, 120]
    y_pred = [90, 110]
    assert warning_f1(y_true, y_pred, threshold=30) == pytest.approx(1.0)


def test_full_report_has_all_expected_keys():
    report = full_report([10, 20], [11, 19])
    assert set(report.keys()) == {"rmse", "mae", "nasa_score", "f1_at_threshold", "n"}
    assert report["n"] == 2
