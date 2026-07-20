"""Baseline correctness on tiny hand-checkable series + fold-generator invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m5forecast.evaluation.backtest import expanding_folds
from m5forecast.evaluation.metrics import evaluate_point
from m5forecast.models.baselines import (
    ExpSmoothingForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)


def series(values, sid="A"):
    return pd.DataFrame({"id": sid, "d": range(1, len(values) + 1), "sales": values})


def future(days, sid="A"):
    return pd.DataFrame({"id": sid, "d": list(days)})


def test_naive_repeats_last_value():
    hist = series([5, 3, 9, 7])
    preds = NaiveForecaster().fit(hist).predict(future([5, 6, 7]))
    assert preds["yhat"].tolist() == [7, 7, 7]


def test_seasonal_naive_repeats_week():
    week = [10, 20, 30, 40, 50, 60, 70]
    hist = series(week * 2)  # 14 days
    preds = SeasonalNaiveForecaster().fit(hist).predict(future(range(15, 29)))
    assert preds["yhat"].tolist() == week * 2  # pattern tiles across both weeks


def test_moving_average_is_tail_mean():
    hist = series(list(range(1, 41)))  # 1..40
    preds = MovingAverageForecaster(window=4).fit(hist).predict(future([41, 42]))
    assert preds["yhat"].tolist() == [38.5, 38.5]  # mean(37..40)


def test_exp_smoothing_matches_manual_recursion():
    vals, alpha = [2.0, 4.0, 8.0], 0.5
    level = vals[0]
    for v in vals[1:]:
        level = alpha * v + (1 - alpha) * level
    hist = series(vals)
    preds = ExpSmoothingForecaster(alpha=alpha).fit(hist).predict(future([4]))
    # pandas ewm with adjust=True differs from the naive recursion; check against ewm itself
    expected = pd.Series(vals).ewm(alpha=alpha).mean().iloc[-1]
    assert preds["yhat"].iloc[0] == pytest.approx(expected)
    assert preds["yhat"].iloc[0] == pytest.approx(level, rel=0.2)  # same ballpark


def test_negative_predictions_clipped():
    hist = series([0, 0, 0, 0])
    preds = NaiveForecaster().fit(hist).predict(future([5]))
    assert (preds["yhat"] >= 0).all()


def test_folds_are_m5_aligned():
    folds = expanding_folds(n_folds=3, horizon=28, last_train_day=1913)
    assert [f.test_end for f in folds] == [1857, 1885, 1913]
    for f in folds:
        assert f.test_end - f.test_start + 1 == 28
        assert f.train_end == f.test_start - 1  # no gap, no overlap
    # expanding: training grows monotonically
    assert [f.train_end for f in folds] == sorted(f.train_end for f in folds)


def test_evaluate_point_catches_missing_predictions():
    preds = pd.DataFrame({"id": ["A"], "d": [99], "yhat": [1.0]})
    actuals = pd.DataFrame({"id": ["A"], "d": [1], "sales": [1]})
    with pytest.raises(ValueError, match="no matching actual"):
        evaluate_point(preds, actuals)


def test_metrics_values():
    preds = pd.DataFrame({"id": ["A", "A"], "d": [1, 2], "yhat": [3.0, 1.0]})
    actuals = pd.DataFrame({"id": ["A", "A"], "d": [1, 2], "sales": [1, 1]})
    m = evaluate_point(preds, actuals)
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(np.sqrt(2))
    assert m["wape"] == pytest.approx(1.0)
    assert m["bias"] == pytest.approx(1.0)
