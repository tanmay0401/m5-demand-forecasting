"""The humble models everything must beat (92% of M5 teams didn't).

Each baseline answers "what if we assume almost nothing?":
    naive            demand tomorrow = demand today (random-walk assumption)
    seasonal_naive_7 demand = same weekday last week (weekly cycle is all)
    moving_avg_28    demand = last month's average level (level is all)
    exp_smoothing    demand = exponentially-decayed level (Phase 2, SES)
    linear_reg       linear map over the Phase 7 features (are interactions
                     and non-linearity even needed?)

All are global across 30,490 series but trivially per-series (vectorized
groupbys — no loops over series).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from m5forecast.models.base import ForecastModel
from m5forecast.utils.logging import get_logger

log = get_logger(__name__)


class NaiveForecaster(ForecastModel):
    name = "naive"

    def fit(self, history, features=None):
        last_day = history["d"].max()
        self._level = history[history["d"] == last_day].set_index("id")["sales"]
        return self

    def predict(self, future):
        return self._finalize(future, future["id"].map(self._level))


class SeasonalNaiveForecaster(ForecastModel):
    """y(T+h) = y(T + h - 7*ceil(h/7)) — the most recent same weekday."""

    name = "seasonal_naive_7"

    def fit(self, history, features=None):
        self._train_end = int(history["d"].max())
        tail = history[history["d"] > self._train_end - 7]
        self._week = tail.set_index(["id", "d"])["sales"]
        return self

    def predict(self, future):
        h = future["d"] - self._train_end
        k = np.ceil(h / 7).astype(int)
        source_d = future["d"] - 7 * k
        idx = pd.MultiIndex.from_arrays([future["id"], source_d])
        return self._finalize(future, pd.Series(self._week.reindex(idx).to_numpy(), index=future.index))


class MovingAverageForecaster(ForecastModel):
    name = "moving_avg_28"

    def __init__(self, window: int = 28):
        self.window = window

    def fit(self, history, features=None):
        train_end = history["d"].max()
        tail = history[history["d"] > train_end - self.window]
        self._level = tail.groupby("id", observed=True)["sales"].mean()
        return self

    def predict(self, future):
        return self._finalize(future, future["id"].map(self._level))


class ExpSmoothingForecaster(ForecastModel):
    """Simple exponential smoothing: flat forecast at the final smoothed level.

    alpha fixed at 0.2 (~ balances the last two weeks against deeper history);
    per-series alpha selection is a documented possible refinement, not done
    here — baselines should stay dumb.
    """

    name = "exp_smoothing"

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def fit(self, history, features=None):
        g = history.sort_values("d").groupby("id", observed=True)["sales"]
        self._level = g.apply(lambda s: s.ewm(alpha=self.alpha).mean().iloc[-1])
        return self

    def predict(self, future):
        return self._finalize(future, future["id"].map(self._level))


class LinearRegressionForecaster(ForecastModel):
    """Ordinary least squares on the Phase 7 feature table.

    The bridge baseline: same features as LightGBM, but no interactions and
    no non-linearity — the gap between this and Phase 9 IS the measured value
    of trees. Trains on the trailing `train_days` of history (OLS on the full
    46M-row table would exceed RAM for no accuracy gain).
    """

    name = "linear_reg"

    def __init__(self, train_days: int = 180):
        self.train_days = train_days

    def fit(self, history, features=None):
        from sklearn.linear_model import LinearRegression

        if features is None:
            raise ValueError("linear_reg needs the feature table")
        from m5forecast.features.build import numeric_feature_columns

        self._cols = numeric_feature_columns(features)
        train_end = history["d"].max()
        rows = features[(features["d"] > train_end - self.train_days) & (features["d"] <= train_end)]
        x = rows[self._cols].fillna(0.0).to_numpy(dtype="float32")
        y = rows["sales"].to_numpy(dtype="float32")
        self._model = LinearRegression().fit(x, y)
        self._features = features
        return self

    def predict(self, future):
        rows = future.merge(self._features, on=["id", "d"], how="left", suffixes=("", "_f"))
        x = rows[self._cols].fillna(0.0).to_numpy(dtype="float32")
        return self._finalize(future, pd.Series(self._model.predict(x), index=future.index))


REGISTRY = {
    m.name: m
    for m in [NaiveForecaster, SeasonalNaiveForecaster, MovingAverageForecaster, ExpSmoothingForecaster, LinearRegressionForecaster]
}


def make_baseline(name: str) -> ForecastModel:
    if name not in REGISTRY:
        raise KeyError(f"unknown baseline '{name}' (have: {sorted(REGISTRY)})")
    return REGISTRY[name]()
