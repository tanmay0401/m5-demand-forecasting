"""GBM forecaster tests on a small synthetic feature table (fast configs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from m5forecast.models.factory import build_runs, feature_lookback_days
from m5forecast.models.gbm import LightGBMForecaster

N_DAYS, N_IDS = 220, 30
HORIZON = 28

FAST = OmegaConf.create(
    {
        "name": "lightgbm",
        "tweedie_variance_power": 1.1,
        "num_leaves": 15,
        "learning_rate": 0.1,
        "n_estimators": 30,
        "early_stopping_rounds": 10,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "train_days": 150,
        "valid_days": 28,
    }
)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(N_IDS):
        base = rng.uniform(0.5, 6.0)
        for d in range(1, N_DAYS + 1):
            dow = d % 7
            mu = base * (1.6 if dow in (0, 6) else 1.0)
            rows.append((f"ID_{i}", d, rng.poisson(mu), dow, base))
    df = pd.DataFrame(rows, columns=["id", "d", "sales", "dow", "base_level"])
    for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        df[c] = pd.Categorical([f"{c}_{i % 3}" for i in range(len(df))])
    # one honest engineered feature so the model has signal
    df["sales_lag_28"] = df.groupby("id")["sales"].shift(HORIZON).astype("float32")
    df["id"] = df["id"].astype("category")
    return df


def test_lightgbm_end_to_end(data):
    history = data[data["d"] <= 190][["id", "d", "sales"]]
    future = data[(data["d"] > 190) & (data["d"] <= 200)][["id", "d"]]
    model = LightGBMForecaster(FAST).fit(history, data)
    preds = model.predict(future)
    assert len(preds) == len(future)
    assert (preds["yhat"] >= 0).all()
    assert preds["yhat"].std() > 0  # not a constant predictor
    assert model.importance_.iloc[0]["gain"] > 0


def test_lightgbm_learns_weekly_pattern(data):
    """Weekend rows (built with 1.6x demand) must get higher predictions."""
    history = data[data["d"] <= 190][["id", "d", "sales"]]
    future = data[(data["d"] > 190) & (data["d"] <= 204)][["id", "d"]]
    model = LightGBMForecaster(FAST).fit(history, data)
    preds = model.predict(future).merge(data[["id", "d", "dow"]], on=["id", "d"])
    weekend = preds[preds["dow"].isin([0, 6])]["yhat"].mean()
    midweek = preds[~preds["dow"].isin([0, 6])]["yhat"].mean()
    assert weekend > midweek * 1.2


def test_factory_builds_groups():
    runs = build_runs(FAST)
    assert runs[0][0] == "lightgbm"
    assert feature_lookback_days(FAST) == 160
    baselines = OmegaConf.create({"name": "baselines", "members": ["naive"]})
    assert feature_lookback_days(baselines) is None
