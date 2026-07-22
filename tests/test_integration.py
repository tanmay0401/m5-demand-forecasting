"""End-to-end smoke test: synthetic raw CSVs -> panel -> features -> baseline
forecast -> WRMSSE, all through the real modules. Catches cross-stage
contract breaks (column names, dtypes, id ordering) that unit tests miss.

Deliberately tiny (4 series x 120 days) so it runs in well under a second and
needs no downloaded data — this is what a CI runner executes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from m5forecast.data.preprocess import join_calendar, join_prices, melt_sales
from m5forecast.data.validate import validate_panel
from m5forecast.evaluation.backtest import expanding_folds
from m5forecast.evaluation.metrics import evaluate_point
from m5forecast.evaluation.wrmsse import level_weights, series_scales, wrmsse
from m5forecast.features.build import build_store_features, numeric_feature_columns
from m5forecast.hierarchy.aggregation import build_hierarchy
from m5forecast.models.baselines import MovingAverageForecaster

N_DAYS = 120
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _raw():
    rng = np.random.default_rng(0)
    series = [
        ("FOODS_1_001_CA_1_evaluation", "FOODS_1_001", "FOODS_1", "FOODS", "CA_1", "CA"),
        ("FOODS_1_002_CA_1_evaluation", "FOODS_1_002", "FOODS_1", "FOODS", "CA_1", "CA"),
        ("HOBBIES_1_001_TX_1_evaluation", "HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "TX_1", "TX"),
        ("HOBBIES_1_002_TX_1_evaluation", "HOBBIES_1_002", "HOBBIES_1", "HOBBIES", "TX_1", "TX"),
    ]
    sales = pd.DataFrame(series, columns=ID_COLS)
    for c in ID_COLS:
        sales[c] = sales[c].astype("category")
    day_cols = pd.DataFrame(
        {f"d_{i}": rng.poisson(3, len(series)).astype("int16") for i in range(1, N_DAYS + 1)}
    )
    sales = pd.concat([sales, day_cols], axis=1)  # build day columns at once (no fragmentation)

    weeks = [11101 + (i // 7) for i in range(N_DAYS)]
    cal = pd.DataFrame({
        "d": [f"d_{i+1}" for i in range(N_DAYS)],
        "date": pd.date_range("2011-01-29", periods=N_DAYS),
        "wm_yr_wk": pd.array(weeks, dtype="int32"),
        "event_name_1": pd.Categorical([None] * N_DAYS),
        "event_type_1": pd.Categorical([None] * N_DAYS),
        "event_name_2": pd.Categorical([None] * N_DAYS),
        "event_type_2": pd.Categorical([None] * N_DAYS),
        "snap_CA": pd.array([i % 3 == 0 for i in range(N_DAYS)], dtype="int8"),
        "snap_TX": pd.array([i % 4 == 0 for i in range(N_DAYS)], dtype="int8"),
        "snap_WI": pd.array([0] * N_DAYS, dtype="int8"),
    })
    prices = pd.DataFrame([
        {"store_id": s, "item_id": it, "wm_yr_wk": w, "sell_price": np.float32(3.0)}
        for s, it in {("CA_1", "FOODS_1_001"), ("CA_1", "FOODS_1_002"),
                      ("TX_1", "HOBBIES_1_001"), ("TX_1", "HOBBIES_1_002")}
        for w in sorted(set(weeks))
    ])
    return sales, cal, prices


CFG = OmegaConf.create({
    "data": {"horizon": 28},
    "features": {"lags": [28, 35], "rolling": {"shift": 28, "windows": [7, 28],
                 "stats": ["mean", "std", "zero_frac"]}, "ewm": {"alphas": [0.1]}},
})


def test_pipeline_end_to_end():
    sales, cal, prices = _raw()

    # data stage
    panel = join_prices(join_calendar(melt_sales(sales), cal), prices).sort_values(["id", "d"], ignore_index=True)
    validate_panel(panel, n_series=4, n_days=N_DAYS)

    # feature stage (single store slice path used by the real per-store build)
    feats = build_store_features(panel[panel["store_id"] == "CA_1"], CFG)
    assert numeric_feature_columns(feats)                      # some features exist
    assert feats.filter(like="sales_lag").notna().any().any()  # lags populated

    # model + backtest fold on the full panel
    fold = expanding_folds(n_folds=1, horizon=28, last_train_day=N_DAYS - 28)[0]
    hist = panel[panel["d"] <= fold.train_end][["id", "d", "sales"]]
    test = panel[(panel["d"] >= fold.test_start) & (panel["d"] <= fold.test_end)]
    preds = MovingAverageForecaster().fit(hist).predict(test[["id", "d"]])
    m = evaluate_point(preds, test[["id", "d", "sales"]])
    assert 0 < m["wape"] < 5 and np.isfinite(m["rmse"])

    # hierarchy + WRMSSE close the loop
    h = build_hierarchy(panel[ID_COLS].drop_duplicates("id"))
    assert h.n_nodes > 4                                        # aggregates above the 4 leaves
    order = h.bottom_ids

    def grid(df, col):
        return df.pivot_table(index="id", columns="d", values=col, fill_value=0,
                              observed=True).reindex(order).to_numpy(dtype="float64")

    scale = series_scales(grid(panel[panel["d"] <= fold.train_end], "sales"), h)
    weights = level_weights(np.ones(4), h)
    score, per_level = wrmsse(grid(preds, "yhat"), grid(test, "sales"), scale, weights, h)
    assert np.isfinite(score) and score >= 0
    assert set(per_level) == set(h.level_slices)                # every level scored
