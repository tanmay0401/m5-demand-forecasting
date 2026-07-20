"""Feature tests. The centerpiece is the leakage test: corrupt every
sales value inside the forecast window (t-27..t) and assert that no
target-derived feature at day t changes. If any feature moves, it saw
the future — the exact bug that produces too-good-to-be-true validation
scores (Phase 2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m5forecast.features.calendar import add_calendar_features
from m5forecast.features.lags import (
    add_ewm_features,
    add_expanding_mean,
    add_lag_features,
    add_momentum,
    add_rolling_features,
)
from m5forecast.features.price import add_price_features

HORIZON = 28
N = 140  # days per series


def make_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for sid, (store, dept) in enumerate([("CA_1", "FOODS_1"), ("CA_1", "HOBBIES_1")]):
        d = np.arange(1, N + 1, dtype="int16")
        dates = pd.date_range("2011-01-29", periods=N)
        frames.append(
            pd.DataFrame(
                {
                    "id": f"ITEM_{sid}",
                    "store_id": store,
                    "dept_id": dept,
                    "d": d,
                    "date": dates,
                    "wm_yr_wk": (11101 + (np.arange(N) // 7)).astype("int32"),
                    "sales": rng.poisson(3, N).astype("int16"),
                    "sell_price": np.float32(4.0),
                    "event_name_1": [None] * N,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["id"] = df["id"].astype("category")
    return df.sort_values(["id", "d"], ignore_index=True)


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(df, [28, 35], HORIZON)
    df = add_rolling_features(df, [7, 28], ["mean", "std", "zero_frac"], HORIZON, HORIZON)
    df = add_ewm_features(df, [0.1], HORIZON, HORIZON)
    df = add_expanding_mean(df, HORIZON, HORIZON)
    return add_momentum(df)


TARGET_FEATURES = [
    "sales_lag_28", "sales_lag_35", "r_mean_7", "r_std_7", "r_zero_frac_7",
    "r_mean_28", "r_std_28", "r_zero_frac_28", "ewm_a1", "hist_mean", "momentum_7_28",
]


def test_future_perturbation_changes_nothing():
    """Corrupt sales inside (t-28, t] -> every feature at day t must be identical."""
    t = 120
    clean = featurize(make_panel())
    dirty_panel = make_panel()
    window = (dirty_panel["d"] > t - HORIZON) & (dirty_panel["d"] <= t)
    dirty_panel.loc[window, "sales"] += 1000
    dirty = featurize(dirty_panel)

    at_t_clean = clean[clean["d"] == t][TARGET_FEATURES].reset_index(drop=True)
    at_t_dirty = dirty[dirty["d"] == t][TARGET_FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(at_t_clean, at_t_dirty)


def test_lag_values_correct():
    df = featurize(make_panel())
    one = df[df["id"] == "ITEM_0"].set_index("d")
    assert one.loc[100, "sales_lag_28"] == one.loc[72, "sales"]
    assert np.isnan(one.loc[28, "sales_lag_28"])  # no history yet


def test_rolling_mean_matches_manual():
    df = featurize(make_panel())
    one = df[df["id"] == "ITEM_0"].set_index("d")
    manual = one.loc[100 - HORIZON - 6 : 100 - HORIZON, "sales"].mean()
    assert one.loc[100, "r_mean_7"] == pytest.approx(manual)


def test_lag_below_horizon_rejected():
    with pytest.raises(ValueError, match="leak"):
        add_lag_features(make_panel(), [7], HORIZON)


def test_momentum_is_ratio():
    df = featurize(make_panel())
    row = df[(df["id"] == "ITEM_0") & (df["d"] == 120)].iloc[0]
    assert row["momentum_7_28"] == pytest.approx(row["r_mean_7"] / row["r_mean_28"])


def test_event_distances():
    df = make_panel()
    df.loc[df["d"] == 10, "event_name_1"] = "SuperBowl"
    out = add_calendar_features(df)
    one = out[out["id"] == "ITEM_0"].set_index("d")
    assert one.loc[7, "days_to_event"] == 3
    assert one.loc[10, "days_to_event"] == 0 and one.loc[10, "is_event"] == 1
    assert one.loc[12, "days_since_event"] == 2


def test_calendar_basics():
    out = add_calendar_features(make_panel())
    one = out[out["id"] == "ITEM_0"].set_index("d")
    assert one.loc[1, "dow"] == 5  # 2011-01-29 was a Saturday
    assert one.loc[1, "is_weekend"] == 1
    assert one["is_christmas"].sum() == 0  # window ends before December


def test_promo_flag_fires_on_price_cut():
    df = make_panel()
    # weeks are 7 days: cut price 25% for days 106-112 (one full week)
    cut = (df["id"] == "ITEM_0") & (df["d"].between(106, 112))
    df.loc[cut, "sell_price"] = np.float32(3.0)
    out = add_price_features(df)
    one = out[out["id"] == "ITEM_0"].set_index("d")
    assert one.loc[108, "is_promo"] == 1
    assert one.loc[95, "is_promo"] == 0
    assert one.loc[108, "price_rel_med"] == pytest.approx(0.75, abs=0.01)


def test_price_rel_dept_cross_section():
    df = make_panel()
    df.loc[df["id"] == "ITEM_1", "dept_id"] = "FOODS_1"  # same shelf as ITEM_0
    df.loc[df["id"] == "ITEM_1", "sell_price"] = np.float32(8.0)
    out = add_price_features(df)
    day = out[out["d"] == 50].set_index("id")
    assert day.loc["ITEM_0", "price_rel_dept"] == pytest.approx(4.0 / 6.0)
    assert day.loc["ITEM_1", "price_rel_dept"] == pytest.approx(8.0 / 6.0)
