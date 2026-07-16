"""Data pipeline tests on a synthetic 2-series x 10-day mini-M5.

The fixtures mirror the real schema exactly (wide sales with d_ columns,
calendar with per-state SNAP flags, weekly prices) so every join is
exercised, including the tricky cases: SNAP resolution per state and
missing prices before an item's launch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m5forecast.data.preprocess import join_calendar, join_prices, melt_sales
from m5forecast.data.validate import DataValidationError, validate_panel

N_DAYS = 10
DATES = pd.date_range("2011-01-29", periods=N_DAYS)
# 10 days spanning two Walmart weeks: 7 days in week 11101, 3 in 11102
WEEKS = [11101] * 7 + [11102] * 3


@pytest.fixture
def sales_wide() -> pd.DataFrame:
    rows = [
        # CA series sells every day; TX series launches in week 2 (zeros before)
        ["FOODS_1_001_CA_1_evaluation", "FOODS_1_001", "FOODS_1", "FOODS", "CA_1", "CA"],
        ["HOBBIES_1_002_TX_1_evaluation", "HOBBIES_1_002", "HOBBIES_1", "HOBBIES", "TX_1", "TX"],
    ]
    df = pd.DataFrame(rows, columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
    for c in df.columns:
        df[c] = df[c].astype("category")
    sales_ca = [3, 0, 5, 2, 1, 0, 4, 2, 6, 1]
    sales_tx = [0, 0, 0, 0, 0, 0, 0, 1, 0, 2]  # launches on day 8
    for i in range(N_DAYS):
        df[f"d_{i + 1}"] = np.array([sales_ca[i], sales_tx[i]], dtype="int16")
    return df


@pytest.fixture
def calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "d": [f"d_{i + 1}" for i in range(N_DAYS)],
            "date": DATES,
            "wm_yr_wk": pd.array(WEEKS, dtype="int32"),
            "event_name_1": pd.Categorical([None] * 8 + ["SuperBowl", None]),
            "event_type_1": pd.Categorical([None] * 8 + ["Sporting", None]),
            "event_name_2": pd.Categorical([None] * N_DAYS),
            "event_type_2": pd.Categorical([None] * N_DAYS),
            "snap_CA": pd.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype="int8"),
            "snap_TX": pd.array([0, 1, 0, 1, 0, 0, 0, 0, 1, 0], dtype="int8"),
            "snap_WI": pd.array([0] * N_DAYS, dtype="int8"),
        }
    )


@pytest.fixture
def prices() -> pd.DataFrame:
    # CA item priced in both weeks; TX item priced only from week 11102 (launch)
    return pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1", "TX_1"],
            "item_id": ["FOODS_1_001", "FOODS_1_001", "HOBBIES_1_002"],
            "wm_yr_wk": pd.array([11101, 11102, 11102], dtype="int32"),
            "sell_price": pd.array([2.50, 2.24, 9.99], dtype="float32"),
        }
    )


@pytest.fixture
def panel(sales_wide, calendar, prices) -> pd.DataFrame:
    long = melt_sales(sales_wide)
    long = join_calendar(long, calendar)
    long = join_prices(long, prices)
    return long.sort_values(["id", "d"], ignore_index=True)


def test_melt_shape_and_dtypes(sales_wide):
    long = melt_sales(sales_wide)
    assert len(long) == 2 * N_DAYS
    assert long["d"].dtype == "int16"
    assert long["sales"].dtype == "int16"


def test_melt_preserves_values(sales_wide):
    long = melt_sales(sales_wide)
    ca_day3 = long[(long["item_id"] == "FOODS_1_001") & (long["d"] == 3)]["sales"].item()
    assert ca_day3 == 5


def test_snap_resolved_per_state(panel):
    ca = panel[panel["state_id"] == "CA"].sort_values("d")
    tx = panel[panel["state_id"] == "TX"].sort_values("d")
    assert ca["snap"].tolist() == [1, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    assert tx["snap"].tolist() == [0, 1, 0, 1, 0, 0, 0, 0, 1, 0]
    assert "snap_CA" not in panel.columns


def test_calendar_join_attaches_events(panel):
    day9 = panel[panel["d"] == 9]
    assert (day9["event_name_1"] == "SuperBowl").all()
    assert (panel[panel["d"] == 1]["date"] == DATES[0]).all()


def test_price_join_weekly_to_daily(panel):
    ca = panel[panel["state_id"] == "CA"].sort_values("d")
    assert ca["sell_price"].tolist() == pytest.approx([2.50] * 7 + [2.24] * 3)


def test_price_missing_before_launch(panel):
    tx = panel[panel["state_id"] == "TX"].sort_values("d")
    assert tx["sell_price"].head(7).isna().all()  # week 11101: not offered
    assert tx["sell_price"].tail(3).notna().all()


def test_validate_accepts_good_panel(panel):
    summary = validate_panel(panel, n_series=2, n_days=N_DAYS)
    assert summary["rows"] == 2 * N_DAYS
    assert 0 < summary["zero_sales_frac"] < 1


def test_validate_rejects_wrong_row_count(panel):
    with pytest.raises(DataValidationError, match="row count"):
        validate_panel(panel.iloc[:-1], n_series=2, n_days=N_DAYS)


def test_validate_rejects_negative_sales(panel):
    bad = panel.copy()
    bad.loc[0, "sales"] = -1
    with pytest.raises(DataValidationError, match="negative"):
        validate_panel(bad, n_series=2, n_days=N_DAYS)


def test_validate_rejects_sold_without_price(panel):
    bad = panel.copy()
    idx = bad.index[(bad["state_id"] == "TX") & (bad["d"] == 2)]
    bad.loc[idx, "sales"] = 3  # sold during a week with no price row
    with pytest.raises(DataValidationError, match="no price"):
        validate_panel(bad, n_series=2, n_days=N_DAYS)
