"""Typed loaders for the three raw M5 files.

Dtype discipline is not optional here: the sales file is 30,490 rows x
1,947 columns and the melted panel is ~59M rows. Loading day columns as
int16 instead of the pandas default int64 cuts sales memory 4x; loading
ids as category stores each 59M-row id column as small integer codes
plus one lookup table instead of 59M Python strings.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from m5forecast.utils.logging import get_logger

log = get_logger(__name__)

#: The six identifier columns of the sales file, hierarchy encoded in ids:
#: item_id = FOODS_3_090, dept_id = FOODS_3, cat_id = FOODS,
#: store_id = CA_1, state_id = CA, id = item_id + store_id + split suffix.
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

CALENDAR_KEEP = [
    "d",
    "date",
    "wm_yr_wk",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]


def load_sales(raw_dir: str | Path, filename: str = "sales_train_evaluation.csv") -> pd.DataFrame:
    """Load wide-format sales: one row per item-store, one column per day (d_1...).

    Day columns are int16 (max daily sales in M5 is 763 — comfortably in range).
    """
    path = Path(raw_dir) / filename
    day_cols = [c for c in pd.read_csv(path, nrows=0).columns if c.startswith("d_")]
    dtypes: dict[str, str] = {c: "int16" for c in day_cols}
    dtypes.update({c: "category" for c in ID_COLS})
    sales = pd.read_csv(path, dtype=dtypes)
    log.info("sales: %d series x %d days", len(sales), len(day_cols))
    return sales


def load_calendar(raw_dir: str | Path, filename: str = "calendar.csv") -> pd.DataFrame:
    """Load the day-level calendar: dates, Walmart week ids, events, SNAP flags."""
    path = Path(raw_dir) / filename
    cal = pd.read_csv(
        path,
        parse_dates=["date"],
        dtype={
            "event_name_1": "category",
            "event_type_1": "category",
            "event_name_2": "category",
            "event_type_2": "category",
            "snap_CA": "int8",
            "snap_TX": "int8",
            "snap_WI": "int8",
            "wm_yr_wk": "int32",
        },
    )
    log.info("calendar: %d days (%s to %s)", len(cal), cal["date"].min().date(), cal["date"].max().date())
    return cal[CALENDAR_KEEP]


def load_prices(raw_dir: str | Path, filename: str = "sell_prices.csv") -> pd.DataFrame:
    """Load weekly prices per (store, item, wm_yr_wk).

    A missing (store, item, week) row means the item was NOT offered for
    sale that week — this is signal (pre-launch periods), not dirt.
    """
    path = Path(raw_dir) / filename
    prices = pd.read_csv(
        path,
        dtype={"wm_yr_wk": "int32", "sell_price": "float32"},
    )
    log.info("prices: %d (store, item, week) rows", len(prices))
    return prices
