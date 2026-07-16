"""Wide sales -> long panel: melt, calendar join, price join.

Output contract (``panel.parquet``): exactly one row per (series, day),
sorted by (id, d), with columns

    id, item_id, dept_id, cat_id, store_id, state_id   -- identity (category)
    d (int16), date (datetime64), wm_yr_wk (int32)      -- time index
    sales (int16)                                       -- the target
    sell_price (float32, NaN = not offered that week)
    snap (int8)                                         -- SNAP flag for the row's OWN state
    event_name_1/2, event_type_1/2 (category)

Everything downstream (features, models, hierarchy) reads only this file,
never the raw CSVs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from m5forecast.data.loading import ID_COLS, load_calendar, load_prices, load_sales
from m5forecast.data.validate import validate_panel
from m5forecast.utils.io import write_parquet
from m5forecast.utils.logging import get_logger

log = get_logger(__name__)


def melt_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Wide (one row per series) -> long (one row per series-day).

    Day columns are renamed to plain ints before melting so the resulting
    ``d`` column is numeric from the start — melting 59M string labels
    like "d_1913" and slicing them afterwards would allocate gigabytes of
    throwaway Python strings.
    """
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    renamed = sales.rename(columns={c: int(c[2:]) for c in day_cols})
    long = renamed.melt(
        id_vars=ID_COLS,
        value_vars=[int(c[2:]) for c in day_cols],
        var_name="d",
        value_name="sales",
    )
    long["d"] = long["d"].astype("int16")
    long["sales"] = long["sales"].astype("int16")
    log.info("melted: %d rows (%d series x %d days)", len(long), sales.shape[0], len(day_cols))
    return long


def join_calendar(long: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Attach date, week id, events; resolve the SNAP flag for each row's state.

    The raw calendar carries snap_CA/snap_TX/snap_WI side by side; a row for a
    Texas store only ever needs snap_TX, so we collapse them into one ``snap``
    column keyed on the row's state — one honest feature instead of three
    two-thirds-irrelevant ones.
    """
    cal = calendar.copy()
    cal["d"] = cal["d"].str[2:].astype("int16")  # calendar is only 1,969 rows — cheap here
    merged = long.merge(cal, on="d", how="left")

    state = merged["state_id"].astype(str)
    merged["snap"] = np.select(
        [state == "CA", state == "TX", state == "WI"],
        [merged["snap_CA"], merged["snap_TX"], merged["snap_WI"]],
    ).astype("int8")
    return merged.drop(columns=["snap_CA", "snap_TX", "snap_WI"])


def join_prices(long: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Attach the weekly sell price on (store_id, item_id, wm_yr_wk).

    Prices are weekly, the panel is daily: every day of a week gets that
    week's price. Rows with no price row remain NaN = item not offered
    (pre-launch); validation asserts nothing was ever SOLD without a price.

    Merge keys are aligned to the panel's category dtypes first so pandas
    takes the fast categorical merge path instead of falling back to object.
    """
    aligned = prices.copy()
    for col in ("store_id", "item_id"):
        aligned[col] = aligned[col].astype(long[col].dtype)
    return long.merge(aligned, on=["store_id", "item_id", "wm_yr_wk"], how="left")


def build_panel(
    raw_dir: str | Path,
    out_path: str | Path | None = None,
    sales_file: str = "sales_train_evaluation.csv",
    n_series: int = 30490,
    n_days: int = 1941,
) -> pd.DataFrame:
    """Full data pipeline: load -> melt -> joins -> validate -> (optionally) write."""
    sales = load_sales(raw_dir, sales_file)
    calendar = load_calendar(raw_dir)
    prices = load_prices(raw_dir)

    panel = melt_sales(sales)
    panel = join_calendar(panel, calendar)
    panel = join_prices(panel, prices)
    panel = panel.sort_values(["id", "d"], ignore_index=True)

    summary = validate_panel(panel, n_series=n_series, n_days=n_days)
    log.info("panel valid: %s", summary)

    if out_path is not None:
        write_parquet(panel, out_path)
    return panel
