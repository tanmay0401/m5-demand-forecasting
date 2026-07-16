"""Structural validation of the long panel.

Silent data corruption is the classic pipeline killer: a bad join
silently duplicating rows, a dtype overflow turning sales negative, a
missing week of calendar. Every check here raises loudly instead of
letting a corrupted artifact flow downstream into features and models.
"""

from __future__ import annotations

import pandas as pd


class DataValidationError(ValueError):
    """Raised when the panel violates its output contract."""


def validate_panel(panel: pd.DataFrame, n_series: int, n_days: int) -> dict:
    """Assert the panel contract; return a summary dict on success.

    Checks:
      1. exactly n_series * n_days rows (a bad join multiplies or drops rows)
      2. target completeness: no null, no negative sales
      3. time index integrity: n_days distinct days and distinct dates
      4. price sanity: no row with sales > 0 but no price
         (missing price = "not offered" and must imply zero sales)
    """
    problems: list[str] = []

    expected = n_series * n_days
    if len(panel) != expected:
        problems.append(f"row count {len(panel):,} != {n_series:,} series x {n_days:,} days = {expected:,}")

    if panel["sales"].isna().any():
        problems.append("null sales values present")
    if (panel["sales"] < 0).any():
        problems.append("negative sales values present (dtype overflow or bad source)")

    if panel["d"].nunique() != n_days:
        problems.append(f"expected {n_days} distinct day indices, found {panel['d'].nunique()}")
    if panel["date"].nunique() != n_days:
        problems.append(f"expected {n_days} distinct dates, found {panel['date'].nunique()}")

    sold_without_price = int(((panel["sales"] > 0) & panel["sell_price"].isna()).sum())
    if sold_without_price:
        problems.append(f"{sold_without_price:,} rows sold units with no price (broken price join)")

    if problems:
        raise DataValidationError("panel validation failed:\n  - " + "\n  - ".join(problems))

    return {
        "rows": len(panel),
        "series": n_series,
        "days": n_days,
        "date_range": f"{panel['date'].min().date()} to {panel['date'].max().date()}",
        "zero_sales_frac": round(float((panel["sales"] == 0).mean()), 4),
        "price_missing_frac": round(float(panel["sell_price"].isna().mean()), 4),
    }
