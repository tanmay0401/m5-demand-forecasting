"""Price & promotion features — the project's namesake family.

Leakage stance, reasoned once and documented: prices are EXOGENOUS and
KNOWN IN ADVANCE (the retailer sets them; M5 ships sell_prices for the
forecast horizon too). So current-day price needs no shift. What we still
avoid is FULL-HISTORY statistics (an item's median price over all five
years peeks at future price policy), hence trailing windows only.

Features (EDA-validated where noted):
  price_chg_7     price vs 7 days ago (week-over-week move)
  price_rel_med   price / trailing-52-week median  <- the promo detector
  is_promo        price_rel_med < 0.85 (the 85% rule validated in Phase 6:
                  dips below it produced up to 10x demand spikes)
  price_rel_dept  price / same-day (store, dept) mean price — is this item
                  cheap or premium within its shelf? (cross-sectional,
                  same-day, exogenous: no leakage possible)

Price elasticity is deliberately NOT a precomputed feature: elasticity
(d log demand / d log price) is what the MODEL learns from is_promo /
price_rel_med x demand covariation; a naive per-item regression estimate
would be noise for 73%-zero series. We measure realized elasticity in
Phase 14 instead.
"""

from __future__ import annotations

import pandas as pd

PROMO_THRESHOLD = 0.85
_MEDIAN_WEEKS = 52


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("id", observed=True)["sell_price"]
    df["price_chg_7"] = (df["sell_price"] / g.shift(7) - 1.0).astype("float32")

    # Trailing median at weekly granularity (prices only change weekly, so
    # computing on ~282 weeks/series instead of 1,941 days is ~7x cheaper).
    weekly = (
        df[["id", "wm_yr_wk", "sell_price"]]
        .dropna(subset=["sell_price"])
        .drop_duplicates(["id", "wm_yr_wk"])
        .sort_values(["id", "wm_yr_wk"])
    )
    weekly["trail_med"] = (
        weekly.groupby("id", observed=True)["sell_price"]
        .transform(lambda s: s.rolling(_MEDIAN_WEEKS, min_periods=4).median())
        .astype("float32")
    )
    df = df.merge(weekly[["id", "wm_yr_wk", "trail_med"]], on=["id", "wm_yr_wk"], how="left")

    df["price_rel_med"] = (df["sell_price"] / df["trail_med"]).astype("float32")
    df["is_promo"] = (df["price_rel_med"] < PROMO_THRESHOLD).fillna(False).astype("int8")

    dept_mean = df.groupby(["store_id", "dept_id", "d"], observed=True)["sell_price"].transform("mean")
    df["price_rel_dept"] = (df["sell_price"] / dept_mean).astype("float32")

    return df.drop(columns=["trail_med"])
