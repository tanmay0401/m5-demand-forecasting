"""Automated error taxonomy: classify every series by failure mode from its
training history, then measure where forecast error concentrates.

The point is diagnosis, not a single number: two models with equal WRMSSE can
fail on completely different series. Bucketing error by regime tells you what
to fix — and which failures are the model's fault vs the data's (a stockout
zero is an unforecastable label, not a modelling miss).

Primary category is assigned by priority (a series can be several things; we
pick the most diagnostic):
    cold_start   < 90 days of history       (no pattern to learn yet)
    dormant      sold before, last 28 train days all zero (delist/stockout)
    sparse       >= 85% zero days           (intermittent long tail)
    demand_shock test level != recent level (regime change / shock)
    dense_stable everything else            (the "easy" series)
"""

from __future__ import annotations

import pandas as pd

TRAIN_END = 1885
TAIL = 28


def training_profile(panel_train: pd.DataFrame) -> pd.DataFrame:
    """Per-series training stats used for classification. panel_train: id,d,sales (d<=TRAIN_END)."""
    tr = panel_train
    nz = tr.assign(nz=(tr["sales"] > 0).astype("int8"))
    g = nz.groupby("id", observed=True)
    first_sale = tr[tr["sales"] > 0].groupby("id", observed=True)["d"].min()
    tail = tr[tr["d"] > TRAIN_END - TAIL].groupby("id", observed=True)["sales"].sum()

    prof = pd.DataFrame({"nonzero": g["nz"].sum(), "tail_sum": tail}).fillna(0)
    prof["first_sale"] = first_sale.reindex(prof.index).fillna(TRAIN_END + 1)
    prof["hist_days"] = (TRAIN_END - prof["first_sale"] + 1).clip(lower=0)
    prof["zero_frac"] = 1 - prof["nonzero"] / prof["hist_days"].clip(lower=1)
    prof["recent_mean"] = prof["tail_sum"] / TAIL
    return prof


def classify(prof: pd.DataFrame, test_mean: pd.Series) -> pd.Series:
    """Assign each series its primary failure category (priority order)."""
    prof = prof.copy()
    prof["test_mean"] = test_mean.reindex(prof.index).fillna(0)
    cat = pd.Series("dense_stable", index=prof.index, dtype=object)

    shock = (
        (prof["recent_mean"] > 0.2)
        & ((prof["test_mean"] > 2 * prof["recent_mean"]) | (prof["test_mean"] < 0.5 * prof["recent_mean"]))
    )
    cat[shock] = "demand_shock"
    cat[prof["zero_frac"] >= 0.85] = "sparse"
    cat[(prof["tail_sum"] == 0) & (prof["nonzero"] > 0)] = "dormant"
    cat[prof["hist_days"] < 90] = "cold_start"
    return cat


def error_taxonomy(forecast: pd.DataFrame, actual: pd.DataFrame, category: pd.Series) -> pd.DataFrame:
    """Per-category error mass. forecast: id,d,yhat ; actual: id,d,sales."""
    j = forecast.merge(actual, on=["id", "d"], how="inner")
    j["abs_err"] = (j["yhat"] - j["sales"]).abs()
    j["cat"] = j["id"].map(category)

    total_abs = j["abs_err"].sum()
    total_sales = j["sales"].sum()
    rows = []
    for c, g in j.groupby("cat"):
        n_series = g["id"].nunique()
        rows.append({
            "category": c,
            "n_series": n_series,
            "series_pct": round(100 * n_series / j["id"].nunique(), 1),
            "abs_err": round(float(g["abs_err"].sum()), 0),
            "err_share_pct": round(100 * g["abs_err"].sum() / total_abs, 1),
            "sales_share_pct": round(100 * g["sales"].sum() / total_sales, 1),
            "wape": round(float(g["abs_err"].sum() / max(g["sales"].sum(), 1)), 3),
            "mean_actual": round(float(g["sales"].mean()), 3),
        })
    return pd.DataFrame(rows).sort_values("err_share_pct", ascending=False, ignore_index=True)


def worst_series(forecast: pd.DataFrame, actual: pd.DataFrame, category: pd.Series, n: int = 10) -> pd.DataFrame:
    """The n series with the largest total absolute error, with their category."""
    j = forecast.merge(actual, on=["id", "d"], how="inner")
    j["abs_err"] = (j["yhat"] - j["sales"]).abs()
    per = j.groupby("id", observed=True).agg(abs_err=("abs_err", "sum"),
                                             actual=("sales", "sum"),
                                             fc=("yhat", "sum")).sort_values("abs_err", ascending=False)
    per["category"] = per.index.map(category)
    return per.head(n).reset_index()
