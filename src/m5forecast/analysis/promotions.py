"""Promotions & events analysis — the project's namesake.

Three questions:
  1. On promotion days, do the models under-forecast the spike, and which
     family handles promos best? (GBMs get an explicit is_promo feature;
     DeepAR/TFT get only raw log-price — who wins?)
  2. Around calendar events, how badly does the champion under-forecast the
     build-up? (Quantifies the Phase 9 event-window flag.)
  3. What is the realized price elasticity by category — and does the naive
     "price < 85% of median = promo" flag actually capture promotions, or
     clearance markdowns on dead stock?
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def segment_metrics(pred: pd.DataFrame, truth: pd.DataFrame, mask_col: str) -> dict:
    """WAPE/bias split by a 0/1 mask column (e.g. is_promo). truth carries
    sales + the mask + cat_id; pred carries yhat. Joined on (id, d)."""
    j = pred.merge(truth, on=["id", "d"], how="inner")
    out = {}
    for label, sel in [("on", j[mask_col] == 1), ("off", j[mask_col] == 0)]:
        s = j[sel]
        err = s["yhat"] - s["sales"]
        denom = s["sales"].sum()
        out[label] = {
            "wape": float(err.abs().sum() / denom) if denom else float("nan"),
            "bias": float(err.mean()),
            "mean_actual": float(s["sales"].mean()),
            "n": int(len(s)),
        }
    return out


def error_by_event_distance(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Mean signed error (yhat - actual) bucketed by days_to_event, over the
    ±7-day window around events. Negative = under-forecast."""
    j = pred.merge(truth, on=["id", "d"], how="inner")
    j = j[j["days_to_event"] <= 7]  # within a week before an event
    j["err"] = j["yhat"] - j["sales"]
    g = j.groupby("days_to_event").agg(mean_err=("err", "mean"),
                                       mean_actual=("sales", "mean"),
                                       n=("err", "size")).reset_index()
    return g


def elasticity_by_category(store_frames) -> pd.DataFrame:
    """Realized promo lift and implied price elasticity per category.

    Streamed over per-store frames to bound memory. Accumulates, per
    (cat_id, is_promo): sum(sales), count; and on promo rows, sum(price_rel_med).
    elasticity ≈ %ΔQ / %ΔP  where  %ΔQ = lift − 1,  %ΔP = mean price ratio − 1.
    """
    agg: dict = {}
    for df in store_frames:
        for (cat, promo), g in df.groupby(["cat_id", "is_promo"], observed=True):
            a = agg.setdefault((cat, int(promo)), {"sales": 0.0, "n": 0, "prel": 0.0, "nprel": 0})
            a["sales"] += float(g["sales"].sum())
            a["n"] += int(len(g))
            if promo:
                pr = g["price_rel_med"].dropna()
                a["prel"] += float(pr.sum()); a["nprel"] += int(len(pr))

    rows = []
    cats = {c for c, _ in agg}
    for cat in cats:
        on, off = agg.get((cat, 1)), agg.get((cat, 0))
        if not on or not off or on["n"] == 0 or off["n"] == 0:
            continue
        mean_on, mean_off = on["sales"] / on["n"], off["sales"] / off["n"]
        lift = mean_on / mean_off if mean_off else float("nan")
        price_ratio = on["prel"] / on["nprel"] if on["nprel"] else float("nan")
        pct_dp = price_ratio - 1.0
        elasticity = (lift - 1.0) / pct_dp if pct_dp else float("nan")
        rows.append({"cat_id": cat, "mean_off": round(mean_off, 3), "mean_on": round(mean_on, 3),
                     "lift": round(lift, 3), "mean_price_ratio": round(price_ratio, 3),
                     "elasticity": round(elasticity, 2), "promo_days": on["n"]})
    return pd.DataFrame(rows).sort_values("cat_id", ignore_index=True)
