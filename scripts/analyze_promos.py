"""Phase 14 orchestration: promo model comparison, event errors, elasticity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from m5forecast.analysis.promotions import (
    elasticity_by_category,
    error_by_event_distance,
    segment_metrics,
)
from m5forecast.utils.config import REPO_ROOT, load_config
from m5forecast.utils.logging import get_logger

log = get_logger("scripts.analyze_promos")

WINDOW = range(1886, 1914)          # common eval window (all 5 models)
MODELS = {"moving_avg_28": "fold3", "lightgbm": "fold3", "xgboost": "fold3",
          "deepar": "fold1", "tft": "fold1"}
EVENT_FOLDS = {"fold1": range(1830, 1858), "fold2": range(1858, 1886)}  # LightGBM, has events


def main():
    cfg = load_config()
    fc = REPO_ROOT / cfg.paths.outputs / "forecasts"
    figs = Path(REPO_ROOT / cfg.paths.figures)
    from m5forecast.features.build import read_features, STORES

    # ---- 1. promo model comparison on FOODS, common window ----
    feats = read_features("data/processed/features",
                          columns=["id", "d", "is_promo", "cat_id", "sales"])
    win = feats[feats["d"].isin(WINDOW)]
    foods = win[win["cat_id"] == "FOODS"][["id", "d", "sales", "is_promo"]]

    promo_cmp = {}
    for name, fold in MODELS.items():
        pred = pd.read_parquet(fc / name / f"{fold}.parquet")
        promo_cmp[name] = segment_metrics(pred, foods, "is_promo")
        m = promo_cmp[name]
        log.info("%-14s FOODS promo bias=%+.3f (n=%d) | non-promo bias=%+.3f",
                 name, m["on"]["bias"], m["on"]["n"], m["off"]["bias"])

    # ---- 2. event-window error (LightGBM, folds with events) ----
    ev_feats = read_features("data/processed/features",
                             columns=["id", "d", "sales", "days_to_event"])
    ev_parts = []
    for fold, days in EVENT_FOLDS.items():
        pred = pd.read_parquet(fc / "lightgbm" / f"{fold}.parquet")
        truth = ev_feats[ev_feats["d"].isin(days)][["id", "d", "sales", "days_to_event"]]
        ev_parts.append(error_by_event_distance(pred, truth))
    event_err = pd.concat(ev_parts).groupby("days_to_event").agg(
        mean_err=("mean_err", "mean"), mean_actual=("mean_actual", "mean")).reset_index()
    log.info("event-window error by days_to_event:\n%s", event_err.round(3).to_string(index=False))

    # ---- 3. elasticity by category (streamed over full history) ----
    def store_frames():
        for s in STORES:
            yield pd.read_parquet(f"data/processed/features/store={s}.parquet",
                                  columns=["cat_id", "is_promo", "price_rel_med", "sales"])
    elas = elasticity_by_category(store_frames())
    log.info("elasticity by category:\n%s", elas.to_string(index=False))

    results = {
        "promo_comparison_foods": promo_cmp,
        "event_window_error": event_err.round(4).to_dict(orient="records"),
        "elasticity": elas.to_dict(orient="records"),
    }
    (REPO_ROOT / cfg.paths.outputs / "promo_analysis.json").write_text(json.dumps(results, indent=2))
    _figures(promo_cmp, event_err, elas, figs)


def _figures(promo_cmp, event_err, elas, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from m5forecast.analysis.eda import (BLUE, GREEN, MAGENTA, ORANGE, MUTED, INK_2, _save, _style)

    _style()
    # A: promo vs non-promo bias by model
    models = list(promo_cmp)
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - w / 2, [promo_cmp[m]["off"]["bias"] for m in models], w, color=MUTED, label="non-promo")
    ax.bar(x + w / 2, [promo_cmp[m]["on"]["bias"] for m in models], w, color=ORANGE, label="promo")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(x, models, fontsize=8)
    ax.set_ylabel("forecast bias (yhat − actual)")
    ax.set_title("FOODS: forecast bias on promo vs non-promo days (− = under-forecast)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, figs, "14_promo_bias.png")

    # B: event-window error by days-to-event
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(event_err["days_to_event"], event_err["mean_err"], color=BLUE, lw=1.8, marker="o", ms=4)
    ax.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax.invert_xaxis()
    ax.set_xlabel("days before event")
    ax.set_ylabel("mean error (yhat − actual)")
    ax.set_title("LightGBM: forecast error approaching events (− = under-forecast)")
    _save(fig, figs, "14_event_error.png")

    # C: promo lift by category
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    colors = [BLUE if l > 1 else MAGENTA for l in elas["lift"]]
    ax.bar(elas["cat_id"], elas["lift"], color=colors, width=0.6)
    ax.axhline(1.0, color=INK_2, lw=0.8, ls="--")
    for i, r in elas.iterrows():
        ax.text(i, r["lift"] + 0.03, f"{r['lift']:.2f}x", ha="center", fontsize=8, color=INK_2)
    ax.set_ylabel("demand lift on promo days (×)")
    ax.set_title("Promo demand lift by category (dashed = no effect)")
    _save(fig, figs, "14_elasticity.png")


if __name__ == "__main__":
    main()
