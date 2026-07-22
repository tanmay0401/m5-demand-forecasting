"""Phase 15 orchestration: build the error taxonomy for the champion model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from m5forecast.analysis.errors import (
    classify,
    error_taxonomy,
    training_profile,
    worst_series,
)
from m5forecast.utils.config import REPO_ROOT, load_config
from m5forecast.utils.logging import get_logger

log = get_logger("scripts.analyze_errors")

TRAIN_END, HORIZON = 1885, 28
MODEL, FOLD = "lightgbm", "fold3"


def main():
    cfg = load_config()
    panel = pd.read_parquet(REPO_ROOT / cfg.paths.interim / "panel.parquet", columns=["id", "d", "sales"])
    prof = training_profile(panel[panel["d"] <= TRAIN_END])

    test = panel[(panel["d"] > TRAIN_END) & (panel["d"] <= TRAIN_END + HORIZON)]
    test_mean = test.groupby("id", observed=True)["sales"].mean()
    category = classify(prof, test_mean)
    log.info("category counts:\n%s", category.value_counts().to_string())

    fc = pd.read_parquet(REPO_ROOT / cfg.paths.outputs / "forecasts" / MODEL / f"{FOLD}.parquet")
    actual = test[["id", "d", "sales"]]

    tax = error_taxonomy(fc, actual, category)
    log.info("error taxonomy (%s):\n%s", MODEL, tax.to_string(index=False))
    worst = worst_series(fc, actual, category, n=10)
    log.info("worst 10 series by abs error:\n%s", worst.to_string(index=False))

    results = {"model": MODEL, "taxonomy": tax.to_dict(orient="records"),
               "worst_series": worst.round(2).to_dict(orient="records")}
    (REPO_ROOT / cfg.paths.outputs / "error_analysis.json").write_text(json.dumps(results, indent=2))
    _figure(tax, Path(REPO_ROOT / cfg.paths.figures))


def _figure(tax, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from m5forecast.analysis.eda import BLUE, INK_2, MUTED, _save, _style

    _style()
    t = tax.sort_values("err_share_pct")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.barh(y - 0.2, t["err_share_pct"], 0.38, color=BLUE, label="% of total error")
    ax.barh(y + 0.2, t["series_pct"], 0.38, color=MUTED, label="% of series")
    ax.set_yticks(y, t["category"])
    for i, r in enumerate(t.itertuples()):
        ax.text(r.err_share_pct + 0.6, i - 0.2, f"{r.err_share_pct:.0f}%", va="center", fontsize=7.5, color=INK_2)
    ax.set_xlabel("percent")
    ax.set_title("Where LightGBM's error concentrates, by failure mode")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _save(fig, figs, "15_error_taxonomy.png")


if __name__ == "__main__":
    main()
