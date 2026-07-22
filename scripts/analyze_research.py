"""Phase 16 orchestration: per-regime model comparison + per-series winners."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from m5forecast.analysis.errors import classify, training_profile
from m5forecast.analysis.research import per_regime_wape, per_series_winners
from m5forecast.utils.config import REPO_ROOT, load_config
from m5forecast.utils.logging import get_logger

log = get_logger("scripts.analyze_research")

TRAIN_END, HORIZON = 1885, 28
MODELS = {"moving_avg_28": "fold3", "lightgbm": "fold3", "xgboost": "fold3",
          "deepar": "fold1", "tft": "fold1"}


def main():
    cfg = load_config()
    fc_dir = REPO_ROOT / cfg.paths.outputs / "forecasts"
    panel = pd.read_parquet(REPO_ROOT / cfg.paths.interim / "panel.parquet", columns=["id", "d", "sales"])

    prof = training_profile(panel[panel["d"] <= TRAIN_END])
    test = panel[(panel["d"] > TRAIN_END) & (panel["d"] <= TRAIN_END + HORIZON)]
    category = classify(prof, test.groupby("id", observed=True)["sales"].mean())
    actual = test[["id", "d", "sales"]]

    forecasts = {m: pd.read_parquet(fc_dir / m / f"{f}.parquet")[["id", "d", "yhat"]] for m, f in MODELS.items()}

    regime_wape = per_regime_wape(forecasts, actual, category)
    log.info("per-regime WAPE by model:\n%s", regime_wape.to_string())
    winners = per_series_winners(forecasts, actual, category)
    log.info("absolute-error (WAPE) winner %%: %s", winners["absolute_error_winner_pct"])
    log.info("squared-error (WRMSSE) winner %%: %s", winners["squared_error_winner_pct"])

    results = {"per_regime_wape": regime_wape.reset_index().to_dict(orient="records"), "winners": winners}
    (REPO_ROOT / cfg.paths.outputs / "research_analysis.json").write_text(json.dumps(results, indent=2))
    _figure(regime_wape, Path(REPO_ROOT / cfg.paths.figures))


def _figure(regime_wape, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from m5forecast.analysis.eda import BLUE, GREEN, MAGENTA, ORANGE, MUTED, INK_2, _save, _style

    _style()
    colors = {"moving_avg_28": ORANGE, "lightgbm": BLUE, "xgboost": MUTED, "deepar": GREEN, "tft": MAGENTA}
    regimes = list(regime_wape.index)
    models = [c for c in regime_wape.columns if c != "n"]
    x = np.arange(len(regimes)); w = 0.16
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    for i, m in enumerate(models):
        ax.bar(x + (i - 2) * w, regime_wape[m], w, color=colors.get(m, MUTED), label=m)
    ax.set_xticks(x, regimes, fontsize=8)
    ax.set_ylabel("WAPE (lower better)")
    ax.set_title("Per-regime WAPE by model — when does each family win?")
    ax.legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center")
    _save(fig, figs, "16_per_regime_wape.png")


if __name__ == "__main__":
    main()
