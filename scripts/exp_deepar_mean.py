"""Phase 13 experiment: does DeepAR's WRMSSE recover if its point forecast is
the predictive MEAN instead of the median?

Hypothesis (from the per-level WRMSSE breakdown): the median under-forecasts
the right-skewed demand; bottom-up ACCUMULATES that systematic bias, so the
money-weighted aggregate levels crater. The mean is unbiased, so bottom-up
should cancel error instead. Retrains DeepAR once (fold 3 window) and scores
both point functionals on WRMSSE.
"""

from __future__ import annotations

import pandas as pd

from m5forecast.evaluation.backtest import expanding_folds
from m5forecast.evaluation.metrics import evaluate_point
from m5forecast.evaluation.wrmsse import level_weights, series_scales, wrmsse
from m5forecast.hierarchy.aggregation import build_hierarchy
from m5forecast.models.deepar.model import DeepARForecaster
from m5forecast.utils.config import REPO_ROOT, load_config
from m5forecast.utils.logging import get_logger
from m5forecast.utils.seed import set_seed
from scripts.train import DEEPAR_COLS, load_feature_table

log = get_logger("scripts.exp_deepar_mean")
TRAIN_END, HORIZON = 1885, 28
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def score(bottom_fc, actual_bottom, scale, weights, h):
    return wrmsse(bottom_fc, actual_bottom, scale, weights, h)[0]


def main():
    cfg = load_config(["model=deepar"])
    set_seed(cfg.seed)
    fold = expanding_folds(1, HORIZON, TRAIN_END)[0]

    feats = load_feature_table(cfg, 1, DEEPAR_COLS)
    panel = pd.read_parquet(REPO_ROOT / cfg.paths.interim / "panel.parquet",
                            columns=ID_COLS + ["d", "sales", "sell_price"])
    h = build_hierarchy(panel[ID_COLS].drop_duplicates("id"))

    def pivot(days, values="sales"):
        w = panel[panel["d"].isin(days)].pivot_table(index="id", columns="d", values=values,
                                                      fill_value=0, observed=True)
        return w.reindex(h.bottom_ids).to_numpy(dtype="float64")

    scale = series_scales(pivot(range(1, TRAIN_END + 1)), h)
    rev = panel[panel["d"].isin(range(TRAIN_END - 27, TRAIN_END + 1))].copy()
    rev["r"] = rev["sales"] * rev["sell_price"].fillna(0)
    bottom_rev = rev.groupby("id", observed=True)["r"].sum().reindex(h.bottom_ids).fillna(0).to_numpy()
    weights = level_weights(bottom_rev, h)
    actual_bottom = pivot(range(TRAIN_END + 1, TRAIN_END + HORIZON + 1))

    history = panel[panel["d"] <= fold.train_end][["id", "d", "sales"]]
    test = panel[(panel["d"] >= fold.test_start) & (panel["d"] <= fold.test_end)]
    future = test[["id", "d"]]

    model = DeepARForecaster(cfg.model).fit(history, feats)
    median_pred = model.predict(future)       # median point forecast
    mean_pred = model.mean_forecast_          # predictive mean

    def to_bottom(df):
        w = df.pivot_table(index="id", columns="d", values="yhat", fill_value=0, observed=True)
        return w.reindex(h.bottom_ids).fillna(0).to_numpy(dtype="float64")

    for label, pred in [("median", median_pred), ("mean", mean_pred)]:
        pt = evaluate_point(pred, test[["id", "d", "sales"]])
        w = score(to_bottom(pred), actual_bottom, scale, weights, h)
        log.info("deepar %-6s point: WAPE=%.4f bias=%+.4f  WRMSSE=%.4f", label, pt["wape"], pt["bias"], w)


if __name__ == "__main__":
    main()
