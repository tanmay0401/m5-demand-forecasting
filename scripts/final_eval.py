"""Phase 18: the single-touch held-out evaluation.

Every model-development decision (features, hyperparameters, functional choice)
was made on backtest folds ending at d1913. The window d1914-1941 — the M5
public-validation block — was never trained on, tuned on, or looked at. This
script trains each model on d<=1913, forecasts d1914-1941 ONCE, and reports
WRMSSE. If it roughly matches the backtest numbers, we did not overfit the
validation process; if it is much worse, we did.

(The true M5 private block d1942-1969 is not in sales_train_evaluation.csv —
its labels were released post-competition — so d1914-1941 is the honest
held-out set available in the public data.)

Writes outputs/final_eval.json. Forecasts go to outputs/forecasts_heldout/ so
the backtest artifacts used by the analysis phases are not clobbered.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from m5forecast.evaluation.metrics import quantile_report
from m5forecast.evaluation.wrmsse import level_weights, series_scales, wrmsse
from m5forecast.hierarchy.aggregation import build_hierarchy
from m5forecast.models.baselines import make_baseline
from m5forecast.models.factory import build_runs, feature_lookback_days
from m5forecast.utils.config import REPO_ROOT, load_config
from m5forecast.utils.logging import get_logger
from m5forecast.utils.seed import set_seed
from scripts.train import DEEPAR_COLS, load_feature_table

log = get_logger("scripts.final_eval")

TRAIN_END, HORIZON = 1913, 28
TEST = range(TRAIN_END + 1, TRAIN_END + HORIZON + 1)   # d1914-1941
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
GROUPS = ["moving_avg_28", "lightgbm", "xgboost", "deepar", "tft"]
QUANTILES = [0.05, 0.165, 0.25, 0.5, 0.75, 0.835, 0.95]


def main():
    # optional CLI: run only the named models (one-per-process avoids OOM when
    # the full-history scale grid + feature table + a model coexist in RAM),
    # appending to any existing final_eval.json
    only = sys.argv[1:] or GROUPS
    base_cfg = load_config()
    panel = pd.read_parquet(REPO_ROOT / base_cfg.paths.interim / "panel.parquet",
                            columns=ID_COLS + ["d", "sales", "sell_price"])
    h = build_hierarchy(panel[ID_COLS].drop_duplicates("id"))

    def grid(df, days, col):
        sub = df[df["d"].isin(days)]
        w = sub.pivot_table(index="id", columns="d", values=col, fill_value=0, observed=True)
        return w.reindex(h.bottom_ids).fillna(0).to_numpy(dtype="float64")

    log.info("building scales + weights (train <= d%d) ...", TRAIN_END)
    scale = series_scales(grid(panel, range(1, TRAIN_END + 1), "sales"), h)
    rev = panel[panel["d"].isin(range(TRAIN_END - 27, TRAIN_END + 1))].copy()
    rev["r"] = rev["sales"] * rev["sell_price"].fillna(0)
    bottom_rev = rev.groupby("id", observed=True)["r"].sum().reindex(h.bottom_ids).fillna(0).to_numpy()
    weights = level_weights(bottom_rev, h)
    actual_bottom = grid(panel, TEST, "sales")

    history = panel[panel["d"] <= TRAIN_END][["id", "d", "sales"]]
    future = panel[panel["d"].isin(TEST)][["id", "d"]]
    out_dir = REPO_ROOT / base_cfg.paths.outputs / "forecasts_heldout"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_path = REPO_ROOT / base_cfg.paths.outputs / "final_eval.json"
    results = json.loads(eval_path.read_text()) if eval_path.exists() else \
        {"window": "d1914-d1941 (held-out, single touch)", "models": {}}
    for group in only:
        cfg = load_config([f"model={group}"] if group not in ("moving_avg_28",) else ["model=baselines"])
        set_seed(cfg.seed)

        if group == "moving_avg_28":
            model = make_baseline("moving_avg_28"); model.fit(history)
        else:
            lb = feature_lookback_days(cfg.model)
            cols = DEEPAR_COLS if group in ("deepar", "tft") else None
            feats = load_feature_table(cfg, TRAIN_END - lb, cols)
            _, builder = build_runs(cfg.model)[0]
            model = builder(); model.fit(history, feats)
            del feats

        preds = model.predict(future)
        preds.to_parquet(out_dir / f"{group}.parquet", index=False)
        bottom_fc = preds.pivot_table(index="id", columns="d", values="yhat",
                                      fill_value=0, observed=True).reindex(h.bottom_ids).fillna(0).to_numpy("float64")
        score, _ = wrmsse(bottom_fc, actual_bottom, scale, weights, h)
        entry = {"wrmsse": round(score, 4)}

        if hasattr(model, "quantiles_"):
            q = model.quantiles_.merge(panel[["id", "d", "sales"]], on=["id", "d"])
            preds_q = {lv: q[f"q{lv}"].to_numpy() for lv in QUANTILES if f"q{lv}" in q.columns}
            entry["mean_pinball"] = round(quantile_report(q["sales"].to_numpy("float64"), preds_q)["mean_pinball"], 4)
        results["models"][group] = entry
        log.info("%-14s held-out WRMSSE=%.4f%s", group, score,
                 f"  pinball={entry['mean_pinball']}" if "mean_pinball" in entry else "")

    eval_path.write_text(json.dumps(results, indent=2))
    print("\n=== HELD-OUT (d1914-1941), single touch ===")
    for g, e in sorted(results["models"].items(), key=lambda kv: kv[1]["wrmsse"]):
        print(f"  {g:14s} WRMSSE={e['wrmsse']:.4f}")


if __name__ == "__main__":
    main()
