"""Phase 13: score every model on WRMSSE (+ pinball for probabilistic ones).

All models' stored fold forecasts cover the SAME window d1886-1913, so this
is an apples-to-apples final comparison on the official M5 metric. Point
models are scored bottom-up (their bottom forecast summed to every level);
DeepAR and TFT additionally get pinball loss + coverage on their stored
quantiles.

Writes outputs/evaluation.json and reports/figures/13_wrmsse.png.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from m5forecast.evaluation.metrics import quantile_report
from m5forecast.evaluation.wrmsse import level_weights, series_scales, wrmsse
from m5forecast.hierarchy.aggregation import build_hierarchy
from m5forecast.utils.config import REPO_ROOT, load_config
from m5forecast.utils.logging import get_logger

log = get_logger("scripts.evaluate")

TRAIN_END, HORIZON = 1885, 28
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
# model name -> stored fold file (deep models ran a single fold, id 1)
MODELS = {
    "moving_avg_28": "fold3", "lightgbm": "fold3", "xgboost": "fold3",
    "deepar": "fold1", "tft": "fold1",
}
QUANTILES = [0.05, 0.165, 0.25, 0.5, 0.75, 0.835, 0.95]


def main() -> None:
    cfg = load_config()
    fc_dir = REPO_ROOT / cfg.paths.outputs / "forecasts"
    panel = pd.read_parquet(REPO_ROOT / cfg.paths.interim / "panel.parquet",
                            columns=ID_COLS + ["d", "sales", "sell_price"])
    h = build_hierarchy(panel[ID_COLS].drop_duplicates("id"))

    def pivot(days, values="sales"):
        sub = panel[panel["d"].isin(days)]
        w = sub.pivot_table(index="id", columns="d", values=values, fill_value=0, observed=True)
        return w.reindex(h.bottom_ids).to_numpy(dtype="float64")

    log.info("building scales and weights ...")
    bottom_hist = pivot(range(1, TRAIN_END + 1))                          # full training history
    actual_bottom = pivot(range(TRAIN_END + 1, TRAIN_END + HORIZON + 1))
    scale = series_scales(bottom_hist, h)
    del bottom_hist

    rev28 = panel[panel["d"].isin(range(TRAIN_END - 27, TRAIN_END + 1))].copy()
    rev28["rev"] = rev28["sales"] * rev28["sell_price"].fillna(0)
    bottom_rev = rev28.groupby("id", observed=True)["rev"].sum().reindex(h.bottom_ids).fillna(0).to_numpy()
    weights = level_weights(bottom_rev, h)

    results = {"window": f"d{TRAIN_END+1}-d{TRAIN_END+HORIZON}", "models": {}}
    for name, fold in MODELS.items():
        f = pd.read_parquet(fc_dir / name / f"{fold}.parquet")
        fc = f.pivot_table(index="id", columns="d", values="yhat", fill_value=0, observed=True)
        bottom_fc = fc.reindex(h.bottom_ids).fillna(0).to_numpy(dtype="float64")
        score, per_level = wrmsse(bottom_fc, actual_bottom, scale, weights, h)
        entry = {"wrmsse": round(score, 4), "per_level": {k: round(v, 4) for k, v in per_level.items()}}

        qfile = fc_dir / name / f"quantiles_{fold}.parquet"
        if qfile.exists():
            q = pd.read_parquet(qfile)
            j = q.merge(panel[["id", "d", "sales"]], on=["id", "d"])
            preds = {lv: j[f"q{lv}"].to_numpy() for lv in QUANTILES if f"q{lv}" in q.columns}
            rep = quantile_report(j["sales"].to_numpy(dtype="float64"), preds)
            entry["mean_pinball"] = round(rep["mean_pinball"], 4)
            entry["coverage"] = {str(k): round(v["coverage"], 3) for k, v in rep["per_quantile"].items()}
        results["models"][name] = entry
        log.info("%-14s WRMSSE=%.4f%s", name, score,
                 f"  pinball={entry['mean_pinball']}" if "mean_pinball" in entry else "")

    out = REPO_ROOT / cfg.paths.outputs / "evaluation.json"
    out.write_text(json.dumps(results, indent=2))
    _figure(results, h, cfg)
    _summary(results)


def _figure(results, h, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    from m5forecast.analysis.eda import BLUE, GREEN, MAGENTA, ORANGE, INK, INK_2, _save, _style

    _style()
    models = list(results["models"])
    scores = [results["models"][m]["wrmsse"] for m in models]
    order = np.argsort(scores)
    models = [models[i] for i in order]; scores = [scores[i] for i in order]
    colors = [GREEN if m in ("deepar", "tft") else (ORANGE if m == "moving_avg_28" else BLUE) for m in models]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    bars = ax.barh(models, scores, color=colors, height=0.6)
    ax.invert_yaxis()
    for m, s in zip(models, scores):
        ax.text(s + 0.005, m, f"{s:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("WRMSSE (lower is better)")
    ax.set_title("Final model comparison — WRMSSE (bottom-up, all 12 levels)")
    ax.set_xlim(0, max(scores) * 1.15)
    _save(fig, Path(REPO_ROOT / cfg.paths.figures), "13_wrmsse.png")


def _summary(results):
    print(f"\n=== WRMSSE on {results['window']} (bottom-up over 12 levels) ===")
    rows = sorted(results["models"].items(), key=lambda kv: kv[1]["wrmsse"])
    for name, e in rows:
        extra = f"  pinball={e.get('mean_pinball', '  -  ')}" if "mean_pinball" in e else ""
        print(f"  {name:14s} WRMSSE={e['wrmsse']:.4f}{extra}")


if __name__ == "__main__":
    main()
