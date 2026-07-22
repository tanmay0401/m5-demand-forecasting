"""Training/backtest entry point.

    python scripts/train.py model=baselines          # all five baselines
    python scripts/train.py model=baselines training.backtest.n_folds=1

Writes per-fold forecasts to outputs/forecasts/<model>/fold<k>.parquet,
logs params+metrics to MLflow (outputs/mlruns), prints a summary table,
and saves it to outputs/metrics_<model-group>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from m5forecast.evaluation.backtest import expanding_folds
from m5forecast.evaluation.metrics import evaluate_point
from m5forecast.models.factory import build_runs, feature_lookback_days
from m5forecast.utils.config import REPO_ROOT, load_config
from m5forecast.utils.logging import get_logger
from m5forecast.utils.seed import set_seed

log = get_logger("scripts.train")


DEEPAR_COLS = ["id", "item_id", "dept_id", "store_id", "d", "sales",
               "dow", "dom", "month", "snap", "is_event", "sell_price"]


def load_feature_table(cfg, min_day: int, columns: list[str] | None = None) -> pd.DataFrame:
    """Feature rows from min_day on, streamed per store to bound memory."""
    feat_dir = REPO_ROOT / cfg.paths.processed / "features"
    parts = []
    for p in sorted(Path(feat_dir).glob("store=*.parquet")):
        df = pd.read_parquet(p, columns=columns)
        parts.append(df[df["d"] >= min_day])
    out = pd.concat(parts, ignore_index=True)
    # concat of per-store categoricals can silently object-ify; re-assert
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype("category")
    return out


def main(overrides: list[str]) -> None:
    cfg = load_config(overrides)
    set_seed(cfg.seed)

    import mlflow

    # sqlite backend: mlflow 3.x deprecated the ./mlruns file store; forward
    # slashes required in the sqlite URI even on Windows
    db = (REPO_ROOT / cfg.mlflow.db_path).as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    mlflow.set_experiment(cfg.mlflow.experiment)

    panel = pd.read_parquet(REPO_ROOT / cfg.paths.interim / "panel.parquet", columns=["id", "d", "sales"])
    folds = expanding_folds(
        n_folds=int(cfg.training.backtest.n_folds),
        horizon=int(cfg.training.horizon),
        last_train_day=int(cfg.data.train_end_day),
    )

    runs = build_runs(cfg.model)
    lookback = feature_lookback_days(cfg.model)
    features = None
    if lookback is not None:
        min_day = max(folds[0].train_end - lookback, 1)
        columns = DEEPAR_COLS if cfg.model.name in ("deepar", "tft") else None
        log.info("loading feature table from d=%d (columns=%s)", min_day, "subset" if columns else "all")
        features = load_feature_table(cfg, min_day, columns)

    results: list[dict] = []
    for fold in folds:
        history = panel[panel["d"] <= fold.train_end]
        test = panel[(panel["d"] >= fold.test_start) & (panel["d"] <= fold.test_end)]
        future, actuals = test[["id", "d"]], test[["id", "d", "sales"]]

        for name, builder in runs:
            model = builder()
            model.fit(history, features)
            preds = model.predict(future)

            out = REPO_ROOT / cfg.paths.outputs / "forecasts" / name
            out.mkdir(parents=True, exist_ok=True)
            preds.to_parquet(out / f"fold{fold.fold_id}.parquet", index=False)
            if hasattr(model, "importance_"):
                model.importance_.to_csv(out / f"importance_fold{fold.fold_id}.csv", index=False)
            if hasattr(model, "quantiles_"):
                model.quantiles_.to_parquet(out / f"quantiles_fold{fold.fold_id}.parquet", index=False)
            if hasattr(model, "attention_"):
                np.save(out / f"attention_fold{fold.fold_id}.npy", model.attention_)

            m = evaluate_point(preds, actuals)
            results.append({"model": name, "fold": fold.fold_id, **m})
            log.info("fold %d %-16s mae=%.4f rmse=%.4f wape=%.4f bias=%+.4f",
                     fold.fold_id, name, m["mae"], m["rmse"], m["wape"], m["bias"])

            with mlflow.start_run(run_name=f"{name}_fold{fold.fold_id}"):
                mlflow.log_params({"model": name, "fold": fold.fold_id,
                                   "train_end": fold.train_end, "horizon": cfg.training.horizon})
                mlflow.log_metrics({k: v for k, v in m.items() if k != "n"})

    table = pd.DataFrame(results)
    summary = table.groupby("model")[["mae", "rmse", "wape", "bias"]].mean().sort_values("wape")
    print("\n=== mean over folds ===")
    print(summary.round(4).to_string())

    out_json = REPO_ROOT / cfg.paths.outputs / f"metrics_{cfg.model.name}.json"
    out_json.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", out_json)


if __name__ == "__main__":
    main(sys.argv[1:])
