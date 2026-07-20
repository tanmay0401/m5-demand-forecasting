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

import pandas as pd

from m5forecast.evaluation.backtest import expanding_folds
from m5forecast.evaluation.metrics import evaluate_point
from m5forecast.models.baselines import make_baseline
from m5forecast.utils.config import REPO_ROOT, load_config
from m5forecast.utils.logging import get_logger
from m5forecast.utils.seed import set_seed

log = get_logger("scripts.train")


def load_feature_table(cfg, min_day: int) -> pd.DataFrame:
    """Feature rows from min_day on, streamed per store to bound memory."""
    from m5forecast.features.build import read_features

    feat_dir = REPO_ROOT / cfg.paths.processed / "features"
    parts = []
    for p in sorted(Path(feat_dir).glob("store=*.parquet")):
        df = pd.read_parquet(p)
        parts.append(df[df["d"] >= min_day])
    return pd.concat(parts, ignore_index=True)


def main(overrides: list[str]) -> None:
    cfg = load_config(overrides)
    set_seed(cfg.seed)

    import mlflow

    mlflow.set_tracking_uri(str(REPO_ROOT / cfg.mlflow.tracking_uri))
    mlflow.set_experiment(cfg.mlflow.experiment)

    panel = pd.read_parquet(REPO_ROOT / cfg.paths.interim / "panel.parquet", columns=["id", "d", "sales"])
    folds = expanding_folds(
        n_folds=int(cfg.training.backtest.n_folds),
        horizon=int(cfg.training.horizon),
        last_train_day=int(cfg.data.train_end_day),
    )

    members = list(cfg.model.members)
    needs_features = "linear_reg" in members
    features = None
    if needs_features:
        min_day = folds[0].train_end - 200  # LR trains on trailing 180d
        log.info("loading feature table from d=%d for linear_reg", min_day)
        features = load_feature_table(cfg, min_day)

    results: list[dict] = []
    for fold in folds:
        history = panel[panel["d"] <= fold.train_end]
        test = panel[(panel["d"] >= fold.test_start) & (panel["d"] <= fold.test_end)]
        future, actuals = test[["id", "d"]], test[["id", "d", "sales"]]

        for name in members:
            model = make_baseline(name)
            model.fit(history, features)
            preds = model.predict(future)

            out = REPO_ROOT / cfg.paths.outputs / "forecasts" / name
            out.mkdir(parents=True, exist_ok=True)
            preds.to_parquet(out / f"fold{fold.fold_id}.parquet", index=False)

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
