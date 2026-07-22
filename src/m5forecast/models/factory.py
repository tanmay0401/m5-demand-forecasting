"""Model construction from config — the only place model names map to classes."""

from __future__ import annotations

from collections.abc import Callable

from m5forecast.models.base import ForecastModel
from m5forecast.models.baselines import make_baseline


def build_runs(model_cfg) -> list[tuple[str, Callable[[], ForecastModel]]]:
    """(name, builder) pairs for the fold loop. Builders are lazy so each
    fold gets a fresh model."""
    if model_cfg.name == "baselines":
        return [(n, lambda n=n: make_baseline(n)) for n in model_cfg.members]
    if model_cfg.name == "lightgbm":
        from m5forecast.models.gbm import LightGBMForecaster

        return [("lightgbm", lambda: LightGBMForecaster(model_cfg))]
    if model_cfg.name == "xgboost":
        from m5forecast.models.gbm import XGBoostForecaster

        return [("xgboost", lambda: XGBoostForecaster(model_cfg))]
    if model_cfg.name == "deepar":
        from m5forecast.models.deepar.model import DeepARForecaster

        return [("deepar", lambda: DeepARForecaster(model_cfg))]
    if model_cfg.name == "tft":
        from m5forecast.models.tft.model import TFTForecaster

        return [("tft", lambda: TFTForecaster(model_cfg))]
    raise KeyError(f"unknown model group '{model_cfg.name}'")


def feature_lookback_days(model_cfg) -> int | None:
    """How much feature history the group needs before the earliest fold cutoff.
    None = no feature table needed at all."""
    if model_cfg.name == "baselines":
        return 200 if "linear_reg" in model_cfg.members else None
    if model_cfg.name in ("deepar", "tft"):
        # deep models need full history for their dense arrays (windows sample
        # anywhere in the training region) — signal "everything"
        return 10_000
    return int(model_cfg.train_days) + 10
