"""Gradient-boosted tree forecasters: LightGBM (primary) and XGBoost (comparison).

Both use the direct multi-step strategy: every feature is already
horizon-safe (Phase 7), so one model predicts any of the 28 days from its
row alone — no recursion, no error feedback.

Loss: Tweedie (variance_power 1.1) — a compound Poisson-Gamma likelihood
with probability mass at exactly zero, matching 68%-zeros retail counts
(the M5 winner's choice, Phase 3). Phase 8 showed why loss choice matters:
OLS chased the mean and lost WAPE; Tweedie's implied prediction sits lower,
respecting the zero mass.

Training window: trailing `train_days` before each fold's cutoff. This is
a RAM-bound compromise (16GB machine): 365 days x 30,490 series ~ 11M rows.
Top M5 solutions used 2-3x more history for ~1-2% gains; documented, not
hidden. Early stopping validates on the last `valid_days` of the window —
legitimate because all features at day t use data <= t-28 (no peeking).
"""

from __future__ import annotations

import pandas as pd

from m5forecast.features.build import numeric_feature_columns
from m5forecast.models.base import ForecastModel
from m5forecast.utils.logging import get_logger

log = get_logger(__name__)

CATEGORICALS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _model_frame(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """X columns = engineered numerics + native categoricals."""
    num = numeric_feature_columns(features)
    cols = num + CATEGORICALS
    return features[cols], cols


class LightGBMForecaster(ForecastModel):
    name = "lightgbm"

    def __init__(self, params):
        self.p = params

    def fit(self, history, features=None):
        import lightgbm as lgb

        if features is None:
            raise ValueError("lightgbm needs the feature table")
        train_end = int(history["d"].max())
        window = features[
            (features["d"] > train_end - int(self.p.train_days)) & (features["d"] <= train_end)
        ]
        valid_start = train_end - int(self.p.valid_days)
        tr, va = window[window["d"] <= valid_start], window[window["d"] > valid_start]

        x_tr, self._cols = _model_frame(tr)
        x_va, _ = _model_frame(va)
        dtrain = lgb.Dataset(x_tr, label=tr["sales"], categorical_feature=CATEGORICALS, free_raw_data=True)
        dvalid = lgb.Dataset(x_va, label=va["sales"], reference=dtrain, free_raw_data=True)

        params = {
            "objective": "tweedie",
            "tweedie_variance_power": float(self.p.tweedie_variance_power),
            # early-stop metric is configurable: tweedie NLL proved a noisy
            # stopping signal (runs halted at ~70 trees, underfit); rmse on the
            # validation slice tracks point accuracy more smoothly
            "metric": str(self.p.get("metric", "rmse")),
            "num_leaves": int(self.p.num_leaves),
            "min_data_in_leaf": int(self.p.get("min_data_in_leaf", 100)),
            "lambda_l2": float(self.p.get("lambda_l2", 0.0)),
            "learning_rate": float(self.p.learning_rate),
            "feature_fraction": float(self.p.feature_fraction),
            "bagging_fraction": float(self.p.bagging_fraction),
            "bagging_freq": int(self.p.bagging_freq),
            "seed": 42,
            "verbosity": -1,
        }
        self._model = lgb.train(
            params,
            dtrain,
            num_boost_round=int(self.p.n_estimators),
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(int(self.p.early_stopping_rounds), verbose=False)],
        )
        log.info("lightgbm: best_iter=%d (of %d max)", self._model.best_iteration, self.p.n_estimators)

        self.importance_ = pd.DataFrame(
            {
                "feature": self._cols,
                "gain": self._model.feature_importance("gain"),
                "splits": self._model.feature_importance("split"),
            }
        ).sort_values("gain", ascending=False, ignore_index=True)

        self._features = features
        return self

    def predict(self, future):
        rows = future.merge(self._features, on=["id", "d"], how="left")
        yhat = self._model.predict(rows[self._cols], num_iteration=self._model.best_iteration)
        return self._finalize(future, pd.Series(yhat, index=future.index))


class XGBoostForecaster(ForecastModel):
    name = "xgboost"

    def __init__(self, params):
        self.p = params

    def fit(self, history, features=None):
        import xgboost as xgb

        if features is None:
            raise ValueError("xgboost needs the feature table")
        train_end = int(history["d"].max())
        window = features[
            (features["d"] > train_end - int(self.p.train_days)) & (features["d"] <= train_end)
        ]
        valid_start = train_end - int(self.p.valid_days)
        tr, va = window[window["d"] <= valid_start], window[window["d"] > valid_start]

        x_tr, self._cols = _model_frame(tr)
        x_va, _ = _model_frame(va)

        self._model = xgb.XGBRegressor(
            objective="reg:tweedie",
            tweedie_variance_power=float(self.p.tweedie_variance_power),
            tree_method="hist",
            grow_policy="lossguide",
            max_leaves=int(self.p.num_leaves),
            max_depth=0,
            learning_rate=float(self.p.learning_rate),
            n_estimators=int(self.p.n_estimators),
            subsample=float(self.p.bagging_fraction),
            colsample_bytree=float(self.p.feature_fraction),
            early_stopping_rounds=int(self.p.early_stopping_rounds),
            enable_categorical=True,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self._model.fit(x_tr, tr["sales"], eval_set=[(x_va, va["sales"])], verbose=False)
        log.info("xgboost: best_iter=%d", self._model.best_iteration)
        self._features = features
        return self

    def predict(self, future):
        rows = future.merge(self._features, on=["id", "d"], how="left")
        yhat = self._model.predict(rows[self._cols])
        return self._finalize(future, pd.Series(yhat, index=future.index))
