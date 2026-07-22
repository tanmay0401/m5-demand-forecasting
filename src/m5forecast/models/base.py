"""The one interface every model implements.

The evaluation loop can only call what's defined here — it cannot
special-case a model, which is what keeps the Phase 16 comparisons fair
(Phase 4 design rule). A model that can't produce quantiles simply
doesn't override predict_quantiles and is excluded from probabilistic
scoring rather than faked.

Contracts:
    fit(history, features)   history: (id, d, sales) rows, train period only.
                             features: the feature table for models that use
                             it (baselines ignore it; LR/LightGBM need it).
    predict(future)          future: (id, d) rows to forecast (28 days per
                             series). Returns the same frame + 'yhat'
                             (float32, >= 0).
    predict_quantiles(...)   same, one column per requested quantile.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ForecastModel(ABC):
    name: str = "abstract"

    @abstractmethod
    def fit(self, history: pd.DataFrame, features: pd.DataFrame | None = None) -> ForecastModel:
        ...

    @abstractmethod
    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        ...

    def predict_quantiles(self, future: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} does not produce quantile forecasts")

    @staticmethod
    def _finalize(future: pd.DataFrame, yhat: pd.Series) -> pd.DataFrame:
        """Common post-processing: clip negatives (demand is a count), cast float32.

        Coerce to numeric first: mapping a categorical id column can yield a
        categorical Series, which cannot be clipped — so force a float array
        regardless of how the caller produced yhat.
        """
        out = future[["id", "d"]].copy()
        vals = pd.to_numeric(pd.Series(yhat).reset_index(drop=True), errors="coerce").to_numpy(dtype="float32")
        out["yhat"] = np.clip(vals, 0.0, None)
        return out
