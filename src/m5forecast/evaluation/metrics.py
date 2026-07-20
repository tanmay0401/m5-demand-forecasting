"""Point-forecast metrics for Phase 8+. WRMSSE and quantile loss arrive in
Phase 13 with the hierarchy; these three carry us until then.

MAPE is deliberately absent: 68% of actuals are zero (Phase 6) and MAPE
divides by them. WAPE (sum of absolute errors over sum of actuals) is the
scale-free aggregate that survives zeros.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_point(preds: pd.DataFrame, actuals: pd.DataFrame) -> dict[str, float]:
    """preds: (id, d, yhat); actuals: (id, d, sales). Inner-joined then scored."""
    j = preds.merge(actuals, on=["id", "d"], how="inner", validate="one_to_one")
    if len(j) != len(preds):
        raise ValueError(f"{len(preds) - len(j)} predictions had no matching actual")
    err = j["yhat"] - j["sales"]
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "wape": float(err.abs().sum() / j["sales"].sum()),
        "bias": float(err.mean()),  # + = over-forecast (overstock), - = under (stockouts)
        "n": int(len(j)),
    }
