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


def pinball_loss(actual: np.ndarray, pred_q: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss for one quantile level q.

    PL_q = mean over points of  max( q*(y - yhat_q), (q-1)*(y - yhat_q) ).
    Asymmetric: under-predicting a high quantile is cheap, over-predicting it
    is expensive — which is exactly how inventory service levels behave.
    """
    e = actual - pred_q
    return float(np.maximum(q * e, (q - 1) * e).mean())


def quantile_report(actual: np.ndarray, quantile_preds: dict[float, np.ndarray]) -> dict:
    """Per-quantile pinball + empirical coverage, plus the mean pinball.

    quantile_preds: {level -> [n] predicted quantile}. actual: [n].
    Coverage = fraction of actuals <= the predicted quantile (nominal ~ level).
    """
    per_q = {}
    for level, pred in sorted(quantile_preds.items()):
        per_q[level] = {
            "pinball": pinball_loss(actual, pred, level),
            "coverage": float((actual <= pred).mean()),
        }
    mean_pinball = float(np.mean([v["pinball"] for v in per_q.values()]))
    return {"mean_pinball": mean_pinball, "per_quantile": per_q}
