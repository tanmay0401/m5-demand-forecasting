"""WRMSSE — the official M5 accuracy metric.

WRMSSE = sum over all 42,840 series of  W_i * RMSSE_i, where

    RMSSE_i = sqrt(  mean_{t in horizon} (y_t - yhat_t)^2
                   / mean_{t in train, after 1st sale} (y_t - y_{t-1})^2 )

The denominator is the MSE of the ONE-STEP NAIVE forecast on the training
series (scale-free: dividing by it means a model "beats naive" iff RMSSE<1),
counted only from each series' first non-zero demand so leading zeros don't
shrink the scale.

Weights W_i encode business importance: each series' share of dollar revenue
(units x price) over the last 28 training days, normalized to sum to 1 WITHIN
each level, then each of the 12 levels weighted equally (1/12). So a
high-revenue item-store and the grand total both matter; a dead SKU barely
does. This is why WRMSSE, not WAPE, is the metric that ranks this project:
it weights by money and evaluates coherence across the whole hierarchy.

Reconciliation-free scoring uses bottom-up aggregation (S @ bottom) so any
bottom-level forecast is scored at every level at once.
"""

from __future__ import annotations

import numpy as np

from m5forecast.hierarchy.aggregation import Hierarchy


def _naive_scale(agg: np.ndarray) -> np.ndarray:
    """Per-series 1/(m-1) * sum (y_t - y_{t-1})^2 from the first non-zero day."""
    n, T = agg.shape
    diffs = np.diff(agg, axis=1)                     # [n, T-1]; diff j is day j+1
    nz = agg > 0
    first = np.argmax(nz, axis=1)                    # first non-zero column
    has_sale = nz.any(axis=1)
    keep = np.arange(T - 1)[None, :] >= first[:, None]  # diffs strictly after first sale
    cnt = keep.sum(axis=1)
    sq = np.where(keep, diffs * diffs, 0.0).sum(axis=1)
    scale = np.where(cnt > 0, sq / np.maximum(cnt, 1), 0.0)
    return np.where(has_sale, scale, 0.0)


def series_scales(bottom_hist: np.ndarray, h: Hierarchy) -> np.ndarray:
    """Scale denominator for every node, streamed per level to bound memory."""
    scale = np.zeros(h.n_nodes, dtype="float64")
    for name, sl in h.level_slices.items():
        agg = (h.S[sl] @ bottom_hist).astype("float64")
        scale[sl] = _naive_scale(agg)
    return scale


def level_weights(bottom_revenue: np.ndarray, h: Hierarchy) -> np.ndarray:
    """W_i: revenue share within level, each level 1/n_levels of total mass."""
    rev = h.S @ bottom_revenue
    W = np.zeros(h.n_nodes, dtype="float64")
    n_levels = len(h.level_slices)
    for name, sl in h.level_slices.items():
        s = rev[sl].sum()
        if s > 0:
            W[sl] = (rev[sl] / s) / n_levels
    return W


def wrmsse(
    bottom_forecast: np.ndarray,   # [n_bottom, H], aligned to h.bottom_ids
    bottom_actual: np.ndarray,     # [n_bottom, H]
    scale: np.ndarray,             # [n_nodes] from series_scales
    weights: np.ndarray,           # [n_nodes] from level_weights
    h: Hierarchy,
) -> tuple[float, dict[str, float]]:
    """Return (overall WRMSSE, per-level weighted-RMSSE contribution x n_levels)."""
    f = h.S @ bottom_forecast
    a = h.S @ bottom_actual
    mse = ((f - a) ** 2).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rmsse = np.sqrt(mse / scale)
    ok = (scale > 0) & np.isfinite(rmsse)
    contrib = np.where(ok, weights * rmsse, 0.0)
    overall = float(contrib.sum())

    n_levels = len(h.level_slices)
    per_level = {name: float(contrib[sl].sum() * n_levels) for name, sl in h.level_slices.items()}
    return overall, per_level
