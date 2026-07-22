"""Forecast reconciliation: make incoherent per-level forecasts add up.

The problem (Phase 3 literature): forecast every level independently and
the numbers won't reconcile — 10 store forecasts won't sum to the state
forecast. A reconciliation maps the incoherent base vector b (all nodes)
to a coherent one via a bottom-level projection:  y_rec = S @ G @ b.
Different G = different method:

    bottom_up   G picks the bottom rows of b, ignores aggregate forecasts.
                Coherent by construction; noisy (bottom series are noisiest).
    top_down    G routes the TOTAL forecast down by fixed historical
                proportions. Stable top; smears item-level detail (promos!).
    mint        G = (S'W^-1 S)^-1 S'W^-1  — the minimum-trace projection
                (Wickramasuriya 2019). Uses EVERY level's forecast, weighted
                by base-error covariance W. Provably >= bottom-up; usually
                improves accuracy because errors partially cancel across
                levels. Exact form needs an [n_bottom x n_bottom] solve, so
                it is tractable on modest hierarchies, not the full 30,490
                (documented in the Phase 12 report).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from m5forecast.hierarchy.aggregation import Hierarchy


def bottom_up(base_all: np.ndarray, h: Hierarchy) -> np.ndarray:
    """Sum the bottom-level base forecasts up to every node."""
    return h.S @ base_all[h.bottom_slice]


def historical_proportions(bottom_actual_history: np.ndarray) -> np.ndarray:
    """p_i = mean demand of bottom series i / mean total demand (average
    historical proportions, the classic top-down disaggregation weights)."""
    means = bottom_actual_history.mean(axis=1)
    total = means.sum()
    return means / total if total > 0 else np.full_like(means, 1.0 / len(means))


def top_down(base_all: np.ndarray, h: Hierarchy, proportions: np.ndarray) -> np.ndarray:
    """Disaggregate the total-node forecast to the bottom by proportions, sum up."""
    total = base_all[h.level_slices["total"]]          # [1, H]
    bottom = proportions[:, None] * total              # [n_bottom, H]
    return h.S @ bottom


def mint(base_all: np.ndarray, h: Hierarchy, w_diag: np.ndarray, shrink: float = 0.0,
         residuals: np.ndarray | None = None) -> np.ndarray:
    """Minimum-trace reconciliation.

    w_diag : per-node base-error variance (the WLS weights; MinT-diagonal).
    shrink : 0 -> pure diagonal W; >0 blends in an off-diagonal covariance
             estimated from `residuals` (Schafer-Strimmer style shrink), the
             MinT(shrink) variant. residuals: [n_nodes, T] base errors.
    """
    S = h.S
    n_nodes, n_bottom = S.shape

    if shrink > 0 and residuals is not None:
        # shrink full covariance toward its diagonal
        cov = np.cov(residuals)
        d = np.diag(np.diag(cov))
        W = (1 - shrink) * cov + shrink * d
    else:
        W = np.diag(w_diag)

    Wi = np.linalg.inv(W)
    Sd = S.toarray()
    A = Sd.T @ Wi @ Sd                 # [n_bottom, n_bottom]
    G = np.linalg.solve(A, Sd.T @ Wi)  # [n_bottom, n_nodes]
    bottom = G @ base_all              # [n_bottom, H]
    return S @ bottom


def coherence_error(values_all: np.ndarray, h: Hierarchy) -> float:
    """Max abs discrepancy between a node and the sum of its bottom members —
    0 for any coherent (reconciled) vector, > 0 for raw base forecasts."""
    implied = h.S @ values_all[h.bottom_slice]
    return float(np.abs(values_all - implied).max())
