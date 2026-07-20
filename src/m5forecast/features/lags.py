"""Target-derived features: lags, rolling stats, EWM, expanding mean.

THE leakage rule lives here (Phase 2): we forecast day t with information
available at t-28 (a 28-day horizon, direct strategy). Therefore every
feature in this module starts from ``shift(horizon)`` — the groupby makes
the shift per-series, and the shift happening BEFORE any window operation
makes leakage impossible by construction:

    shift(28) -> roll/expand/decay -> aggregate

All functions assume df is sorted by (id, d) — the panel contract.
"""

from __future__ import annotations

import pandas as pd

_ZERO = "zero_frac"  # rolling stat name handled specially below


def add_lag_features(df: pd.DataFrame, lags: list[int], horizon: int) -> pd.DataFrame:
    """sales_lag_L = sales L days ago. Every L must be >= horizon (asserted)."""
    bad = [lag for lag in lags if lag < horizon]
    if bad:
        raise ValueError(f"lags {bad} < horizon {horizon}: would leak future target")
    g = df.groupby("id", observed=True)["sales"]
    for lag in lags:
        df[f"sales_lag_{lag}"] = g.shift(lag).astype("float32")
    return df


def add_rolling_features(
    df: pd.DataFrame, windows: list[int], stats: list[str], shift: int, horizon: int
) -> pd.DataFrame:
    """r_{stat}_{w} over the w days ending at t-shift (shift >= horizon, asserted)."""
    if shift < horizon:
        raise ValueError(f"rolling shift {shift} < horizon {horizon}: would leak future target")
    g = df.groupby("id", observed=True)["sales"]
    shifted = g.shift(shift)
    is_zero = (shifted == 0).astype("float32")  # NaN stays NaN -> excluded from mean
    for w in windows:
        roll = shifted.rolling(w, min_periods=max(w // 2, 2))
        for stat in stats:
            if stat == _ZERO:
                df[f"r_{_ZERO}_{w}"] = (
                    is_zero.rolling(w, min_periods=max(w // 2, 2)).mean().astype("float32")
                )
            else:
                df[f"r_{stat}_{w}"] = getattr(roll, stat)().astype("float32")
    return df


def add_ewm_features(df: pd.DataFrame, alphas: list[float], shift: int, horizon: int) -> pd.DataFrame:
    """Exponentially weighted demand level (Phase 2's adaptive alternative to plain rolling means)."""
    if shift < horizon:
        raise ValueError(f"ewm shift {shift} < horizon {horizon}")
    shifted = df.groupby("id", observed=True)["sales"].shift(shift)
    for a in alphas:
        name = f"ewm_a{str(a).replace('0.', '')}"
        df[name] = (
            shifted.groupby(df["id"], observed=True).transform(lambda s, a=a: s.ewm(alpha=a).mean())
        ).astype("float32")
    return df


def add_expanding_mean(df: pd.DataFrame, shift: int, horizon: int) -> pd.DataFrame:
    """Per-series historical mean demand up to t-shift.

    This is target encoding done safely: the id's average target computed
    strictly from its own past (expanding, shifted) — never from the full
    column, which would leak the future into every row.
    """
    if shift < horizon:
        raise ValueError(f"expanding shift {shift} < horizon {horizon}")
    shifted = df.groupby("id", observed=True)["sales"].shift(shift)
    df["hist_mean"] = (
        shifted.groupby(df["id"], observed=True).transform(lambda s: s.expanding().mean())
    ).astype("float32")
    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Demand momentum: recent week vs recent month (both already shift-safe).

    > 1 means demand is accelerating; < 1 decaying. Requires r_mean_7 and
    r_mean_28 to exist (build order enforces this).
    """
    df["momentum_7_28"] = (df["r_mean_7"] / df["r_mean_28"].replace(0.0, pd.NA)).astype("float32")
    return df
