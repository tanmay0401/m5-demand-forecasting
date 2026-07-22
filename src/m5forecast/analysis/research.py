"""Research analysis: head-to-head model comparison by series regime, and
per-series winners under two metrics — the experiment that answers "when does
each model family win?".

The key move: score each series by BOTH absolute error (WAPE-aligned, rewards
the median) and squared error (WRMSSE-aligned, rewards the mean). The winner
flips between them, which is the whole point — model choice is inseparable from
metric choice (Phases 8-13).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _merge_all(forecasts: dict[str, pd.DataFrame], actual: pd.DataFrame) -> pd.DataFrame:
    """Wide frame: one row per (id, d) with a yhat column per model + sales."""
    out = actual.copy()
    for name, fc in forecasts.items():
        out = out.merge(fc.rename(columns={"yhat": name}), on=["id", "d"], how="left")
    return out


def per_regime_wape(forecasts, actual, category: pd.Series) -> pd.DataFrame:
    """WAPE per (regime, model)."""
    j = _merge_all(forecasts, actual)
    j["cat"] = j["id"].map(category)
    models = list(forecasts)
    rows = []
    for regime, g in j.groupby("cat"):
        denom = g["sales"].sum()
        row = {"regime": regime, "n": int(len(g))}
        for m in models:
            row[m] = round(float((g[m] - g["sales"]).abs().sum() / denom), 3) if denom else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("regime")


def per_series_winners(forecasts, actual, category: pd.Series) -> dict:
    """For each series, the best model under absolute vs squared error; then
    tabulate winner share overall and by regime."""
    j = _merge_all(forecasts, actual)
    models = list(forecasts)
    for m in models:
        j[f"ae_{m}"] = (j[m] - j["sales"]).abs()
        j[f"se_{m}"] = (j[m] - j["sales"]) ** 2
    per = j.groupby("id", observed=True).agg({**{f"ae_{m}": "sum" for m in models},
                                              **{f"se_{m}": "sum" for m in models}})
    ae_win = per[[f"ae_{m}" for m in models]].idxmin(axis=1).str[3:]
    se_win = per[[f"se_{m}" for m in models]].idxmin(axis=1).str[3:]
    cat = pd.Series(per.index.map(category), index=per.index)

    def share(win):
        return (win.value_counts(normalize=True) * 100).round(1).to_dict()

    by_regime_se = {}
    for regime in cat.unique():
        sel = cat == regime
        by_regime_se[regime] = (se_win[sel].value_counts(normalize=True) * 100).round(1).to_dict()

    return {
        "absolute_error_winner_pct": share(ae_win),   # WAPE-aligned
        "squared_error_winner_pct": share(se_win),    # WRMSSE-aligned
        "squared_winner_by_regime_pct": by_regime_se,
    }
