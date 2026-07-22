"""Phase 12 experiment: reconcile forecasts across the 12 M5 levels.

Design (faithful, and tied to our models):
  * Bottom base = our stored LightGBM fold-3 forecasts (Tweedie -> mean-like,
    the right thing to bottom-up). Aggregate-node base = independent mean of
    each node's last 28 days. These disagree (LightGBM summed != mean-28 of
    the aggregates), so the base is genuinely INCOHERENT — the setup a real
    system faces when different levels are forecast by different models.
    (A purely linear base at every level would already be coherent and leave
    nothing to reconcile — a subtle, important point. And a MEDIAN base makes
    bottom-up collapse, because the median of a 73%-zeros series is 0; that is
    itself why bottom-up needs mean-like bottom forecasts.)
  * Full 12-level hierarchy: bottom-up and top-down (both scale to 30,490).
  * Upper 9-level hierarchy (store x department leaves, 70 series): the place
    exact MinT is computable — compare base / BU / TD / MinT there.

Writes reports/figures/12_reconciliation.png and outputs/reconciliation.json.
Uses the window d1886-1913 (same as the model fold-3 comparison).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from m5forecast.hierarchy.aggregation import UPPER_LEVELS, build_hierarchy
from m5forecast.hierarchy.reconciliation import (
    bottom_up,
    coherence_error,
    historical_proportions,
    mint,
    top_down,
)
from m5forecast.utils.config import REPO_ROOT, load_config
from m5forecast.utils.logging import get_logger

log = get_logger("scripts.reconcile")

TRAIN_END, HORIZON, MED_WINDOW = 1885, 28, 56
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def wape(f: np.ndarray, a: np.ndarray) -> float:
    denom = np.abs(a).sum()
    return float(np.abs(f - a).sum() / denom) if denom else float("nan")


def per_level_wape(values_all: np.ndarray, actual_all: np.ndarray, h) -> dict[str, float]:
    return {name: wape(values_all[sl], actual_all[sl]) for name, sl in h.level_slices.items()}


def main() -> None:
    cfg = load_config()
    panel = pd.read_parquet(REPO_ROOT / cfg.paths.interim / "panel.parquet",
                            columns=ID_COLS + ["d", "sales"])
    meta = panel[ID_COLS].drop_duplicates("id")
    h = build_hierarchy(meta)
    log.info("full hierarchy: %d nodes over %d leaves", h.n_nodes, h.S.shape[1])

    # bottom-level matrices, columns ordered to match h.bottom_ids
    def pivot(days: range) -> np.ndarray:
        sub = panel[panel["d"].isin(days)]
        wide = sub.pivot_table(index="id", columns="d", values="sales", fill_value=0)
        return wide.reindex(h.bottom_ids).to_numpy(dtype="float64")

    hist28 = pivot(range(TRAIN_END - 27, TRAIN_END + 1))                   # [30490, 28]
    actual_bottom = pivot(range(TRAIN_END + 1, TRAIN_END + HORIZON + 1))   # [30490, 28]
    node_hist = h.S @ hist28                                               # [n_nodes, 28]
    actual_all = h.S @ actual_bottom

    # aggregate-node base = independent mean-28; bottom base = LightGBM fold-3
    base_all = np.repeat(node_hist.mean(axis=1)[:, None], HORIZON, axis=1)
    lgbm = pd.read_parquet(REPO_ROOT / cfg.paths.outputs / "forecasts" / "lightgbm" / "fold3.parquet")
    lgbm_wide = lgbm.pivot_table(index="id", columns="d", values="yhat", fill_value=0)
    base_all[h.bottom_slice] = lgbm_wide.reindex(h.bottom_ids).to_numpy(dtype="float64")

    base_coh = coherence_error(base_all, h)
    log.info("base coherence error (should be > 0): %.3f", base_coh)

    props = historical_proportions(hist28)
    rec_bu = bottom_up(base_all, h)
    rec_td = top_down(base_all, h, props)

    results = {
        "window": f"d{TRAIN_END+1}-d{TRAIN_END+HORIZON}",
        "base_coherence_error": round(base_coh, 3),
        "bu_coherence_error": round(coherence_error(rec_bu, h), 6),
        "td_coherence_error": round(coherence_error(rec_td, h), 6),
        "per_level": {
            "base": per_level_wape(base_all, actual_all, h),
            "bottom_up": per_level_wape(rec_bu, actual_all, h),
            "top_down": per_level_wape(rec_td, actual_all, h),
        },
    }
    for method in ("base", "bottom_up", "top_down"):
        pl = results["per_level"][method]
        results["per_level"][method]["ALL_avg"] = float(np.mean(list(pl.values())))

    # ---- exact MinT on the upper 9-level hierarchy (store x dept leaves) ----
    red = meta.copy()
    red["id"] = red["store_id"].astype(str) + "|" + red["dept_id"].astype(str)
    red = red[["id", "cat_id", "dept_id", "store_id", "state_id"]].drop_duplicates("id")
    hR = build_hierarchy(red, UPPER_LEVELS)

    # map full-hierarchy store_dept rows into hR's leaf ordering
    sd_slice = h.level_slices["store_department"]
    sd_keys = h.nodes.iloc[sd_slice]["key"].to_numpy()          # "store|dept"
    order = {k: i for i, k in enumerate(hR.bottom_ids)}
    perm = np.array([order[k] for k in sd_keys])
    leaf_hist = np.zeros((len(perm), HORIZON)); leaf_hist[perm] = node_hist[sd_slice]
    leaf_lgbm = np.zeros((len(perm), HORIZON)); leaf_lgbm[perm] = rec_bu[sd_slice]  # LightGBM summed
    leaf_actual = np.zeros((len(perm), HORIZON)); leaf_actual[perm] = actual_all[sd_slice]

    nodeR_hist = hR.S @ leaf_hist                              # coherent aggregate history
    baseR = np.repeat(nodeR_hist.mean(axis=1)[:, None], HORIZON, axis=1)  # aggregate nodes: mean-28
    baseR[hR.bottom_slice] = leaf_lgbm                         # leaves: LightGBM (incoherent base)
    actualR = hR.S @ leaf_actual
    residR = nodeR_hist - nodeR_hist.mean(axis=1, keepdims=True)
    w_diag = residR.var(axis=1) + 1e-6

    recR = {
        "base": baseR,
        "bottom_up": bottom_up(baseR, hR),
        "top_down": top_down(baseR, hR, historical_proportions(leaf_hist)),
        "mint_diag": mint(baseR, hR, w_diag),
        "mint_shrink": mint(baseR, hR, w_diag, shrink=0.5, residuals=residR),
    }
    results["upper_hierarchy_mint"] = {
        m: {"ALL_avg": float(np.mean(list(per_level_wape(v, actualR, hR).values()))),
            "coherence_error": round(coherence_error(v, hR), 6)}
        for m, v in recR.items()
    }

    out = REPO_ROOT / cfg.paths.outputs / "reconciliation.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", out)
    _figure(results, cfg)
    _print_summary(results)


def _figure(results, cfg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from m5forecast.analysis.eda import BLUE, GREEN, MAGENTA, INK_2, MUTED, _save, _style
    from pathlib import Path

    _style()
    levels = list(results["per_level"]["base"].keys())[:-1]  # drop ALL_avg
    x = range(len(levels))
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    for method, color in [("base", MUTED), ("bottom_up", BLUE), ("top_down", MAGENTA)]:
        ax.plot(x, [results["per_level"][method][l] for l in levels], color=color, lw=1.6,
                marker="o", ms=4, label=method)
    ax.set_xticks(list(x), [l.replace("_", "\n") for l in levels], fontsize=7)
    ax.set_ylabel("WAPE")
    ax.set_title("WAPE by aggregation level: incoherent base vs reconciled")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, Path(REPO_ROOT / cfg.paths.figures), "12_reconciliation.png")


def _print_summary(results) -> None:
    print("\n=== full 12-level hierarchy (avg WAPE over levels) ===")
    for m in ("base", "bottom_up", "top_down"):
        print(f"  {m:12s} {results['per_level'][m]['ALL_avg']:.4f}   "
              f"coherence={results.get(m+'_coherence_error', results['base_coherence_error'] if m=='base' else 0)}")
    print("\n=== upper 9-level hierarchy — exact MinT (avg WAPE) ===")
    for m, r in results["upper_hierarchy_mint"].items():
        print(f"  {m:12s} {r['ALL_avg']:.4f}   coherence={r['coherence_error']}")


if __name__ == "__main__":
    main()
