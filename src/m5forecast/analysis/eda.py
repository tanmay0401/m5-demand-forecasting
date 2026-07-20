"""Phase 6 EDA: the eight figures that prove the dataset's structure.

Each function takes the panel (or a pre-aggregated frame) and writes one
PNG to reports/figures. Design rules applied throughout (dataviz method):
one hue per job, categorical hues in fixed order (FOODS=blue,
HOUSEHOLD=green, HOBBIES=magenta), no dual axes (stacked subplots
instead), recessive grid, thin marks, text in ink colors never in series
colors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from m5forecast.utils.logging import get_logger

log = get_logger(__name__)

# Validated palette (light mode) — see docs/phases/PHASE_06_eda.md
BLUE, BLUE_DARK = "#2a78d6", "#104281"
GREEN, MAGENTA, ORANGE = "#008300", "#d55181", "#eb6834"
INK, INK_2, MUTED, GRID, SURFACE, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
CAT_COLORS = {"FOODS": BLUE, "HOUSEHOLD": GREEN, "HOBBIES": MAGENTA}

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASE,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.labelsize": 9,
            "font.family": "sans-serif",
            "figure.dpi": 150,
        }
    )


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out = Path(out_dir) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("figure: %s", out)
    return out


def daily_totals(panel: pd.DataFrame) -> pd.DataFrame:
    """Total units sold per calendar day — the base series for several figures."""
    daily = panel.groupby("date", observed=True)["sales"].sum().reset_index()
    daily["dow"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    return daily


def fig_total_sales(daily: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(daily["date"], daily["sales"] / 1e3, color=BLUE, lw=0.5, alpha=0.45)
    roll = daily["sales"].rolling(28, center=True).mean() / 1e3
    ax.plot(daily["date"], roll, color=BLUE_DARK, lw=1.8, label="28-day rolling mean")
    xmas = daily[(daily["date"].dt.month == 12) & (daily["date"].dt.day == 25)]
    ax.scatter(xmas["date"], xmas["sales"] / 1e3, color=ORANGE, s=14, zorder=3, label="Christmas (stores closed)")
    ax.set_title("Total daily unit sales, all 10 stores (2011–2016)")
    ax.set_ylabel("units (thousands)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, out_dir, "01_total_daily_sales.png")


def fig_weekly_seasonality(daily: pd.DataFrame, out_dir: Path) -> Path:
    by_dow = daily.groupby("dow")["sales"].mean() / 1e3
    fig, ax = plt.subplots(figsize=(5.2, 3))
    ax.bar(range(7), by_dow.values, color=BLUE, width=0.62)
    ax.set_xticks(range(7), DOW)
    ax.set_title("Average total daily sales by weekday")
    ax.set_ylabel("units (thousands)")
    for i, v in enumerate(by_dow.values):  # direct labels: min & max only
        if v in (by_dow.max(), by_dow.min()):
            ax.text(i, v + 0.5, f"{v:.0f}k", ha="center", fontsize=8, color=INK_2)
    return _save(fig, out_dir, "02_weekly_seasonality.png")


def fig_monthly_seasonality(daily: pd.DataFrame, out_dir: Path) -> Path:
    by_m = daily.groupby("month")["sales"].mean() / 1e3
    fig, ax = plt.subplots(figsize=(5.6, 3))
    ax.bar(range(1, 13), by_m.values, color=BLUE, width=0.62)
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_title("Average total daily sales by month")
    ax.set_ylabel("units (thousands)")
    return _save(fig, out_dir, "03_monthly_seasonality.png")


def fig_snap_effect(panel: pd.DataFrame, out_dir: Path) -> Path:
    foods = panel[panel["cat_id"] == "FOODS"]
    g = foods.groupby(["state_id", "snap"], observed=True)["sales"].mean().unstack()
    states = ["CA", "TX", "WI"]
    x = range(len(states))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    w = 0.36
    ax.bar([i - w / 2 for i in x], [g.loc[s, 0] for s in states], w, color=BASE, label="non-SNAP day")
    ax.bar([i + w / 2 for i in x], [g.loc[s, 1] for s in states], w, color=BLUE, label="SNAP day")
    for i, s in enumerate(states):
        lift = 100 * (g.loc[s, 1] / g.loc[s, 0] - 1)
        ax.text(i + w / 2, g.loc[s, 1] + 0.02, f"+{lift:.0f}%", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(list(x), states)
    ax.set_title("FOODS: mean unit sales per item-store day, SNAP vs non-SNAP")
    ax.set_ylabel("units")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out_dir, "04_snap_effect.png")


def fig_event_windows(panel: pd.DataFrame, daily: pd.DataFrame, out_dir: Path) -> Path:
    """Average total sales in a ±7-day window around three major events."""
    events = {"SuperBowl": BLUE, "Thanksgiving": GREEN, "Christmas": MAGENTA}
    day_sales = daily.set_index("date")["sales"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for name, color in events.items():
        dates = panel.loc[panel["event_name_1"] == name, "date"].unique()
        frames = []
        for d in dates:
            win = day_sales.loc[d - pd.Timedelta(days=7) : d + pd.Timedelta(days=7)]
            if len(win) == 15:
                frames.append(win.to_numpy())
        if frames:
            avg = pd.DataFrame(frames).mean(axis=0) / 1e3
            ax.plot(range(-7, 8), avg, color=color, lw=1.8, label=name)
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_title("Total sales around events (average across years)")
    ax.set_xlabel("days from event")
    ax.set_ylabel("units (thousands)")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out_dir, "05_event_windows.png")


def fig_intermittency(panel: pd.DataFrame, out_dir: Path) -> Path:
    zero_frac = panel.groupby("id", observed=True)["sales"].agg(lambda s: (s == 0).mean())
    fig, ax = plt.subplots(figsize=(5.6, 3))
    ax.hist(zero_frac, bins=40, color=BLUE)
    ax.axvline(zero_frac.median(), color=BLUE_DARK, lw=1.5, ls="--")
    ax.text(zero_frac.median() + 0.01, ax.get_ylim()[1] * 0.9, f"median {zero_frac.median():.0%}", fontsize=8, color=INK_2)
    ax.set_title("Zero-sales days per series (30,490 series)")
    ax.set_xlabel("fraction of days with zero sales")
    ax.set_ylabel("series")
    return _save(fig, out_dir, "06_intermittency.png"), zero_frac.median()


def fig_store_category(panel: pd.DataFrame, out_dir: Path) -> Path:
    g = panel.groupby(["store_id", "cat_id"], observed=True)["sales"].mean().unstack()
    stores = sorted(g.index)
    x = range(len(stores))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    for k, (cat, color) in enumerate(CAT_COLORS.items()):
        ax.bar([i + (k - 1) * w for i in x], [g.loc[s, cat] for s in stores], w, color=color, label=cat)
    ax.set_xticks(list(x), stores)
    ax.set_title("Mean unit sales per item-store day, by store and category")
    ax.set_ylabel("units")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    return _save(fig, out_dir, "07_store_category.png")


def fig_promo_example(panel: pd.DataFrame, out_dir: Path) -> tuple[Path, str]:
    """One item where price drops visibly move demand — two stacked panels, shared x (never a dual axis)."""
    stats = panel.groupby("id", observed=True).agg(
        mean_sales=("sales", "mean"), p_std=("sell_price", "std"), p_mean=("sell_price", "mean")
    )
    stats["p_cv"] = stats["p_std"] / stats["p_mean"]
    pick = stats[stats["mean_sales"] >= 2.0]["p_cv"].idxmax()
    s = panel[panel["id"] == pick].sort_values("date")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True, height_ratios=[2, 1])
    ax1.plot(s["date"], s["sales"].rolling(7, center=True).mean(), color=BLUE, lw=1.2)
    ax1.set_ylabel("units (7-day mean)")
    ax1.set_title(f"Price cuts create demand spikes — {pick.replace('_evaluation', '')}")
    ax2.plot(s["date"], s["sell_price"], color=ORANGE, lw=1.4)
    ax2.set_ylabel("price ($)")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, out_dir, "08_promo_example.png"), pick


def run_all(panel: pd.DataFrame, out_dir: Path) -> dict:
    """Generate every Phase 6 figure; return headline stats for the report."""
    _style()
    daily = daily_totals(panel)
    fig_total_sales(daily, out_dir)
    fig_weekly_seasonality(daily, out_dir)
    fig_monthly_seasonality(daily, out_dir)
    fig_snap_effect(panel, out_dir)
    fig_event_windows(panel, daily, out_dir)
    _, median_zero = fig_intermittency(panel, out_dir)
    fig_store_category(panel, out_dir)
    _, promo_id = fig_promo_example(panel, out_dir)

    xmas = daily.loc[(daily["date"].dt.month == 12) & (daily["date"].dt.day == 25), "sales"]
    stats = {
        "zero_sales_frac_overall": round(float((panel["sales"] == 0).mean()), 4),
        "median_series_zero_frac": round(float(median_zero), 4),
        "weekend_vs_midweek_lift": round(
            float(daily[daily.dow >= 5]["sales"].mean() / daily[daily.dow.isin([1, 2])]["sales"].mean() - 1), 4
        ),
        "christmas_vs_avg_day": round(float(xmas.mean() / daily["sales"].mean()), 4),
        "promo_example_id": promo_id,
    }
    log.info("EDA stats: %s", stats)
    return stats
