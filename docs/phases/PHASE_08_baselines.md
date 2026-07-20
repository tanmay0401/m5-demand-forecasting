# Phase 8 — Baselines

> Status: ✅ Complete · First real forecasts of the project. Results table at the bottom is generated from `outputs/metrics_baselines.json`.

---

## 1. Why baselines are not optional

Three reasons, in increasing order of importance:

1. **The M5 humiliation stat:** ~92% of 5,558 teams — data scientists entering a forecasting competition on purpose — failed to beat a simple exponential-smoothing benchmark. Complexity without discipline *loses* to good baselines.
2. **Baselines price the problem.** The gap between naive and seasonal-naive measures *how much weekly seasonality is worth*. The gap between seasonal-naive and LightGBM measures *what ML actually buys you*. Without baselines, "my model gets WRMSSE 0.65" is a number in a vacuum.
3. **Baselines catch broken pipelines.** If a fancy model loses to naive, the model isn't "bad" — your pipeline is leaking, misaligned, or mis-scaled. Baselines are the smoke detector.

## 2. The contenders

| Model | Assumption it tests | Forecast rule |
|---|---|---|
| `naive` | "tomorrow = today" (random walk) | last observed value, flat for 28 days |
| `seasonal_naive_7` | "weekly cycle is everything" | same weekday last week, tiled |
| `moving_avg_28` | "recent level is everything" | mean of last 28 days, flat |
| `exp_smoothing` | "decaying level memory" (SES, α=0.2) | final EWM level, flat |
| `linear_reg` | "the features matter, non-linearity doesn't" | OLS on all 37 Phase 7 features |

`linear_reg` is the *bridge baseline*: it uses the exact feature table LightGBM will use. Whatever LightGBM beats it by in Phase 9 is the measured value of **interactions and non-linearity** — not of features. That decomposition (features vs model class) is an interview-grade experimental design point.

Design details worth defending:
- **Flat forecasts are correct behavior** for level-only models over a 28-day horizon — they have no seasonal information, and pretending otherwise would be adding a model.
- **Negative predictions are clipped to zero** in the shared `_finalize` (demand is a count; OLS can go negative).
- **SES α is fixed at 0.2**, not tuned per series — baselines should stay dumb; per-series α selection is a documented refinement we deliberately skip.
- Everything is vectorized groupbys — no per-series Python loops across 30,490 series.

## 3. The evaluation harness (this phase's real product)

- **`models/base.py`** — the `fit / predict / predict_quantiles` contract. The fold loop calls only this interface, so every future model (LightGBM, DeepAR, TFT) drops into the *same* loop and gets the *same* folds. Comparisons stay fair by construction.
- **`evaluation/backtest.py`** — expanding-window folds, the *only* place split boundaries are computed:
  - fold 1: train ≤ d1829, test d1830–1857
  - fold 2: train ≤ d1857, test d1858–1885
  - fold 3: train ≤ d1885, test d1886–1913
  - **d1914–1941 is reserved** — the final M5 evaluation block gets touched exactly once, at the end.
- **`evaluation/metrics.py`** — MAE, RMSE, WAPE, bias. **MAPE is deliberately absent** (68% zero actuals). WAPE = Σ|error| / Σ actual is the scale-free headline until WRMSSE arrives in Phase 13. **Bias** is tracked because its *sign* is a business statement: positive = systematic overstock, negative = systematic stockouts.
- MLflow logging per (model, fold); forecasts persisted per fold for Phase 12/13 reuse (reconciliation and WRMSSE re-score the *same* stored forecasts — no re-training).

## 4. Results (3-fold mean, 28-day horizon, all 30,490 series)

| model | MAE | RMSE | WAPE | bias |
|---|---|---|---|---|
| **moving_avg_28** | **1.0401** | **2.2849** | **0.7535** | −0.035 |
| exp_smoothing | 1.0642 | 2.3394 | 0.7709 | +0.008 |
| linear_reg | 1.0939 | 2.2946 | 0.7925 | −0.013 |
| seasonal_naive_7 | 1.2141 | 2.7648 | 0.8796 | −0.084 |
| naive | 1.3745 | 3.1927 | 0.9958 | +0.239 |

Reading the table — three genuine surprises, each a lesson:

1. **Seasonal-naive LOSES to a plain 28-day average.** But Phase 6 measured +37% weekend lift?! Resolution: weekly seasonality dominates at the *aggregate* level, but a median SKU-store series is 73% zeros — copying last Tuesday's specific value copies mostly *noise*, while averaging 28 days cancels it. Phase 1's "noise cancels under aggregation" principle, biting in reverse: at the bottom level, **variance reduction beats pattern matching**. The models that will win are the ones that get seasonality *without* giving up smoothing — exactly what regression-style models (day-of-week feature + level features) do.
2. **Linear regression on 37 engineered features loses (on MAE/WAPE) to a dumb average.** Not a feature failure — a *loss-function* mismatch: OLS minimizes squared error (and indeed nearly ties on RMSE, winning fold 3), but with 68% zeros the MAE/WAPE-optimal prediction sits near the conditional *median*, below the conditional mean that OLS targets. First concrete sighting of a theme that owns this project: **which loss you optimize decides which metric you win** (→ Tweedie in Phase 9, quantile losses in Phases 10–13).
3. **Naive's bias is +0.24** — heavily over-forecasting, because the last training day's value carries that specific weekday's level for 28 days. Every other baseline is near-unbiased. WAPE ≈ 1.0 means naive's total absolute error equals total actual demand — the "you'd do as well forecasting zero everywhere" line (predicting all zeros gives WAPE exactly 1.0).

**The bar is set: WAPE 0.7535.** Any model that can't beat a 28-day average has not earned its complexity.

## 5. Interview questions — Phase 8

**Easy**
1. Why start with a naive forecast at all? *(Prices the problem; detects broken pipelines; the M5 92% stat.)*
2. Why is the seasonal-naive forecast "same weekday last week" and not "yesterday"? *(Retail's dominant pattern is weekly; Phase 6 measured +37% weekend lift.)*

**Medium**
3. Why do your level baselines output flat 28-day lines, and is that a bug? *(No — a level-only model has no information to vary the horizon; inventing variation would be adding an unstated model.)*
4. Why WAPE instead of MAPE? *(MAPE divides by actuals; 68% are zero. WAPE aggregates errors before dividing — defined and stable under intermittency.)*
5. What does linear regression on the full feature table isolate, compared to LightGBM on the same table? *(The value of non-linearity/interactions, separately from the value of the features.)*

**Hard**
6. Your fancy model beats naive by 40% but loses to seasonal-naive. Diagnose. *(It learned level but not weekly cycle — check whether day-of-week features exist/survive preprocessing; in deep models check whether the receptive field covers ≥ 7 days.)*
7. Why must the final 28-day block never appear in any fold? *(Every look leaks information via your own decisions — hyperparameters, model choice. One touch = an unbiased generalization estimate; the M5 public/private leaderboard collapse is the cautionary tale.)*
8. Bias is +0.08 on one model and −0.08 on another with equal MAE. Which do you ship? *(Depends on cost asymmetry — retail usually prefers slight over-forecast for availability; but the real answer is: neither number is complete without quantile forecasts — Phase 10+.)*

---

*Next: Phase 9 — Gradient Boosting: LightGBM with Tweedie loss on the full feature table.*
