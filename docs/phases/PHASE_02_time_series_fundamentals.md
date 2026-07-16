# Phase 2 — Time Series Fundamentals

> Status: ✅ Complete · Theory phase — these concepts become code in Phases 5–8

---

## 1. Stationarity

### What it is
A time series is **stationary** if its statistical properties don't change over time. Practically ("weak stationarity"):
1. **Constant mean** — no trend, no level shifts.
2. **Constant variance** — the wiggle size stays the same (no "fanning out").
3. **Autocovariance depends only on lag** — the relationship between today and 7 days ago is the same in 2011 as in 2016.

("Strict stationarity" — the entire joint distribution is time-invariant — is the theoretical version; nobody checks it in practice.)

### Why it exists / what problem it solves
Every model learns patterns from the past and applies them to the future. That is only justified if **the rules of the game don't change** — which is exactly what stationarity formalizes. Classical models (ARIMA) *require* it: their math assumes a stable mean/variance, so you must transform the series to stationary first.

**Intuition:** learning from a non-stationary series is like studying for an exam using last year's syllabus after the syllabus changed. Stationarity = "the syllabus is stable."

### Retail examples
- Roughly stationary: daily sales of salt (stable staple).
- Non-stationary in mean: a growing store (trend).
- Non-stationary in variance: a product whose sales swing wildly at high volume and barely at all at low volume (variance tied to level → multiplicative behavior; fix with log transform).
- Structural break: item gets moved to an end-cap display → permanent level shift.

### How to check
- **Eyeball a plot** (genuinely the most used method): trend? changing spread?
- **Rolling mean/std plot**: if they drift, non-stationary.
- **ADF test** (Augmented Dickey–Fuller): H₀ = unit root (non-stationary). Small p → stationary.
- **KPSS test**: the mirror image, H₀ = stationary. Using both catches disagreements.

### Do WE need stationarity? (important nuance for interviews)
Our project uses ML models with covariates, and the picture changes:
- **Gradient boosting / DeepAR / TFT do not formally require stationarity.** They can *model* trend and seasonality directly via features (day-of-week, time index, lags) or internal state.
- But non-stationarity still bites: **a tree model cannot extrapolate a trend** (trees predict averages of training targets — they've never seen values above the historical max, so they can't predict them). Standard fixes: predict *differences* or *ratios* instead of levels, or include trend-capturing features.
- Neural nets need **scale normalization per series** (DeepAR divides each series by its mean — we'll implement exactly this in Phase 10) because a series selling 500/day and one selling 2/day otherwise blow up shared weights.

**Alternatives to transforming-to-stationary:** feature-based detrending (give the model a time index), per-window normalization, or models with explicit trend components (Holt, Prophet).

---

## 2. Differencing

### What & why
Differencing replaces the series with its **changes**: `y'ₜ = yₜ − yₜ₋₁` (first difference). A series with a steady trend becomes a series wobbling around a constant — i.e., differencing **removes trend** and is the standard route to stationarity. That's the "I" in ARIMA (Integrated = how many times you differenced).

**Seasonal differencing** subtracts the value one season ago: `y'ₜ = yₜ − yₜ₋₇` (for weekly patterns). "How much better is this Saturday than last Saturday?" — removes stable weekly seasonality.

**Intuition:** instead of asking "how many units will we sell Tuesday?" ask "how many *more or fewer* than Monday?" Changes are usually much more stable than levels.

### Trade-offs
- Differencing amplifies noise (you subtract two noisy numbers).
- Over-differencing creates artificial negative autocorrelation.
- For **intermittent** retail data (`0,0,1,0,2,…`), differencing is nearly useless — the "trend" of a series that's mostly zeros isn't the problem. We will *not* difference our target; instead we use lag/rolling features and per-series scaling. Know differencing because (a) interviews, (b) it's the backbone of classical forecasting.

---

## 3. Lag Features

### What
A lag feature hands the model a **past value of the target as an input column**:
`lag_7` at date t = sales at date t−7.

This is the bridge that turns forecasting into ordinary supervised learning — the trick our entire LightGBM pipeline (Phase 9) rests on. A time series `y₁…yₜ` becomes a table:

| date | lag_1 | lag_7 | lag_28 | day_of_week | target |
|---|---|---|---|---|---|
| Mar 10 | 12 | 15 | 9 | Tue | 14 |
| Mar 11 | 14 | 11 | 13 | Wed | 10 |

### Why these particular lags
- `lag_1`: momentum — what happened yesterday.
- `lag_7`: same weekday last week — captures weekly seasonality *as a feature*.
- `lag_28`: same weekday four weeks ago — a "monthly-ish" anchor, and (not a coincidence) the M5 horizon.
- `lag_364`: same weekday last year (365 would land on a different weekday!).

### THE leakage rule (the most important paragraph in this phase)
When you forecast **h days ahead**, at prediction time you only know sales up to today. So a model that must predict day t+28 **cannot use lag_1 of day t+28** (that's day t+27 — unknown!). Two clean strategies:

1. **Direct forecasting:** only use lags ≥ horizon (lag_28 and older). Simple, safe, one model can predict all 28 days. Loses recent-history sharpness.
2. **Recursive forecasting:** use lag_1…lag_7, predict day t+1, **feed the prediction back** as a fake lag to predict t+2, and so on. Uses fresh information but **errors compound** — a bad day-1 prediction poisons days 2–28.

M5 winners used both; we will implement direct (primary, safer) and discuss recursive. **If your validation score looks too good to be true, you leaked a lag.** This is the first thing to audit.

---

## 4. Rolling Windows, Moving Averages, Rolling Statistics

### What
A rolling (moving) window slides over the series and computes a statistic on the last *w* observations:
- **Rolling mean** (= moving average): recent typical level. Smooths noise; the bigger *w*, the smoother and the more it lags behind turns.
- **Rolling std:** recent volatility — "is this item's demand erratic right now?" Great feature for intermittent items and for uncertainty.
- **Rolling median:** like the mean but robust to one-off spikes — a promotion day doesn't drag it.
- **Rolling min/max, rolling count-of-zeros** (how many of the last 28 days sold nothing — a beautiful intermittency feature).

### Two distinct uses — don't confuse them
1. **As a smoother/visual tool:** plot the 7-day moving average to see trend through weekly noise (we'll do this constantly in EDA, Phase 6).
2. **As model features:** `rolling_mean_7`, `rolling_mean_28`, `rolling_std_28` computed **only from data available at prediction time**.

### Leakage rule #2
The rolling window must **end at or before the last known day** — never include the target day. In pandas this is the canonical pattern:
```python
df["r_mean_7"] = df.groupby("id")["sales"].transform(
    lambda s: s.shift(SHIFT).rolling(7).mean())   # SHIFT ≥ forecast horizon for direct models
```
`shift` first, `rolling` second. Reversing the order (or forgetting `shift`) silently leaks the target into the feature. Memorize the order: **shift → roll → aggregate**.

---

## 5. Exponential Smoothing

### What & why
The moving average has a flaw: it weights the last *w* days equally and ignores everything older — day 7 counts fully, day 8 not at all. **Simple Exponential Smoothing (SES)** fixes this with geometrically decaying weights over *all* history:

```
ŷₜ₊₁ = α·yₜ + (1−α)·ŷₜ ,   0 < α ≤ 1
```

Unrolled: `ŷₜ₊₁ = α·yₜ + α(1−α)·yₜ₋₁ + α(1−α)²·yₜ₋₂ + …` — yesterday matters most, last week a little, last year almost none.

- **α near 1** → reactive, trusts recent data (good for fast-drifting demand).
- **α near 0** → stable, long memory (good for steady staples).
- α is fit by minimizing historical one-step error.

**Analogy:** your opinion of a restaurant — the latest visit moves it most, old visits fade but never vanish.

### The family
- **SES:** level only → forecasts a flat line. Baseline for no-trend/no-seasonality data.
- **Holt:** adds a smoothed **trend** term → forecasts a sloped line.
- **Holt-Winters:** adds a smoothed **seasonal** component → the classical workhorse; still shockingly competitive. In the M4 competition (M5's predecessor), a hybrid of exponential smoothing + RNN won outright — classical structure + learned residuals.

### Why we care in this project
1. SES is a **baseline** family (Phase 8) — advanced models must beat it or they're not earning their complexity.
2. The exponentially-weighted mean (`ewm` in pandas) makes a great **feature** — a "current demand level" estimate that adapts faster than a plain rolling mean.
3. Conceptually, DeepAR's hidden state is doing learned, nonlinear exponential smoothing — the RNN decides its own forgetting rates.

---

## 6. Forecast Horizon & Multi-step Strategies

**Horizon (h)** = how far ahead you predict. M5: **h = 28 days** (a retailer's ordering cycle). Difficulty grows with h: day 1 rides on fresh momentum, day 28 is mostly seasonality + level.

Producing 28 numbers from one model — three strategies:

| Strategy | How | Pros | Cons |
|---|---|---|---|
| **Recursive** | 1-step model, feed predictions back | Uses freshest lags; one model | Error compounding |
| **Direct** | Features restricted to horizon-safe lags; predict any day directly (optionally one model per step) | No compounding; robust | Stale lags; per-step models cost 28× |
| **Multi-output / seq2seq** | Model emits all 28 values at once (DeepAR decodes day-by-day sampling; TFT emits all quantiles for all steps in one shot) | Learns cross-step structure | Only for models with native sequence output |

Our project: LightGBM → direct; DeepAR → recursive-by-sampling (that's literally its design); TFT → multi-output. Comparing the three *strategies* is as instructive as comparing the three *models*.

---

## 7. Backtesting & Time-based Validation

### Why random K-fold is forbidden
Random folds put future rows in training while past rows sit in test — the model "remembers the future" through lag/rolling features and autocorrelation. Scores look great; production fails. **Backtesting** = evaluating the way you'll actually predict: train on past → forecast a *contiguous future* block → slide forward → repeat. Each fold simulates a real deployment day.

### Expanding vs sliding window
- **Expanding window:** every fold trains on *all* history up to the cutoff. Origin grows: train d1–800 → test 801–828; train d1–828 → test 829–856; …
  - ✅ maximum data (deep models are data-hungry) · ❌ old, possibly stale data keeps influencing; folds are unequal-sized.
- **Sliding (rolling) window:** fixed-length train window slides forward: train d201–800 → test 801–828; train d229–828 → test 829–856; …
  - ✅ adapts to drift, equal fold sizes · ❌ throws away old data.

Rule of thumb: **long stable history → expanding; drifting regime → sliding.** We'll use expanding windows with ~3 folds whose test blocks are 28-day slices ending at the M5 validation edge — mimicking exactly how the competition scored.

### M5's own split (we adopt it)
- Days d_1 … d_1913 → training
- d_1914 … d_1941 (28 days) → validation (public leaderboard)
- d_1942 … d_1969 (28 days) → evaluation (final, hidden until competition end)

### Fine points
- **Gap/embargo:** if features contain windows that could straddle the split, leave a gap between train end and test start. Our shift-first feature discipline makes this mostly unnecessary, but know the concept.
- **Never tune on the final test block.** Tune on earlier folds; touch d_1942–1969 once, at the very end. "How many times did you evaluate on your test set?" — the only right answer is *once*.

---

## 8. Feature Engineering — the map (fully built in Phase 7)

Everything above feeds one table per (item, store, day):

| Family | Examples | Captures |
|---|---|---|
| Lags | lag_28, lag_35, lag_42 (horizon-safe) | Autocorrelation, weekly pattern |
| Rolling stats | r_mean_{7,28,90}, r_std_28, r_zero_frac_28 (all shifted) | Level, volatility, intermittency |
| EWM | ewm_α=0.1 of sales (shifted) | Adaptive demand level |
| Calendar | day-of-week, month, SNAP flag, event name/type, days-to/from-event | Seasonality, holidays |
| Price | current price, price change %, price relative to item average (→ promo inference), price relative to category | Promotions, elasticity |
| Identity | item/store/category/department ids (categorical or embedded) | Series-specific level & behavior |

The **model families consume history differently** — this is the thesis of the whole comparison:
- LightGBM: history only via engineered lags/rollings (we do the memory work).
- DeepAR: raw recent history through a recurrent hidden state (the model does the memory work).
- TFT: raw history through attention (the model *chooses* which past days to look at).

---

## 9. Interview Questions — Phase 2

**Easy**
1. What is stationarity, and name two ways to check it. *(Time-invariant mean/variance/autocovariance; plot + rolling stats, ADF/KPSS.)*
2. What's the difference between a moving average and exponential smoothing? *(Equal weights over a fixed window vs geometrically decaying weights over all history.)*
3. What is a forecast horizon? *(How far ahead you predict; 28 days in M5.)*

**Medium**
4. Why can't you use standard K-fold cross-validation on time series? *(Temporal leakage via autocorrelation and lag features; must backtest chronologically.)*
5. Expanding vs sliding window validation — when do you pick which? *(Expanding when history is stable/data-hungry models; sliding under regime drift.)*
6. You add lag_1 to a model with a 28-day horizon and validation error halves. What happened? *(Leakage — lag_1 of day t+28 isn't known at prediction time. Either restrict to lags ≥ 28 (direct) or go recursive with feedback.)*
7. Why do tree models struggle with trends, and what are two fixes? *(Trees can't extrapolate beyond seen target values; predict differences/ratios, or add trend features / target transforms.)*

**Hard**
8. Recursive vs direct multi-step forecasting — trade-offs, and which does DeepAR use? *(Recursive: fresh lags but compounding errors; direct: no compounding but stale features. DeepAR is recursive by ancestral sampling — it feeds sampled values back, and the sampling actually propagates uncertainty honestly.)*
9. Your rolling-mean feature was computed with `rolling(7).mean()` then `shift(1)` in one pipeline, and `shift(28)` then rolling in another. Which is correct for a 28-day direct model and why? *(Shift-then-roll with shift ≥ horizon; roll-then-shift-by-1 leaks days t+1…t+27 into features for day t+28.)*
10. Does LightGBM require a stationary target? *(No formal requirement, but non-stationarity hurts via non-extrapolation; handle with per-series scaling, difference/ratio targets, or trend features. Contrast ARIMA, which requires it.)*

---

*Next: Phase 3 — Literature Review (M5 competition & winners, DeepAR, TFT, hierarchical forecasting).*
