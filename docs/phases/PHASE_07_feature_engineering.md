# Phase 7 — Feature Engineering

> Status: ✅ Complete · The EDA findings become model inputs. Everything here feeds the LightGBM table (Phases 8–9); the deep models (10–11) consume mostly raw history + calendar/price and learn their own representations — that contrast is the point of the comparison.

---

## 1. The one-sentence philosophy

A gradient-boosted tree has **no memory** — it sees one row at a time. Feature engineering is the act of **compressing each series' history into that row**, without ever letting information from after the prediction-information cutoff leak in. Our cutoff: to predict day *t* we may use target information up to *t − 28* (direct strategy, 28-day horizon) — but *calendar and price* information up to *t* itself, because those are known in advance.

That two-tier rule (target: shift ≥ 28; exogenous known-future: no shift) is the intellectual core of this phase, and it maps exactly onto TFT's input taxonomy from Phase 3: *past-observed* vs *known-future* inputs.

## 2. Feature families (what, why, leakage stance)

### 2.1 Lags — `sales_lag_{28,35,42,49,56,364}`
Raw autocorrelation handles. All ≥ 28 (the code **raises** on smaller lags — the constraint is enforced, not hoped). 28/35/42/49/56 are the same weekday 4–8 weeks back (weekly seasonality as features); 364 is the same weekday last year (365 would land on the wrong weekday). Early rows where a lag reaches before the series start are NaN — LightGBM handles NaN natively, no imputation lies needed.

### 2.2 Rolling statistics — `r_{mean,std,median,zero_frac}_{7,28,90}`
Computed on the series **shifted by 28 first** (`shift → roll → aggregate`, Phase 2's discipline, now code):
- `r_mean_*`: demand level at three time scales (sharp week / stable month / quarter).
- `r_std_*`: volatility — separates steady staples from erratic items; also a natural uncertainty input.
- `r_median_*`: level estimate a single promo spike can't drag (robustness).
- `r_zero_frac_*`: **the intermittency dial** — fraction of zero days in the window. EDA showed the median series is 73% zeros; this feature tells the model *which regime* each row lives in (dense staple vs sparse long-tail), which changes what a "2" means.

### 2.3 Exponentially weighted mean — `ewm_a1`
Phase 2's exponential smoothing as a feature: all history, geometrically decaying weights (α=0.1 ≈ 10-day effective memory). Adapts to level shifts faster than an equal-weight window while staying smooth.

### 2.4 Expanding mean — `hist_mean` (target encoding, done safely)
Classic target encoding ("replace the id with its average target") is a leakage machine if computed over the full column — every row then contains the future. Our version is the *only* safe form for time series: **per-series expanding mean of the shifted target** — the item-store's lifetime average demand *as known 28 days ago*. It gives the model a stable per-series level without any of the 30,490-way categorical sparsity.

### 2.5 Demand momentum — `momentum_7_28`
`r_mean_7 / r_mean_28`: >1 accelerating, <1 decaying. Ratios of already-safe features are safe; trees can build this themselves but giving the ratio directly saves splits and shows up in importances (Phase 9 will check).

### 2.6 Calendar — `dow, is_weekend, dom, week, month, year, is_christmas, snap, days_to_event, days_since_event, is_event, event_name_1, event_type_1`
No shifting — the calendar is known years ahead (this is what "known-future input" means). Design driven directly by EDA findings:
- Weekend lift +37% → `dow`, `is_weekend`.
- Events have *shapes* (Super Bowl spikes the day before, Thanksgiving two days before) → signed **distance features** `days_to_event`/`days_since_event` (capped at 30), not just a flag. A flag can't see build-up; distance can.
- Christmas = 0.05% of a normal day → explicit `is_christmas` so the model learns "closed", and the metric code (Phase 13) can exclude the day from scale denominators.
- SNAP +10/16/30% by state → per-state `snap` flag from Phase 5 rides along.

### 2.7 Price & promotions — `sell_price, price_chg_7, price_rel_med, is_promo, price_rel_dept`
Prices are exogenous and set in advance (M5 publishes them for the horizon), so current-day price is fair game; what we avoid is *full-history* statistics (an item's all-time median peeks at future pricing policy) — hence a **trailing 52-week median**, computed at weekly granularity for a 7× speedup.
- `price_rel_med` + `is_promo` (< 0.85 of trailing median): the promo detector the EDA validated — dips below it produced up to 10× spikes.
- `price_chg_7`: week-over-week move — catches the *moment* of a price change (both cuts and the post-promo snap-back).
- `price_rel_dept`: same-day cross-sectional position on the shelf (cheap vs premium within store×department).

**Where's price elasticity?** Deliberately not a precomputed feature. Elasticity (∂log demand/∂log price) is what the model *learns* from price×demand covariation; estimating it per-item first with 73%-zero series would inject noise. We *measure realized* elasticity in Phase 14 as an analysis output. (Interview-ready answer.)

### 2.8 Categorical identity — `item_id, dept_id, cat_id, store_id, state_id`
Kept as raw categoricals: LightGBM consumes them natively (no 3,049-column one-hot), and the deep models will embed them. Store heterogeneity (CA_3 = 2.7× CA_4) is why identity features matter this much.

## 3. Engineering: streaming build

The full table is ~59M rows × ~35 features ≈ 8GB float32 — more than this machine's free RAM. `build.py` therefore streams **one store at a time** (~5.9M rows each) using pyarrow filter pushdown, writing `data/processed/features/store=XX.parquet`. Ten files, read back as one dataset. This was the Phase 4 risk-register mitigation; now it's exercised code, and it's the same pattern (partition + stream) that scales to Walmart-sized data on Spark.

## 4. The leakage test (the most important test in the repo)

Correctness tests check formulas; the leakage test checks the *property that matters*:

> Corrupt every sales value inside the forecast window (t−27 … t) by +1000, rebuild all features, and assert **bit-identical** feature values at day t.

If any target-derived feature moves, it saw the future. This test would have caught the classic `rolling().mean().shift(1)` bug, wrong shift orders, off-by-one windows — the whole class of silent catastrophes. `test_lag_below_horizon_rejected` additionally proves the API *refuses* unsafe configs rather than trusting discipline.

## 5. Interview questions — Phase 7

**Easy**
1. Why do tree models need lag features at all? *(Trees are memoryless row-processors; lags inject the sequence into the row.)*
2. Why is `lag_364` used instead of `lag_365`? *(Same weekday alignment — 364 = 52×7.)*

**Medium**
3. Why don't price features need a 28-day shift when sales features do? *(Prices are exogenous and known in advance — the retailer sets them; sales are the target being forecast. Known-future vs past-observed inputs.)*
4. What's wrong with classic target encoding on time series, and what did you do instead? *(Full-column mean leaks future target into every row; used per-series expanding mean of the 28-shifted target.)*
5. Why `days_to_event` rather than an event flag? *(Events have build-up shapes — Super Bowl's spike is the day BEFORE; a flag is blind to approach, distance is not.)*
6. How do you featurize intermittency? *(`r_zero_frac_*` — the fraction of zero days tells the model which demand regime the series is in.)*

**Hard**
7. Your `price_rel_med` used the item's full-history median. What subtle leakage is that? *(Future pricing policy: a permanent price cut in 2015 lowers the median, retroactively marking 2012 prices as "high" — information from the future shapes past features. Trailing window fixes it.)*
8. Describe a test that catches feature leakage in general, not one bug at a time. *(Future-perturbation: corrupt the target inside the forecast window, assert features at the origin are bit-identical.)*
9. The feature table exceeds RAM. What did you do, and how does it scale? *(Partition by a natural key — store — stream one partition at a time with predicate pushdown; identical pattern to Spark/Dask partitioning at industrial scale.)*
10. Which of your features would you expect at the top of LightGBM importance, and why? *(Prediction registered before Phase 9: `r_mean_7/28` and `hist_mean` — level features dominate in count data; then `dow`, then price family on promo-heavy items. We verify next phase.)*

---

*Next: Phase 8 — Baselines: naive, seasonal naive, moving average, exponential smoothing — the bar every model must clear.*
