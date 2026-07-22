# Phase 16 — Research Analysis

> Status: ✅ Complete · Every "when does X win?" answered with an experiment, not an opinion. This is the phase that turns the build into research.

---

## 0. The one experiment that frames everything

For each of the 30,490 series, we found the best model under two error metrics. **The winner flips completely with the metric:**

| best model on… | deepar | tft | moving_avg_28 | lightgbm | xgboost |
|---|---|---|---|---|---|
| **absolute error** (WAPE-aligned) | **55.7%** | 19.2% | 13.0% | 6.5% | 5.6% |
| **squared error** (WRMSSE-aligned) | 11.4% | 12.2% | **29.7%** | 24.7% | 22.0% |

Deep models win **75% of series on absolute error**; the mean-like models (MA + GBMs) win **76% on squared error** — *the same forecasts on the same series*. This is the project's thesis proven at the finest grain: **model choice is inseparable from metric choice, which is inseparable from the business decision.** Absolute error rewards the median (deep models emit it); squared error rewards the mean (GBMs/MA approximate it). There is no model-ranking question that isn't secretly a loss-function question.

## 1. When do gradient-boosting models beat deep learning?

**Answer: on squared, revenue-weighted, aggregated metrics — i.e. WRMSSE, the metric that matters.** (Phase 13.) LightGBM 0.555 vs DeepAR 1.414. Mechanism (Phase 13): the GBMs' Tweedie-trained forecasts are near-**unbiased**, so bottom-up aggregation *cancels* their error, whereas the deep models' median forecasts are biased low and bottom-up *accumulates* that bias — catastrophic at the money-weighted aggregate levels (total RMSSE 0.27 vs 1.66). GBMs also win on the highest-revenue series, which WRMSSE weights most.

## 2. When does DeepAR beat the tree models?

**Answer: on absolute-error metrics, and on specific hard regimes.** (This phase.) Per-regime WAPE — deep models win *every* regime, but the margin reveals *where* they matter:

| regime | moving_avg | lightgbm | xgboost | deepar | tft |
|---|---|---|---|---|---|
| cold_start | 0.914 | 0.885 | 0.891 | **0.776** | 0.811 |
| demand_shock | 1.383 | 1.276 | 1.264 | 1.126 | **1.123** |
| dense_stable | 0.687 | 0.678 | 0.678 | **0.620** | 0.627 |
| **dormant** | 1.000 | **2.385** | 2.359 | 1.117 | **1.001** |
| sparse | 1.442 | 1.430 | 1.439 | **0.939** | 0.946 |

The standout is **dormant / stockout-recovery series, where the GBMs are catastrophic (WAPE 2.4) and the deep models are 2× better (1.0).** Why: a dormant series' last 28 days are zero, so the GBMs' recency features (`r_mean_7/28`, `ewm`) collapse to ~0 and they forecast zero — but the series sold in the test window (restock/stockout recovery). The deep models carry the series' **identity embedding** and an 84-day context, so they remember its historical level and forecast the return. **This is a concrete, mechanistic reason to prefer a global deep model: it retains per-series memory that recency-weighted tabular features discard.** Deep models also win **cold_start** (embeddings let a thin-history series borrow from similar ones) and **sparse** (the median ≈ 0 is absolute-error-optimal).

## 3. When do Transformers (TFT) beat both?

**Answer: rarely on accuracy, decisively on operational qualities.** (Phase 11.) TFT ≈ DeepAR statistically (WAPE 0.670 vs 0.664; it edges DeepAR on demand_shock and dormant). Its real wins: **single-pass inference ~100× faster** than DeepAR's sampling, **readable attention**, and native direct multi-quantile output. Deploy TFT when inference cost or interpretability matters; DeepAR when you need a coherent sampled joint distribution across horizons.

## 4. How much does hierarchical reconciliation help?

**Answer: enormously for coherence and aggregate accuracy — bottom-up cut total-level WAPE ~5× (0.141 → 0.029) while making all 12 levels coherent** (Phase 12). But the gain is conditional: bottom-up only helps if the bottom forecast is *unbiased* (it amplifies bias — see §1), and MinT only beats bottom-up if the error-covariance weights are well-estimated (ours weren't, so bottom-up won). Reconciliation is not free accuracy; it is a lever whose sign depends on the base forecast's bias and the covariance estimate.

## 5. How do promotions affect model accuracy?

**Answer: every model under-forecasts promo spikes, and price-awareness is what separates them** (Phase 14). The promo-blind moving average has a promo penalty (bias gap) of −0.34, **3× any price-aware model's**. FOODS demand is highly elastic (elasticity −4; a 25% cut doubles demand), so a price-aware model is essential — but the "price < 85% of median" promo signal conflates promotions with clearance markdowns in non-food categories, so the effect is really a FOODS story.

## 6. The meta-finding: the point statistic must match the metric

The single most important practical result (Phase 13, verified controlled): the *same* DeepAR model scored WRMSSE **1.88 with a median point forecast and 0.68 with a mean** — a 2.8× swing from changing nothing but the statistic reported. Median for absolute-error / service-level decisions; **mean (expectation) for squared, weighted, aggregated ones.** Reporting the wrong functional made a good model look terrible.

## 7. Consolidated verdict

| If your objective is… | use | because |
|---|---|---|
| WRMSSE / revenue-weighted point accuracy | **LightGBM** (Tweedie, mean-like) + bottom-up | unbiased → aggregation helps; wins the M5 metric (0.555) |
| absolute-error / typical-day accuracy | **DeepAR/TFT** (median) | median is absolute-error-optimal on zero-heavy data |
| probabilistic / inventory quantiles | **DeepAR or TFT** (pinball 0.30) | calibrated distributions the GBMs don't natively give |
| stockout-recovery / cold-start series | **deep models** | embeddings retain per-series memory GBM features discard |
| fast inference + interpretability | **TFT** | one-pass quantiles, readable attention |
| a strong, cheap, robust baseline | **LightGBM** | 3× faster than XGBoost, ~ties it, beats deep on WRMSSE |

The honest headline, matching M5 itself: **for the competition's own metric, tuned gradient boosting wins.** The deep models earn their place on absolute-error regimes, probabilistic forecasting, and hard per-series memory — not on the headline point score. Knowing *which* is which, and *why*, is the whole point of building all three.

## 8. Interview questions — Phase 16

1. Give one sentence on when GBMs beat deep models and when they don't. *(GBMs win squared/weighted/aggregate metrics via unbiased forecasts; deep models win absolute-error metrics and memory-dependent regimes like stockout recovery.)*
2. Why are the GBMs catastrophic on dormant series? *(Recency features collapse to zero on a dormant series, so they forecast zero; deep embeddings retain the series' historical level and predict the recovery.)*
3. Your per-series winner flips between absolute and squared error. Why is that the whole story? *(Absolute error is minimized by the median, squared by the mean; model ranking is a restatement of which functional the metric rewards — model choice = metric choice = decision.)*
4. How much did reconciliation help and under what condition? *(~5× at the total level via bottom-up, but only because the base was unbiased; it amplifies bias otherwise.)*
5. If a stakeholder wants "the best model", what do you ask back? *(Best for which decision/metric — service level (median/quantile), revenue-weighted accuracy (mean/WRMSSE), or full distribution — because the answer changes the winner.)*

---

*Next: Phase 17 — Engineering hardening (config, tests, logging, reproducibility, experiment tracking) and Phase 18 — the full documentation set.*
