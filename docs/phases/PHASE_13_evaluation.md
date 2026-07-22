# Phase 13 — Evaluation: WRMSSE & Quantile Loss

> Status: ✅ Complete · The metrics that finally rank everything. This is where the GBM-vs-DeepAR-vs-TFT verdict is settled on the measure the M5 competition actually used.

---

## 1. The metric zoo, and why most are wrong here

| metric | formula (per point) | why it fails / fits M5 |
|---|---|---|
| MAE | \|y − ŷ\| | fine, but scale-dependent — can't compare a 500/day item to a 2/day item |
| RMSE | (y − ŷ)² | punishes big misses (good for spikes) but scale-dependent |
| MAPE | \|y − ŷ\|/y | **fails**: 68% of actuals are 0 → division by zero |
| SMAPE | \|y−ŷ\|/(\|y\|+\|ŷ\|) | defined at 0 but unstable and asymmetric near 0 |
| WAPE | Σ\|y−ŷ\|/Σy | scale-free, survives zeros — our interim metric, but unweighted by importance and single-level |
| **WRMSSE** | (below) | **the M5 metric**: scale-free, importance-weighted by money, evaluated across all 12 levels |

The lesson threaded through Phases 8–12 — *which loss you optimize decides which metric you win* — reaches its conclusion here: the right metric is the one aligned with the business, and for M5 that is WRMSSE for point forecasts and (weighted, scaled) pinball loss for probabilistic ones.

## 2. WRMSSE, defined

**WRMSSE = Σᵢ Wᵢ · RMSSEᵢ** over all 42,840 series, where

```
RMSSEᵢ = sqrt(  mean_{t∈horizon} (yₜ − ŷₜ)²
              / mean_{t∈train, after 1st sale} (yₜ − yₜ₋₁)²  )
```

Three ideas, each deliberate:

1. **Scaled** (the "SSE"): the denominator is the MSE of the *one-step naive* forecast ("tomorrow = today") on the training series. Dividing by it makes the metric **scale-free** — a 500/day and a 2/day series contribute comparably — and gives an absolute benchmark: **RMSSE < 1 means you beat naive**, > 1 means you lost to it. Counting only from each series' first non-zero sale stops leading zeros (pre-launch) from shrinking the scale and flattering the model.
2. **Root Squared** (the "RMS"): squared error, so it rewards getting the **spikes** right — exactly the promotion/event demand this project targets, and exactly where the mean-forecasting GBMs should shine over the median-forecasting deep models.
3. **Weighted** (the "W"): each series' weight is its **share of dollar revenue** (units × price) over the last 28 training days, normalized within each level, with all **12 levels weighted equally** (1/12 each). So the grand total and a high-revenue item-store both matter; a dead SKU barely does. **This is the crucial difference from WAPE**: WRMSSE concentrates scoring mass on the dense, high-revenue series where mean-accurate forecasts matter — and it credits performance at every aggregation level, rewarding coherent forecasts.

Any bottom-level forecast is scored at all 12 levels by bottom-up aggregation (`S @ bottom`), tying Phase 12 directly into the metric.

## 3. Quantile (pinball) loss, defined

For quantile level q, `PLq(y, ŷq) = max( q·(y−ŷq), (q−1)·(y−ŷq) )`. It is **asymmetric**: for q = 0.9, under-predicting is cheap and over-predicting is expensive, so minimizing it pushes ŷ toward the true 90th percentile. Averaged over the 7 quantiles it scores a whole predictive distribution; it is the loss TFT trains on directly and the natural scorer for DeepAR's sampled quantiles. **This is the metric behind the resume's "quantile loss"**, and it is where the probabilistic models (DeepAR, TFT) get evaluated on what they were built for — not just their median point forecast.

## 4. Results

**Final comparison on WRMSSE** (window d1886–1913, every model scored bottom-up across all 12 levels):

![WRMSSE comparison](../../reports/figures/13_wrmsse.png)

| model | WRMSSE | mean pinball | (recall WAPE) |
|---|---|---|---|
| **lightgbm** | **0.555** | — | 0.752 |
| xgboost | 0.571 | — | 0.751 |
| moving_avg_28 | 1.097 | — | 0.751 |
| deepar (median pt) | 1.414 | **0.297** | 0.664 |
| tft (median pt) | 1.492 | 0.302 | 0.670 |

**The ranking reversed.** On WAPE the deep models won; on WRMSSE they finish last — behind a moving average — and **LightGBM wins at 0.555, reproducing the real M5 result** (GBMs dominated; winners ≈ 0.52). RMSSE < 1 = "beats naive", which only the GBMs clear.

### Why — and it is NOT that the deep models forecast badly

The per-level WRMSSE breakdown is the smoking gun:

| level | lightgbm | deepar (median) |
|---|---|---|
| total | 0.27 | 1.66 |
| store | 0.44 | 1.48 |
| item | 0.90 | 0.99 |
| item × store (bottom) | 0.85 | **0.86** |

At the **bottom** the deep model matches LightGBM. It detonates only at the **aggregate** levels — and that is the capstone insight of the whole project, tying Phases 10, 12 and 13 together:

> **Bottom-up aggregation cancels random noise but *accumulates systematic bias*.** In Phase 12, summing LightGBM's *unbiased* forecasts made the total 5× more accurate. Here, summing DeepAR's *median* forecasts — each biased low, because the median of right-skewed demand sits below the mean — makes the total 6× *worse*: 30,490 same-direction biases add instead of cancel.

### The fix, verified: match the functional to the metric

Same trained DeepAR model, only the point forecast extracted from its predictive distribution changed:

| DeepAR point forecast | WAPE | bias | WRMSSE |
|---|---|---|---|
| **median** | **0.671** | −0.430 | 1.883 |
| **mean** | 0.764 | **+0.031** | **0.676** |

Switching median → mean **cut WRMSSE 2.8× (1.88 → 0.68)** and erased the bias — while making WAPE *worse*. The mean now scores WRMSSE **0.68**, in the GBMs' league (0.56) and far ahead of the moving average (1.10). **The deep model was never bad; the median functional was mismatched to a squared, aggregated, money-weighted metric.** This is the entire "which loss/statistic you optimize decides which metric you win" thread (Phases 8–13) proven in one controlled table: *the point statistic you report must match the decision the forecast serves.* Median for absolute-error / service-level decisions; mean (expectation) for squared-error, revenue-weighted, aggregated ones.

### Probabilistic scoring

On the metric the deep models were *built* for — pinball loss over 7 quantiles — DeepAR (0.297) edges TFT (0.302), on calibrated, monotone quantiles the point-only GBMs don't natively produce. (The GBMs *can* be made probabilistic via per-quantile Tweedie models — a documented extension.) This is the resume's "quantile loss", and it is where probabilistic forecasting earns its keep: not a better point number, but a usable *distribution* for inventory decisions.

## 5. Interview questions — Phase 13

**Easy**
1. Why can't you use MAPE on M5? *(68% zero actuals → division by zero.)*
2. What does RMSSE < 1 mean? *(You beat the one-step naive benchmark; the scaling denominator is naive's training MSE.)*

**Medium**
3. What are the three ideas in WRMSSE and what does each buy? *(Scaling → scale-free + naive benchmark; RMS → rewards spike accuracy; revenue weighting across 12 levels → business importance + coherence.)*
4. Why does WRMSSE favour the GBMs where WAPE favoured the deep models? *(WRMSSE is squared-error and money-weighted toward dense high-revenue series; the GBMs' mean-like forecasts fit those better, while the deep models' median forecasts won WAPE by nailing the zero-heavy long tail that WRMSSE barely weights.)*
5. What is pinball loss and why is it asymmetric? *(Quantile scoring; the asymmetry encodes that missing a high quantile low ≠ missing it high — matching inventory service-level economics.)*

**Hard**
6. A model has great WAPE but poor WRMSSE. What is it doing and is it useful? *(Predicting near the median/zero — cheap on unweighted absolute error, but under-forecasting the high-revenue, spiky series WRMSSE cares about; useful only if the business truly optimizes unweighted volume, which retail doesn't.)*
7. Why compute the scaling denominator from the first non-zero sale? *(Leading pre-launch zeros make naive's diff-error tiny, inflating RMSSE arbitrarily; starting at first sale scales by the series' active-period volatility.)*
8. How would you turn point-forecast WRMSSE winners into a probabilistic submission? *(Center a predictive distribution on the point forecast — e.g. scale historical residual quantiles, or a NegBin with matched mean — then score with weighted scaled pinball; the M5-uncertainty winners did exactly this.)*
9. Your validation WRMSSE is 0.60 but the final held-out block is 0.75. What happened and what's the rule? *(Overfitting to the validation window via model/feature/hyperparameter choices; the rule is to touch the true hold-out exactly once — which is why d1942–1969 stayed sealed until the end.)*

---

*Next: Phase 14 — Promotions & events analysis: dissecting where each model wins and loses on the demand spikes this project is named for.*
