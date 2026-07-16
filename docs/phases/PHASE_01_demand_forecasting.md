# Phase 1 — Understanding Demand Forecasting

> Status: ✅ Complete · No code in this phase (by design — theory before implementation)

---

## 1. What is Demand Forecasting?

**Demand forecasting is the practice of estimating how much of a product customers will want to buy, at a specific place, over a specific future time window.**

Every word in that sentence matters:

- **How much** — the output is a *quantity* (units of milk, number of phone cases), not a yes/no answer.
- **Customers will *want* to buy** — we forecast *demand*, not *sales*. If a store stocks 10 units and 25 people wanted one, recorded sales = 10 but true demand = 25. Sales data is a *censored* view of demand. This distinction (called the **stockout / censoring problem**) haunts every retail forecasting system and we will meet it again in Phase 15 (error analysis).
- **Specific place** — demand for umbrellas in Mumbai in July ≠ demand in Jaipur. Forecasts are made at a *location granularity* (store, warehouse, region, country).
- **Specific future time window** — "next day", "next week", "next 28 days". This is called the **forecast horizon**.

So a concrete forecasting question looks like:

> "How many units of item `FOODS_3_090` will store `CA_1` sell **each day for the next 28 days**?"

That is *exactly* the question the M5 dataset (our project's dataset) asks — for ~30,000 item-store combinations simultaneously.

---

## 2. Why Businesses Forecast Demand

A forecast is never the end goal. It is an *input to a decision*. The main decisions it drives:

| Decision | Question the forecast answers | Who cares |
|---|---|---|
| **Inventory / replenishment** | How many units to order from the supplier this week? | Store & warehouse ops |
| **Supply chain & logistics** | How many trucks, how much warehouse space, which routes? | Logistics |
| **Workforce planning** | How many cashiers / delivery drivers on Saturday? | HR / ops |
| **Purchasing & production** | How much raw material to buy 3 months ahead? | Manufacturing |
| **Pricing & promotions** | If we discount 20%, how much extra stock do we need? | Marketing |
| **Financial planning** | Expected revenue next quarter? | Finance / investors |

The core tension every retailer lives with:

- **Under-forecast → stockout** → empty shelf → lost sale *today* + customer may permanently switch to a competitor (lost lifetime value). For groceries, industry studies put stockout costs at ~4% of revenue.
- **Over-forecast → overstock** → capital tied up in inventory, warehouse costs, and for perishables (milk, produce — a huge part of M5's FOODS category) the product literally rots and is written off.

Forecasting is how you walk the tightrope between those two failure modes. Notice the two costs are **asymmetric** (throwing away 5 yogurts ≠ losing 5 yogurt customers) — this asymmetry is *the* reason we will build **probabilistic** forecasts (quantiles) instead of single-number forecasts. A store manager doesn't actually want "expected demand = 12.3"; they want "there's a 95% chance demand ≤ 18, so stock 18 and we almost never disappoint a customer." Hold that thought — it becomes Quantile Loss in Phase 13.

---

## 3. Forecasting vs Prediction

People use these interchangeably, but in ML there's a useful distinction:

- **Prediction (general ML):** estimate an unknown value from features. The rows are usually *exchangeable* — shuffling your training set doesn't break anything. Example: predict house price from square footage.
- **Forecasting:** prediction **where time is the fundamental axis**. The value you want lies in the *future*, and the ordering of observations carries information. This changes everything:
  1. **You cannot randomly shuffle train/test splits.** Training on Friday and testing on the previous Monday is *data leakage* — you'd be using the future to predict the past. (This is the #1 rookie mistake in time series ML, and why Phase 2 covers backtesting.)
  2. **Errors compound over the horizon.** Predicting tomorrow is easier than predicting day 28.
  3. **The data-generating process drifts.** Customer behavior in 2016 ≠ 2011 (the M5 data spans 2011–2016). A model must handle *non-stationarity* (Phase 2 concept).
  4. **Autocorrelation:** today's sales are correlated with yesterday's. Standard ML assumes independent samples; forecasting exploits the dependence (via lag features, recurrence, or attention — that's literally our three model families in Phases 9–11).

**One-liner for interviews:** *"All forecasting is prediction, but not all prediction is forecasting — forecasting is prediction under the arrow of time, which breaks the i.i.d. assumption and forces special validation, features, and models."*

---

## 4. What is a Time Series?

A **time series** is a sequence of observations of the same quantity, recorded at successive, (usually) equally-spaced points in time:

```
y₁, y₂, y₃, …, yₜ        (e.g. daily unit sales of one item at one store)
```

Key vocabulary:

- **Frequency / granularity:** daily, weekly, hourly. M5 is **daily**.
- **Univariate vs multivariate:** one series vs many observed together.
- **Panel / cross-sectional time series:** *many related series in parallel* — 30,490 item-store series in M5. This is the modern retail setting, and it's why "global" models (one model trained across all series — LightGBM, DeepAR, TFT) beat "local" models (one ARIMA per series). A single item-store series is short and noisy; 30k series together contain enough signal to learn rich patterns. **This insight is the single biggest shift in forecasting practice of the last decade, and this project is built on it.**
- **Exogenous variables (covariates):** things observed alongside the target that help explain it — price, promotions, holidays, day-of-week. M5 ships these in `calendar.csv` and `sell_prices.csv`.

---

## 5. Components of a Time Series

Classical decomposition views a series as a combination (additive: `y = T + S + C + R`, or multiplicative: `y = T × S × C × R`) of:

### 5.1 Trend (T)
The long-run direction of the series — the level rising, falling, or flat over months/years.
*Retail example:* a store's total sales creeping upward over 5 years as the neighborhood grows; a DVD's sales trending to zero as streaming takes over.

### 5.2 Seasonality (S)
A **fixed, known-period** repeating pattern.
- **Weekly seasonality** — the dominant pattern in retail: Sat/Sun peaks, Mon–Tue troughs. In M5 this is enormous.
- **Yearly seasonality** — ice cream in summer, soup in winter, back-to-school stationery in August.
- **Monthly / paycheck seasonality** — M5's calendar includes SNAP (US food-stamp) days; grocery demand in Walmart stores spikes on the days SNAP benefits are disbursed. This is a beautiful, non-obvious seasonal driver we will visualize in Phase 6.

Key property: seasonality has a *known, fixed period* (7 days, 365 days), which is what separates it from…

### 5.3 Cyclic Patterns (C)
Repeating ups and downs with **no fixed period** — usually multi-year and driven by economics (business cycles, housing booms/busts). A recession suppressing discretionary spending is cyclic, not seasonal, because you can't set your watch by it. Cycles matter less at the daily-SKU level of this project, but the distinction is a classic interview question: **seasonality = calendar-driven & fixed period; cycles = economy-driven & variable period.**

### 5.4 Noise / Irregular component (R)
What remains after trend/seasonality/known effects — genuinely unpredictable variation: a random customer buying 8 jars of pasta sauce, weather nobody logged, a shelf being restocked late. Two crucial facts:
1. **Noise sets the floor on achievable accuracy.** At the single-item-single-store-single-day level, retail data is *mostly* noise — a typical M5 series sells 0–3 units on most days. No model can predict which specific human walks in. This is why hierarchical aggregation (Phase 12) matters: noise cancels out when you sum series, so store-level or category-level forecasts are far more accurate than SKU-level ones.
2. **One model's noise is another model's signal.** A promotion spike looks like noise to a model that doesn't see the price column, and like signal to one that does. "Noise" is partly a statement about your feature set.

### 5.5 Promotions
Deliberate, retailer-controlled demand interventions: price cuts, buy-one-get-one, featured placement. Effects:
- **Spike during the promo** (often 2–10× baseline for deep discounts),
- **Cannibalization** (promo pasta steals sales from full-price pasta),
- **Pantry-loading / post-promo dip** (customers stockpiled, so demand dips after the promo ends).
In M5, we don't get an explicit "promotion" flag — we *infer* promotions from `sell_prices.csv` (a sudden price drop = de facto promotion). Engineering that inference is a key part of Phase 7 and the heart of the project title: *"Demand Forecasting Under Promotions and Events."*

### 5.6 Holidays
Calendar events with massive, *predictable-in-timing* but *hard-in-magnitude* effects: Christmas, Thanksgiving, Super Bowl (all flagged in M5's `calendar.csv`). Subtleties:
- Effects spread over a **window** (demand ramps up *before* Thanksgiving, craters *on* Christmas Day when stores close — in M5, Christmas is literally the one day with ~zero sales, an outlier we must handle).
- **Moving holidays** (Easter, Ramadan-linked Eid) shift dates each year, so "day-of-year" features can't capture them — you need explicit holiday features.

### 5.7 Events
Broader than holidays: sports finals (Super Bowl → snack sales), cultural/religious periods (Lent → M5 flags it), weather shocks, and one-off disruptions (a pandemic being the extreme case). M5's calendar tags events with a **type** (Sporting, Cultural, National, Religious), which becomes a categorical feature for us.

---

## 6. Why Retail Forecasting is Hard

1. **Scale.** Walmart carries ~100k SKUs per store across 4,700 US stores → hundreds of millions of forecasts, refreshed daily. You cannot hand-tune per series; everything must be automated, global, and cheap per-forecast. (Our 30k series is a realistic miniature.)
2. **Intermittent / sparse demand.** Most SKUs sell 0 units most days. The series looks like `0,0,1,0,0,0,2,0…`. Classical methods (ARIMA) assume smooth continuous data and fail here; you need count-aware approaches (DeepAR's Negative Binomial likelihood in Phase 10 exists precisely for this).
3. **Censored demand (stockouts).** Zeros are ambiguous: "no one wanted it" vs "shelf was empty." The model trains on corrupted labels.
4. **Promotions & cannibalization.** Demand is not a passive natural process — the retailer *intervenes* in it constantly, and items compete with each other.
5. **Hierarchy & coherence.** The business needs forecasts at every level (item-store for replenishment, category-store for shelf space, state for logistics, total for finance) and they must **add up consistently** — independent forecasts won't. This is Forecast Reconciliation, Phase 12.
6. **Cold starts.** New products have no history. New stores have no history.
7. **Non-stationarity.** Tastes, prices, competitors, and assortments drift over 5 years of data.
8. **Asymmetric, business-specific costs.** Accuracy metrics are proxies; the true objective is money, and it's asymmetric (stockout vs waste).

### Company examples

- **Amazon** — forecasts at *fulfillment-center × SKU* level for hundreds of millions of products, driving what gets stocked in which warehouse *before* you order (that's how next-day delivery works: the item was already near you). Their scale + sparse-demand problem is exactly why Amazon researchers **invented DeepAR** (Salinas et al., the paper we implement in Phase 10). One global probabilistic model over millions of related series.
- **Walmart** — the source of our M5 dataset. Grocery-heavy, so perishables + SNAP-day seasonality + weekly cycles dominate. Their pain point is store-level replenishment of ~100k SKUs, which is why M5 evaluates at the *bottom* (item-store) level with a weighted hierarchical metric (WRMSSE).
- **Flipkart** — event-driven extreme: the **Big Billion Days** sale compresses a month of demand into ~5 days, with 5–20× spikes. Forecasting the spike drives everything: seller stock-in to warehouses weeks ahead, delivery workforce hiring, even bandwidth planning. Get it wrong and you either cancel orders (reputation damage) or sit on unsold festival inventory.
- **BigBasket** — perishables + hyperlocal + short horizon: forecasting *tomorrow's* demand for bananas *per dark store per city*, where over-forecast = literal spoilage and under-forecast = customer opens the app, sees "out of stock", and orders from a competitor within minutes. They also face slot-capacity forecasting (delivery slots), a demand forecast of a different flavor.

### What happens when forecasts are wrong

| Failure | Immediate cost | Downstream cost | Famous example |
|---|---|---|---|
| Under-forecast | Stockouts, lost sales | Customer churn to competitors, damaged availability reputation | Retail rule of thumb: ~4% of revenue lost to stockouts |
| Over-forecast | Markdowns, spoilage, storage cost | Cash-flow crunch, warehouse gridlock blocking *other* products | Target Canada (2013–15): supply-chain/forecast data chaos → empty shelves *and* overflowing warehouses simultaneously → C$2B loss, full market exit |
| Systematic bias | Consistently wrong ordering | Bullwhip effect: small store-level errors amplify up the supply chain into wild swings at manufacturers | COVID-era toilet-paper whiplash: panic-spike → over-ordering → 2021 glut |

**Bullwhip effect** (interview favorite): each supply-chain layer forecasts from the noisy orders of the layer below and adds safety margin, so a 10% blip at the shelf can become a 50% swing at the factory. Better (and *probabilistic*) forecasts at the edge damp the whip.

---

## 7. How this maps to our project (bridge to Phase 2+)

| Concept from this phase | Where it becomes concrete |
|---|---|
| Demand vs sales, censoring | Error analysis of stockout-like zeros (Phase 15) |
| Panel of related series → global models | LightGBM / DeepAR / TFT, all trained globally (Phases 9–11) |
| Asymmetric costs → need distributions | Probabilistic forecasting, quantile loss (Phases 10, 13) |
| Noise cancels under aggregation; business needs coherent levels | 12-level hierarchy + reconciliation (Phase 12) |
| Weekly/yearly/SNAP seasonality, holidays, events | Calendar features (Phase 7), EDA (Phase 6) |
| Promotions inferred from price | Price features & elasticity (Phase 7), promo analysis (Phase 14) |
| Errors measured relative to scale & importance | WRMSSE (Phase 13) |

---

## 8. Interview Questions — Phase 1

**Easy**
1. What is demand forecasting and why do retailers need it? *(Estimate future quantity demanded per product/location/time to drive inventory, staffing, and supply-chain decisions.)*
2. What's the difference between trend, seasonality, and cyclic patterns? *(Long-run direction; fixed-period calendar pattern; variable-period economic swings.)*
3. What is a forecast horizon? *(How far into the future you predict — 28 days in M5.)*

**Medium**
4. Why is sales data not the same as demand data? *(Stockouts censor demand — sales = min(demand, stock). Models trained naively on sales under-forecast popular items.)*
5. Why can't you use random train/test splits for forecasting? *(Temporal leakage — future information contaminates training; must split by time / backtest.)*
6. Why are forecasts more accurate at higher aggregation levels? *(Independent noise across series partially cancels when summed — variance of the sum grows slower than the sum of variances relative to the mean level.)*
7. Why do we want probabilistic forecasts instead of point forecasts in retail? *(Costs are asymmetric; inventory decisions are quantile decisions — e.g. stock to the 95th percentile of demand.)*

**Hard**
8. Your model treats promotion spikes as outliers and smooths them away. Is that noise or signal, and what do you do? *(Signal, if you can observe the promotion — add price/promo covariates so the "outlier" becomes explainable; noise only if the intervention is unobservable.)*
9. Explain the bullwhip effect and how forecasting mitigates it. *(Order-variance amplification up the chain from layered reactive forecasting + safety stock; sharing point-of-sale-level probabilistic forecasts across the chain damps it.)*
10. A new product launches next week. Your models have no history for it. What do you do? *(Cold start: forecast from attributes — category/price/store — using a global model that generalizes across items; borrow the launch curves of similar past items; widen uncertainty intervals; update rapidly as first sales arrive.)*

**Sound-bite answers to have ready**
- *30-second project pitch:* "I built a hierarchical demand-forecasting system on Walmart's M5 dataset — 30k daily item-store series over 5 years. I compared gradient boosting, DeepAR-style probabilistic models, and temporal transformers, reconciled forecasts across 12 aggregation levels so they're business-coherent, and evaluated with WRMSSE and quantile loss, focusing on how each model family handles promotion- and event-driven demand spikes."

---

*Next: Phase 2 — Time Series Fundamentals (stationarity, lags, rolling windows, backtesting).*
