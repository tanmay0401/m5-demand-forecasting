# Phase 3 — Literature Review

> Status: ✅ Complete · Why read papers before writing code? Because every design decision in Phases 4–13 is *borrowed* from this literature, and in interviews "I chose X because the M5 results / the DeepAR paper showed Y" beats "I chose X because a tutorial used it."

---

## 1. The M-competitions: 40 years of "what actually works"

Forecasting has a unique empirical tradition: since 1982, Spyros Makridakis has run open competitions (M1…M6) where methods compete on *thousands of real series* with held-out futures. They repeatedly embarrassed conventional wisdom:

| Competition | Year | Series | Headline finding |
|---|---|---|---|
| M1 | 1982 | 1,001 | Simple methods (exponential smoothing) ≈ or beat complex statistical ones |
| M3 | 2000 | 3,003 | Confirmed M1; combinations of methods win; became *the* benchmark dataset for 20 years |
| M4 | 2018 | 100,000 | First ML breakthrough — winner was Slawek Smyl's (Uber) **hybrid exponential-smoothing + RNN**; pure ML methods mostly *lost* to statistical benchmarks |
| **M5** | **2020** | **42,840 (hierarchical)** | **First competition where ML clearly won** — LightGBM-based global models dominated; our project's home turf |
| M6 | 2022–23 | financial | Forecasting accuracy ≠ investment returns (out of scope for us) |

The arc matters for interviews: ML did **not** always win in forecasting. It started winning when (a) datasets became *large panels of related series* (global models can share learning) and (b) rich covariates (price, calendar) became available. M5 is precisely that setting — which is why *this project* uses ML rather than ARIMA.

## 2. The M5 competition

- **Organizers:** Makridakis Open Forecasting Center (University of Nicosia) + Kaggle, 2020. Data from **Walmart**.
- **Data:** 30,490 bottom-level series = 3,049 products × 10 stores (3 states: CA, TX, WI), daily unit sales over 1,941 days (2011-01-29 → 2016-05-22), plus `calendar.csv` (events, SNAP days) and `sell_prices.csv` (weekly prices).
- **Hierarchy:** series aggregate into **12 levels** (total → state → store → category → department → …→ item×store), 42,840 series in total. Forecasts judged across *all* levels.
- **Task:** forecast 28 days ahead.
- **Two parallel tracks:**
  - **Accuracy** (point forecasts) — metric: **WRMSSE** (weighted root mean squared scaled error; Phase 13).
  - **Uncertainty** (probabilistic) — predict **9 quantiles** (0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995) per series; metric: **WSPL** (weighted scaled pinball loss). Our project deliberately spans *both* tracks.

### Official findings (from the results papers — see references)
1. **ML won decisively for the first time.** The top accuracy solutions were overwhelmingly **LightGBM-based global models** — one model (or a few pooled ones) trained across many series, with calendar + price features.
2. **~92% of teams failed to beat the best simple benchmark** (an exponential-smoothing bottom-up baseline). Sophistication without discipline loses to a good baseline — this is why our Phase 8 exists.
3. **Combinations/ensembles beat single models**, continuing a 40-year M-competition pattern.
4. **External information (prices, events, SNAP) carried real signal** — methods using it outperformed pure-history methods.
5. **Cross-validation discipline separated winners from leaderboard-overfitters** — many teams who tuned to the public leaderboard crashed on the final month.

### Winning solutions worth knowing by name
- **Accuracy, 1st place — Yeonjun Im** (undergraduate, Kyung Hee University, South Korea — encouraging precedent for a student project!). An **equal-weighted ensemble of ~220 LightGBM models**: pooled per store (10), per store×category (30), per store×department (70), each trained in **recursive and non-recursive** variants, with **Tweedie loss** (a distribution with probability mass at zero — built for intermittent retail counts). Simple features (calendar, price, lags/rollings). Takeaways we adopt: Tweedie objective, store-level pooling, recursive-vs-direct comparison.
- **Top accuracy solutions generally:** several blended in neural nets (notably **N-BEATS**) as ensemble members; a controversial trick was the **"magic multiplier"** — scaling all forecasts by a constant (~0.95–0.97) because recent demand ran below the historical average, i.e., a crude trend correction. It worked, and it's a great interview discussion topic about the gap between competition tricks and production practice.
- **Uncertainty, 1st place** (team including Russ Wolfinger & David Lander): **separate LightGBM models per aggregation level and per quantile**, with distributional features (rolling quantiles, skewness, kurtosis). Beat the ARIMA benchmark by ~25%. A second influential pattern in that track: derive quantiles from point forecasts by **scaling historical residual distributions** — cheap and strong; we'll use it as a probabilistic baseline. Notably, at the SKU level a *purely statistical* negative-binomial approach (Lokad) was #1 — at the noisy bottom level, distributional assumptions matter more than model flexibility.

## 3. DeepAR (Salinas, Flunkert, Gasthaus — Amazon, 2017; IJF 2020)

**"DeepAR: Probabilistic forecasting with autoregressive recurrent networks"** — the paper that industrialized deep probabilistic forecasting; it powers Amazon's internal demand systems and is the flagship model of GluonTS.

Problem it solves: Amazon has *millions* of related, mostly short, mostly intermittent series. Per-series classical models can't share learning and give poor point estimates with no usable uncertainty.

Key ideas (each becomes code in Phase 10):
1. **One global LSTM** trained on *all* series; series identity via covariates/embeddings.
2. **Predict a distribution, not a number:** the network outputs *parameters* of a likelihood at each step — Gaussian for continuous data, **Negative Binomial for retail counts** (handles overdispersion and zeros).
3. **Training = maximize likelihood** of observed history (teacher forcing).
4. **Prediction = ancestral sampling:** sample tomorrow's demand from the predicted distribution, feed the sample back as input, repeat 28 steps, do this a few hundred times → a *Monte Carlo cloud of futures* from which any quantile can be read. Recursive forecasting where uncertainty honestly compounds.
5. **Scale handling:** divide each series by its average; also *sample training windows proportionally to scale* so a few high-volume items don't dominate.

Limitations (interview gold): recursive sampling is slow over long horizons; error can compound; the LSTM is a black box; a fixed likelihood family can mis-fit (motivates quantile-based models like MQ-RNN/TFT).

## 4. Temporal Fusion Transformer (Lim, Arık, Loeff, Pfister — Oxford/Google, 2019; IJF 2021)

**"Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting."** Where DeepAR asks "what distribution comes next?", TFT redesigns the whole architecture around the *structure of forecasting inputs*:

- It distinguishes **static covariates** (store id, category), **known-future inputs** (calendar, planned promos — we *know* next week has Christmas), and **past-observed inputs** (sales, price history). Most earlier models mash these together; TFT routes each properly. This taxonomy alone is worth stealing for any forecasting system.
- **Variable Selection Networks** learn per-timestep soft feature importance.
- **LSTM encoder-decoder** handles *local* patterns; **interpretable multi-head attention** handles *long-range* dependencies (which past days matter for this prediction?).
- **Gated Residual Networks (GRNs)** everywhere let the model skip unneeded components.
- **Direct multi-horizon quantile output:** all 28 days × several quantiles in one forward pass, trained on **quantile (pinball) loss** — no sampling, no error feedback.
- **Interpretability:** attention weights reveal *which* history the model used (e.g., attention spiking on last year's Christmas); variable importances reveal *what* features matter. In the paper, TFT beat DeepAR and MQ-RNN on retail benchmarks.

Our Phase 11 model is TFT-*style*: static/known/observed input routing, attention over the encoded past, direct multi-quantile output — full TFT's every bell and whistle is not required to demonstrate (or interview about) the ideas.

## 5. Gradient boosting: LightGBM & XGBoost

- **XGBoost** (Chen & Guestrin, KDD 2016): made GBM industrial — regularized objective, second-order (Newton) optimization of arbitrary losses, sparsity-aware splits, out-of-core training.
- **LightGBM** (Ke et al., NeurIPS 2017): made GBM *fast at panel scale* — histogram-based split finding, **GOSS** (keep large-gradient samples, subsample small-gradient ones), **EFB** (bundle mutually-exclusive sparse features), **leaf-wise** tree growth (deeper where loss says so, vs XGBoost's level-wise).

Why LightGBM owns tabular forecasting competitions: our M5 training table is ~46M rows × dozens of features; LightGBM trains on it in minutes on CPU, handles categoricals natively (no one-hot explosion over 3,049 item ids), and ships **Tweedie and quantile objectives** out of the box. Why GBMs beat deep nets on tabular data generally (Grinsztajn et al. 2022): trees are robust to uninformative features, capture sharp non-smooth interactions (exactly what promo × weekday × price effects look like), and need no feature scaling. The trade: no native sequence awareness (we hand-craft the memory as lag/rolling features), no native distribution output (we train one model per quantile), and no extrapolation beyond seen target values.

## 6. Hierarchical forecasting literature

The problem: forecasts at different aggregation levels, made independently, **won't add up** — 10 store forecasts won't sum to the state forecast. Incoherent numbers cause real organizational damage (finance plans to one number, ops to another).

- **Bottom-up (BU):** forecast the 30,490 bottom series, sum upward. Coherent by construction; but bottom series are the noisiest.
- **Top-down (TD)** (proportions per Gross & Sohl, 1990): forecast the stable total, split by historical proportions. Stable top, but item-level detail (promos!) is smeared away.
- **Middle-out:** forecast at a middle level (e.g., store×dept), aggregate up, disaggregate down.
- **Optimal combination / trace minimization:** the modern answer. Hyndman et al. (2011) reframed reconciliation as *regression*: forecast **every** level, then project the incoherent forecast vector onto the coherent subspace ( ŷ_reconciled = S·G·ŷ, with S the summing matrix). **MinT** (Wickramasuriya, Athanasopoulos & Hyndman, 2019) chooses G to minimize total forecast error variance using the covariance of base-forecast errors — provably at least as good as BU/TD *and* it improves accuracy, because errors at different levels partially cancel. Phase 12 implements BU, TD, and MinT (shrinkage variant) and measures the WRMSSE delta.
- **Probabilistic reconciliation** (Ben Taieb et al.) extends this to distributions — active research; we'll discuss it in limitations.
- Tooling landscape: R's `hts`/`fable` (Hyndman school), Python's Nixtla `hierarchicalforecast` (we implement the math ourselves, then can cross-check against it).

## 7. Broader retail forecasting research

- **Intermittent demand:** Croston (1972) — split demand into size and interval, smooth separately; refined by Syntetos-Boylan (SBA). The Tweedie/NegBin choices in our models are the modern descendants of this insight.
- **Amazon's quantile line:** MQ-RNN/MQ-CNN (Wen et al., 2017) — *direct multi-horizon quantile* forecasting (no sampling), the conceptual bridge from DeepAR to TFT.
- **N-BEATS** (Oreshkin et al., 2020): pure fully-connected basis-expansion model; strong in M4/M5 ensembles.
- **Foundation models (the current frontier, 2023→):** zero-shot pretrained forecasters — Nixtla **TimeGPT**, Amazon **Chronos** (series → tokens → language model), Google **TimesFM**, Salesforce **Moirai**, **Lag-Llama**. As of 2026 they are competitive zero-shot on generic benchmarks but still generally lose to tuned domain models *with covariates* (a promotion is invisible to a history-only model). Being able to say this places you at the field's leading edge in an interview.

## 8. Where this project sits

This project is a **faithful, student-scale replication of the M5 problem, evaluated the M5 way, comparing the three model families that dominate industry practice** — Walmart-style GBM stacks, Amazon-style DeepAR, Google-style TFT — plus the reconciliation layer that most Kaggle solutions skipped (the competition's summing was handled by bottom-up; we study reconciliation explicitly, which is *more* than the winners did and squarely in the Hyndman research tradition). The promotions/events focus mirrors where industrial effort actually goes: baseline demand is easy; interventions are the hard, valuable part.

## 9. Interview questions — Phase 3

**Easy**
1. What is the M5 dataset and where does it come from? *(Walmart daily unit sales, 30,490 item-store series, 5.4 years, hierarchical, with prices and calendar events; M-competition #5, Kaggle 2020.)*
2. What are the two M5 tracks and their metrics? *(Accuracy → WRMSSE; Uncertainty → weighted scaled pinball loss over 9 quantiles.)*

**Medium**
3. What was the headline result of M5? *(First M-competition where ML clearly won — LightGBM global models dominated; yet ~92% of teams lost to a simple exponential-smoothing benchmark.)*
4. Why did ML win M5 after losing/tying for decades? *(Large panel of related series → global models share learning across series; rich covariates like price/events carry signal classical univariate models can't use.)*
5. What is Tweedie loss and why did the M5 winner use it? *(A compound Poisson-Gamma likelihood with point mass at zero — matches intermittent sales; optimizing it handles many-zeros count data better than MSE.)*
6. DeepAR vs TFT in one sentence each. *(DeepAR: global LSTM emitting likelihood parameters, forecasts by sampling recursively. TFT: attention-based, routes static/known/observed inputs separately, outputs all horizons × quantiles directly with interpretable attention.)*

**Hard**
7. Why is MinT reconciliation provably at least as good as bottom-up? *(BU is a special case of the linear projection ŷ=SGŷ with a particular G; MinT picks the G minimizing total error variance among all unbiased coherent projections, so it can only match or improve.)*
8. The M5 winner multiplied all forecasts by ~0.97. Defend and attack this. *(Defend: crude correction for recent below-average demand level / GBM's inability to extrapolate a downtrend; validated on holdout. Attack: unprincipled constant, wouldn't survive regime change, unusable in production without governance — better to fix via trend features or target scaling.)*
9. Why do gradient-boosted trees beat deep nets on most tabular problems? *(Robustness to uninformative features, sharp non-smooth interaction modeling, no scaling needed, strong with modest data; deep nets close the gap when there's sequence/spatial structure or massive data — which is exactly why we ALSO test DeepAR/TFT on the sequential aspect.)*
10. Would a time-series foundation model (Chronos/TimesFM) solve M5 zero-shot? *(Competitive on baseline demand, but history-only zero-shot models can't see prices/promos/SNAP — the covariate-driven spikes that matter most in retail; fine-tuning with covariates or hybrid approaches would be needed.)*

## References

- Makridakis, Spiliotis, Assimakopoulos — [M5 accuracy competition: Results, findings, and conclusions](https://www.sciencedirect.com/science/article/pii/S0169207021001874) (IJF 2022)
- Makridakis et al. — [The M5 uncertainty competition: Results, findings and conclusions](https://www.sciencedirect.com/science/article/pii/S0169207021001722) (IJF 2022)
- [M5 Accuracy 1st place write-up (Yeonjun Im)](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163684)
- Salinas, Flunkert, Gasthaus — DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks (arXiv:1704.04110)
- Lim, Arık, Loeff, Pfister — Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting (arXiv:1912.09363)
- Ke et al. — LightGBM: A Highly Efficient Gradient Boosting Decision Tree (NeurIPS 2017)
- Chen, Guestrin — XGBoost: A Scalable Tree Boosting System (KDD 2016)
- Hyndman, Ahmed, Athanasopoulos, Shang — Optimal combination forecasts for hierarchical time series (CSDA 2011)
- Wickramasuriya, Athanasopoulos, Hyndman — Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization (JASA 2019)
- Wen et al. — A Multi-Horizon Quantile Recurrent Forecaster (arXiv:1711.11053)
- Grinsztajn, Oyallon, Varoquaux — Why do tree-based models still outperform deep learning on tabular data? (NeurIPS 2022)

---

*Next: Phase 4 — Project Planning (architecture, folder structure, pipelines, module dependency graph).*
