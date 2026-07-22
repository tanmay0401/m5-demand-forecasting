# Interview Preparation — M5 Demand Forecasting

Explanations at every altitude, plus the questions you must be able to answer. Per-phase question banks (easy/medium/hard) live in each `docs/phases/PHASE_*.md`.

---

## 1. The explanations, by length

### 30-second (the resume pitch)
"I built a hierarchical demand-forecasting system on Walmart's M5 dataset — 30,000 daily item-store series over five years. I implemented and compared three model families from scratch: gradient boosting, a DeepAR-style probabilistic RNN, and a temporal transformer. I reconciled forecasts across all 12 aggregation levels so they're business-coherent, and evaluated with WRMSSE and quantile loss, focusing on promotion- and event-driven demand. The headline finding is that the *metric* decides the winner — gradient boosting wins the competition's WRMSSE while the deep models win absolute-error and probabilistic metrics."

### 2-minute
Add: the data is intermittent (68% zeros) and censored by stockouts, so I engineered leakage-safe lag/rolling/price features and inferred promotions from price drops. All models sit behind one interface and share expanding-window backtest folds. The key results: (1) WRMSSE reverses the WAPE ranking because it's squared and revenue-weighted — LightGBM 0.555 vs deep models ~1.4; (2) the reversal is a *functional* effect — switching DeepAR's point forecast from median to mean cut its WRMSSE 2.8×, because bottom-up aggregation accumulates the median's systematic bias; (3) reconciliation via bottom-up cut total-level error ~5× while enforcing coherence; (4) promotions are essentially a FOODS phenomenon (elasticity −4), and the promo flag conflates promotions with clearance markdowns elsewhere. It's 60 tests, CI, MLflow-tracked, reproducible.

### Deep technical (the whiteboard version)
Draw: the 12-level hierarchy and summing matrix S; `y_reconciled = S·G·ŷ` with G = bottom-up / top-down / MinT `(S'W⁻¹S)⁻¹S'W⁻¹`. Then WRMSSE = Σ Wᵢ·RMSSEᵢ, RMSSEᵢ = √(horizon MSE / naive training MSE), Wᵢ = revenue share × 1/12 per level. Explain DeepAR (LSTM → NegBin(μ,α) per step, teacher-forced NLL training, ancestral sampling for quantiles) vs TFT (static/known/observed routing → LSTM → interpretable attention → direct pinball-loss quantile head). Then the punchline: median minimizes absolute error, mean minimizes squared error, so the model ranking is a restatement of which functional the metric rewards.

### Resume-line defense
Each bullet, backed: "30,000+ SKU series over 1,900+ days" → 30,490 × 1,941, validated. "Compared gradient boosting, DeepAR-style, and temporal transformers" → all three implemented from scratch, compared on identical folds. "12 aggregation levels" → sparse summing matrix, BU/TD/MinT, measured. "WRMSSE and quantile loss" → both implemented and tested; WRMSSE 0.555 (LightGBM), pinball 0.30 (DeepAR). "Optimized for inventory during promotions/events" → promo/event error analysis + elasticity + the mean-vs-median functional result for inventory decisions.

### HR / non-technical
"Stores must decide how much to stock. Too little loses sales; too much rots on shelves. I built and compared several AI approaches to predict demand for 30,000 products, made sure the predictions add up correctly across every level of the business, and measured them the way the industry does — including how they handle sales spikes during promotions. The main insight was that there's no single 'best' model; the right choice depends on what business decision you're making."

---

## 2. The questions you WILL be asked (and crisp answers)

**"Why did LightGBM win if you also built deep models?"**
On the competition's metric (WRMSSE — squared, revenue-weighted, hierarchical), yes. It's near-unbiased, so summing it up the hierarchy cancels error; and trees excel on tabular data with sharp promo×weekday interactions. The deep models win *absolute-error* metrics and provide the probabilistic quantiles GBMs don't natively give. The winner is a function of the metric, which is a function of the decision.

**"Why DeepAR instead of a plain LSTM?"**
DeepAR *is* an LSTM — but one that outputs the *parameters of a distribution* (Negative Binomial for counts) rather than a point, trains by maximum likelihood, and forecasts by sampling so uncertainty compounds honestly. That gives calibrated quantiles for inventory, which a point-LSTM can't.

**"Why a transformer if it just ties DeepAR?"**
On accuracy it ties; its wins are operational — single-pass inference ~100× faster than DeepAR's sampling loop, and readable attention weights. And its input routing (static/known/observed) makes future leakage structurally impossible.

**"What is WRMSSE and why does M5 use it?"**
Weighted Root Mean Squared Scaled Error. *Scaled* by the naive forecast's training error (scale-free, RMSSE<1 beats naive); *squared* (rewards spike accuracy); *weighted* by dollar revenue across all 12 levels (business importance + coherence). It ranks models the way money does.

**"What is quantile loss?"**
Pinball loss: asymmetric per-quantile scoring that pushes a prediction to the true quantile. Minimizing it gives calibrated intervals — which is what inventory decisions actually need (stock to the 95th percentile, not the mean).

**"What is forecast reconciliation and why does it matter?"**
Independent per-level forecasts don't add up. Reconciliation projects them onto the coherent subspace (`y = S·G·ŷ`). It matters because finance, logistics, and replenishment must plan against consistent numbers — and, done right, it also improves accuracy because errors partially cancel across levels.

**"How would this scale to Amazon/Walmart?"**
The layered file-based pipeline maps onto a distributed store (Parquet → Spark/warehouse), scheduled retraining, and a model registry. Bottom-up reconciliation scales trivially; full MinT needs sparse solvers. The global-model design is already the industrial pattern.

**"Biggest weakness of your project?"**
Honest ones: deep models on a single fold (compute); the promo flag conflates promotions with clearance markdowns outside FOODS; stockout-censored zeros are trained as true demand. Each has a documented fix in the technical report.

**"What did you learn?"**
That the metric and the point statistic are not details — they *are* the model-selection problem. A "worse" model was actually a mismatched functional; a 2.8× WRMSSE improvement came from reporting the mean instead of the median. And that honest error analysis (where does error live, is it the model's fault or the data's) matters more than one headline number.

---

## 3. Traps to avoid in the interview

- Don't claim the deep models "failed" — they won different metrics; say *which* and *why*.
- Don't say feature importance proves causation — the `dow`/attention findings show it measures marginal contribution within the feature set (Phases 9, 11).
- Don't oversell MinT — be ready to explain why bottom-up beat it here (misspecified W).
- Don't claim you solved cold-start — M5's fixed catalog under-represents it.

---

*The consolidated results are in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md); the theory + line-by-line reasoning behind every claim is in the per-phase docs.*
