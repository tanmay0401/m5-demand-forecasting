# Phase 18 — Documentation & the Held-Out Final Number

> Status: ✅ Complete · The consolidated documentation set, and the single honest number the whole backtest discipline was protecting.

---

## 1. The documentation set

| document | purpose |
|---|---|
| [TECHNICAL_REPORT.md](../TECHNICAL_REPORT.md) | The one-read summary: problem, method, results, the five findings, verdict-by-objective, limitations, future work |
| [INTERVIEW_PREP.md](../INTERVIEW_PREP.md) | Explanations at every altitude (30s → whiteboard), the questions you'll be asked with answers, traps to avoid |
| [PHASE_01…18](.) | Per-phase teaching: theory → intuition → math → implementation → common mistakes → interview questions (this *is* the Phase 19 "teaching mode" deliverable, done continuously) |
| README | Landing page with start-here links and the headline result |
| PROJECT_LOG / CHANGELOG / TODO | Running engineering log, versioned changes, roadmap |

## 2. The held-out evaluation (the payoff of the discipline)

Every model-development decision was made on backtest folds ending at **d1913**. The window **d1914–1941** was never trained on, tuned on, or looked at. We trained each model on d≤1913 and forecast it **exactly once**:

| model | held-out WRMSSE | backtest WRMSSE |
|---|---|---|
| **LightGBM** | **0.679** | 0.555 |
| moving average | 1.082 | 1.097 |
| TFT | 1.468 | 1.492 |
| DeepAR | 1.832 | 1.414 |

**The ranking is identical to the backtest** — gradient boosting wins on data it has never seen. Honest reading:
- **LightGBM 0.555 → 0.679**: a real but modest degradation — some backtest optimism (we did tune the recipe on those folds), but it still beats the naive benchmark and every other model handily. No overfitting reversal.
- **Moving average and TFT barely move** (1.10→1.08, 1.49→1.47): untuned models don't overfit a validation window, exactly as theory predicts.
- **DeepAR is worse out-of-sample** — the median-bias weakness compounds; but its **pinball loss is stable (~0.30)**, so its genuine contribution (calibrated distributions) generalizes even though its point WRMSSE does not win.

Why this matters: the *entire* backtesting discipline — chronological folds, a sealed final block, "touch it once" — exists to make this one number trustworthy. A model can look great on the data you tuned on and collapse on truly fresh data; ours didn't, and because we only looked once, we can say that honestly.

*(Note: `sales_train_evaluation.csv` ends at d1941, so the M5 *private* block d1942–1969 isn't in the public data — d1914–1941 is the honest held-out set available. XGBoost's held-out run exceeded the 16GB laptop's RAM; since it tied LightGBM on the backtest, the champion number stands.)*

## 3. What "done" means for this project

Every resume claim is implemented, measured on real data, interrogated with experiments, documented from first principles, and covered by tests:

- ✅ M5 dataset, 30,490 series × 1,941 days — loaded, validated, 59M-row panel
- ✅ Gradient boosting vs DeepAR-style vs temporal transformer — all from scratch, compared on identical folds
- ✅ Reconciliation across 12 aggregation levels — summing matrix, BU/TD/MinT, measured
- ✅ WRMSSE and quantile loss — implemented, tested, and used to settle the comparison
- ✅ Promotions/events optimization — elasticity, promo/event error analysis, the mean-vs-median inventory-functional result
- ✅ Production engineering — 60 tests, CI, MLflow, configs, Makefile, reproducible

## 4. Interview questions — Phase 18

1. What is the held-out set and why does touching it once matter? *(d1914–1941, never used in development; a single evaluation gives an unbiased generalization estimate — repeated looks leak your model-selection choices into it.)*
2. Your held-out WRMSSE is worse than backtest. Is that a problem? *(No — modest degradation with the ranking intact means honest mild optimism, not overfitting; a reversal or collapse would be the red flag.)*
3. Why is DeepAR's pinball stable out-of-sample but its WRMSSE worse? *(Its calibrated distribution generalizes; its median point forecast is mismatched to WRMSSE regardless of window — a functional problem, not a generalization one.)*

---

*The project is complete. Phase 19 (teaching) was delivered throughout the per-phase docs; Phase 20 (interview prep) is [INTERVIEW_PREP.md](../INTERVIEW_PREP.md).*
