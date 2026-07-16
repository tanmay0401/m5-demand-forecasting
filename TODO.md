# TODO / Roadmap

## Next up
- [ ] **Phase 3 — Literature review**: M5 competition & winning solutions, DeepAR paper, TFT paper, LightGBM forecasting, hierarchical forecasting literature.

## Backlog (phases 4–18)
- [ ] Phase 4 — Architecture, folder structure, pipeline & module dependency design
- [ ] Phase 5 — Download M5 data; dataset walkthrough; preprocessing pipeline
- [ ] Phase 6 — EDA (trends, weekly/monthly seasonality, holidays, SNAP, promos, store differences)
- [ ] Phase 7 — Feature engineering (lags, rolling stats, calendar, price/promo, elasticity, encodings)
- [ ] Phase 8 — Baselines (naive, seasonal naive, moving average, linear regression)
- [ ] Phase 9 — Gradient boosting pipeline (LightGBM primary, XGBoost comparison)
- [ ] Phase 10 — DeepAR-style probabilistic model (PyTorch, Negative Binomial likelihood)
- [ ] Phase 11 — Temporal transformer (TFT-style)
- [ ] Phase 12 — 12-level hierarchy construction + reconciliation (BU / TD / MinT)
- [ ] Phase 13 — Metrics: WRMSSE, quantile/pinball loss, coverage, calibration; model comparison
- [ ] Phase 14 — Promotions & events analysis
- [ ] Phase 15 — Automated error analysis & failure taxonomy
- [ ] Phase 16 — Research questions & experiment-backed conclusions
- [ ] Phase 17 — Engineering hardening (configs, tests, logging, experiment tracking, seeds)
- [ ] Phase 18 — Full documentation set (technical report, guides, limitations)

## Open questions / decisions to make later
- [ ] Data source for M5 (Kaggle API vs manual download) — decide in Phase 5
- [ ] Experiment tracking: MLflow vs W&B — decide in Phase 4
- [ ] Compute budget for deep models (subset of series vs all 30k for DeepAR/TFT training) — decide in Phase 4/10
