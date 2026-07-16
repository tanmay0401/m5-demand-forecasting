# TODO / Roadmap

## Next up
- [ ] **Phase 5 — Dataset**: download M5 via Kaggle API, teach the three files and their joins, build data pipeline (loading, long-format melt, calendar/price joins, dtypes, validation), first unit tests.

## Backlog (phases 6–18)
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

## Open questions / decisions
- [x] Experiment tracking → **MLflow** (local-first, no account; see Phase 4 doc §6)
- [x] Configs → **Hydra**; data interchange → **Parquet**; DL → **PyTorch from scratch** (learning goal)
- [x] Deep-model compute → all series with sampled training windows; GPU if available (fallback documented)
- [ ] Data source for M5: Kaggle API preferred — verify user's Kaggle credentials in Phase 5
