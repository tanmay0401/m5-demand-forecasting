# TODO / Roadmap

## Next up
- [ ] **Phase 7 — Feature engineering**: lags (≥28 only), rolling stats (shift→roll discipline), EWM, calendar features (`days_to_event` — EDA showed events have *shapes*), price/promo features (85%-of-median promo rule validated in EDA), categorical encodings; leakage unit tests.

## Backlog (phases 8–18)
- [ ] Phase 8 — Baselines (naive, seasonal naive, moving average, exponential smoothing, linear regression)
- [ ] Phase 9 — Gradient boosting pipeline (LightGBM primary, XGBoost comparison)
- [ ] Phase 10 — DeepAR-style probabilistic model (PyTorch, Negative Binomial likelihood; install torch)
- [ ] Phase 11 — Temporal transformer (TFT-style)
- [ ] Phase 12 — 12-level hierarchy construction + reconciliation (BU / TD / MinT)
- [ ] Phase 13 — Metrics: WRMSSE, quantile/pinball loss, coverage, calibration; model comparison
- [ ] Phase 14 — Promotions & events analysis
- [ ] Phase 15 — Automated error analysis & failure taxonomy
- [ ] Phase 16 — Research questions & experiment-backed conclusions
- [ ] Phase 17 — Engineering hardening (configs, tests, logging, experiment tracking, seeds)
- [ ] Phase 18 — Full documentation set (technical report, guides, limitations)

## Resolved decisions
- [x] Experiment tracking → **MLflow** (local-first, no account; Phase 4 doc §6)
- [x] Configs → YAML tree composed by **OmegaConf** (Hydra CLI broken on py3.14 — see PROJECT_LOG 2026-07-21); data interchange → **Parquet**; DL → **PyTorch from scratch**
- [x] Deep-model compute → all series with sampled training windows; GPU if available (fallback documented)
- [x] M5 data acquired via browser download (Kaggle account phone-verification was the hidden blocker); panel built & validated 2026-07-21

## Watch list
- [ ] Christmas outlier: exclude from WRMSSE scale denominators (Phase 13), add `is_christmas` flag (Phase 7)
- [ ] torch + Python 3.14 wheel availability — verify before Phase 10
