# TODO / Roadmap

## Next up
- [ ] **Phase 12 — Hierarchical reconciliation**: build the 12-level M5 hierarchy as a summing matrix S; teach bottom-up / top-down / middle-out / MinT; reconcile the stored base forecasts across all levels; measure WRMSSE delta before/after; visualize coherence. (This is a resume-claim centerpiece: "reconciliation across 12 aggregation levels".)

## Deferred to later phases
- [ ] Exercise `predict_quantiles` for LightGBM (per-quantile objectives) → Phase 13
- [ ] GBM fold-1 event-window under-forecast (Super Bowl, bias −0.11) → dissect in Phase 14
- [ ] Direct-model feature staleness vs recursive/per-horizon alternatives → Phase 16 research question

## Backlog (phases 11–18)
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
