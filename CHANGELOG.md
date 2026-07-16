# Changelog

## [0.5.0] — 2026-07-16 — Phase 5
### Added
- Data pipeline (`src/m5forecast/data/`): typed loaders, wide→long melt, calendar/price joins, SNAP resolution, structural validation.
- Utils layer (`src/m5forecast/utils/`): seeding, logging, parquet IO.
- Entry points: `scripts/download_data.py`, `scripts/build_panel.py`.
- Test suite: 10 data-pipeline tests on synthetic mini-M5 fixtures.
- `pyproject.toml` (editable src-layout install), Phase 5 dataset doc.

## [0.4.0] — 2026-07-16 — Phase 4
### Added
- Architecture & planning doc (`docs/phases/PHASE_04_project_planning.md`).
- Package skeleton `src/m5forecast/` (data, features, models{deepar,tft}, hierarchy, evaluation, analysis, utils).
- Hydra config tree (`configs/`), `requirements.txt`, `.gitignore`.
### Changed
- README: layout section + setup instructions; TODO decisions resolved (MLflow, Hydra, Parquet, PyTorch).

## [0.3.0] — 2026-07-16 — Phase 3
### Added
- Phase 3 literature review (`docs/phases/PHASE_03_literature_review.md`): M-competition history, M5 competition & winning solutions, DeepAR & TFT paper summaries, gradient-boosting papers, hierarchical reconciliation literature, foundation-model frontier, interview Q&A, references.

## [0.2.0] — 2026-07-16 — Phase 2
### Added
- Phase 2 teaching document (`docs/phases/PHASE_02_time_series_fundamentals.md`): stationarity, differencing, lag features, rolling statistics, exponential smoothing, forecast horizons & multi-step strategies, backtesting (expanding vs sliding windows), leakage rules, feature-engineering map, interview Q&A.
### Changed
- README phase table, TODO roadmap, PROJECT_LOG updated for Phase 2.

## [0.1.0] — 2026-07-16 — Phase 1
### Added
- Repository scaffolding: README, PROJECT_LOG, TODO, CHANGELOG.
- Phase 1 teaching document (`docs/phases/PHASE_01_demand_forecasting.md`): demand forecasting fundamentals, time series components, promotions/holidays/events, why retail forecasting is hard, cost of forecast errors, Phase 1 interview questions.
