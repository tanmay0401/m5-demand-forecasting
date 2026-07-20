# Changelog

## [0.8.0] — 2026-07-21 — Phase 8
### Added
- Shared model interface (`models/base.py`); five baselines (`models/baselines.py`).
- Backtest folds (`evaluation/backtest.py`), point metrics (`evaluation/metrics.py`).
- `scripts/train.py` fold loop with MLflow (sqlite) logging and forecast persistence.
- 8 tests (31 total); Phase 8 doc.
### Changed
- MLflow backend: sqlite (`outputs/mlflow.db`) — 3.x deprecated the file store.

## [0.7.0] — 2026-07-21 — Phase 7
### Added
- Feature engineering package (`features/`): lags, rolling stats, EWM, expanding mean, momentum, calendar + event distances, price/promo family; per-store streaming build to partitioned parquet.
- `scripts/build_features.py`; 9 feature tests including the future-perturbation leakage test.
- Phase 7 teaching doc.
### Fixed
- Momentum dtype crash on zero-heavy real series (`.replace(0, pd.NA)` → `.where`); test fixture now intermittent.

## [0.6.0] — 2026-07-21 — Phase 6
### Added
- EDA module (`analysis/eda.py`) + `scripts/run_eda.py`: 8 figures + headline stats JSON.
- Phase 6 EDA report with measured findings; README data showcase section.
- Config composer (`utils/config.py`, OmegaConf) + 4 tests.
### Changed
- Entry points no longer use Hydra CLI (broken on Python 3.14); same configs, same override syntax.
- requirements.txt: hydra-core → omegaconf (with explanation).
### Data
- M5 raw CSVs in `data/raw/` (gitignored); `panel.parquet` built & validated (59,181,090 rows).

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
