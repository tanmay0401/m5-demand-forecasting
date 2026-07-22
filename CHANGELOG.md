# Changelog

## [0.15.0] — 2026-07-22 — Phase 15
### Added
- `analysis/errors.py` + `scripts/analyze_errors.py`: failure-regime classification and per-regime error taxonomy; worst-series diagnostics.
- Error taxonomy figure (15); Phase 15 doc.
### Findings
- Demand shocks are the highest-leverage failure (11% error / 8% series); sparse/dormant hard but low-impact; cold-start negligible in M5's fixed catalog.

## [0.14.0] — 2026-07-22 — Phase 14
### Added
- `analysis/promotions.py` + `scripts/analyze_promos.py`: promo model comparison, event-window error, category elasticity.
- 3 figures (promo bias, event error, elasticity); Phase 14 doc.
### Findings
- Promo-blind MA has 3× the promo penalty of price-aware models; FOODS elasticity −4.06; HOBBIES/HOUSEHOLD price cuts are clearance, not promos.

## [0.13.0] — 2026-07-22 — Phase 13
### Added
- `evaluation/wrmsse.py` (official M5 WRMSSE), pinball loss + quantile report in `metrics.py`.
- `scripts/evaluate.py` (final model comparison), `scripts/exp_deepar_mean.py` (mean-vs-median functional experiment).
- WRMSSE figure (13); DeepAR `mean_forecast_`; 7 metric tests; Phase 13 doc.
### Result
- LightGBM wins WRMSSE (0.555); WAPE ranking reversed; median→mean point functional cuts DeepAR WRMSSE 2.8×.

## [0.12.0] — 2026-07-22 — Phase 12
### Added
- Hierarchy package (`hierarchy/`): 12-level M5 summing matrix, bottom-up / top-down / MinT reconciliation, coherence check.
- `scripts/reconcile.py` experiment (LightGBM base + mean-28 aggregates); reconciliation figure (12); 7 hierarchy tests.
- Phase 12 teaching doc with per-level results.
### Fixed
- Leaf-level row/column misalignment in the summing matrix (bottom block must be identity in column order).

## [0.11.0] — 2026-07-22 — Phase 11
### Added
- TFT-style temporal transformer (`models/tft/`): GRN blocks, input routing, LSTM encoder-decoder, interpretable attention, direct multi-quantile head (pinball loss); attention persisted per fold.
- Attention-over-history figure (11); 6 TFT tests (45 total); Phase 11 teaching doc.
### Notes
- Third and final model family. All three (GBM / DeepAR / TFT) now comparable on the same fold.

## [0.10.0] — 2026-07-22 — Phase 10
### Added
- DeepAR-style probabilistic forecaster (`models/deepar/`): windowed dataset with per-series scaling, LSTM + Negative Binomial network, ancestral-sampling prediction with quantile output.
- torch 2.13 (CUDA) to the environment; deep models train on GPU (RTX 4050).
- 5 DeepAR tests (39 total); Phase 10 teaching doc.
### Changed
- `scripts/train.py`: column-subset feature loading for deepar; per-fold quantile persistence.
- `models/factory.py`: deepar wired into the build/lookback registry.

## [0.9.0] — 2026-07-21 — Phase 9
### Added
- `models/gbm.py`: LightGBM + XGBoost forecasters (tweedie, direct multi-step, native categoricals, per-fold gain importances); `models/factory.py`; `configs/model/xgboost.yaml`; 3 tests.
- Feature-importance figure (09) and full results/interpretation in the Phase 9 doc.
### Changed
- LightGBM recipe tuned: rmse early-stop metric, min_data_in_leaf=200 (experiments logged).
- xgboost added to the environment/requirements.

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
