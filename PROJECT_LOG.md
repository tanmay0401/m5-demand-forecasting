# Project Log

Running engineering/learning log. Newest entries at the top.

## 2026-07-21 — Real data in; Phase 6 EDA complete

- **Data acquisition saga** (worth remembering): Kaggle API token never existed; browser downloads failed silently because the account (a) wasn't signed in on the automated Chrome profile and (b) had never done **phone verification**, which Kaggle silently requires before joining any competition. Resolution: user verified phone → joined competition (rules accepted with user's explicit OK) → browser download → extracted CSVs moved to `data/raw/`.
- **Hydra is broken on Python 3.14** (argparse strictness rejects its help strings, crash before main). Replaced entry-point layer with `utils/config.py`: OmegaConf composition of the same configs/ tree, same override semantics (`model=tft`, `a.b=c`). 4 new tests; 14 total green.
- **Panel built and validated on first real run**: 59,181,090 rows exactly, 68.0% zero-sales cells, 20.8% price-missing (pre-launch), 233MB Parquet, 51s build.
- **Phase 6 EDA shipped** ([report](docs/phases/PHASE_06_eda.md), 8 figures in reports/figures/):
  - Weekend lift **+36.9%**; monthly variation only ~8% peak-to-trough.
  - SNAP lift on FOODS: **CA +10%, TX +16%, WI +30%**.
  - Median series **73.3% zeros**; distribution modes at 85–95%.
  - Christmas = **0.05%** of an average day (stores closed).
  - Promo example: price $4→$3 (≥15% below median = inferred promo) → spikes up to **10×** baseline.
  - Lesson learned: two naive "pick an illustrative item" heuristics selected artifacts (price-CV found a one-off price blip; adding a price floor found a stockout story). Final selector defines the phenomenon (promo days) and maximizes measured promo-day lift. *Define the phenomenon, then select on it.*

## 2026-07-16 — Phase 5: dataset teaching + data pipeline code

- Wrote [docs/phases/PHASE_05_dataset.md](docs/phases/PHASE_05_dataset.md) (the three files, join graph, SNAP, pipeline contract).
- First real code shipped:
  - `utils/` (seed, logging, parquet IO), `pyproject.toml` (editable install, src layout).
  - `data/loading.py` — dtype-disciplined readers (int16 day columns, category ids): ~2GB panel instead of ~15GB naive.
  - `data/preprocess.py` — melt (day columns renamed to ints *before* melting to avoid 59M throwaway strings), calendar join + per-state SNAP resolution, weekly price join with categorical keys aligned for the fast merge path.
  - `data/validate.py` — DataValidationError on: wrong row count, null/negative sales, broken time index, sold-without-price.
  - `scripts/download_data.py` (Kaggle CLI), `scripts/build_panel.py` (Hydra entry).
  - `tests/test_data_pipeline.py` — 10 tests on a synthetic 2-series × 10-day mini-M5 (snap resolution, price NaN before launch, all validation failure modes). All pass on Python 3.14.
- Environment: Python 3.14 venv; core deps installed (torch deferred to Phase 10).
- **Blocked on user action:** Kaggle API token needed to download the real data (instructions in TODO.md). Pipeline run on real data + EDA happen next session.

## 2026-07-16 — Phase 4 complete: Planning, architecture, scaffold

- Wrote [docs/phases/PHASE_04_project_planning.md](docs/phases/PHASE_04_project_planning.md): dataflow architecture (7 stages connected by Parquet artifacts), module rationale, dependency graph, risk register.
- Scaffolded the repo: `src/m5forecast` layered package, Hydra config tree (`configs/`), `requirements.txt`, `.gitignore` (data/outputs excluded with .gitkeep placeholders — learned the gitignore negation subtlety: ignore dir *contents*, not the dir, or `!.gitkeep` is dead).
- Tooling locked: **Hydra** configs, **MLflow** local tracking, **Parquet** interchange, **PyTorch from scratch** for DeepAR/TFT (learning goal — GluonTS/pytorch-forecasting would be the production shortcut).
- Key invariants declared in configs already: horizon=28, all feature shifts ≥ 28, expanding backtest with 3 folds ending at d_1913, seed=42.

## 2026-07-16 — Phase 3 complete: Literature Review

- Wrote [docs/phases/PHASE_03_literature_review.md](docs/phases/PHASE_03_literature_review.md) (M-competitions, M5 results & winners, DeepAR, TFT, LightGBM/XGBoost, hierarchical reconciliation, foundation models).
- Design decisions borrowed from the literature and locked in:
  - **Tweedie objective** for LightGBM (M5 accuracy winner's choice — matches intermittent counts).
  - **Negative Binomial likelihood** for DeepAR (paper's recommendation for retail counts).
  - **Quantile (pinball) loss + direct multi-horizon output** for the TFT-style model.
  - **Residual-scaling quantiles from point forecasts** as the probabilistic baseline (top M5-uncertainty pattern).
  - **Reconciliation: implement BU, TD, and MinT-shrinkage**, measure WRMSSE delta (goes beyond what Kaggle winners did).
  - Verified competition facts against the official IJF results papers before writing.

## 2026-07-16 — Phase 2 complete: Time Series Fundamentals

- Wrote Phase 2 teaching doc: [docs/phases/PHASE_02_time_series_fundamentals.md](docs/phases/PHASE_02_time_series_fundamentals.md).
- Decisions locked in for later phases:
  - **No differencing of the target** (intermittent data) — lag/rolling features + per-series scaling instead.
  - **Feature discipline: shift → roll → aggregate**, with shift ≥ horizon for direct models. This is our #1 leakage guard.
  - **Multi-step strategy per model**: LightGBM = direct (horizon-safe lags), DeepAR = recursive by sampling, TFT = multi-output.
  - **Validation**: expanding-window backtest, ~3 folds of 28-day test blocks, aligned to M5's d_1914–1941 validation / d_1942–1969 evaluation split. Final evaluation block touched exactly once.
- Repo housekeeping: user wants granular commits (contribution graph) — committing per artifact from now on.
- Published to GitHub: https://github.com/tanmay0401/m5-demand-forecasting (public). Every commit is pushed immediately.

## 2026-07-16 — Phase 1 complete: Understanding Demand Forecasting

- Initialized repository and documentation scaffolding (README, TODO, CHANGELOG, PROJECT_LOG).
- Wrote Phase 1 teaching document: [docs/phases/PHASE_01_demand_forecasting.md](docs/phases/PHASE_01_demand_forecasting.md).
- Key ideas established that shape the rest of the project:
  - **Demand ≠ sales** (stockout censoring) — will matter in error analysis.
  - **Asymmetric inventory costs → probabilistic (quantile) forecasts**, not point forecasts.
  - **Panel of 30k related series → global models** (LightGBM / DeepAR / TFT), not per-series classical models.
  - **Noise cancels under aggregation** → hierarchy + reconciliation is both a business need and an accuracy lever.
  - **Promotions are inferred from price changes** in M5 (no explicit promo flag) — core feature-engineering task.
- No code yet (intentional — theory before implementation).
