# Project Log

Running engineering/learning log. Newest entries at the top.

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
