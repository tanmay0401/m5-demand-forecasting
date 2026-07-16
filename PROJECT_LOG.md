# Project Log

Running engineering/learning log. Newest entries at the top.

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
