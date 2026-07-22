# Project Log

Running engineering/learning log. Newest entries at the top.

## 2026-07-22 — Phase 18: documentation + held-out final number (PROJECT COMPLETE)

- Shipped the documentation set: `docs/TECHNICAL_REPORT.md`, `docs/INTERVIEW_PREP.md`, Phase 18 doc, README landing page.
- **Single-touch held-out (d1914–1941, never seen in development):** LightGBM **0.679**, moving_avg 1.082, TFT 1.468, DeepAR 1.832 (pinball ~0.30 stable). **Ranking identical to backtest** — gradient boosting wins on unseen data, no overfitting reversal. LightGBM 0.555→0.679 = honest mild backtest optimism; MA/TFT barely move (untuned); DeepAR worse (median-bias compounds) but pinball stable (calibration generalizes). The "touch once" discipline delivered a trustworthy number.
- `final_eval.py` runs one model per process (OOM fix after XGBoost exceeded 16GB; XGBoost held-out skipped — tied LightGBM on backtest so champion stands).
- All 20 phases complete. Phase 19 (teaching) delivered throughout per-phase docs; Phase 20 (interview prep) = INTERVIEW_PREP.md. 44+ commits, 60 tests, CI green, on GitHub.

## 2026-07-22 — Phase 17: engineering hardening

- Added `Makefile` (pipeline + dev targets), `.github/workflows/tests.yml` (CI: pytest + ruff, CPU torch, no data needed), ruff config, and **an end-to-end integration test**.
- **Integration test caught a real bug on first run:** `ForecastModel._finalize` clipped a Series that could be categorical (when mapping a categorical id) → `TypeError: Unordered Categoricals`. Unit tests missed it (string-id fixtures); real runs missed it (parquet changes id dtype). Fixed `_finalize` to coerce to a float array regardless of input dtype. *This is exactly the inter-module seam integration tests exist to catch.*
- Reproducibility verified: `pip install -e .` clean, all 20 modules import, 60 tests green with no downloaded data (synthetic fixtures; deep tests importorskip torch). Promoted `scipy` to an explicit dependency.
- Documented honest remaining gaps (single-fold deep runs, upper-only exact MinT, no Dockerfile).

## 2026-07-22 — Phase 16: research analysis (experiment-backed conclusions)

- **Winner-flip experiment (headline):** per series, deep models win **75%** on absolute error; mean-like models (MA+GBM) win **76%** on squared error — same forecasts, opposite winners. Model choice = metric choice = decision, proven at series grain.
- **Per-regime WAPE:** deep models win every regime; standout is **dormant/stockout-recovery where GBMs are catastrophic (WAPE 2.4) vs deep 1.0** — GBM recency features collapse to 0 on dormant series while deep embeddings retain per-series level. Deep also win cold_start (embeddings generalize) and sparse (median≈0).
- Consolidated all cross-phase findings into experiment-backed answers: GBMs win WRMSSE (unbiased→aggregation helps); deep win absolute-error + memory regimes; TFT≈DeepAR but 100× faster inference; reconciliation ~5× at top but bias-dependent; promo-awareness worth 3× penalty; median→mean functional worth 2.8× WRMSSE. Verdict table by objective.
- `analysis/research.py`, `scripts/analyze_research.py`, per-regime figure (16).

## 2026-07-22 — Phase 15: automated error taxonomy

- `analysis/errors.py`: classify every series by failure regime from training history (cold_start/dormant/sparse/demand_shock/dense_stable), then measure LightGBM error mass per regime.
- **Findings:** dense_stable = 71% series / 90% sales / 81.5% error but BEST WAPE (0.68) — error follows volume, not weakness. **demand_shock** = 8% series but 11% error at WAPE 1.28 (error share > sales share) — the real, highest-leverage failure. sparse/dormant WAPE 1.4/2.4 (worse than naive) but only ~4% error each (low volume). cold_start negligible (24 series) — M5's fixed catalog, a dataset artifact not a solved problem. Dormant = stockout recoveries where training zeros were censored demand (ties Phase 1).
- Worst 10 series: high-volume FOODS_3 concentrated in store WI_2 (worst: forecast 993 vs actual 1862) — exactly what revenue-weighted WRMSSE punishes.
- `scripts/analyze_errors.py`, taxonomy figure (15).

## 2026-07-22 — Phase 14: promotions & events analysis (the namesake)

- **Promo handling (FOODS, d1886–1913):** all models under-forecast promos; the promo-blind moving average is worst (promo penalty −0.34, 3× any price-aware model) — the cleanest justification for the price/promo features. GBMs give the best absolute promo forecasts (−0.16/−0.19); deep models respond to price (promo penalty ~0) but are dragged by their median bias.
- **Event windows (LightGBM, Feb–Mar 2016 folds):** systematic under-forecast of the pre-event build-up, worst 1 day before (−0.15 ≈ 10% of actual) — quantifies the Phase 9 flag.
- **Elasticity by category (headline):** FOODS elasticity **−4.06** (25% cut → demand doubles, 2× lift) — real promotional behaviour. HOBBIES/HOUSEHOLD show *positive* elasticity (demand falls when price falls) because their price cuts are **clearance markdowns on declining stock, not promotions** — the "price<85% of median = promo" heuristic conflates the two in non-food categories. Sophisticated, honest caveat; promo-driven forecasting in M5 is really a FOODS story.
- `analysis/promotions.py` (segment metrics, event-distance error, streamed elasticity), `scripts/analyze_promos.py`, 3 figures.

## 2026-07-22 — Phase 13: WRMSSE + quantile loss (the capstone metric)

- Implemented official M5 **WRMSSE** (`evaluation/wrmsse.py`): RMSSE (naive-scaled from first sale, streamed per level) × revenue weights (per-level normalized, 1/12 each level); scores any bottom forecast bottom-up over all 12 levels. Plus pinball loss + coverage in `metrics.py`. 7 tests.
- **Final comparison (WRMSSE, d1886–1913):** lightgbm **0.555** (wins, ~ real M5), xgboost 0.571, moving_avg_28 1.097, deepar 1.414, tft 1.492. **Complete reversal of the WAPE ranking** (where deep models won 0.66 vs GBM 0.75).
- **Mechanism nailed by per-level breakdown:** deep models match LightGBM at the bottom (RMSSE 0.86) but explode at aggregates (total 1.66 vs 0.27). *Bottom-up cancels random noise but accumulates systematic bias* — the median under-forecast, summed over 30,490 series, craters the money-weighted aggregate levels. This is the flip side of Phase 12's bottom-up win.
- **Verified the fix (controlled experiment, same trained DeepAR):** median→mean point forecast cut WRMSSE **1.88 → 0.68** (2.8×) and bias −0.43 → +0.03, while WAPE worsened 0.67 → 0.76. The point statistic must match the metric: median for absolute-error, mean for squared/weighted/aggregated. Added `mean_forecast_` to DeepAR.
- **Pinball:** deepar 0.297 < tft 0.302 — the probabilistic metric the deep models were built for; calibrated monotone quantiles the GBMs don't natively give.

## 2026-07-22 — Phase 12: hierarchical reconciliation (12 levels)

- Built the 12-level M5 hierarchy (42,840 series) as a sparse summing matrix S; BU / TD / MinT(diag, shrink) with coherence check. Tests cover exact M5 cardinalities and the MinT→BU reduction.
- **Bug caught by inspecting coherence numbers** (not by a test — the tests passed because their ids sorted the same as their keys): leaf-level S rows were key-sorted while columns are id-sorted → bottom block was a permutation, not identity → scrambled bottom indexing / coherence / BU. Fixed to force leaf identity in column order; added a regression test with ids that sort differently from keys. *Lesson: a passing test on a symmetric fixture proves nothing about the asymmetric case.*
- **Base design**: LightGBM fold-3 at the bottom + independent mean-28 at aggregates (genuinely incoherent; median base was rejected because median-of-zeros=0 collapses bottom-up — itself a documented insight).
- **Results (d1886–1913):** coherence base 12,433 → BU/TD exactly 0. Avg WAPE over 12 levels: base 0.247 → **BU 0.174**, TD 0.247 (no help). Total-level WAPE base 0.141 → **BU 0.029** (~5×): "noise cancels under aggregation" made quantitative. Upper-9 exact MinT: BU 0.059 best; MinT 0.114 beats base/TD but not BU because our variance-based diagonal W is misspecified (doesn't know the LightGBM leaves beat the mean-28 aggregates) — MinT≥BU holds only with true W. Honest, sophisticated result.

## 2026-07-22 — Phase 11: TFT-style temporal transformer

- Implemented from scratch (`models/tft/network.py` + `model.py`, ~180 lines): GRN blocks, static-context conditioning, observed/known input routing (decoder physically can't see sales → structural leakage prevention), LSTM encoder-decoder, head-averaged interpretable attention with per-call causal mask, direct multi-quantile head on pinball loss. Reuses DeepAR's dataset unchanged.
- Documented simplifications vs full TFT: VSNs → projected input blocks; interpretable attention → head-averaged standard attention. Honest, not hidden.
- **Bug caught by tests**: attention mask was hard-coded to horizon=28 but predict can use a shorter horizon (test used 20) → built the causal mask per forward pass from actual sizes.
- **Results (test d1886–1913):** TFT MAE 0.928 / RMSE 2.145 / WAPE 0.670 / bias −0.338 — a statistical tie with DeepAR (0.664), same profile (median-optimal WAPE, big negative bias). TFT's edge is operational: single-pass inference ~100× faster than DeepAR's sampling; plus readable attention.
- **Attention finding (honest negative result, fig 11):** predicted weekly spikes (lag 7/14/21); got a recency ramp + mild ~7-day sawtooth instead. Cause: `dow` is an explicit decoder covariate, so the model reads weekly seasonality from that easy channel and spends attention on recent level. Same lesson as Phase 9's dow-importance trap, from the attention side. 45 tests green.

## 2026-07-22 — Phase 10: DeepAR-style probabilistic model

- Implemented DeepAR from scratch in PyTorch across three modules (`deepar/dataset.py`, `network.py`, `model.py`) — the learning goal vs calling GluonTS.
  - **Network**: item(16)/dept(4)/store(4) embeddings + 2-layer LSTM + Negative Binomial head. Parametrization `r=1/alpha, logits=log(mu*alpha)`; unit-tested that `dist.mean == mu*nu`.
  - **Scale handling** (paper 3.3): inputs / nu, head restores (mu*nu, alpha/sqrt(nu)); windows sampled proportional to series volume.
  - **Prediction = ancestral sampling**: 200 paths x 28 days, feed each sampled draw back as next input → uncertainty compounds honestly; point forecast = per-day sample median; 7 quantiles persisted per fold for Phase 13.
  - Leakage guards: window starts constrained below fold cutoff (tested); post-cutoff sales zeroed in dense arrays at fit time.
- **Hardware**: laptop has RTX 4050 (6GB, CUDA 13.1). torch 2.13.0+cu126 wheels exist for py3.14 — installed, `cuda.is_available()==True`. Deep models train on GPU.
- **Test note**: first `test_training_reduces_loss` compared epoch-total NLL across epochs and failed — the tiny synthetic net converges within the first epoch's 8 steps, so epoch 1 total already ≈ epoch 4 total. Fixed to compare a fixed eval-batch NLL before-vs-after training (untrained vs trained), which is the property actually meant. 39 tests green.
- Deep-model compute honesty: running **fold 3 only** (latest fold, aligns with GBM's best fold) rather than all 3 — documented, not hidden.
- **Results (test d1886–1913, same window as GBM fold 3):** DeepAR MAE 0.921 / RMSE 2.134 / WAPE **0.664** / bias **−0.318**. First model to beat the MA bar (0.751 WAPE) decisively — but the −0.318 bias is the flip side: the sample-median point forecast under-shoots the right-skewed mean, so it would stock out on spikes. RMSE only ties the GBMs. Metric-vs-decision tension escalated; final verdict → WRMSSE + pinball (Phase 13).
- **Calibration** (fig 10): quantiles monotone-ordered; every empirical coverage above nominal. Low-quantile over-coverage (5%→57%) is a zero-inflation discreteness artifact (predicted q05=0, and P(y≤0)=P(y=0) is large), not a bug. Trained ~2 min on GPU; NLL 2.06→1.85 over 20 epochs.

## 2026-07-21 — Phase 9: gradient boosting (LightGBM + XGBoost)

- Built `models/gbm.py` (both libraries behind the Phase 8 interface, mirrored recipes), `models/factory.py`, extended `scripts/train.py`. 34 tests green. xgboost had been missing from the env (added).
- **Tuning experiments** (all one-override reproducible):
  - tweedie-NLL early stopping → halted ~70–120 trees, underfit → **rmse stopping** (194 trees, fold-3 WAPE 0.7521, RMSE 2.130 vs MA's 2.219).
  - `train_days` 550 < 365 (0.7542 vs 0.7521) — recent regime beats volume.
- **Results (3-fold mean):** XGBoost 0.7694 WAPE ≈ LightGBM 0.7704, but LightGBM 3× faster (2.5 vs 8 min/fold). Both beat linear_reg (0.7925) clearly = measured value of non-linearity. Neither cleanly beats MA(28) on WAPE yet (0.7535): documented three-cause reading (direct-model feature staleness; WAPE's geometry favoring smoothers on sparse series — GBMs already win fold-3 RMSE by 4%; fold-1 event window under-forecast). Final verdict deferred to WRMSSE (Phase 13) by design.
- **Importance findings** (figure 09): `ewm_a1` 46% + `r_mean_90` 23% of gain — Phase 7 prediction half-right (levels dominate; wrong about which). `dow` at 0.58% despite +37% weekend lift, and the underfitting explanation was falsified (194 trees, unchanged) — real cause: all lags are same-weekday multiples of 7, so weekday signal lives in the lag structure; importance is marginal-within-feature-set, not causal. `item_id`: 8% gain via 14k splits (native categoricals earning keep).

## 2026-07-21 — Phase 8: baselines + evaluation harness

- Shipped the evaluation backbone ([doc](docs/phases/PHASE_08_baselines.md)):
  - `models/base.py` — fit/predict/predict_quantiles contract; fold loop only ever calls the interface → fair comparisons by construction.
  - `evaluation/backtest.py` — expanding folds (test blocks d1830–1857, 1858–1885, 1886–1913); **d1914–1941 reserved for exactly one final touch**.
  - `evaluation/metrics.py` — MAE/RMSE/WAPE/bias; MAPE banned (68% zeros).
  - Five baselines, all vectorized: naive, seasonal-naive(7), MA(28), SES(α=0.2), linear regression on the Phase 7 features (the "bridge baseline" isolating non-linearity's value for Phase 9).
- Windows/tooling papercuts fixed & documented: MLflow rejects bare `D:\` paths as URIs; MLflow 3.x deprecated the ./mlruns file store → **sqlite backend** (`outputs/mlflow.db`).
- 31 tests green.

## 2026-07-21 — Phase 7: feature engineering

- Shipped four feature families ([doc](docs/phases/PHASE_07_feature_engineering.md)):
  - Target-derived (all shift ≥ 28, enforced with ValueError): lags {28,35,42,49,56,364}, rolling mean/std/median/zero_frac over {7,28,90}, EWM(α=0.1), expanding `hist_mean` (leakage-safe target encoding), `momentum_7_28`.
  - Calendar (no shift — known future): date parts, `is_christmas`, `days_to/since_event` (±30 cap; EDA: events have shapes).
  - Price/promo (no shift — exogenous, but trailing-window stats only): `price_chg_7`, `price_rel_med` (trailing 52-week median), `is_promo` (<0.85), `price_rel_dept`. Elasticity deliberately left to the model; measured in Phase 14 instead.
  - Identity categoricals kept raw for LightGBM/native embeddings.
- **Leakage test**: corrupt sales inside (t−28, t] with +1000 → assert features at t bit-identical. Plus API-refusal test for lags < horizon.
- **Real-data bug caught after synthetic tests passed**: `.replace(0, pd.NA)` in momentum → object dtype → astype crash, because real rolling means ARE zero (73%-zeros median series) while the Poisson(3) fixture's never were. Fix: `.where(denom > 0)`. Fixture now includes a 90%-zeros series. *Lesson: synthetic fixtures must mirror the data's pathologies, not its happy path.*
- Build streams per-store (~5.9M rows/chunk) → `data/processed/features/store=XX.parquet` (Phase 4 risk-register mitigation, exercised). 23 tests green.

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
