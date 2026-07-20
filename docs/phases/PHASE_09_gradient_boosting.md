# Phase 9 — Gradient Boosting

> Status: ✅ Complete · The primary model family (and the M5 winner's). Results and importances at the bottom are generated from the real backtest.

---

## 1. Decision trees from first principles

A decision tree predicts by asking a sequence of yes/no questions about features and outputting a constant in each terminal region ("leaf"): *"is r_mean_28 > 3.2? → is dow a weekend? → predict 5.1"*. Training greedily picks, at each node, the (feature, threshold) split that most reduces loss; the leaf value is the average (more precisely, the loss-minimizing constant) of its training rows.

Why trees fit tabular retail data so well:
- **Sharp interactions for free**: "high recent demand AND weekend AND on-promo" is three splits — no feature crosses needed. Promo × weekday × item-type effects are exactly this shape.
- **Scale-free**: splits care only about order, so no normalization; robust to outliers in features.
- **Native missing-value routing**: our early-history NaN lags just get routed to whichever child helps loss.

And the two structural weaknesses we've already met: **no extrapolation** (a leaf can't output a value it never saw — Phase 2's trend problem) and **no memory** (hence the whole Phase 7 feature apparatus).

## 2. From one tree to boosting

One deep tree overfits (memorizes rows); one shallow tree underfits. **Gradient boosting** builds an *additive* model: trees are added one at a time, each fitted to the **negative gradient of the loss at the current predictions** — the direction each prediction should move. For squared loss the gradient is just the residual, so intuition: *each new tree predicts the errors of the ensemble so far*, scaled by a small learning rate.

That gradient view is the key generalization: swap the loss, and the same machinery optimizes **any differentiable objective** — which is precisely how we get Tweedie boosting (and, later, quantile boosting). Phase 8's lesson ("which loss you optimize decides which metric you win") becomes an engineering dial.

## 3. The library landscape

- **XGBoost** (2016): industrialized GBM — second-order (Newton) optimization, explicit L1/L2 regularization on leaf values, sparsity-aware splits.
- **LightGBM** (2017): made GBM fast at panel scale — **histogram binning** (features quantized to 255 bins; split search touches bins, not rows), **leaf-wise growth** (always split the leaf with the largest loss reduction, rather than filling levels — deeper where the data wants it), **GOSS** (bias split search toward large-gradient rows), **EFB** (bundle mutually exclusive sparse features), and **native categoricals** (optimal split of category *sets*, no one-hot explosion over 3,049 item ids).
- **CatBoost** (2017): ordered target statistics for categoricals + symmetric trees; strongest when high-cardinality categoricals dominate and leakage via target stats is the main risk. We teach it, don't run it — LightGBM already handles our categoricals natively and the M5 evidence base is LightGBM's.

## 4. Our design (and every parameter's why)

| Choice | Value | Why |
|---|---|---|
| objective | `tweedie`, power=1.1 | Compound Poisson-Gamma: point mass at zero + continuous positive part = intermittent retail counts. Power→1 is Poisson-like, →2 Gamma-like; 1.1 (near-Poisson with overdispersion) is the M5-winning setting. |
| strategy | direct multi-step | All features are ≥28-day-safe (Phase 7), so one model serves all 28 horizons; no recursive error compounding. |
| `num_leaves` 128 | leaf-wise depth budget | ~2^7-equivalent complexity; the standard M5 range (top solutions: 100–200). |
| `learning_rate` 0.05 + up to 1500 trees + early stopping (100) | slow learning, data decides depth of ensemble | The canonical robust recipe; early stopping on the last 28 train days (legal: features can't see those targets). |
| `feature_fraction` / `bagging_fraction` 0.8 | decorrelate trees | Variance reduction, mild regularization. |
| `train_days` 365 | trailing window | RAM-bound honesty on a 16GB machine (11M rows in-window); top M5 teams used 2–3 years for ~1–2% more — documented trade, not hidden. |
| categoricals | native (`item_id`… as category) | No 3,049-wide one-hot; LightGBM finds optimal category-set splits. |
| XGBoost run | same recipe, `reg:tweedie`, hist, lossguide | Mirrored config isolates the *library* as the only variable. |

## 5. Results (3-fold mean, vs the Phase 8 bar)

<!-- RESULTS -->

## 6. Feature importance — checking the Phase 7 prediction

<!-- IMPORTANCE -->

## 7. Interview questions — Phase 9

**Easy**
1. Why do gradient-boosted trees dominate tabular forecasting? *(Sharp interactions natively, no scaling, missing-value routing, strong with modest data — Grinsztajn 2022.)*
2. What does the learning rate do in boosting? *(Shrinks each tree's contribution; small rate + more trees = smoother, better-generalizing fit.)*

**Medium**
3. Why Tweedie loss instead of MSE here? *(68% zeros: MSE targets the conditional mean and can't represent zero-inflation; Tweedie has probability mass at exactly zero — its gradients push predictions to respect it. Phase 8 showed OLS/MSE losing WAPE for this reason.)*
4. Leaf-wise vs level-wise growth? *(Leaf-wise always splits the max-gain leaf → deeper, asymmetric trees, better loss per leaf count; needs num_leaves capping to avoid overfit. Level-wise (classic XGBoost) is more conservative.)*
5. Why is validating on the last 28 training days not leakage for this model? *(Every feature at day t is built from data ≤ t−28; the validation targets are never visible to any feature row used in training.)*
6. How does LightGBM handle 3,049 item ids without one-hot? *(Native categorical splits: orders categories by gradient statistics, finds optimal set partition — approximately optimal split in O(k log k).)*

**Hard**
7. Your LightGBM beats XGBoost by 2% with identical settings. Is LightGBM "better"? *(No general claim: same recipe ≠ same optimum per library; differences come from growth policy details, binning, categorical algorithms. The honest statement is about this dataset/recipe.)*
8. Boosting with Tweedie: what exactly does each successive tree fit? *(The negative gradient of Tweedie NLL at current predictions — not raw residuals; for Tweedie the gradient is (exp-scale) µ^{1−p}(µ−y)-shaped, so zero-heavy rows push predictions down in a likelihood-weighted way.)*
9. Why not just log1p-transform sales and use MSE? *(A real M5 alternative! But: bias when back-transforming (Jensen), zeros still special-cased, and Tweedie handled it better empirically in the competition. Good ablation candidate.)*
10. When would you actually pick CatBoost here? *(If target-statistic encodings of high-cardinality ids were the main signal and leakage-in-encoding the main risk — CatBoost's ordered boosting solves exactly that.)*

---

*Next: Phase 10 — DeepAR-style probabilistic forecasting (PyTorch, Negative Binomial likelihood, ancestral sampling).*
