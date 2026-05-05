# TODO

## Project Flow

### Step 1 — Data Cleaning (`dataclean.ipynb`) ✅
- [x] Load raw Spotify CSV (KaggleHub fallback if missing)
- [x] Clean: dedup, clip [0,1] features, drop zero-tempo rows
- [x] Save `data/processed/spotify_clean.csv`
- [x] Save train/val/test splits (70/15/15) to `data/processed/`
- [x] Plot feature distributions

### Step 2 — Baseline Models (`baseline.ipynb`) ✅
- [x] Mean baseline (DummyRegressor) on danceability
- [x] Linear regression with StandardScaler on danceability
- [x] 5-fold cross-validation
- [x] Save metrics → `outputs/baseline_danceability_metrics.csv`
- [x] Save coefficients → `outputs/baseline_danceability_coefficients.csv`
- [x] Scatter plot: predicted vs actual (danceability only)
- [x] **Investigate overflow/divide-by-zero warnings** during cross_validate — resolved; no warnings emitted on fresh runs

### Step 3 — Regularized Models (`regularized_models.ipynb`) ✅
- [x] Run Dummy, Linear, Ridge (RidgeCV), Lasso (LassoCV) for all 6 targets
- [x] Save results → `outputs/regularized_results.csv`, `regularized_test_results.csv`, `regularized_best_params.csv`
- [x] Save best model per target → `outputs/best_model_by_target.csv`
- [x] **Add scatter plots (predicted vs actual) for all 6 targets** — saved to `outputs/scatter_predicted_vs_actual.png`
- [x] **Add Lasso coefficient/feature importance analysis** — saved `lasso_coefficients.csv`, `lasso_sparsity_summary.csv`, `lasso_feature_rank.csv`, plus heatmap PNGs
- [ ] Add K-fold CV for all targets (currently only danceability has CV columns)

### Step 4 — Tree-Based Extension (`tree_models.ipynb`) ✅
- [x] Implement Random Forest **and** XGBoost for all 6 targets
- [x] Compare R² against linear/regularized models — `final.ipynb` auto-merges `tree_test_results.csv`
- [x] Extract feature importances — saved to `outputs/tree_feature_importances.csv` and visualized in `outputs/tree_feature_importance_heatmap.png`

### Step 5 — Final Assembly (`final.ipynb`) ✅
- [x] **Re-run steps 1–3 to regenerate `outputs/` files**
- [x] R² heatmap / model-vs-target chart (cell 4 of `final.ipynb`)
- [x] Best-model summary table → `outputs/final/best_model_by_target.csv`
- [x] Save `outputs/final/all_test_model_results.csv` and `r2_comparison_pivot.csv`
- [ ] Written summary statements for report

### Step 6 — Report / Presentation
- [ ] Key plots from `final.ipynb`
- [ ] Discuss feature redundancy using R² rankings (energy and loudness most predictable; tempo least)
- [ ] Discuss Lasso feature selection findings
- [ ] Include `outputs/final/best_model_by_target.csv` in write-up
- [ ] (Bonus) Artist clustering analysis — do songs by the same artist cluster in feature space?
