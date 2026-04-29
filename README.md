# Danceable: CPSC 381 Final Project (Spring 2026)
*generated readme for the other group members to update and keep up with work*



This repo implements our project proposal:

**"What Makes a Song Danceable? A Predictability Analysis Among Audio Features"**

We study whether Spotify audio features can predict each other (and therefore reveal feature redundancy), with a focus on:
- `danceability`
- `energy`
- `valence`
- `acousticness`
- `loudness`
- `tempo`

## Proposal Alignment

This structure follows the proposal milestones:
- **Week 1-2:** dataset setup + cleaning (`dataclean.ipynb`)
- **Week 3:** baseline + linear model (`baseline.ipynb`)
- **Week 4-5:** Ridge/Lasso + multi-target comparison (`regularized_models.ipynb`)
- **Week 6:** final plots/charts/comparison summary (`final.ipynb`)

## Project Structure

- `dataclean.ipynb`  
  Loads Spotify data, cleans it, saves processed dataset + train/val/test splits.
- `baseline.ipynb`  
  Danceability-only baseline (`mean`) vs linear regression with CV + metrics.
- `regularized_models.ipynb`  
  Multi-target modeling with `Dummy`, `Linear`, `Ridge`, `Lasso`.
- `final.ipynb`  
  Final comparison plots/tables and report-ready summary outputs.
- `src/project_utils.py`  
  Shared helper functions (loading, cleaning, splitting, metrics).
- `data/processed/`  
  Generated cleaned dataset and splits.
- `outputs/`  
  Saved metrics and model comparison files.

## Setup

From repo root:

```bash
python3 -m pip install pandas numpy scikit-learn matplotlib kagglehub
```

Notes:
- If your dataset CSV already exists under `data/`, the notebooks use it directly.
- If not, they attempt a KaggleHub fallback to `tomigelo/spotify-audio-features`.

## Run Order (Important)

1. `dataclean.ipynb`
2. `baseline.ipynb`
3. `regularized_models.ipynb`
4. `final.ipynb`

The later notebooks expect output files created by earlier notebooks.

## Metrics Used (From Proposal)

Every model comparison reports:
- **MSE** (primary training/comparison objective)
- **MAE** (easy to interpret in target units)
- **R²** (variance explained vs mean baseline)

## Main Output Files

- `outputs/baseline_danceability_metrics.csv`
- `outputs/baseline_danceability_coefficients.csv`
- `outputs/regularized_results.csv`
- `outputs/regularized_test_results.csv`
- `outputs/regularized_best_params.csv`
- `outputs/best_model_by_target.csv`
- `outputs/final/all_test_model_results.csv`
- `outputs/final/best_model_by_target.csv`
- `outputs/final/r2_comparison_pivot.csv`

## What To Submit

Use `final.ipynb` as your final assembly notebook for:
- plots/charts,
- model-vs-target comparison,
- final summary statements for your report/presentation.

This keeps everything simple and intuitive while matching the proposal workflow.



## Submission Checklist
- Include key plots from this notebook.
- Include `outputs/final/best_model_by_target.csv` in your write-up.
- Discuss feature redundancy using which targets are easiest to predict (highest R²).