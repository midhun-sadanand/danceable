"""
EDA script for the Danceable project.

Investigates:
  1. Feature distributions and outliers
  2. Correlation matrix and multicollinearity (VIF)
  3. Condition number of the scaled feature matrix
  4. Cross-validation fold simulation — which fold triggers overflow warnings
  5. Per-fold coefficient magnitudes (to catch exploding coefficients)

Run from the danceable/ directory:
    python3 src/eda.py
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SRC_PATH = Path(__file__).parent
sys.path.insert(0, str(SRC_PATH))
from project_utils import ensure_processed_data

OUT = Path("outputs/eda")
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "danceability"

# ── Load data ─────────────────────────────────────────────────────────────────

parts = ensure_processed_data()
train = parts["train"]
feature_cols = [c for c in train.columns if c != TARGET]
X_raw = train[feature_cols]
y = train[TARGET]

print(f"Train shape: {train.shape}")
print(f"Features: {feature_cols}\n")

# ── 1. Descriptive stats ──────────────────────────────────────────────────────

print("=" * 60)
print("1. DESCRIPTIVE STATS")
print("=" * 60)
stats = train.describe().T[["mean", "std", "min", "max"]]
print(stats.to_string())
stats.to_csv(OUT / "descriptive_stats.csv")

# ── 2. Outlier analysis ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. OUTLIER ANALYSIS (|z| > 5)")
print("=" * 60)
z_scores = (X_raw - X_raw.mean()) / X_raw.std()
extreme = (z_scores.abs() > 5).sum().sort_values(ascending=False)
print(extreme[extreme > 0].to_string())
print(f"\nTotal rows with any |z| > 5: {(z_scores.abs() > 5).any(axis=1).sum()}")
print(f"Total rows with any |z| > 10: {(z_scores.abs() > 10).any(axis=1).sum()}")

outlier_detail = {}
for col in feature_cols:
    z = z_scores[col]
    extreme_rows = (z.abs() > 5).sum()
    if extreme_rows > 0:
        outlier_detail[col] = {
            "n_extreme_z5": extreme_rows,
            "n_extreme_z10": (z.abs() > 10).sum(),
            "max_z": z.abs().max().round(1),
            "raw_max": X_raw[col].max(),
            "raw_min": X_raw[col].min(),
        }
pd.DataFrame(outlier_detail).T.to_csv(OUT / "outlier_summary.csv")

# ── 3. Correlation matrix ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. CORRELATION MATRIX (sorted by |corr with target|)")
print("=" * 60)
corr = train[feature_cols + [TARGET]].corr()
target_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print(target_corr.round(3).to_string())
target_corr.to_csv(OUT / "target_correlations.csv")

# Top inter-feature correlations (potential multicollinearity)
print("\nTop inter-feature correlations (|r| > 0.5):")
feat_corr = corr.loc[feature_cols, feature_cols]
mask = np.triu(np.ones_like(feat_corr, dtype=bool), k=1)
high_corr = (
    feat_corr.where(mask)
    .stack()
    .reset_index()
    .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "r"})
)
high_corr = high_corr[high_corr["r"].abs() > 0.5].sort_values("r", key=abs, ascending=False)
print(high_corr.to_string(index=False) if len(high_corr) else "  None above threshold.")
high_corr.to_csv(OUT / "high_inter_feature_correlations.csv", index=False)

# ── 4. VIF (Variance Inflation Factor) ───────────────────────────────────────

print("\n" + "=" * 60)
print("4. VIF (multicollinearity; VIF > 10 is concerning)")
print("=" * 60)
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    vif_data = pd.DataFrame({
        "feature": feature_cols,
        "VIF": [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])],
    }).sort_values("VIF", ascending=False)
    print(vif_data.to_string(index=False))
    vif_data.to_csv(OUT / "vif.csv", index=False)
except ImportError:
    print("  statsmodels not installed — skipping VIF (pip install statsmodels)")

# ── 5. Condition number of scaled feature matrix ──────────────────────────────

print("\n" + "=" * 60)
print("5. CONDITION NUMBER OF SCALED FEATURE MATRIX")
print("=" * 60)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
cond = np.linalg.cond(X_scaled)
print(f"Condition number: {cond:.2e}")
print("  < 100   → well-conditioned")
print("  100–1k  → moderate ill-conditioning")
print("  > 1k    → ill-conditioned (can cause numerical instability)")

# ── 6. Per-fold coefficient magnitudes ───────────────────────────────────────

print("\n" + "=" * 60)
print("6. PER-FOLD COEFFICIENT MAGNITUDES (5-fold CV)")
print("=" * 60)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_coef_norms = []
fold_warnings = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_raw)):
    X_tr, X_vl = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
    y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_vl)

    coef_norm = np.linalg.norm(pipe.named_steps["lr"].coef_)
    n_warns = len(caught)
    fold_warnings.append(n_warns)
    fold_coef_norms.append(coef_norm)

    status = f"  {n_warns} warning(s)" if n_warns else ""
    print(f"  Fold {fold+1}: coef L2-norm = {coef_norm:.4f}{status}")
    if n_warns:
        for w in caught:
            print(f"    -> {w.category.__name__}: {w.message}")

fold_summary = pd.DataFrame({
    "fold": range(1, 6),
    "coef_l2_norm": fold_coef_norms,
    "n_warnings": fold_warnings,
})
fold_summary.to_csv(OUT / "fold_coef_norms.csv", index=False)

# ── 7. Distribution plots ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("7. SAVING DISTRIBUTION PLOTS")
print("=" * 60)

# Raw distributions with outlier threshold lines
fig, axes = plt.subplots(3, 5, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    axes[i].hist(train[col], bins=40, edgecolor="none")
    axes[i].set_title(col, fontsize=9)
    z = z_scores[col]
    if z.abs().max() > 5:
        axes[i].axvline(
            X_raw[col].mean() + 5 * X_raw[col].std(), color="red", linestyle="--", linewidth=0.8
        )
        axes[i].axvline(
            X_raw[col].mean() - 5 * X_raw[col].std(), color="red", linestyle="--", linewidth=0.8
        )
for j in range(i + 1, len(axes)):
    axes[j].axis("off")
plt.suptitle("Feature Distributions (red dashed = ±5σ)", y=1.01)
plt.tight_layout()
plt.savefig(OUT / "feature_distributions.png", dpi=120, bbox_inches="tight")
plt.close()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=ax)
ax.set_title("Correlation Matrix")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()

print(f"\nAll outputs saved to: {OUT.resolve()}")
print("Done.")
