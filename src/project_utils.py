from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EXPECTED_AUDIO_COLS = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "loudness",
    "tempo",
    "speechiness",
    "instrumentalness",
    "liveness",
    "popularity",
]

BOUNDED_01_COLS = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "liveness",
]


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_column_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_column_name(col) for col in out.columns]
    return out


def find_spotify_csv(data_dir: str | Path = "data") -> Path | None:
    data_dir = Path(data_dir)
    candidates = sorted(data_dir.rglob("*.csv"))
    if not candidates:
        return None

    best_path = None
    best_score = -1
    for csv_path in candidates:
        try:
            head = pd.read_csv(csv_path, nrows=200)
        except Exception:
            continue
        cols = {normalize_column_name(c) for c in head.columns}
        score = len(set(EXPECTED_AUDIO_COLS).intersection(cols))
        if "spotify" in csv_path.name.lower():
            score += 1
        if score > best_score:
            best_score = score
            best_path = csv_path

    return best_path


def _download_kaggle_fallback() -> Path | None:
    try:
        import kagglehub  # type: ignore

        path = Path(kagglehub.dataset_download("tomigelo/spotify-audio-features"))
    except Exception:
        return None

    candidates = sorted(path.rglob("*.csv"))
    if not candidates:
        return None

    best_path = None
    best_score = -1
    for csv_path in candidates:
        try:
            head = pd.read_csv(csv_path, nrows=200)
        except Exception:
            continue
        cols = {normalize_column_name(c) for c in head.columns}
        score = len(set(EXPECTED_AUDIO_COLS).intersection(cols))
        if score > best_score:
            best_score = score
            best_path = csv_path
    return best_path


def load_spotify_dataframe(
    data_dir: str | Path = "data",
    csv_path: str | Path | None = None,
    allow_download: bool = True,
) -> Tuple[pd.DataFrame, Path]:
    if csv_path is not None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
    else:
        csv_path = find_spotify_csv(data_dir)

    if csv_path is None and allow_download:
        csv_path = _download_kaggle_fallback()

    if csv_path is None:
        raise FileNotFoundError(
            "No CSV found. Put the Spotify dataset CSV under data/ or install kagglehub."
        )

    df = pd.read_csv(csv_path)
    return df, csv_path


def coerce_numeric_columns(df: pd.DataFrame, min_numeric_ratio: float = 0.8) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]):
            original_non_null = out[col].notna().sum()
            converted = pd.to_numeric(out[col], errors="coerce")
            converted_non_null = converted.notna().sum()
            ratio = (converted_non_null / original_non_null) if original_non_null else 0.0
            if ratio >= min_numeric_ratio:
                out[col] = converted
    return out


def clean_spotify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clean = standardize_columns(df)

    unnamed_cols = [c for c in clean.columns if c.startswith("unnamed")]
    if unnamed_cols:
        clean = clean.drop(columns=unnamed_cols)

    clean = clean.drop_duplicates()
    clean = coerce_numeric_columns(clean)

    # Keep only rows where core proposal features are present.
    core_cols = [c for c in EXPECTED_AUDIO_COLS if c in clean.columns]
    if core_cols:
        clean = clean.dropna(subset=core_cols)

    for col in BOUNDED_01_COLS:
        if col in clean.columns:
            clean[col] = clean[col].clip(0, 1)

    if "tempo" in clean.columns:
        clean = clean[clean["tempo"] > 0]
    if "duration_ms" in clean.columns:
        clean = clean[clean["duration_ms"] > 0]

    clean = clean.reset_index(drop=True)
    return clean


def get_numeric_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols].copy()


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
    val_ratio_of_train_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio_of_train_val, random_state=random_state
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def compute_regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"mse": mse, "mae": mae, "r2": r2}


def ensure_processed_data(
    data_dir: str | Path = "data",
    processed_dir: str | Path = "data/processed",
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    processed_dir = ensure_dir(processed_dir)

    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"
    clean_path = processed_dir / "spotify_clean.csv"

    if train_path.exists() and val_path.exists() and test_path.exists() and clean_path.exists():
        return {
            "clean": pd.read_csv(clean_path),
            "train": pd.read_csv(train_path),
            "val": pd.read_csv(val_path),
            "test": pd.read_csv(test_path),
        }

    raw_df, source_path = load_spotify_dataframe(data_dir=data_dir)
    clean_df = clean_spotify_dataframe(raw_df)
    model_df = get_numeric_model_frame(clean_df)
    train_df, val_df, test_df = split_data(model_df, random_state=random_state)

    clean_df.to_csv(clean_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved processed data from: {source_path}")

    return {
        "clean": clean_df,
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
