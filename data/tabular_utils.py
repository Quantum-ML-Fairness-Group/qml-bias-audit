"""
data/tabular_utils.py

Shared helpers for local tabular fairness datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return repo_root().parent


def find_data_file(
    dataset_dir: str,
    filenames: list[str],
    data_dir: str | Path | None = None,
) -> Path:
    """Find the first matching local data file for a dataset."""
    if data_dir is not None:
        roots = [Path(data_dir)]
    else:
        roots = [
            repo_root() / "data" / "raw" / dataset_dir,
            repo_root() / "data" / dataset_dir,
            workspace_root() / "data" / dataset_dir,
            Path.cwd() / "data" / "raw" / dataset_dir,
            Path.cwd() / "data" / dataset_dir,
        ]

    for root in roots:
        for filename in filenames:
            path = root / filename
            if path.exists():
                return path

    checked = "\n  - ".join(str(root / name) for root in roots for name in filenames)
    raise FileNotFoundError(
        f"Could not find data file for {dataset_dir!r}. Expected one of:\n  - {checked}"
    )


def read_csv_auto(path: Path) -> pd.DataFrame:
    """Read comma- or semicolon-delimited CSV files."""
    return pd.read_csv(path, sep=None, engine="python", na_values=["?", "NA", "N/A", ""])


def find_column(df: pd.DataFrame, aliases: list[str]) -> str:
    """Find a column by case-insensitive alias."""
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        key = alias.strip().lower()
        if key in normalized:
            return normalized[key]
    raise KeyError(f"Could not find any of {aliases}. Available columns: {list(df.columns)}")


def binary_from_strings(series: pd.Series, positive_values: set[str]) -> np.ndarray:
    vals = series.astype(str).str.strip().str.lower()
    positives = {v.lower() for v in positive_values}
    return vals.isin(positives).astype(int).values


def sex_to_binary(series: pd.Series) -> np.ndarray:
    """Encode Male=1, Female=0 for text or common numeric encodings."""
    if pd.api.types.is_numeric_dtype(series):
        # Common Kaggle cardiovascular coding: 1=female, 2=male.
        uniq = set(series.dropna().astype(int).unique())
        if uniq.issubset({1, 2}):
            return (series.astype(int) == 2).astype(int).values
        return (series.astype(float) > series.astype(float).median()).astype(int).values
    return binary_from_strings(series, {"male", "m", "man"})


def race_white_vs_other(series: pd.Series) -> np.ndarray:
    """Encode White/Caucasian=1, all other known races=0."""
    vals = series.astype(str).str.strip().str.lower()
    return vals.isin({"white", "caucasian"}).astype(int).values


def clean_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values and one-hot encode categorical columns."""
    X = X.copy()
    X = X.replace(["?", "NA", "N/A", ""], np.nan)
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].astype("object").where(X[col].notna(), "Unknown")
    return pd.get_dummies(X, dummy_na=False)


def safe_train_test_split(*arrays, stratify, **kwargs):
    """Use stratified splitting when possible, otherwise fall back gracefully."""
    try:
        return train_test_split(*arrays, stratify=stratify, **kwargs)
    except ValueError as exc:
        print(f"[data] Stratified split unavailable ({exc}); using unstratified split.")
        return train_test_split(*arrays, stratify=None, **kwargs)


def make_split_dict(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: dict[str, np.ndarray],
    feature_drop_cols: list[str],
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scale: bool = True,
) -> dict:
    """
    Convert a processed DataFrame, binary target, and group arrays into the
    standard experiment split dict.
    """
    keep = np.isfinite(y)
    for values in groups.values():
        keep &= pd.Series(values).notna().values

    df = df.loc[keep].reset_index(drop=True)
    y = y[keep].astype(int)
    groups = {k: np.asarray(v)[keep].astype(int) for k, v in groups.items()}

    X_df = clean_feature_frame(df.drop(columns=feature_drop_cols, errors="ignore"))
    feature_names = list(X_df.columns)
    X = X_df.values.astype(np.float32)

    primary_group = groups["race"] if "race" in groups else next(iter(groups.values()))
    strat_key = y * 2 + primary_group

    arrays = [X, y] + [groups[k] for k in groups]
    split = safe_train_test_split(
        *arrays,
        test_size=test_size,
        stratify=strat_key,
        random_state=random_state,
    )

    X_temp, X_test, y_temp, y_test = split[0], split[1], split[2], split[3]
    group_keys = list(groups.keys())
    temp_groups = {}
    test_groups = {}
    offset = 4
    for key in group_keys:
        temp_groups[key] = split[offset]
        test_groups[key] = split[offset + 1]
        offset += 2

    strat_val = y_temp * 2 + (
        temp_groups["race"] if "race" in temp_groups else next(iter(temp_groups.values()))
    )
    val_frac = val_size / (1 - test_size)
    arrays = [X_temp, y_temp] + [temp_groups[k] for k in group_keys]
    split = safe_train_test_split(
        *arrays,
        test_size=val_frac,
        stratify=strat_val,
        random_state=random_state,
    )

    X_train, X_val, y_train, y_val = split[0], split[1], split[2], split[3]
    train_groups = {}
    val_groups = {}
    offset = 4
    for key in group_keys:
        train_groups[key] = split[offset]
        val_groups[key] = split[offset + 1]
        offset += 2

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    return {
        "X_train": X_train.astype(np.float32),
        "X_val": X_val.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(int),
        "y_val": y_val.astype(int),
        "y_test": y_test.astype(int),
        "groups_train": train_groups,
        "groups_val": val_groups,
        "groups_test": test_groups,
        "scaler": scaler,
        "feature_names": feature_names,
        "df_full": df,
    }


def load_single_csv_dataset(
    *,
    dataset_name: str,
    dataset_dir: str,
    filenames: list[str],
    target_aliases: list[str],
    protected_builders: dict[str, tuple[list[str], Callable[[pd.Series], np.ndarray]]],
    target_builder: Callable[[pd.Series], np.ndarray],
    drop_aliases: list[str] | None = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scale: bool = True,
    data_dir: str | Path | None = None,
) -> dict:
    """Load one local CSV and return the standard split dict."""
    path = find_data_file(dataset_dir, filenames, data_dir=data_dir)
    df = read_csv_auto(path)

    target_col = find_column(df, target_aliases)
    y = target_builder(df[target_col])

    groups = {}
    protected_cols = []
    for attr, (aliases, builder) in protected_builders.items():
        col = find_column(df, aliases)
        protected_cols.append(col)
        groups[attr] = builder(df[col])

    drop_cols = [target_col] + protected_cols
    for aliases in drop_aliases or []:
        try:
            drop_cols.append(find_column(df, [aliases]))
        except KeyError:
            pass

    splits = make_split_dict(
        df,
        y,
        groups,
        drop_cols,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        scale=scale,
    )
    print(
        f"[data] Preprocessed {dataset_name}: train={len(splits['X_train'])} | "
        f"val={len(splits['X_val'])} | test={len(splits['X_test'])} | "
        f"features={splits['X_train'].shape[1]}"
    )
    return splits
