"""
data/load_adult.py

Loads the UCI Adult Income dataset from local files, applies preprocessing, and
returns the same split-dict API used by the COMPAS loader.

Expected local layout, relative to either this repo or qml-adult-bias-study:

    data/adult/adult.data
    data/adult/adult.test

Target: income >50K.
Protected attributes: race and sex.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

PROTECTED_ATTRS = ["race", "sex"]
TARGET = "income"


def _candidate_data_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    workspace_root = repo_root.parent
    return [
        repo_root / "data" / "adult",
        workspace_root / "qml-adult-bias-study" / "data" / "adult",
        workspace_root / "Adversarial_Debiasing" / "data" / "adult",
        Path.cwd() / "data" / "adult",
    ]


def find_adult_data_dir(data_dir: str | Path | None = None) -> Path:
    """Return a directory containing adult.data and adult.test."""
    candidates = [Path(data_dir)] if data_dir is not None else _candidate_data_dirs()
    for path in candidates:
        if (path / "adult.data").exists() and (path / "adult.test").exists():
            return path

    checked = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Adult dataset files were not found. Expected adult.data and adult.test "
        f"in one of:\n  - {checked}"
    )


def load_adult_raw(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the Adult train/test files."""
    root = find_adult_data_dir(data_dir)
    train_df = pd.read_csv(
        root / "adult.data",
        names=COLUMNS,
        skipinitialspace=True,
        na_values="?",
    )
    test_df = pd.read_csv(
        root / "adult.test",
        names=COLUMNS,
        skipinitialspace=True,
        na_values="?",
        skiprows=1,
    )
    return train_df, test_df


def preprocess_adult(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scale: bool = True,
) -> dict:
    """
    Clean, encode, align one-hot columns, and scale numeric features.

    Sensitive attributes are removed from X and returned in groups_*.
    race is binary encoded as White=1, non-White=0.
    sex is binary encoded as Male=1, Female=0.
    """
    train_df = train_df.dropna().copy()
    test_df = test_df.dropna().copy()

    train_df[TARGET] = (
        train_df[TARGET].str.replace(".", "", regex=False).map({"<=50K": 0, ">50K": 1})
    )
    test_df[TARGET] = (
        test_df[TARGET].str.replace(".", "", regex=False).map({"<=50K": 0, ">50K": 1})
    )

    train_df = train_df.dropna(subset=[TARGET])
    test_df = test_df.dropna(subset=[TARGET])

    groups_train = {
        "race": (train_df["race"] == "White").astype(int).values,
        "sex": (train_df["sex"] == "Male").astype(int).values,
    }
    groups_test = {
        "race": (test_df["race"] == "White").astype(int).values,
        "sex": (test_df["sex"] == "Male").astype(int).values,
    }

    y_train_full = train_df[TARGET].astype(int).values
    y_test = test_df[TARGET].astype(int).values

    X_train_df = train_df.drop(columns=[TARGET] + PROTECTED_ATTRS)
    X_test_df = test_df.drop(columns=[TARGET] + PROTECTED_ATTRS)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train_df[NUMERIC_FEATURES] = scaler.fit_transform(X_train_df[NUMERIC_FEATURES])
        X_test_df[NUMERIC_FEATURES] = scaler.transform(X_test_df[NUMERIC_FEATURES])

    X_train_df = pd.get_dummies(X_train_df)
    X_test_df = pd.get_dummies(X_test_df)
    X_train_df, X_test_df = X_train_df.align(X_test_df, join="inner", axis=1)

    return {
        "X_train_full": X_train_df.values.astype(np.float32),
        "X_test": X_test_df.values.astype(np.float32),
        "y_train_full": y_train_full,
        "y_test": y_test,
        "groups_train_full": groups_train,
        "groups_test": groups_test,
        "scaler": scaler,
        "feature_names": list(X_train_df.columns),
        "df_full": pd.concat([train_df, test_df], ignore_index=True),
    }


def load_adult_splits(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scale: bool = True,
    data_dir: str | Path | None = None,
) -> dict:
    """
    Full pipeline returning the split dict used by experiments.

    The UCI files already provide a train/test split, so test_size is accepted for
    registry compatibility but not used. A validation split is carved from train.
    """
    del test_size
    train_df, test_df = load_adult_raw(data_dir)
    data = preprocess_adult(train_df, test_df, scale=scale)

    y = data["y_train_full"]
    groups = data["groups_train_full"]
    strat_key = y * 2 + groups["race"]

    val_frac = val_size
    X_train, X_val, y_train, y_val, g_train_race, g_val_race, g_train_sex, g_val_sex = (
        train_test_split(
            data["X_train_full"],
            y,
            groups["race"],
            groups["sex"],
            test_size=val_frac,
            stratify=strat_key,
            random_state=random_state,
        )
    )

    print(
        f"[data] Preprocessed Adult: train={len(X_train)} | val={len(X_val)} | "
        f"test={len(data['X_test'])} | features={X_train.shape[1]}"
    )

    return {
        "X_train": X_train.astype(np.float32),
        "X_val": X_val.astype(np.float32),
        "X_test": data["X_test"].astype(np.float32),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": data["y_test"],
        "groups_train": {"race": g_train_race, "sex": g_train_sex},
        "groups_val": {"race": g_val_race, "sex": g_val_sex},
        "groups_test": data["groups_test"],
        "scaler": data["scaler"],
        "feature_names": data["feature_names"],
        "df_full": data["df_full"],
    }


if __name__ == "__main__":
    splits = load_adult_splits()
    print("Feature count:", len(splits["feature_names"]))
    print("X_train shape:", splits["X_train"].shape)
