"""
data/load_draft_datasets.py

Loaders for the extra datasets listed in the project draft.

These loaders expect local CSV files. They intentionally do not download from
Kaggle/UCI because those sources usually require manual download or credentials.
Place files under qml-bias-audit/data/raw/<dataset_key>/ or pass data_dir=...
when calling the loader directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.tabular_utils import (
    binary_from_strings,
    load_single_csv_dataset,
    race_white_vs_other,
    sex_to_binary,
)


def _numeric_binary(series: pd.Series) -> np.ndarray:
    return series.astype(float).round().astype(int).values


def _yes_no(series: pd.Series) -> np.ndarray:
    return binary_from_strings(series, {"yes", "y", "true", "1"})


def _readmitted(series: pd.Series) -> np.ndarray:
    vals = series.astype(str).str.strip().str.lower()
    return vals.isin({"<30", ">30"}).astype(int).values


def _glioma_grade(series: pd.Series) -> np.ndarray:
    """Encode higher grade / GBM as positive when the column is textual."""
    if pd.api.types.is_numeric_dtype(series):
        values = series.astype(int).values
        if set(pd.Series(values).dropna().unique()).issubset({0, 1}):
            return values
        return (values >= np.nanmedian(values)).astype(int)

    vals = series.astype(str).str.strip().str.lower()
    positive = {
        "gbm",
        "high",
        "high grade",
        "hgg",
        "grade iv",
        "iv",
        "4",
        "glioblastoma",
    }
    return vals.isin(positive).astype(int).values


def _glioma_race(series: pd.Series) -> np.ndarray:
    """UCI glioma file uses numeric race codes; 0 corresponds to White."""
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(int) == 0).astype(int).values
    return race_white_vs_other(series)


def load_diabetes_prediction_splits(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scale: bool = True,
    data_dir: str | Path | None = None,
) -> dict:
    """Kaggle Diabetes Prediction Dataset. Protected attribute: gender/sex."""
    return load_single_csv_dataset(
        dataset_name="Diabetes Prediction",
        dataset_dir="diabetes_prediction",
        filenames=["diabetes_prediction_dataset.csv", "diabetes.csv"],
        target_aliases=["diabetes"],
        protected_builders={
            "sex": (["gender", "sex"], sex_to_binary),
        },
        target_builder=_numeric_binary,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        scale=scale,
        data_dir=data_dir,
    )


def load_cardiovascular_splits(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scale: bool = True,
    data_dir: str | Path | None = None,
) -> dict:
    """Kaggle Cardiovascular Disease Dataset. Protected attribute: gender."""
    return load_single_csv_dataset(
        dataset_name="Cardiovascular Disease",
        dataset_dir="cardiovascular",
        filenames=["cardio_train.csv", "cardiovascular_disease_dataset.csv"],
        target_aliases=["cardio", "target"],
        protected_builders={
            "sex": (["gender", "sex"], sex_to_binary),
        },
        target_builder=_numeric_binary,
        drop_aliases=["id"],
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        scale=scale,
        data_dir=data_dir,
    )


def load_heart_indicators_splits(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scale: bool = True,
    data_dir: str | Path | None = None,
) -> dict:
    """Personal Key Indicators of Heart Disease Dataset. Protected attribute: sex."""
    return load_single_csv_dataset(
        dataset_name="Heart Disease Indicators",
        dataset_dir="heart_indicators",
        filenames=[
            "heart_2020_cleaned.csv",
            "heart_2022_cleaned.csv",
            "personal_key_indicators_heart_disease.csv",
        ],
        target_aliases=["HeartDisease", "heartdisease", "HadHeartAttack"],
        protected_builders={
            "sex": (["Sex", "sex", "gender"], sex_to_binary),
        },
        target_builder=_yes_no,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        scale=scale,
        data_dir=data_dir,
    )


def load_diabetes_hospital_splits(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scale: bool = True,
    data_dir: str | Path | None = None,
) -> dict:
    """Diabetes 130-US Hospitals Dataset. Protected attribute: race."""
    return load_single_csv_dataset(
        dataset_name="Diabetes 130-US Hospitals",
        dataset_dir="diabetes_hospital",
        filenames=["diabetic_data.csv", "diabetes_130_us_hospitals.csv"],
        target_aliases=["readmitted"],
        protected_builders={
            "race": (["race"], race_white_vs_other),
        },
        target_builder=_readmitted,
        drop_aliases=["encounter_id", "patient_nbr"],
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        scale=scale,
        data_dir=data_dir,
    )


def load_glioma_splits(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scale: bool = True,
    data_dir: str | Path | None = None,
) -> dict:
    """UCI Glioma Grading Clinical and Mutation Features. Protected attribute: race."""
    return load_single_csv_dataset(
        dataset_name="Glioma Grading",
        dataset_dir="glioma",
        filenames=[
            "TCGA_InfoWithGrade.csv",
            "glioma_grading.csv",
            "Glioma Grading Clinical and Mutation Features.csv",
        ],
        target_aliases=["Grade", "grade"],
        protected_builders={
            "race": (["Race", "race"], _glioma_race),
        },
        target_builder=_glioma_grade,
        drop_aliases=["Project", "Case_ID", "Case ID", "ID"],
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        scale=scale,
        data_dir=data_dir,
    )
