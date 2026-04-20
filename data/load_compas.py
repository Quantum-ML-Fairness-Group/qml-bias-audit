"""
data/load_compas.py

Loads the ProPublica COMPAS recidivism dataset, applies standard preprocessing,
and exposes a clean API for the rest of the pipeline.

COMPAS features used:
  - age, priors_count, days_b_screening_arrest, juv_fel_count, juv_misd_count,
    juv_other_count, c_charge_degree (binary)
Protected attributes: race (binary: African-American vs Caucasian), sex
Target: two_year_recid
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)

CACHE_PATH = Path(__file__).parent / "compas_raw.csv"
LOCAL_CANDIDATES = [
    Path(__file__).parent / "compas_raw.csv",
    Path(__file__).parent / "raw" / "compas" / "compas-scores-two-years.csv",
    Path(__file__).parent / "raw" / "compas" / "cox-violent-parsed.csv",
    Path(__file__).parent / "raw" / "compas" / "cox-violent-parsed_filt.csv",
]

# Features used in ProPublica's analysis (standard in fairness literature)
NUMERIC_FEATURES = [
    "age",
    "priors_count",
    "days_b_screening_arrest",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
]
CATEGORICAL_FEATURES = ["c_charge_degree"]  # F or M (felony/misdemeanor)
PROTECTED_ATTRS = ["race", "sex"]
TARGET = "two_year_recid"

# For binary race split (most studied in COMPAS literature)
BINARY_RACE_GROUPS = {"African-American": 0, "Caucasian": 1}


def download_compas(cache: bool = True) -> pd.DataFrame:
    """Download raw COMPAS CSV, optionally cache locally."""
    for path in LOCAL_CANDIDATES:
        if path.exists():
            print(f"[data] Loading local COMPAS from {path}")
            return pd.read_csv(path)
    print(f"[data] Fetching COMPAS from ProPublica GitHub...")
    df = pd.read_csv(COMPAS_URL)
    if cache:
        df.to_csv(CACHE_PATH, index=False)
        print(f"[data] Cached to {CACHE_PATH}")
    return df


def preprocess_compas(
    df: pd.DataFrame,
    binary_race: bool = True,
    clip_days: float = 30.0,
) -> pd.DataFrame:
    """
    Apply standard COMPAS preprocessing following ProPublica methodology.

    Args:
        df: Raw DataFrame from download_compas()
        binary_race: If True, keep only African-American and Caucasian rows
                     and encode race as 0/1. If False, keep all races encoded
                     as integer codes.
        clip_days: Clip days_b_screening_arrest to [-clip_days, clip_days]
                   (removes clearly erroneous entries).

    Returns:
        Cleaned DataFrame with columns:
          age, priors_count, days_b_screening_arrest,
          juv_fel_count, juv_misd_count, juv_other_count,
          c_charge_degree,          # 0=F, 1=M
          race,                      # 0=African-American, 1=Caucasian (if binary_race)
          sex,                       # 0=Female, 1=Male
          two_year_recid             # target: 0/1
    """
    # --- Standard ProPublica filter ---
    df = df[df["days_b_screening_arrest"] <= 30].copy()
    df = df[df["days_b_screening_arrest"] >= -30].copy()
    if TARGET not in df.columns and "is_recid" in df.columns:
        df[TARGET] = df["is_recid"]

    df = df[df["is_recid"] != -1].copy()
    df = df[df["c_charge_degree"] != "O"].copy()
    df = df[df["score_text"] != "N/A"].copy()

    if binary_race:
        df = df[df["race"].isin(BINARY_RACE_GROUPS.keys())].copy()
        df["race"] = df["race"].map(BINARY_RACE_GROUPS)
    else:
        df["race"] = LabelEncoder().fit_transform(df["race"])

    # Encode sex: Female=0, Male=1
    df["sex"] = (df["sex"] == "Male").astype(int)

    # Encode charge degree: Felony=0, Misdemeanor=1
    df["c_charge_degree"] = (df["c_charge_degree"] == "M").astype(int)

    # Clip days
    df["days_b_screening_arrest"] = df["days_b_screening_arrest"].clip(
        -clip_days, clip_days
    )

    keep_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + PROTECTED_ATTRS + [TARGET]
    df = df[keep_cols].dropna().reset_index(drop=True)

    print(
        f"[data] Preprocessed COMPAS: {len(df)} rows | "
        f"recidivism rate: {df[TARGET].mean():.3f} | "
        f"race=0 (AA): {(df['race']==0).sum()} | "
        f"race=1 (Cau): {(df['race']==1).sum()}"
    )
    return df


def get_features_and_labels(df: pd.DataFrame):
    """
    Split DataFrame into X (features), y (target), and group vectors.

    Returns:
        X       : np.ndarray (n_samples, n_features), float32
        y       : np.ndarray (n_samples,), int
        groups  : dict {"race": np.ndarray, "sex": np.ndarray}
        feat_names : list[str]
    """
    feat_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feat_names].values.astype(np.float32)
    y = df[TARGET].values.astype(int)
    groups = {
        "race": df["race"].values.astype(int),
        "sex": df["sex"].values.astype(int),
    }
    return X, y, groups, feat_names


def load_compas_splits(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scale: bool = True,
    binary_race: bool = True,
):
    """
    Full pipeline: download → preprocess → split → (optionally) scale.

    Returns a dict with keys:
        X_train, X_val, X_test      : np.ndarray
        y_train, y_val, y_test      : np.ndarray
        groups_train, groups_val, groups_test  : dict of np.ndarray
        scaler                      : fitted StandardScaler (or None)
        feature_names               : list[str]
        df_full                     : full preprocessed DataFrame
    """
    raw = download_compas()
    df = preprocess_compas(raw, binary_race=binary_race)
    X, y, groups, feat_names = get_features_and_labels(df)

    # Stratify on (y, race) jointly to preserve group balance
    strat_key = y * 2 + groups["race"]

    X_temp, X_test, y_temp, y_test, g_temp_race, g_test_race, g_temp_sex, g_test_sex = (
        train_test_split(
            X, y, groups["race"], groups["sex"],
            test_size=test_size,
            stratify=strat_key,
            random_state=random_state,
        )
    )

    strat_val = y_temp * 2 + g_temp_race
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, g_train_race, g_val_race, g_train_sex, g_val_sex = (
        train_test_split(
            X_temp, y_temp, g_temp_race, g_temp_sex,
            test_size=val_frac,
            stratify=strat_val,
            random_state=random_state,
        )
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    print(
        f"[data] Split sizes — train: {len(X_train)} | "
        f"val: {len(X_val)} | test: {len(X_test)}"
    )

    return {
        "X_train": X_train.astype(np.float32),
        "X_val": X_val.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "groups_train": {"race": g_train_race, "sex": g_train_sex},
        "groups_val": {"race": g_val_race, "sex": g_val_sex},
        "groups_test": {"race": g_test_race, "sex": g_test_sex},
        "scaler": scaler,
        "feature_names": feat_names,
        "df_full": df,
    }


if __name__ == "__main__":
    splits = load_compas_splits()
    print("Feature names:", splits["feature_names"])
    print("X_train shape:", splits["X_train"].shape)
    print("Race distribution in test set:")
    for v, label in [(0, "African-American"), (1, "Caucasian")]:
        mask = splits["groups_test"]["race"] == v
        print(f"  {label}: {mask.sum()} samples, recid rate={splits['y_test'][mask].mean():.3f}")
