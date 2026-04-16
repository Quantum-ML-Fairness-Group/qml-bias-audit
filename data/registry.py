"""
data/registry.py

Dataset loaders for experiments. Each loader must return a **split dict** with the
following keys (numpy arrays unless noted):

    X_train, X_val, X_test : np.ndarray, float32, shape (n, n_features)
    y_train, y_val, y_test : np.ndarray, int, shape (n,)
    groups_train, groups_val, groups_test : dict[str, np.ndarray]
        Each value is shape (n,) with integer group codes per protected attribute
        (e.g. {"race": ..., "sex": ...}).
    scaler : sklearn StandardScaler or None
    feature_names : list[str]
    df_full : optional pandas DataFrame of the preprocessed tabular data

Fairness evaluation uses ``groups_*`` together with predictions; protected
attributes must not be duplicated inside X unless you intentionally model them.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from data.load_compas import load_compas_splits

# name -> callable with same kwargs signature as load_compas_splits where possible
DATASETS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "compas": load_compas_splits,
}


def get_dataset_loader(name: str) -> Callable[..., Dict[str, Any]]:
    key = name.lower()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Registered: {list(DATASETS.keys())}")
    return DATASETS[key]


def list_datasets() -> list[str]:
    return list(DATASETS.keys())
