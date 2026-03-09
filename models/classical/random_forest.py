"""
models/classical/random_forest.py
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def build_random_forest(tune: bool = False, X_train=None, y_train=None):
    """
    Random Forest with optional GridSearch hyperparameter tuning.

    Args:
        tune: If True, runs 5-fold CV grid search over key hyperparams.
              Requires X_train and y_train to be provided.
    """
    if tune and X_train is not None and y_train is not None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 5],
            "max_features": ["sqrt", "log2"],
        }
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid = GridSearchCV(rf, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        print(f"[RF] Best params: {grid.best_params_} | CV AUC: {grid.best_score_:.4f}")
        return grid.best_estimator_

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
