"""
models/classical/logistic_regression.py
models/classical/random_forest.py
models/classical/mlp.py

Classical baseline models — all expose a unified sklearn-compatible interface
plus a .predict_proba() method for AUC computation.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

def build_logistic_regression(C: float = 1.0, calibrate: bool = True):
    """
    L2-regularized logistic regression.
    Calibrated with isotonic regression for better-quality probability estimates.
    """
    base = LogisticRegression(
        C=C,
        l1_ratio=0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    if calibrate:
        return CalibratedClassifierCV(base, method="isotonic", cv=5)
    return base
