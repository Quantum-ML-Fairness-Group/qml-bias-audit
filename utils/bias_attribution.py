"""
utils/bias_attribution.py

Shapley-based bias attribution for classical models, and
circuit-parameter sensitivity analysis for quantum models.

For classical models:
  - SHAP TreeExplainer / KernelExplainer
  - Group-conditional SHAP: compare feature importance between privileged
    and unprivileged subgroups to identify disparate feature reliance

For quantum models:
  - Parameter gradient sensitivity (∂output/∂θ) as proxy for importance
  - Encoding sensitivity: measure how output changes with each feature

Reference: Lundberg & Lee (2017). "A unified approach to interpreting model predictions."
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable


# ─── Classical SHAP Attribution ──────────────────────────────────────────────

def compute_shap_values(model, X: np.ndarray, feature_names: list, model_type: str = "tree"):
    """
    Compute SHAP values for a fitted classical model.

    Args:
        model       : Fitted sklearn model
        X           : Input array, shape (n, d)
        feature_names: Feature names for display
        model_type  : "tree" (RF), "linear" (LR), or "kernel" (MLP/generic)

    Returns:
        shap_values: np.ndarray shape (n, d)
        explainer  : fitted SHAP explainer object
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap: pip install shap")

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        # For binary classifiers, sv is a list [class0, class1]; take class1
        if isinstance(sv, list):
            sv = sv[1]
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X)
        sv = explainer.shap_values(X)
    else:
        background = shap.kmeans(X, 50)
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1], background
        )
        sv = explainer.shap_values(X, nsamples=100)

    return sv, explainer


def group_shap_comparison(
    shap_values: np.ndarray,
    groups: np.ndarray,
    feature_names: list,
    group_labels: Dict[int, str] = None,
) -> pd.DataFrame:
    """
    Compare mean absolute SHAP values between privileged and unprivileged groups.

    Returns DataFrame with columns: feature, mean_abs_shap_priv, mean_abs_shap_unpriv,
    shap_diff, shap_ratio
    """
    if group_labels is None:
        group_labels = {0: "Unprivileged", 1: "Privileged"}

    unique_groups = sorted(np.unique(groups))
    group_shap = {}
    for g in unique_groups:
        mask = groups == g
        group_shap[g] = np.abs(shap_values[mask]).mean(axis=0)

    rows = []
    for i, feat in enumerate(feature_names):
        row = {"feature": feat}
        for g in unique_groups:
            row[f"mean_abs_shap_{group_labels[g]}"] = group_shap[g][i]
        if len(unique_groups) == 2:
            g0, g1 = unique_groups
            diff = group_shap[g1][i] - group_shap[g0][i]
            ratio = group_shap[g1][i] / (group_shap[g0][i] + 1e-10)
            row["shap_diff"] = diff
            row["shap_ratio"] = ratio
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("shap_diff", ascending=False)
    return df


# ─── Quantum Sensitivity Attribution ─────────────────────────────────────────

def quantum_encoding_sensitivity(
    model,
    X: np.ndarray,
    feature_names: list,
    epsilon: float = 0.01,
    n_samples: int = 100,
) -> pd.DataFrame:
    """
    Estimate input feature sensitivity for a quantum VQC via finite differences.

    For each feature j, compute:
        S_j = mean_i |f(x_i + ε·e_j) - f(x_i - ε·e_j)| / (2ε)

    This is the quantum analog of gradient-based feature importance.
    Run separately for each group to identify disparate sensitivity.

    Args:
        model        : Fitted VQC (must have predict_proba method)
        X            : Input data (n_samples, n_features)
        feature_names: Names for display
        epsilon      : Finite difference step
        n_samples    : How many samples to average over (use subset for speed)

    Returns:
        DataFrame with feature sensitivities
    """
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_sub = X[idx]

    n_features = X_sub.shape[1]
    sensitivities = np.zeros(n_features)

    for j in range(n_features):
        diffs = []
        for x in X_sub:
            x_plus = x.copy(); x_plus[j] += epsilon
            x_minus = x.copy(); x_minus[j] -= epsilon
            f_plus = model.predict_proba(x_plus[None, :])[0, 1]
            f_minus = model.predict_proba(x_minus[None, :])[0, 1]
            diffs.append(abs(f_plus - f_minus) / (2 * epsilon))
        sensitivities[j] = np.mean(diffs)
        print(f"  Sensitivity [{feature_names[j]}]: {sensitivities[j]:.4f}")

    return pd.DataFrame({
        "feature": feature_names,
        "sensitivity": sensitivities,
    }).sort_values("sensitivity", ascending=False)


def quantum_group_sensitivity(
    model,
    X: np.ndarray,
    groups: np.ndarray,
    feature_names: list,
    group_labels: Dict[int, str] = None,
    epsilon: float = 0.01,
    n_samples: int = 50,
) -> pd.DataFrame:
    """
    Compute encoding sensitivity separately per group.
    Disparities in sensitivity indicate that the circuit treats groups differently.
    """
    if group_labels is None:
        group_labels = {0: "Unprivileged", 1: "Privileged"}

    rows = []
    for g, label in group_labels.items():
        mask = groups == g
        X_g = X[mask]
        if len(X_g) == 0:
            continue
        print(f"\n[sensitivity] Group: {label} (n={len(X_g)})")
        df_g = quantum_encoding_sensitivity(model, X_g, feature_names,
                                             epsilon=epsilon, n_samples=n_samples)
        df_g["group"] = label
        rows.append(df_g)

    combined = pd.concat(rows, ignore_index=True)
    # Pivot for comparison
    pivot = combined.pivot(index="feature", columns="group", values="sensitivity")
    pivot["sensitivity_diff"] = pivot.get("Privileged", 0) - pivot.get("Unprivileged", 0)
    return pivot.reset_index().sort_values("sensitivity_diff", ascending=False)


# ─── Parameter Gradient Analysis ─────────────────────────────────────────────

def parameter_gradient_norm(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int = 50,
) -> Dict[str, float]:
    """
    Compute mean gradient norm of loss w.r.t. circuit parameters.
    Very small norms indicate barren plateaus — a known QML bias source.

    Returns:
        {"mean_grad_norm": float, "std_grad_norm": float, "barren_plateau": bool}
    """
    import pennylane as qml
    from pennylane import numpy as pnp

    if model.weights_ is None:
        raise RuntimeError("Model not fitted.")

    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), min(n_samples, len(X)), replace=False)
    X_sub, y_sub = X[idx], y[idx]

    weights = pnp.array(model.weights_, requires_grad=True)

    grad_norms = []
    for xi, yi in zip(X_sub, y_sub):
        def loss_fn(w):
            probs = (model._circuit(w, xi) + 1) / 2
            probs = pnp.clip(probs, 1e-7, 1 - 1e-7)
            return -(yi * pnp.log(probs) + (1 - yi) * pnp.log(1 - probs))

        grad = qml.grad(loss_fn)(weights)
        grad_norms.append(float(pnp.linalg.norm(grad)))

    mean_norm = np.mean(grad_norms)
    std_norm = np.std(grad_norms)

    return {
        "mean_grad_norm": mean_norm,
        "std_grad_norm": std_norm,
        "barren_plateau": mean_norm < 1e-3,
    }
