"""
utils/fairness_metrics.py

Comprehensive fairness metric suite for binary classification.
All metrics follow the Hardt et al. (2016) and Feldman et al. (2015) definitions.

Usage:
    from utils.fairness_metrics import FairnessEvaluator
    evaluator = FairnessEvaluator(y_true, y_pred, groups)
    report = evaluator.full_report()
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, brier_score_loss,
)


@dataclass
class GroupMetrics:
    """Per-group performance and fairness statistics."""
    group_name: str
    group_value: int
    n_samples: int
    prevalence: float          # P(y=1 | group)
    accuracy: float
    tpr: float                 # True Positive Rate (recall)
    fpr: float                 # False Positive Rate
    ppv: float                 # Positive Predictive Value (precision)
    f1: float
    selection_rate: float      # P(ŷ=1 | group)
    roc_auc: float


@dataclass
class FairnessReport:
    """Full fairness evaluation for a model on one protected attribute."""
    attribute: str
    privileged_value: int
    unprivileged_value: int

    # Per-group metrics
    privileged: GroupMetrics
    unprivileged: GroupMetrics

    # Group fairness metrics
    demographic_parity_diff: float      # |SR_priv - SR_unpriv|
    disparate_impact: float             # SR_unpriv / SR_priv
    equalized_odds_diff: float          # max(|TPR_diff|, |FPR_diff|)
    equal_opportunity_diff: float       # |TPR_priv - TPR_unpriv|
    predictive_parity_diff: float       # |PPV_priv - PPV_unpriv|
    accuracy_diff: float                # |Acc_priv - Acc_unpriv|

    # Overall (pooled) metrics
    overall_accuracy: float
    overall_f1: float
    overall_roc_auc: float

    # Interpretations
    def __post_init__(self):
        self.is_fair_dp = abs(self.demographic_parity_diff) < 0.1
        self.is_fair_eo = abs(self.equalized_odds_diff) < 0.1
        self.is_fair_di = 0.8 <= self.disparate_impact <= 1.25


def _safe_divide(num: float, denom: float, fallback: float = 0.0) -> float:
    return num / denom if denom > 0 else fallback


def _group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    group_name: str,
    group_value: int,
) -> GroupMetrics:
    """Compute all per-group metrics."""
    n = len(y_true)
    if n == 0:
        raise ValueError(f"No samples for group {group_name}={group_value}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    tpr = _safe_divide(tp, tp + fn)
    fpr = _safe_divide(fp, fp + tn)
    ppv = _safe_divide(tp, tp + fp)
    selection_rate = y_pred.mean()

    if y_score is not None and len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = float("nan")

    return GroupMetrics(
        group_name=group_name,
        group_value=group_value,
        n_samples=n,
        prevalence=y_true.mean(),
        accuracy=accuracy_score(y_true, y_pred),
        tpr=tpr,
        fpr=fpr,
        ppv=ppv,
        f1=f1_score(y_true, y_pred, zero_division=0),
        selection_rate=selection_rate,
        roc_auc=auc,
    )


class FairnessEvaluator:
    """
    Compute group fairness metrics for a binary classifier.

    Args:
        y_true    : Ground truth labels (0/1), shape (n,)
        y_pred    : Binary predictions (0/1), shape (n,)
        groups    : Dict mapping attribute name → array of group memberships
        y_score   : Optional continuous prediction scores (for AUC), shape (n,)
        privileged: Dict mapping attribute name → privileged group value
                    Defaults to 1 for all attributes.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Dict[str, np.ndarray],
        y_score: Optional[np.ndarray] = None,
        privileged: Optional[Dict[str, int]] = None,
    ):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.groups = {k: np.asarray(v) for k, v in groups.items()}
        self.y_score = np.asarray(y_score) if y_score is not None else None
        self.privileged = privileged or {k: 1 for k in groups}

    def evaluate_attribute(self, attr: str) -> FairnessReport:
        """Compute full fairness report for a single protected attribute."""
        group_arr = self.groups[attr]
        priv_val = self.privileged[attr]
        unpriv_val = 1 - priv_val  # works for binary groups

        priv_mask = group_arr == priv_val
        unpriv_mask = group_arr == unpriv_val

        def _score_subset(mask):
            return self.y_score[mask] if self.y_score is not None else None

        priv_metrics = _group_metrics(
            self.y_true[priv_mask], self.y_pred[priv_mask],
            _score_subset(priv_mask), attr, priv_val,
        )
        unpriv_metrics = _group_metrics(
            self.y_true[unpriv_mask], self.y_pred[unpriv_mask],
            _score_subset(unpriv_mask), attr, unpriv_val,
        )

        sr_priv = priv_metrics.selection_rate
        sr_unpriv = unpriv_metrics.selection_rate

        overall_auc = float("nan")
        if self.y_score is not None and len(np.unique(self.y_true)) > 1:
            overall_auc = roc_auc_score(self.y_true, self.y_score)

        return FairnessReport(
            attribute=attr,
            privileged_value=priv_val,
            unprivileged_value=unpriv_val,
            privileged=priv_metrics,
            unprivileged=unpriv_metrics,
            demographic_parity_diff=abs(sr_priv - sr_unpriv),
            disparate_impact=_safe_divide(sr_unpriv, sr_priv, fallback=0.0),
            equalized_odds_diff=max(
                abs(priv_metrics.tpr - unpriv_metrics.tpr),
                abs(priv_metrics.fpr - unpriv_metrics.fpr),
            ),
            equal_opportunity_diff=abs(priv_metrics.tpr - unpriv_metrics.tpr),
            predictive_parity_diff=abs(priv_metrics.ppv - unpriv_metrics.ppv),
            accuracy_diff=abs(priv_metrics.accuracy - unpriv_metrics.accuracy),
            overall_accuracy=accuracy_score(self.y_true, self.y_pred),
            overall_f1=f1_score(self.y_true, self.y_pred, zero_division=0),
            overall_roc_auc=overall_auc,
        )

    def full_report(self) -> Dict[str, FairnessReport]:
        """Evaluate all provided protected attributes."""
        return {attr: self.evaluate_attribute(attr) for attr in self.groups}

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten full report into a tidy DataFrame for easy comparison."""
        rows = []
        for attr, report in self.full_report().items():
            rows.append({
                "attribute": attr,
                "overall_accuracy": report.overall_accuracy,
                "overall_f1": report.overall_f1,
                "overall_roc_auc": report.overall_roc_auc,
                "demographic_parity_diff": report.demographic_parity_diff,
                "disparate_impact": report.disparate_impact,
                "equalized_odds_diff": report.equalized_odds_diff,
                "equal_opportunity_diff": report.equal_opportunity_diff,
                "predictive_parity_diff": report.predictive_parity_diff,
                "accuracy_diff": report.accuracy_diff,
                "fair_dp": report.is_fair_dp,
                "fair_eo": report.is_fair_eo,
                "fair_di": report.is_fair_di,
                # Per-group
                f"acc_{attr}_priv": report.privileged.accuracy,
                f"acc_{attr}_unpriv": report.unprivileged.accuracy,
                f"tpr_{attr}_priv": report.privileged.tpr,
                f"tpr_{attr}_unpriv": report.unprivileged.tpr,
                f"fpr_{attr}_priv": report.privileged.fpr,
                f"fpr_{attr}_unpriv": report.unprivileged.fpr,
                f"sr_{attr}_priv": report.privileged.selection_rate,
                f"sr_{attr}_unpriv": report.unprivileged.selection_rate,
            })
        return pd.DataFrame(rows)


def compare_models(
    model_results: Dict[str, Dict],
    attribute: str = "race",
) -> pd.DataFrame:
    """
    Build a comparison DataFrame from multiple model results.

    Args:
        model_results: {model_name: {"evaluator": FairnessEvaluator}} or
                       {model_name: FairnessReport}
    """
    rows = []
    for model_name, obj in model_results.items():
        if isinstance(obj, dict) and "evaluator" in obj:
            report = obj["evaluator"].evaluate_attribute(attribute)
        elif isinstance(obj, FairnessReport):
            report = obj
        else:
            raise ValueError(f"Unexpected type for {model_name}: {type(obj)}")

        rows.append({
            "model": model_name,
            "accuracy": report.overall_accuracy,
            "f1": report.overall_f1,
            "roc_auc": report.overall_roc_auc,
            "DPD": report.demographic_parity_diff,
            "DI": report.disparate_impact,
            "EOD": report.equalized_odds_diff,
            "EOpp": report.equal_opportunity_diff,
            "PPD": report.predictive_parity_diff,
        })
    return pd.DataFrame(rows).set_index("model")
