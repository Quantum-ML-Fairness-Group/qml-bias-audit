"""
experiments/run_classical.py

Train and evaluate all classical baseline models on COMPAS.
Saves per-model results to results/classical_results.json
and prints a summary comparison table.

Usage:
    python experiments/run_classical.py
    python experiments/run_classical.py --tune        # enable RF grid search
    python experiments/run_classical.py --threshold_search  # optimize decision threshold
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_compas import load_compas_splits
from models.classical.logistic_regression import build_logistic_regression
from models.classical.random_forest import build_random_forest
from models.classical.mlp import build_mlp
from utils.fairness_metrics import FairnessEvaluator, compare_models


RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate_model(name, model, splits, threshold_search=False):
    """Train, predict, and compute full fairness evaluation for a single model."""
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    groups_test = splits["groups_test"]

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    # Train
    if name == "MLP":
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        model.fit(X_train, y_train)

    # Get predictions
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = None

    # Threshold optimization on validation set
    if threshold_search and y_score is not None:
        y_score_val = model.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.3, 0.7, 41)
        from sklearn.metrics import f1_score as sk_f1
        best_t = max(thresholds, key=lambda t: sk_f1(y_val, (y_score_val >= t).astype(int)))
        print(f"  Optimal threshold (val F1): {best_t:.3f}")
    else:
        best_t = 0.5

    y_pred = (y_score >= best_t).astype(int) if y_score is not None else model.predict(X_test)

    # Fairness evaluation
    evaluator = FairnessEvaluator(y_test, y_pred, groups_test, y_score=y_score)
    df = evaluator.to_dataframe()

    # Print per-attribute results
    for _, row in df.iterrows():
        attr = row["attribute"]
        print(f"\n  [{attr.upper()}]")
        print(f"    Accuracy:              {row['overall_accuracy']:.4f}")
        print(f"    F1 Score:              {row['overall_f1']:.4f}")
        print(f"    ROC-AUC:               {row['overall_roc_auc']:.4f}")
        print(f"    Demographic Parity Δ:  {row['demographic_parity_diff']:.4f}  {'✓' if row['fair_dp'] else '✗'}")
        print(f"    Disparate Impact:      {row['disparate_impact']:.4f}  {'✓' if row['fair_di'] else '✗'}")
        print(f"    Equalized Odds Δ:      {row['equalized_odds_diff']:.4f}  {'✓' if row['fair_eo'] else '✗'}")
        print(f"    Equal Opportunity Δ:   {row['equal_opportunity_diff']:.4f}")
        print(f"    Predictive Parity Δ:   {row['predictive_parity_diff']:.4f}")

    return {
        "model": name,
        "threshold": best_t,
        "fairness_df": df.to_dict(orient="records"),
        "evaluator": evaluator,
    }


def main(args):
    print("Loading COMPAS dataset...")
    splits = load_compas_splits(random_state=42)

    n_features = splits["X_train"].shape[1]
    print(f"Input dimension: {n_features} features")

    models = {
        "Logistic Regression": build_logistic_regression(C=1.0, calibrate=True),
        "Random Forest": build_random_forest(
            tune=args.tune,
            X_train=splits["X_train"] if args.tune else None,
            y_train=splits["y_train"] if args.tune else None,
        ),
        "MLP": build_mlp(
            input_dim=n_features,
            hidden=(128, 64, 32),
            dropout=0.3,
            lr=1e-3,
            epochs=100,
            patience=15,
        ),
    }

    all_results = {}
    for name, model in models.items():
        result = evaluate_model(name, model, splits, threshold_search=args.threshold_search)
        all_results[name] = result

    # Summary comparison table
    print("\n" + "=" * 70)
    print("SUMMARY: Classical Models on COMPAS (Race Attribute)")
    print("=" * 70)
    comp_df = compare_models(all_results, attribute="race")
    print(comp_df.round(4).to_string())

    # Save results
    save_data = {
        name: {
            "threshold": res["threshold"],
            "fairness": res["fairness_df"],
        }
        for name, res in all_results.items()
    }
    out_path = RESULTS_DIR / "classical_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[✓] Results saved to {out_path}")

    # Also save comparison CSV
    comp_df.reset_index().to_csv(RESULTS_DIR / "classical_comparison.csv", index=False)
    print(f"[✓] Comparison table saved to {RESULTS_DIR / 'classical_comparison.csv'}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Enable RF grid search")
    parser.add_argument("--threshold_search", action="store_true",
                        help="Optimize classification threshold on val set")
    args = parser.parse_args()
    main(args)
