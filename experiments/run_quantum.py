"""
experiments/run_quantum.py

Train and evaluate all three QML models on COMPAS.
By default runs on CPU simulation (default.qubit).
Swap to qiskit.aer for noise-model experiments.

Usage:
    python experiments/run_quantum.py                    # all 3 VQCs
    python experiments/run_quantum.py --model angle      # single model
    python experiments/run_quantum.py --n_epochs 30      # quick run
    python experiments/run_quantum.py --subsample 300    # small data for testing
    python experiments/run_quantum.py --noise fixed --noise_strength 0.01
    python experiments/run_quantum.py --noise random --noise_strength 0.05

Runtime note:
  Full run on CPU: ~30-60 min total
  With --subsample 300 --n_epochs 20: ~5-10 min
"""

import sys
import json
import argparse
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_compas import load_compas_splits
from models.quantum.vqc_angle import VQCAngle
from models.quantum.vqc_amplitude import VQCAmplitude
from models.quantum.vqc_iqp import VQCIQP
from utils.fairness_metrics import FairnessEvaluator, compare_models

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def subsample_splits(splits, n: int, seed: int = 42):
    """Downsample training set for quick experimentation."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(splits["X_train"]), size=min(n, len(splits["X_train"])), replace=False)
    out = dict(splits)
    out["X_train"] = splits["X_train"][idx]
    out["y_train"] = splits["y_train"][idx]
    out["groups_train"] = {k: v[idx] for k, v in splits["groups_train"].items()}
    return out


def build_quantum_models(n_features: int, n_epochs: int, noise_type=None, noise_strength=0.0):
    """Instantiate all QML models with matched hyperparameters."""
    return {
        "VQC-Angle": VQCAngle(
            n_qubits=n_features,
            n_layers=3,
            n_epochs=n_epochs,
            lr=0.02,
            batch_size=32,
            random_state=42,
            noise_type=noise_type,
            noise_strength=noise_strength,
        ),
        "VQC-Amplitude": VQCAmplitude(
            n_features=n_features,
            n_layers=3,
            n_epochs=n_epochs,
            lr=0.02,
            batch_size=32,
            random_state=42,
            noise_type=noise_type,
            noise_strength=noise_strength,
        ),
        "VQC-IQP": VQCIQP(
            n_qubits=n_features,
            n_layers=3,
            n_encoding_reps=2,
            n_epochs=n_epochs,
            lr=0.015,
            batch_size=32,
            random_state=42,
            noise_type=noise_type,
            noise_strength=noise_strength,
        ),
    }


def evaluate_quantum_model(name, model, splits):
    """Train and fully evaluate a single QML model."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"  Parameters: {model.parameter_count()}")
    print(f"{'='*60}")

    t0 = time.time()
    model.fit(
        splits["X_train"], splits["y_train"],
        X_val=splits["X_val"], y_val=splits["y_val"],
    )
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Predictions
    y_score = model.predict_proba(splits["X_test"])[:, 1]
    y_pred = model.predict(splits["X_test"])

    # Fairness evaluation
    evaluator = FairnessEvaluator(
        splits["y_test"], y_pred, splits["groups_test"], y_score=y_score
    )
    df = evaluator.to_dataframe()

    for _, row in df.iterrows():
        attr = row["attribute"]
        print(f"\n  [{attr.upper()}]")
        print(f"    Accuracy:              {row['overall_accuracy']:.4f}  ↑")
        print(f"    F1 Score:              {row['overall_f1']:.4f}  ↑")
        print(f"    ROC-AUC:               {row['overall_roc_auc']:.4f}  ↑")
        print(f"    Demographic Parity Δ:  {row['demographic_parity_diff']:.4f}  ↓  {'✓' if row['fair_dp'] else '✗'}")
        print(f"    Disparate Impact:      {row['disparate_impact']:.4f}  →1  {'✓' if row['fair_di'] else '✗'}")
        print(f"    Equalized Odds Δ:      {row['equalized_odds_diff']:.4f}  ↓  {'✓' if row['fair_eo'] else '✗'}")
        print(f"    Equal Opportunity Δ:   {row['equal_opportunity_diff']:.4f}  ↓")

    return {
        "model": name,
        "train_time_s": train_time,
        "loss_history": [float(x) for x in model.loss_history_],
        "val_loss_history": [float(x) for x in getattr(model, "val_loss_history_", [])],
        "fairness_df": df.to_dict(orient="records"),
        "evaluator": evaluator,
    }


def main(args):
    print("Loading COMPAS dataset...")
    splits = load_compas_splits(random_state=42)

    if args.subsample:
        splits = subsample_splits(splits, args.subsample)
        print(f"[!] Subsampled training set to {len(splits['X_train'])} samples")

    n_features = splits["X_train"].shape[1]
    print(f"Input dimension: {n_features} features")

    all_models = build_quantum_models(n_features, args.n_epochs, args.noise, args.noise_strength)

    # Filter if specific model requested
    if args.model:
        model_map = {
            "angle": "VQC-Angle",
            "amplitude": "VQC-Amplitude",
            "iqp": "VQC-IQP",
        }
        key = model_map.get(args.model.lower())
        if key:
            all_models = {key: all_models[key]}
        else:
            print(f"Unknown model: {args.model}. Choose from: angle, amplitude, iqp")
            sys.exit(1)

    all_results = {}
    for name, model in all_models.items():
        result = evaluate_quantum_model(name, model, splits)
        all_results[name] = result

    # Summary
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY: Quantum Models on COMPAS (Race Attribute)")
        print("=" * 70)
        comp_df = compare_models(all_results, attribute="race")
        print(comp_df.round(4).to_string())

    # Save
    save_data = {
        name: {
            "train_time_s": res["train_time_s"],
            "loss_history": res["loss_history"],
            "val_loss_history": res["val_loss_history"],
            "fairness": res["fairness_df"],
        }
        for name, res in all_results.items()
    }
    out_path = RESULTS_DIR / "quantum_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[✓] Results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Run single model: angle | amplitude | iqp")
    parser.add_argument("--n_epochs", type=int, default=60,
                        help="Training epochs (default 60; use 20 for quick test)")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample N training points for speed")
    parser.add_argument("--noise", type=str, default=None, choices=["fixed", "random"],
                        help="Noise type: fixed (constant depolarizing) or random (per-batch sampled)")
    parser.add_argument("--noise_strength", type=float, default=0.01,
                        help="Max depolarizing noise probability (default 0.0)")
    args = parser.parse_args()
    main(args)
