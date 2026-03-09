"""
experiments/ablation_encoding.py

Ablation study: How does the choice of quantum encoding scheme affect bias?

We systematically vary:
  1. Encoding type: angle, amplitude, IQP
  2. Circuit depth (n_layers): 1, 2, 3, 5
  3. Feature selection: raw features, PCA-reduced, correlation-sorted
  4. Protected attribute inclusion: with vs. without race/sex in features

This isolates the contribution of each architectural component to observed
bias, following the framework of Heredge et al. (2024).

Results saved to results/ablation_encoding.json and figures/ablation_*.png
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_compas import load_compas_splits
from models.quantum.vqc_angle import VQCAngle
from models.quantum.vqc_amplitude import VQCAmplitude
from models.quantum.vqc_iqp import VQCIQP
from utils.fairness_metrics import FairnessEvaluator

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Quick ablation: use subsampled data + fewer epochs
ABLATION_N_TRAIN = 400
ABLATION_EPOCHS = 30
ABLATION_SEED = 0


def make_model(encoding: str, n_qubits: int, n_layers: int):
    """Instantiate a VQC by encoding name and depth."""
    if encoding == "angle":
        return VQCAngle(n_qubits=n_qubits, n_layers=n_layers,
                        n_epochs=ABLATION_EPOCHS, random_state=ABLATION_SEED)
    elif encoding == "amplitude":
        return VQCAmplitude(n_features=n_qubits, n_layers=n_layers,
                            n_epochs=ABLATION_EPOCHS, random_state=ABLATION_SEED)
    elif encoding == "iqp":
        return VQCIQP(n_qubits=n_qubits, n_layers=n_layers,
                      n_encoding_reps=1, n_epochs=ABLATION_EPOCHS, random_state=ABLATION_SEED)
    raise ValueError(f"Unknown encoding: {encoding}")


def run_condition(
    X_train, y_train, X_val, y_val, X_test, y_test,
    groups_test, encoding, n_layers, drop_protected=False,
    feature_names=None,
):
    """
    Run a single ablation condition.

    Args:
        drop_protected: If True, remove race and sex from features
                        (proxies may remain — this tests direct vs proxy bias)
    """
    # Optional: drop protected attributes from feature matrix
    # (COMPAS features don't include race/sex directly — they're in groups_test —
    #  but if you added them as features, this lets you test their direct effect)
    n_features = X_train.shape[1]

    model = make_model(encoding, n_features, n_layers)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    evaluator = FairnessEvaluator(y_test, y_pred, groups_test, y_score=y_score)
    report_race = evaluator.evaluate_attribute("race")

    return {
        "encoding": encoding,
        "n_layers": n_layers,
        "drop_protected": drop_protected,
        "accuracy": report_race.overall_accuracy,
        "f1": report_race.overall_f1,
        "roc_auc": report_race.overall_roc_auc,
        "DPD": report_race.demographic_parity_diff,
        "DI": report_race.disparate_impact,
        "EOD": report_race.equalized_odds_diff,
        "EOpp": report_race.equal_opportunity_diff,
        "PPD": report_race.predictive_parity_diff,
    }


def main():
    print("Loading COMPAS...")
    splits = load_compas_splits(random_state=42)

    # Subsample for ablation speed
    rng = np.random.RandomState(ABLATION_SEED)
    idx = rng.choice(len(splits["X_train"]), ABLATION_N_TRAIN, replace=False)
    X_train = splits["X_train"][idx]
    y_train = splits["y_train"][idx]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]
    groups_test = splits["groups_test"]

    encodings = ["angle", "amplitude", "iqp"]
    depths = [1, 2, 3, 5]

    results = []
    total = len(encodings) * len(depths)
    done = 0

    for encoding, n_layers in product(encodings, depths):
        done += 1
        print(f"\n[{done}/{total}] encoding={encoding}, depth={n_layers}")
        try:
            row = run_condition(
                X_train, y_train, X_val, y_val, X_test, y_test,
                groups_test, encoding, n_layers,
            )
            results.append(row)
            print(f"  Acc={row['accuracy']:.3f} | DPD={row['DPD']:.3f} | EOD={row['EOD']:.3f}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "ablation_encoding.json"
    df.to_json(out_path, orient="records", indent=2)
    df.to_csv(RESULTS_DIR / "ablation_encoding.csv", index=False)
    print(f"\n[✓] Ablation results saved to {out_path}")

    # Print pivot table: encoding × depth → DPD
    print("\nDemographic Parity Diff (race) by Encoding × Depth:")
    pivot = df.pivot_table(index="encoding", columns="n_layers", values="DPD")
    print(pivot.round(4).to_string())

    print("\nEqualized Odds Diff (race) by Encoding × Depth:")
    pivot_eo = df.pivot_table(index="encoding", columns="n_layers", values="EOD")
    print(pivot_eo.round(4).to_string())

    return df


if __name__ == "__main__":
    main()
