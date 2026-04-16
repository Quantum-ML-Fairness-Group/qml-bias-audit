"""
experiments/run_sweep.py

Systematic sweeps over dataset × quantum encoding × ansatz × depth (and optional subsampling).

Usage:
    python experiments/run_sweep.py --datasets compas --encodings angle,amplitude,iqp \\
        --ansatze strongly_entangling,basic_entangler --depths 1,2,3 --n_epochs 20 --subsample 400

Results are written under results/sweep/ as CSV and JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.registry import get_dataset_loader, list_datasets
from models.quantum.ansatze import list_ansatze
from models.quantum.configurable_vqc import ConfigurableVQC
from models.quantum.encodings import list_encodings
from utils.fairness_metrics import FairnessEvaluator


RESULTS_DIR = ROOT / "results" / "sweep"


def subsample_splits(splits: Dict[str, Any], n: int, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    n_train = len(splits["X_train"])
    idx = rng.choice(n_train, size=min(n, n_train), replace=False)
    out = dict(splits)
    out["X_train"] = splits["X_train"][idx]
    out["y_train"] = splits["y_train"][idx]
    out["groups_train"] = {k: v[idx] for k, v in splits["groups_train"].items()}
    return out


def run_one(
    splits: Dict[str, Any],
    dataset_name: str,
    encoding: str,
    ansatz: str,
    n_layers: int,
    n_epochs: int,
    random_state: int,
) -> Dict[str, Any]:
    n_features = splits["X_train"].shape[1]
    measurement = "zz_01" if encoding == "iqp" else "z0"
    lr = 0.015 if encoding == "iqp" else 0.02

    model = ConfigurableVQC(
        encoding=encoding,
        ansatz=ansatz,
        measurement=measurement,
        n_layers=n_layers,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=32,
        random_state=random_state,
        user_n_qubits=n_features if encoding in ("angle", "iqp") else None,
        n_encoding_reps=2,
        n_features_expected=n_features,
        model_label=f"{dataset_name}-{encoding}-{ansatz}-L{n_layers}",
    )

    t0 = pd.Timestamp.now()
    model.fit(
        splits["X_train"],
        splits["y_train"],
        X_val=splits["X_val"],
        y_val=splits["y_val"],
    )
    train_time_s = (pd.Timestamp.now() - t0).total_seconds()

    y_score = model.predict_proba(splits["X_test"])[:, 1]
    y_pred = model.predict(splits["X_test"])

    evaluator = FairnessEvaluator(
        splits["y_test"],
        y_pred,
        splits["groups_test"],
        y_score=y_score,
    )
    rep_race = evaluator.evaluate_attribute("race")

    return {
        "dataset": dataset_name,
        "encoding": encoding,
        "ansatz": ansatz,
        "n_layers": n_layers,
        "n_epochs": n_epochs,
        "n_qubits": model.encoding_spec_.n_qubits if model.encoding_spec_ else None,
        "n_parameters": model.parameter_count(),
        "train_time_s": train_time_s,
        "accuracy": rep_race.overall_accuracy,
        "f1": rep_race.overall_f1,
        "roc_auc": rep_race.overall_roc_auc,
        "DPD": rep_race.demographic_parity_diff,
        "DI": rep_race.disparate_impact,
        "EOD": rep_race.equalized_odds_diff,
        "EOpp": rep_race.equal_opportunity_diff,
        "PPD": rep_race.predictive_parity_diff,
    }


def parse_list(arg: str) -> List[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


def parse_int_list(arg: str) -> List[int]:
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Sweep encodings × ansätze × depth on registered datasets.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="compas",
        help=f"Comma-separated names. Available: {list_datasets()}",
    )
    parser.add_argument(
        "--encodings",
        type=str,
        default="angle,amplitude,iqp",
        help=f"Comma-separated. Available: {list_encodings()}",
    )
    parser.add_argument(
        "--ansatze",
        type=str,
        default="strongly_entangling,basic_entangler",
        help=f"Comma-separated. Available: {list_ansatze()}",
    )
    parser.add_argument("--depths", type=str, default="1,2,3", help="Comma-separated n_layers values.")
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--subsample", type=int, default=None, help="Limit training set size for speed.")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--seed", type=int, default=None, help="Alias for random_state.")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else args.random_state

    datasets = parse_list(args.datasets)
    encodings = parse_list(args.encodings)
    ansatze = parse_list(args.ansatze)
    depths = parse_int_list(args.depths)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for ds_name in datasets:
        loader = get_dataset_loader(ds_name)
        splits = loader(random_state=seed)
        if args.subsample:
            splits = subsample_splits(splits, args.subsample, seed=seed)
            print(f"[sweep] {ds_name}: subsampled train to {len(splits['X_train'])}")

        for encoding, ansatz, n_layers in product(encodings, ansatze, depths):
            key = f"{ds_name}_{encoding}_{ansatz}_L{n_layers}"
            print(f"\n[sweep] Running {key} ...")
            try:
                row = run_one(
                    splits,
                    ds_name,
                    encoding,
                    ansatz,
                    n_layers,
                    args.n_epochs,
                    seed,
                )
                rows.append(row)
                print(
                    f"  acc={row['accuracy']:.4f}  DPD={row['DPD']:.4f}  "
                    f"time={row['train_time_s']:.1f}s"
                )
            except Exception as e:
                print(f"  [ERROR] {e}")
                rows.append(
                    {
                        "dataset": ds_name,
                        "encoding": encoding,
                        "ansatz": ansatz,
                        "n_layers": n_layers,
                        "error": str(e),
                    }
                )

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "sweep_results.csv"
    json_path = RESULTS_DIR / "sweep_results.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[sweep] Wrote {csv_path}")
    print(f"[sweep] Wrote {json_path}")


if __name__ == "__main__":
    main()
