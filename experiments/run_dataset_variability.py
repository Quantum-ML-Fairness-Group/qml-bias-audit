"""
experiments/run_dataset_variability.py

Dataset ablation for the project assignment:
vary the dataset while keeping model architecture, encoding, noise setting, and
fairness metric fixed.

Default condition:
  - architecture: VQC
  - encoding: angle
  - noise: none, exact default.qubit simulator
  - fairness metric: demographic_parity_diff
  - protected attribute: auto (race when available, otherwise sex)

Results are saved to:
  results/dataset_variability.csv
  results/dataset_variability.json
  figures/dataset_variability_<metric>.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.registry import get_dataset_loader, list_datasets
from models.quantum.vqc_angle import VQCAngle
from utils.fairness_metrics import FairnessEvaluator


RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


METRIC_TO_COLUMN = {
    "demographic_parity_diff": "demographic_parity_diff",
    "equalized_odds_diff": "equalized_odds_diff",
    "equal_opportunity_diff": "equal_opportunity_diff",
    "predictive_parity_diff": "predictive_parity_diff",
    "accuracy_diff": "accuracy_diff",
}


def _sample_indices(
    n_total: int,
    n_keep: int | None,
    seed: int,
    groups: dict[str, np.ndarray] | None = None,
) -> np.ndarray | None:
    if n_keep is None or n_keep >= n_total:
        return None
    rng = np.random.RandomState(seed)
    if groups:
        primary = groups["race"] if "race" in groups else next(iter(groups.values()))
        required = []
        for value in np.unique(primary):
            candidates = np.flatnonzero(primary == value)
            if len(candidates) > 0:
                required.append(rng.choice(candidates))
        required = np.unique(required)
        if len(required) < n_keep:
            remaining = np.setdiff1d(np.arange(n_total), required, assume_unique=False)
            fill = rng.choice(remaining, size=n_keep - len(required), replace=False)
            return np.concatenate([required, fill])
        return required[:n_keep]
    return rng.choice(n_total, size=n_keep, replace=False)


def subsample_splits(
    splits: dict,
    *,
    n_train: int | None,
    n_val: int | None,
    n_test: int | None,
    seed: int,
) -> dict:
    """Subsample train/validation/test for faster quantum simulation."""
    out = dict(splits)
    for split_name, n_keep in [("train", n_train), ("val", n_val), ("test", n_test)]:
        idx = _sample_indices(
            len(splits[f"X_{split_name}"]),
            n_keep,
            seed,
            groups=splits[f"groups_{split_name}"],
        )
        if idx is None:
            continue
        out[f"X_{split_name}"] = splits[f"X_{split_name}"][idx]
        out[f"y_{split_name}"] = splits[f"y_{split_name}"][idx]
        out[f"groups_{split_name}"] = {
            k: v[idx] for k, v in splits[f"groups_{split_name}"].items()
        }
    return out


def pca_reduce_splits(splits: dict, n_components: int, seed: int) -> dict:
    """Apply the same PCA dimensionality reduction protocol to every dataset."""
    max_components = min(n_components, splits["X_train"].shape[1])
    pca = PCA(n_components=max_components, random_state=seed)

    out = dict(splits)
    out["X_train"] = pca.fit_transform(splits["X_train"]).astype(np.float32)
    out["X_val"] = pca.transform(splits["X_val"]).astype(np.float32)
    out["X_test"] = pca.transform(splits["X_test"]).astype(np.float32)
    out["feature_names"] = [f"pc{i + 1}" for i in range(max_components)]
    out["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
    return out


def resolve_attribute(groups_test: dict, requested: str) -> str:
    if requested != "auto":
        if requested not in groups_test:
            available = ", ".join(groups_test.keys())
            raise ValueError(
                f"Requested protected attribute {requested!r} is not available. "
                f"Available: {available}"
            )
        return requested
    if "race" in groups_test:
        return "race"
    if "sex" in groups_test:
        return "sex"
    return next(iter(groups_test.keys()))


def run_one_dataset(args, dataset: str) -> dict:
    loader = get_dataset_loader(dataset)
    splits = loader(random_state=args.seed)
    splits = subsample_splits(
        splits,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
    )
    splits = pca_reduce_splits(splits, args.n_qubits, args.seed)

    attribute = resolve_attribute(splits["groups_test"], args.attribute)

    model = VQCAngle(
        n_qubits=splits["X_train"].shape[1],
        n_layers=args.n_layers,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device="default.qubit",
        shots=None,
        random_state=args.seed,
    )

    print(
        f"\n[{dataset}] VQC-Angle | qubits={splits['X_train'].shape[1]} | "
        f"layers={args.n_layers} | epochs={args.n_epochs} | train={len(splits['X_train'])}"
    )
    t0 = time.time()
    model.fit(
        splits["X_train"],
        splits["y_train"],
        X_val=splits["X_val"],
        y_val=splits["y_val"],
    )
    train_time = time.time() - t0

    y_score = model.predict_proba(splits["X_test"])[:, 1]
    y_pred = model.predict(splits["X_test"])
    evaluator = FairnessEvaluator(
        splits["y_test"],
        y_pred,
        {attribute: splits["groups_test"][attribute]},
        y_score=y_score,
    )
    row = evaluator.to_dataframe().iloc[0].to_dict()
    row.update(
        {
            "dataset": dataset,
            "resolved_attribute": attribute,
            "model": "VQC-Angle",
            "encoding": "angle",
            "noise": "none",
            "n_qubits": splits["X_train"].shape[1],
            "n_layers": args.n_layers,
            "n_epochs": args.n_epochs,
            "n_train": len(splits["X_train"]),
            "n_val": len(splits["X_val"]),
            "n_test": len(splits["X_test"]),
            "train_time_s": train_time,
            "pca_explained_variance_sum": float(
                sum(splits["pca_explained_variance_ratio"])
            ),
        }
    )
    return row


def save_plot(df: pd.DataFrame, metric: str, attribute: str) -> Path:
    plot_df = df.sort_values("dataset")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(
        plot_df["dataset"],
        plot_df[metric],
        color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(plot_df)],
        edgecolor="white",
    )
    ax.axhline(0.1, color="gray", linestyle="--", linewidth=1, label="0.1 threshold")
    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (lower is fairer)")
    ax.set_xlabel("Dataset")
    title_attr = attribute if attribute != "auto" else "available protected attribute"
    ax.set_title(f"Dataset Variability: Fixed VQC-Angle on {title_attr}")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", labelrotation=25)

    for bar, acc, attr in zip(
        bars, plot_df["overall_accuracy"], plot_df["resolved_attribute"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{attr}\nacc={acc:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, max(0.25, float(plot_df[metric].max()) * 1.25))
    plt.tight_layout()
    out = FIGURES_DIR / f"dataset_variability_{metric}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main(args):
    if args.metric not in METRIC_TO_COLUMN:
        raise ValueError(f"Unknown metric {args.metric!r}. Choose from {list(METRIC_TO_COLUMN)}")

    rows = []
    for dataset in args.datasets:
        rows.append(run_one_dataset(args, dataset))

    df = pd.DataFrame(rows)
    metric_col = METRIC_TO_COLUMN[args.metric]

    csv_path = RESULTS_DIR / "dataset_variability.csv"
    json_path = RESULTS_DIR / "dataset_variability.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    fig_path = save_plot(df, metric_col, args.attribute)

    cols = [
        "dataset",
        "overall_accuracy",
        "overall_f1",
        metric_col,
        "equalized_odds_diff",
        "n_train",
        "n_val",
        "n_test",
        "pca_explained_variance_sum",
        "train_time_s",
    ]
    print("\nDataset variability summary:")
    print(df[cols].round(4).to_string(index=False))
    print(f"\n[✓] Results saved to {csv_path}")
    print(f"[✓] Plot saved to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["compas", "adult"],
        choices=list_datasets(),
        help="Datasets to compare.",
    )
    parser.add_argument(
        "--attribute",
        default="auto",
        help="Protected attribute to evaluate, or 'auto' for race when available else sex.",
    )
    parser.add_argument(
        "--metric",
        default="demographic_parity_diff",
        choices=list(METRIC_TO_COLUMN),
        help="Fairness metric to plot.",
    )
    parser.add_argument("--n_qubits", type=int, default=4, help="Fixed PCA dimension/qubit count.")
    parser.add_argument("--n_layers", type=int, default=2, help="Fixed VQC ansatz depth.")
    parser.add_argument("--n_epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--n_train", type=int, default=300, help="Training subsample size.")
    parser.add_argument("--n_val", type=int, default=300, help="Validation subsample size.")
    parser.add_argument("--n_test", type=int, default=1000, help="Test subsample size.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
