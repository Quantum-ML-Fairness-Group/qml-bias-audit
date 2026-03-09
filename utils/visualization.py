"""
utils/visualization.py

All plotting utilities for the QML fairness project.
Generates publication-quality figures for classical vs. quantum comparison.

Usage:
    from utils.visualization import FairnessPlotter
    plotter = FairnessPlotter(results_dir="results", figures_dir="figures")
    plotter.plot_all(classical_results, quantum_results, splits)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional


# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE = {
    "Logistic Regression": "#2C7BB6",
    "Random Forest":        "#1A9641",
    "MLP":                  "#ABD9E9",
    "VQC-Angle":            "#D7191C",
    "VQC-Amplitude":        "#FDAE61",
    "VQC-IQP":              "#9E0142",
}

CLASSICAL_MODELS = ["Logistic Regression", "Random Forest", "MLP"]
QUANTUM_MODELS   = ["VQC-Angle", "VQC-Amplitude", "VQC-IQP"]

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


class FairnessPlotter:
    def __init__(self, results_dir: str = "results", figures_dir: str = "figures"):
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)

    # ── 1. Main Comparison Bar Chart ──────────────────────────────────────────
    def plot_bias_comparison(self, comparison_df: pd.DataFrame, attribute: str = "race"):
        """
        Side-by-side grouped bar chart: classical vs quantum for all fairness metrics.
        comparison_df: output of compare_models(), index=model_name
        """
        metrics = ["DPD", "EOD", "EOpp", "PPD"]
        metric_labels = {
            "DPD": "Demographic\nParity Diff",
            "EOD": "Equalized\nOdds Diff",
            "EOpp": "Equal\nOpportunity Diff",
            "PPD": "Predictive\nParity Diff",
        }
        models = comparison_df.index.tolist()

        x = np.arange(len(metrics))
        n_models = len(models)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, model in enumerate(models):
            vals = [comparison_df.loc[model, m] for m in metrics]
            offset = (i - n_models / 2 + 0.5) * width
            color = PALETTE.get(model, f"C{i}")
            hatch = "//" if model in QUANTUM_MODELS else ""
            bars = ax.bar(x + offset, vals, width * 0.9, label=model,
                          color=color, hatch=hatch, edgecolor="white", linewidth=0.5)

        ax.axhline(0.1, color="gray", linestyle="--", linewidth=1.0, alpha=0.7,
                   label="Fairness threshold (0.1)")
        ax.set_xticks(x)
        ax.set_xticklabels([metric_labels[m] for m in metrics])
        ax.set_ylabel("Bias Metric Value (lower = fairer)")
        ax.set_title(f"Fairness Metrics: Classical vs Quantum — Protected Attribute: {attribute.capitalize()}")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.set_ylim(0, max(comparison_df[metrics].values.max() * 1.25, 0.35))

        # Shade quantum region
        q_start = len(CLASSICAL_MODELS) - 0.5
        ax.axvspan(q_start / len(metrics), 1.0, alpha=0.03, color="red", transform=ax.transAxes)

        plt.tight_layout()
        out = self.figures_dir / f"bias_comparison_{attribute}.pdf"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        print(f"  [fig] {out}")
        plt.close()

    # ── 2. Accuracy vs Fairness Scatter ───────────────────────────────────────
    def plot_accuracy_vs_fairness(self, comparison_df: pd.DataFrame):
        """
        Pareto scatter: accuracy on x-axis, DPD on y-axis.
        Ideal model: top-left (high accuracy, low bias).
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, (metric, label) in zip(axes, [("DPD", "Demographic Parity Diff"),
                                               ("EOD", "Equalized Odds Diff")]):
            for model in comparison_df.index:
                acc = comparison_df.loc[model, "accuracy"]
                val = comparison_df.loc[model, metric]
                color = PALETTE.get(model, "gray")
                marker = "D" if model in QUANTUM_MODELS else "o"
                ax.scatter(acc, val, c=color, marker=marker, s=120,
                           zorder=3, edgecolors="black", linewidths=0.5)
                ax.annotate(model, (acc, val),
                            textcoords="offset points", xytext=(6, 3), fontsize=8)

            ax.axhline(0.1, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Overall Accuracy")
            ax.set_ylabel(label)
            ax.set_title(f"Accuracy–Fairness Tradeoff ({metric})")

        # Legend
        classical_patch = mpatches.Patch(color="gray", label="Classical (circle)")
        quantum_patch = mpatches.Patch(color="gray", hatch="//", label="Quantum (diamond)")
        fig.legend(handles=[classical_patch, quantum_patch], loc="lower center",
                   ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout()
        out = self.figures_dir / "accuracy_vs_fairness.pdf"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        print(f"  [fig] {out}")
        plt.close()

    # ── 3. TPR/FPR Group Breakdown ─────────────────────────────────────────────
    def plot_tpr_fpr_breakdown(self, all_results: Dict, attribute: str = "race"):
        """Grouped bar chart of TPR and FPR per group per model."""
        rows = []
        for model_name, res in all_results.items():
            report = res["evaluator"].evaluate_attribute(attribute)
            rows.append({
                "model": model_name,
                "group": "Privileged",
                "TPR": report.privileged.tpr,
                "FPR": report.privileged.fpr,
            })
            rows.append({
                "model": model_name,
                "group": "Unprivileged",
                "TPR": report.unprivileged.tpr,
                "FPR": report.unprivileged.fpr,
            })

        df = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

        for ax, metric in zip(axes, ["TPR", "FPR"]):
            pivot = df.pivot(index="model", columns="group", values=metric)
            pivot.plot(kind="bar", ax=ax, color=["#2166AC", "#D6604D"],
                       edgecolor="white", width=0.65)
            ax.set_title(f"{metric} by Group — {attribute.capitalize()}")
            ax.set_xlabel("")
            ax.set_ylabel(metric)
            ax.set_xticklabels(pivot.index, rotation=30, ha="right")
            ax.legend(title="Group")
            if metric == "TPR":
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(0, 0.6)

        plt.suptitle(f"Per-Group TPR and FPR: Classical vs Quantum", fontsize=13, y=1.02)
        plt.tight_layout()
        out = self.figures_dir / f"tpr_fpr_{attribute}.pdf"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        print(f"  [fig] {out}")
        plt.close()

    # ── 4. Training Loss Curves ────────────────────────────────────────────────
    def plot_training_curves(self, quantum_results: Dict):
        """Training and validation loss curves for quantum models."""
        fig, axes = plt.subplots(1, len(quantum_results), figsize=(5 * len(quantum_results), 4),
                                 sharey=False)
        if len(quantum_results) == 1:
            axes = [axes]

        for ax, (name, res) in zip(axes, quantum_results.items()):
            train_loss = res.get("loss_history", [])
            val_loss = res.get("val_loss_history", [])

            if train_loss:
                ax.plot(train_loss, label="Train", color=PALETTE.get(name, "C0"))
            if val_loss:
                ax.plot(val_loss, label="Val", color=PALETTE.get(name, "C0"),
                        linestyle="--", alpha=0.7)

            ax.set_title(name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()

        plt.suptitle("Quantum Model Training Curves", fontsize=13)
        plt.tight_layout()
        out = self.figures_dir / "quantum_training_curves.pdf"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        print(f"  [fig] {out}")
        plt.close()

    # ── 5. Ablation Heatmap ────────────────────────────────────────────────────
    def plot_ablation_heatmap(self, ablation_df: pd.DataFrame):
        """Heatmap of DPD and EOD over encoding × circuit depth."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for ax, metric in zip(axes, ["DPD", "EOD"]):
            pivot = ablation_df.pivot_table(index="encoding", columns="n_layers", values=metric)
            sns.heatmap(
                pivot, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn_r",
                vmin=0, vmax=0.3, linewidths=0.5, cbar_kws={"label": metric},
            )
            ax.set_title(f"{metric} by Encoding × Depth")
            ax.set_xlabel("Circuit Depth (n_layers)")
            ax.set_ylabel("Encoding Scheme")

        plt.suptitle("Ablation Study: Bias vs. Encoding and Depth", fontsize=13)
        plt.tight_layout()
        out = self.figures_dir / "ablation_heatmap.pdf"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        print(f"  [fig] {out}")
        plt.close()

    # ── Full pipeline ──────────────────────────────────────────────────────────
    def plot_all(self, classical_results: Dict, quantum_results: Dict, splits,
                 ablation_df: Optional[pd.DataFrame] = None):
        """Generate all figures."""
        from utils.fairness_metrics import compare_models

        all_results = {**classical_results, **quantum_results}
        comp_df = compare_models(all_results, attribute="race")

        print("\n[viz] Generating figures...")
        self.plot_bias_comparison(comp_df, attribute="race")
        self.plot_accuracy_vs_fairness(comp_df)
        self.plot_tpr_fpr_breakdown(all_results, attribute="race")

        # Training curves for quantum (needs loss_history in results)
        q_with_history = {
            k: v for k, v in quantum_results.items()
            if "loss_history" in v and v["loss_history"]
        }
        if q_with_history:
            self.plot_training_curves(q_with_history)

        if ablation_df is not None:
            self.plot_ablation_heatmap(ablation_df)

        print(f"[viz] All figures saved to {self.figures_dir}/")
