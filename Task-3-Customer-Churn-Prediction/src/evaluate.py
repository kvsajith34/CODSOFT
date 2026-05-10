"""
evaluate.py

Makes all the reports and plots after training.
Prints metrics, saves confusion matrices, ROC curves, feature importance etc.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# colors for the plots - picked some nice ones
PALETTE = {
    "Logistic Regression": "#4E8FD4",
    "Random Forest": "#27AE60",
    "Gradient Boosting": "#E74C3C",
}
GREY = "#95a5a6"
BG = "#FAFAFA"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.8,
})


def print_report(name, y_true, y_pred, y_prob):
    # print the usual sklearn report and grab the numbers we care about
    print(f"\n{'-'*50}")
    print(name)
    print(f"{'-'*50}")
    print(classification_report(y_true, y_pred, target_names=["Stayed", "Churned"]))

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


def plot_confusion_matrices(models: dict, X_test, y_test, output_dir: str) -> None:
    """One confusion matrix per model, saved as a single figure."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        color = PALETTE.get(name, "#555")

        sns.heatmap(
            cm,
            annot=True, fmt="d",
            cmap=sns.light_palette(color, as_cmap=True),
            xticklabels=["Stayed", "Churned"],
            yticklabels=["Stayed", "Churned"],
            ax=ax,
            linewidths=0.5,
            cbar=False,
        )
        ax.set_title(name, fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Actual", fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)

    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_roc_curves(models: dict, X_test, y_test, output_dir: str) -> None:
    """Overlay all ROC curves on a single chart."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            label=f"{name}  (AUC = {roc_auc:.3f})",
            color=PALETTE.get(name, "#555"),
            linewidth=2.2,
        )

    ax.plot([0, 1], [0, 1], "--", color=GREY, linewidth=1.2, label="Random guess")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True)

    path = os.path.join(output_dir, "roc_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_feature_importance(models: dict, feature_names: list, output_dir: str) -> None:
    """
    Bar chart of top-15 features for each tree-based model.
    Logistic Regression uses |coefficient| as a proxy for importance.
    """
    for name, model in models.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            title_suffix = "Feature Importances (Gini)"
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
            title_suffix = "Feature Importances (|Coefficient|)"
        else:
            continue

        top_n = min(15, len(importances))
        indices = np.argsort(importances)[::-1][:top_n]
        vals    = importances[indices]
        labels  = [feature_names[i] for i in indices]

        color = PALETTE.get(name, "#555")
        bars = ax.barh(range(top_n), vals[::-1], color=color, alpha=0.82, edgecolor="white")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(labels[::-1], fontsize=9)
        ax.set_xlabel("Importance", fontsize=9)
        ax.set_title(f"{name} — {title_suffix}", fontsize=11, fontweight="bold")
        ax.grid(axis="x")

        fig.tight_layout()
        fname = name.lower().replace(" ", "_") + "_importance.png"
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path}")


def plot_metrics_comparison(summary: list, output_dir: str) -> None:
    """Grouped bar chart comparing Accuracy, Precision, Recall, F1, ROC-AUC."""
    df = pd.DataFrame(summary).set_index("Model")
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    df = df[metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    x    = np.arange(len(metrics))
    w    = 0.22
    n    = len(df)
    offsets = np.linspace(-(n - 1) * w / 2, (n - 1) * w / 2, n)

    for i, (idx, row) in enumerate(df.iterrows()):
        bars = ax.bar(
            x + offsets[i], row.values, w,
            label=idx,
            color=PALETTE.get(idx, "#888"),
            alpha=0.85,
            edgecolor="white",
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Model Comparison — All Metrics", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_churn_probability_dist(models: dict, X_test, y_test, output_dir: str) -> None:
    """
    Histogram of predicted churn probabilities, split by actual class.
    Useful for understanding model calibration.
    """
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        probs = model.predict_proba(X_test)[:, 1]
        color = PALETTE.get(name, "#555")

        ax.hist(probs[y_test == 0], bins=30, alpha=0.6, color="#4E8FD4", label="Stayed",  density=True)
        ax.hist(probs[y_test == 1], bins=30, alpha=0.6, color="#E74C3C", label="Churned", density=True)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Churn Probability", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y")

    fig.suptitle("Churn Probability Distribution by Actual Class", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "probability_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
