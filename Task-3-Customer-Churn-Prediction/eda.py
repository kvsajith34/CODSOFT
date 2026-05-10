"""
eda.py — Exploratory Data Analysis
====================================

Run this before training to understand the data.
Generates a set of diagnostic plots to outputs/eda/.

Usage:
    python eda.py
    python eda.py --data path/to/Churn_Modelling.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))

EDA_DIR  = os.path.join("outputs", "eda")
DATA_PATH = os.path.join("data", "Churn_Modelling.csv")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor":   "#FAFAFA",
    "font.family":      "DejaVu Sans",
})


def plot_target_distribution(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts = df["Exited"].value_counts()
    axes[0].bar(["Stayed", "Churned"], counts.values, color=["#4E8FD4", "#E74C3C"], width=0.5, edgecolor="white")
    axes[0].set_title("Churn Count", fontweight="bold")
    axes[0].set_ylabel("Number of Customers")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")

    axes[1].pie(
        counts.values,
        labels=["Stayed", "Churned"],
        autopct="%1.1f%%",
        colors=["#4E8FD4", "#E74C3C"],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Churn Distribution", fontweight="bold")

    fig.suptitle("Target Variable — Exited", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "target_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → target_distribution.png")


def plot_numerical_distributions(df, outdir):
    num_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        stayed  = df[df["Exited"] == 0][col]
        churned = df[df["Exited"] == 1][col]
        ax.hist(stayed,  bins=30, alpha=0.6, color="#4E8FD4", label="Stayed",  density=True)
        ax.hist(churned, bins=30, alpha=0.6, color="#E74C3C", label="Churned", density=True)
        ax.set_title(col, fontweight="bold")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    axes[-1].axis("off")   # hide the empty 6th subplot
    fig.suptitle("Numerical Feature Distributions by Churn", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "numerical_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → numerical_distributions.png")


def plot_categorical_churn_rates(df, outdir):
    cat_cols = ["Geography", "Gender", "NumOfProducts", "HasCrCard", "IsActiveMember"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        churn_rate = df.groupby(col)["Exited"].mean().reset_index()
        churn_rate.columns = [col, "ChurnRate"]
        sns.barplot(data=churn_rate, x=col, y="ChurnRate", ax=ax, palette="Set2")
        ax.set_title(f"Churn Rate by {col}", fontweight="bold")
        ax.set_ylabel("Churn Rate")
        ax.set_ylim(0, 0.7)
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2, p.get_height() + 0.01),
                ha="center", fontsize=9,
            )

    axes[-1].axis("off")
    fig.suptitle("Churn Rate by Category", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "categorical_churn_rates.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → categorical_churn_rates.png")


def plot_correlation_heatmap(df, outdir):
    num_df = df.select_dtypes(include=[np.number]).drop(columns=["RowNumber", "CustomerId"], errors="ignore")
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → correlation_heatmap.png")


def plot_age_balance_scatter(df, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    stayed  = df[df["Exited"] == 0]
    churned = df[df["Exited"] == 1]

    ax.scatter(stayed["Age"],  stayed["Balance"],  alpha=0.3, s=10, color="#4E8FD4", label="Stayed")
    ax.scatter(churned["Age"], churned["Balance"], alpha=0.4, s=10, color="#E74C3C", label="Churned")
    ax.set_xlabel("Age")
    ax.set_ylabel("Balance")
    ax.set_title("Age vs Balance by Churn Status", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "age_vs_balance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → age_vs_balance.png")


def run_eda(data_path: str) -> None:
    os.makedirs(EDA_DIR, exist_ok=True)

    print(f"\n  Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape}")
    print(f"\n  Basic stats:\n{df.describe().to_string()}\n")

    print("  Generating EDA plots...")
    plot_target_distribution(df,            EDA_DIR)
    plot_numerical_distributions(df,        EDA_DIR)
    plot_categorical_churn_rates(df,        EDA_DIR)
    plot_correlation_heatmap(df,            EDA_DIR)
    plot_age_balance_scatter(df,            EDA_DIR)

    print(f"\n  All EDA plots saved to ./{EDA_DIR}/\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA_PATH)
    args = p.parse_args()

    if not os.path.exists(args.data):
        print(f"\n  ❌  File not found: {args.data}")
        print("  Download the dataset from Kaggle and place it at data/Churn_Modelling.csv\n")
        sys.exit(1)

    run_eda(args.data)
