"""
evaluate.py
-----------
Model evaluation utilities: metrics, reports, and confusion matrix plots.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────
# Core Evaluation
# ──────────────────────────────────────────────

def evaluate_model(classifier, X_test, y_test) -> dict:
    """
    Evaluate a fitted GenreClassifier on test data.

    Returns:
        Dictionary containing:
            - accuracy
            - f1_macro
            - f1_weighted
            - classification_report (string)
            - y_pred (list)
            - y_true (list)
            - classes (list)
    """
    y_pred = classifier.predict(X_test)
    y_true = list(y_test)

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    classes = sorted(set(y_true))

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report,
        'y_pred': list(y_pred),
        'y_true': y_true,
        'classes': classes,
    }


def print_results(results: dict):
    """Pretty-print evaluation metrics."""
    print(f"\n{'─'*45}")
    print(f"  Accuracy   : {results['accuracy']:.4f}  ({results['accuracy']*100:.2f}%)")
    print(f"  F1 (macro) : {results['f1_macro']:.4f}")
    print(f"  F1 (weighted): {results['f1_weighted']:.4f}")
    print(f"{'─'*45}")
    print("\n📋 Classification Report:")
    print(results['classification_report'])


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def plot_confusion_matrix(results: dict, title: str = "Confusion Matrix",
                          save_path: str = None, figsize=(10, 8)):
    """
    Plot a confusion matrix heatmap.

    Args:
        results:   Output from evaluate_model()
        title:     Plot title
        save_path: If provided, saves the figure to this path
        figsize:   Figure dimensions
    """
    cm = confusion_matrix(results['y_true'], results['y_pred'], labels=results['classes'])

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=results['classes'],
        yticklabels=results['classes'],
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Genre', fontsize=11)
    ax.set_ylabel('True Genre', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Confusion matrix saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_model_comparison(results_dict: dict, save_path: str = None):
    """
    Bar chart comparing multiple models on Accuracy and F1.

    Args:
        results_dict: {model_name: evaluate_model output}
        save_path:    Optional save path
    """
    model_names = list(results_dict.keys())
    accuracies = [r['accuracy'] for r in results_dict.values()]
    f1_scores = [r['f1_macro'] for r in results_dict.values()]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 2), 5))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue', alpha=0.85)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 (macro)', color='coral', alpha=0.85)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Accuracy vs F1-Score', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', '\n') for n in model_names], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.bar_label(bars1, fmt='%.3f', padding=3, fontsize=8)
    ax.bar_label(bars2, fmt='%.3f', padding=3, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Model comparison chart saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from src.models import GenreClassifier
    from src.data_preprocessing import load_data, split_data

    parser = argparse.ArgumentParser(description="Evaluate a trained GenreClassifier")
    parser.add_argument('--model', type=str, required=True, help='Path to saved .pkl model')
    parser.add_argument('--data', type=str, default='data/sample_data.csv', help='Test CSV path')
    parser.add_argument('--save-plots', type=str, default=None, help='Directory to save plots')
    args = parser.parse_args()

    df = load_data(args.data)
    _, X_test, _, y_test = split_data(df)

    classifier = GenreClassifier.load(args.model)
    results = evaluate_model(classifier, X_test, y_test)
    print_results(results)

    if args.save_plots:
        plot_confusion_matrix(
            results,
            title=f"Confusion Matrix — {args.model}",
            save_path=os.path.join(args.save_plots, 'confusion_matrix.png')
        )
