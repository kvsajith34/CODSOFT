"""
Train multiple spam-detection classifiers and save the best one.

Models compared:
  • Naive Bayes       (MultinomialNB)
  • Logistic Regression
  • Support Vector Machine (LinearSVC)

Each model is wrapped in a Pipeline with TF-IDF vectorization so a single
.predict(raw_text) call handles everything end-to-end.

Usage:
    python src/train.py               # uses default data path
    python src/train.py data/spam.csv
"""

import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
)

# ── project root ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocess import load_and_prepare

# ── config ──────────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join(ROOT, "data", "spam.csv")
MODELS_DIR   = os.path.join(ROOT, "models")
REPORTS_DIR  = os.path.join(ROOT, "reports")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

TFIDF_PARAMS = dict(
    max_features=10_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)

CLASSIFIERS = {
    "NaiveBayes": MultinomialNB(alpha=0.1),
    "LogisticRegression": LogisticRegression(max_iter=1000, C=5, solver="lbfgs", random_state=42),
    "SVM": LinearSVC(C=1.0, max_iter=2000, random_state=42),
}

RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ── helpers ─────────────────────────────────────────────────────────────────

def build_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf",   clf),
    ])


def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    # LinearSVC has no predict_proba; fall back to decision_function for AUC
    try:
        y_score = pipeline.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_score = pipeline.decision_function(X_test)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_score),
        "report":    classification_report(y_test, y_pred, target_names=["ham", "spam"]),
        "conf_mat":  confusion_matrix(y_test, y_pred).tolist(),
        "y_pred":    y_pred,
    }


def plot_confusion_matrix(conf_mat, model_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, f"cm_{model_name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_comparison(results: dict):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    models  = list(results.keys())
    values  = {m: [results[mod][m] for mod in models] for m in metrics}

    x = np.arange(len(models))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, values[metric], width, label=metric.upper())
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ── main ────────────────────────────────────────────────────────────────────

def train(data_path=None):
    data_path = data_path or DATA_PATH
    print(f"[train] Loading data from {data_path}")
    df = load_and_prepare(data_path)
    X, y = df["clean_text"], df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[train] Train: {len(X_train)}  Test: {len(X_test)}")

    results = {}
    best_name, best_f1, best_pipeline = None, -1, None

    for name, clf in CLASSIFIERS.items():
        print(f"\n── {name} ──────────────────")
        pipeline = build_pipeline(clf)

        # 5-fold CV on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
        print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        results[name] = metrics

        print(f"  Test accuracy : {metrics['accuracy']:.4f}")
        print(f"  Test F1 (spam): {metrics['f1']:.4f}")
        print(f"  Test ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(metrics["report"])

        # save confusion matrix
        plot_confusion_matrix(metrics["conf_mat"], name)

        # track best
        if metrics["f1"] > best_f1:
            best_f1, best_name, best_pipeline = metrics["f1"], name, pipeline

    # save comparison chart
    plot_comparison({k: v for k, v in results.items()})

    # persist every pipeline
    for name, clf in CLASSIFIERS.items():
        pipeline = build_pipeline(clf)
        pipeline.fit(X_train, y_train)
        save_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(pipeline, save_path)
        print(f"[train] Saved {name} → {save_path}")

    # save best model with canonical name
    best_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_pipeline, best_path)
    print(f"\n[train] ✅ Best model: {best_name}  F1={best_f1:.4f}  → {best_path}")

    # save summary JSON
    summary = {
        name: {k: v for k, v in m.items() if k not in ("report", "y_pred", "conf_mat")}
        for name, m in results.items()
    }
    summary["best"] = best_name
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return best_pipeline, results


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    train(path)
