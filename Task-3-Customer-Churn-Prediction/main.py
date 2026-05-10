"""
main.py - Customer Churn Prediction

Simple end-to-end script for the CodSoft task.
Run with python main.py or add --skip-tuning for quick test.
"""

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np

# add src to path so we can import from there
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import load_data, explore_basics, preprocess
from src.train import train_all, save_models
from src.evaluate import (
    print_report, plot_confusion_matrices, plot_roc_curves,
    plot_feature_importance, plot_metrics_comparison, plot_churn_probability_dist
)

DEFAULT_DATA_PATH = os.path.join("data", "Churn_Modelling.csv")
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"


def header(title):
    # quick way to print a nice section title
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def run(data_path, skip_tuning=False):
    start = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    header("Step 1: Load Data")
    df = load_data(data_path)
    explore_basics(df)

    header("Step 2: Preprocess")
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(df)

    header("Step 3: Train Models")
    if skip_tuning:
        print("  [quick mode] skipping grid search, using defaults\n")
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42).fit(X_train, y_train),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42).fit(X_train, y_train),
        }
    else:
        models = train_all(X_train, y_train)

    save_models(models, MODEL_DIR)

    header("Step 4: Evaluate")
    summary = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = print_report(name, y_test, y_pred, y_prob)
        summary.append(metrics)

    print("\nMaking the plots now...")
    plot_confusion_matrices(models, X_test, y_test, OUTPUT_DIR)
    plot_roc_curves(models, X_test, y_test, OUTPUT_DIR)
    plot_feature_importance(models, feature_names, OUTPUT_DIR)
    plot_metrics_comparison(summary, OUTPUT_DIR)
    plot_churn_probability_dist(models, X_test, y_test, OUTPUT_DIR)

    header("Final Results")
    df_summary = pd.DataFrame(summary).set_index("Model")
    print(df_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    best = df_summary["ROC-AUC"].idxmax()
    print(f"\nBest model (by ROC-AUC): {best} ({df_summary.loc[best, 'ROC-AUC']:.4f})")
    print(f"Total time: {time.time() - start:.1f} seconds")
    print(f"Check outputs/ folder for plots and models/ for saved models.\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA_PATH, help="path to the csv")
    p.add_argument("--skip-tuning", action="store_true", help="skip the grid search for faster run")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.data):
        print(f"Data file not found at {args.data}")
        print("Download Churn_Modelling.csv from Kaggle and put it in data/ folder.")
        sys.exit(1)

    run(data_path=args.data, skip_tuning=args.skip_tuning)
