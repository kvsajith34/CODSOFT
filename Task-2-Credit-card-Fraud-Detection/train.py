#!/usr/bin/env python3
"""
train.py

This is the main training script for the credit card fraud detection project.
I built it as Task 2 for the Codesoft internship. It loads the data (real or synthetic),
engineers some features, trains a few different models with proper cross-validation
and resampling inside the folds, tunes the decision threshold using a cost model,
and saves everything plus some diagnostic plots.

It's not the most elegant code in the world but it works and I learned a ton
putting it together. Run with --help to see the options.
"""

import argparse
import sys
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

# Make the src folder importable no matter where we run this from
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_data, CSV_PATH
from src.features import engineer_features, get_feature_columns
from src.models import get_models, cross_validate_model, train_final
from src.evaluate import (
    evaluate,
    tune_threshold,
    plot_roc_pr_curves,
    plot_confusion_matrix,
    plot_class_distribution,
)

MODELS_DIR = Path(__file__).resolve().parent / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Credit Card Fraud Detection Training (Task 2)")
    p.add_argument("--model",     default="all",
                   choices=["logistic", "decision_tree", "random_forest", "gradient_boost", "all"])
    p.add_argument("--resample",  default="combined",
                   choices=["smote", "undersample", "combined"])
    p.add_argument("--no-cv",     action="store_true",
                   help="Skip cross-validation (much faster, good for quick tests)")
    p.add_argument("--data",      default=str(CSV_PATH),
                   help="Path to the fraud dataset CSV")
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  Credit Card Fraud Detection — Training Pipeline (Task 2)")
    print("="*60 + "\n")

    # Load + engineer features
    df = load_data(args.data)
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].values
    y = df["is_fraud"].values

    plot_class_distribution(y)

    # Train/test split with stratification so the fraud ratio stays roughly the same
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=42,
    )

    print(f"\n[split] Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"[split] Fraud in test set: {y_test.sum()} ({y_test.mean()*100:.2f}%)\n")

    # Get the models we want to train
    all_models = get_models()
    if args.model == "all":
        models_to_run = all_models
    else:
        models_to_run = {args.model: all_models[args.model]}

    results = {}
    best_thresholds = {}

    for name, model in models_to_run.items():
        print(f"\n{'='*55}")
        print(f"  MODEL: {name.upper()}")
        print(f"{'='*55}")

        if not args.no_cv:
            print(f"\n[cv] Running {5}-fold stratified CV (resampling only on train folds)...")
            cv_scores = cross_validate_model(
                model, X_train, y_train,
                resample_strategy=args.resample,
            )
            print(
                f"\n  CV AUC-ROC : {cv_scores['auc_roc'].mean():.4f} ± {cv_scores['auc_roc'].std():.4f}"
            )
            print(
                f"  CV AUC-PR  : {cv_scores['auc_pr'].mean():.4f} ± {cv_scores['auc_pr'].std():.4f}"
            )

        # Train final model on all training data (after resampling)
        print(f"\n[train] Fitting {name} on the full training set...")
        model = train_final(model, X_train, y_train, resample_strategy=args.resample)

        # Evaluate on hold-out test set
        res = evaluate(model, X_test, y_test, model_name=name)
        results[name] = res

        # Find the best threshold using our cost model
        thresh = tune_threshold(res["probs"], y_test, model_name=name)
        best_thresholds[name] = thresh

        plot_confusion_matrix(model, X_test, y_test, threshold=thresh, model_name=name)

        # Save the model bundle so we can use it later with predict.py
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump({"model": model, "threshold": thresh, "features": feature_cols}, model_path)
        print(f"\n[saved] Model bundle → {model_path}")

    # If we trained more than one model, show the comparison plots
    if len(results) > 1:
        plot_roc_pr_curves(results, y_test)

    # Nice summary table at the end
    print(f"\n\n{'='*55}")
    print("  FINAL SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Model':<18}  {'AUC-ROC':>8}  {'AUC-PR':>8}  {'Threshold':>10}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*8}  {'-'*10}")
    for name, res in results.items():
        print(
            f"  {name:<18}  {res['auc_roc']:>8.4f}  {res['auc_pr']:>8.4f}"
            f"  {best_thresholds[name]:>10.3f}"
        )
    print(f"\n[done] All plots saved to outputs/plots/")
    print("[done] Trained models saved to outputs/models/\n")


if __name__ == "__main__":
    main()