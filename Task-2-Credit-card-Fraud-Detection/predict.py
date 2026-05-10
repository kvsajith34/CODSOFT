#!/usr/bin/env python3
"""
predict.py

Simple script to score new transactions using one of the models we trained earlier.
Just load the saved .joblib file, run the same feature engineering, and spit out
fraud probabilities + a binary flag based on the tuned threshold.

I made this so you can actually use the models on fresh data without having to
re-train everything.
"""

import argparse
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.features import engineer_features, get_feature_columns

MODELS_DIR = Path(__file__).resolve().parent / "outputs" / "models"


def parse_args():
    p = argparse.ArgumentParser(description="Fraud Detection Inference")
    p.add_argument("--input",     required=True, help="CSV file with new transactions to score")
    p.add_argument("--model",     default="random_forest",
                   choices=["logistic", "decision_tree", "random_forest", "gradient_boost"])
    p.add_argument("--threshold", type=float, default=None,
                   help="Override the saved threshold (0-1). If not given, uses the one saved during training.")
    p.add_argument("--output",    default=None,
                   help="Where to save the scored CSV (default: <input>_scored.csv)")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = MODELS_DIR / f"{args.model}.joblib"
    if not model_path.exists():
        print(f"[error] No saved model found at {model_path}")
        print("        Run train.py first to create the model files.")
        sys.exit(1)

    # Load the bundle we saved earlier
    bundle = joblib.load(model_path)
    model = bundle["model"]
    threshold = args.threshold if args.threshold is not None else bundle["threshold"]
    feature_cols = bundle["features"]

    print(f"[predict] Using model: {args.model}  |  Threshold: {threshold:.3f}")

    # Load new data and engineer the exact same features
    df = pd.read_csv(args.input)
    df = engineer_features(df)

    # Make sure we have all the columns the model expects
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[error] Input CSV is missing these columns: {missing}")
        sys.exit(1)

    X = df[feature_cols].values

    # Predict
    probs = model.predict_proba(X)[:, 1]
    flags = (probs >= threshold).astype(int)

    df["fraud_score"] = np.round(probs, 4)
    df["fraud_flag"] = flags

    flagged = flags.sum()
    print(f"[predict] Scored {len(df):,} transactions  |  {flagged} flagged as fraud")

    # Save results
    if args.output:
        out_path = Path(args.output)
    else:
        in_path = Path(args.input)
        out_path = in_path.with_stem(in_path.stem + "_scored")

    df.to_csv(out_path, index=False)
    print(f"[predict] Results saved to {out_path}")


if __name__ == "__main__":
    main()