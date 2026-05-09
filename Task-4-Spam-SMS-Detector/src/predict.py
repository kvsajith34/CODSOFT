"""
Single-message and batch prediction using the saved best model.

Usage (CLI):
    python src/predict.py "Congratulations! You've won a free iPhone. Call now!"
    python src/predict.py --batch data/spam.csv
"""

import os
import sys
import joblib
import argparse
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocess import clean_text

DEFAULT_MODEL = os.path.join(ROOT, "models", "best_model.pkl")


def load_model(model_path: str = DEFAULT_MODEL):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python src/train.py` first."
        )
    return joblib.load(model_path)


def predict_message(text: str, model=None) -> dict:
    """
    Predict a single SMS message.
    Returns a dict with keys: text, prediction (spam/ham), confidence.
    """
    model = model or load_model()
    cleaned = clean_text(text)
    label_num = model.predict([cleaned])[0]
    label = "spam" if label_num == 1 else "ham"

    try:
        proba = model.predict_proba([cleaned])[0]
        confidence = float(max(proba))
    except AttributeError:
        # LinearSVC → use decision function as proxy
        score = model.decision_function([cleaned])[0]
        confidence = float(1 / (1 + abs(score)))  # rough confidence

    return {"text": text, "prediction": label, "confidence": round(confidence, 4)}


def predict_batch(filepath: str, model=None) -> pd.DataFrame:
    """
    Run predictions on a CSV file.  Expects 'v1' (label) and 'v2' (text) columns
    OR a single column named 'text'.  Returns a DataFrame with results.
    """
    model = model or load_model()
    df = pd.read_csv(filepath, encoding="latin-1")

    if "v2" in df.columns:
        texts = df["v2"].astype(str).tolist()
        true_labels = df["v1"].tolist() if "v1" in df.columns else None
    elif "text" in df.columns:
        texts = df["text"].astype(str).tolist()
        true_labels = df["label"].tolist() if "label" in df.columns else None
    else:
        raise ValueError("CSV must contain a 'v2' or 'text' column.")

    cleaned = [clean_text(t) for t in texts]
    preds   = model.predict(cleaned)

    result_df = pd.DataFrame({
        "text":       texts,
        "prediction": ["spam" if p == 1 else "ham" for p in preds],
    })
    if true_labels is not None:
        result_df["true_label"] = true_labels
        result_df["correct"]    = result_df["true_label"] == result_df["prediction"]

    return result_df


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SMS Spam Predictor")
    parser.add_argument("message", nargs="?", help="Single SMS message to classify")
    parser.add_argument("--batch", metavar="CSV", help="Path to CSV file for batch prediction")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to model .pkl file")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.batch:
        df = predict_batch(args.batch, model)
        print(df.head(20).to_string(index=False))
        if "correct" in df.columns:
            acc = df["correct"].mean()
            print(f"\nBatch accuracy: {acc:.4f}  ({df['correct'].sum()}/{len(df)})")
    elif args.message:
        result = predict_message(args.message, model)
        print(f"\n{'='*50}")
        print(f"  Message    : {result['text'][:80]}")
        print(f"  Prediction : {result['prediction'].upper()}")
        print(f"  Confidence : {result['confidence']:.2%}")
        print(f"{'='*50}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
