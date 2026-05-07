"""
predict.py
----------
Prediction utilities and CLI for the Movie Genre Classifier.

Usage:
    python src/predict.py --text "A detective investigates a haunted lighthouse"
    python src/predict.py --text "A wizard discovers magic" --model models/svm_tfidf.pkl --top-k 3
"""

import sys
import argparse
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import GenreClassifier
from src.data_preprocessing import TextPreprocessor


# ──────────────────────────────────────────────
# Prediction Utilities
# ──────────────────────────────────────────────

def predict_genre(text: str, model_path: str, top_k: int = 3,
                  preprocess: bool = True) -> list:
    """
    Predict genre(s) for a given plot summary.

    Args:
        text:        Raw plot summary string
        model_path:  Path to saved GenreClassifier .pkl
        top_k:       Number of top predictions to return
        preprocess:  Whether to clean the text before prediction

    Returns:
        List of (genre, confidence) tuples sorted by confidence descending
    """
    classifier = GenreClassifier.load(model_path)

    if preprocess:
        preprocessor = TextPreprocessor()
        text = preprocessor.clean_text(text)

    return classifier.predict_top_k(text, k=top_k)


def format_prediction(predictions: list, original_text: str = None) -> str:
    """Format prediction results for display."""
    lines = []
    if original_text:
        preview = original_text[:80] + "..." if len(original_text) > 80 else original_text
        lines.append(f"\n📖 Plot: \"{preview}\"")

    lines.append("\n🎬 Predicted Genres:")
    lines.append("─" * 40)
    for i, (genre, confidence) in enumerate(predictions, 1):
        bar_len = int(confidence * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        lines.append(f"  {i}. {genre:<20} {bar}  {confidence:.1%}")
    lines.append("─" * 40)

    top_genre, top_conf = predictions[0]
    lines.append(f"\n✅ Best Prediction: {top_genre} ({top_conf:.1%} confidence)")
    return "\n".join(lines)


def find_latest_model(model_dir: str = 'models') -> str:
    """Find the most recently modified .pkl model file."""
    model_dir = Path(model_dir)
    pkl_files = list(model_dir.glob('*.pkl'))
    if not pkl_files:
        raise FileNotFoundError(
            f"No .pkl model files found in '{model_dir}'. "
            "Run 'python src/train.py' first."
        )
    # Return most recently modified
    return str(sorted(pkl_files, key=os.path.getmtime)[-1])


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict movie genre from a plot summary",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--text', type=str, required=True,
        help='Movie plot summary to classify'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to trained model .pkl file (auto-detects latest if not given)'
    )
    parser.add_argument(
        '--top-k', type=int, default=3,
        help='Number of top genre predictions to show (default: 3)'
    )
    parser.add_argument(
        '--model-dir', type=str, default='models',
        help='Directory to search for models (default: models/)'
    )
    parser.add_argument(
        '--no-preprocess', action='store_true',
        help='Skip text preprocessing (use if text is already cleaned)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve model path
    model_path = args.model or find_latest_model(args.model_dir)
    print(f"🤖 Using model: {model_path}")

    # Predict
    predictions = predict_genre(
        text=args.text,
        model_path=model_path,
        top_k=args.top_k,
        preprocess=not args.no_preprocess,
    )

    # Display
    print(format_prediction(predictions, original_text=args.text))


if __name__ == "__main__":
    main()
