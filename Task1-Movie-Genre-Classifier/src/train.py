"""
train.py
--------
Training script for the Movie Genre Classifier.

Usage:
    python src/train.py --data data/sample_data.csv --model all --features tfidf
    python src/train.py --data data/sample_data.csv --model logistic_regression --features embeddings
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import load_data, split_data
from src.models import GenreClassifier, MODEL_REGISTRY
from src.evaluate import evaluate_model, print_results


# ──────────────────────────────────────────────
# Training Logic
# ──────────────────────────────────────────────

def train_single(X_train, X_test, y_train, y_test,
                 model_name: str, feature_type: str,
                 save_dir: str = 'models') -> dict:
    """
    Train a single model and return evaluation results.

    Returns:
        Dictionary of metrics.
    """
    classifier = GenreClassifier(
        feature_type=feature_type,
        model_name=model_name,
    )

    # Train
    classifier.fit(X_train, y_train)

    # Evaluate
    results = evaluate_model(classifier, X_test, y_test)

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}_{feature_type}.pkl")
    classifier.save(model_path)

    return results


def train_all(X_train, X_test, y_train, y_test,
              feature_type: str, save_dir: str = 'models') -> dict:
    """
    Train all models and return a results summary.
    """
    all_results = {}
    for model_name in MODEL_REGISTRY.keys():
        print(f"\n{'='*55}")
        print(f"  Model: {model_name.upper()}  |  Features: {feature_type.upper()}")
        print('='*55)
        results = train_single(
            X_train, X_test, y_train, y_test,
            model_name=model_name,
            feature_type=feature_type,
            save_dir=save_dir,
        )
        all_results[f"{model_name}_{feature_type}"] = results
        print_results(results)

    return all_results


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Movie Genre Classifier",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--data', type=str, default='data/sample_data.csv',
        help='Path to CSV dataset (requires "plot" and "genre" columns)'
    )
    parser.add_argument(
        '--model', type=str, default='all',
        choices=['all'] + list(MODEL_REGISTRY.keys()),
        help='Model to train. "all" trains every available model.'
    )
    parser.add_argument(
        '--features', type=str, default='tfidf',
        choices=['tfidf', 'embeddings'],
        help='Feature extraction method.'
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Proportion of data for test split (default: 0.2)'
    )
    parser.add_argument(
        '--save-dir', type=str, default='models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--results-file', type=str, default=None,
        help='Optional path to save results as JSON'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n🎬 Movie Genre Classifier — Training Pipeline")
    print("=" * 55)

    # Load and preprocess data
    df = load_data(args.data)

    # Split
    X_train, X_test, y_train, y_test = split_data(df, test_size=args.test_size)

    # Train
    if args.model == 'all':
        results = train_all(
            X_train, X_test, y_train, y_test,
            feature_type=args.features,
            save_dir=args.save_dir,
        )
        print("\n\n📊 SUMMARY TABLE")
        print("-" * 45)
        print(f"{'Model':<35} {'Accuracy':>8} {'F1 (macro)':>12}")
        print("-" * 45)
        for name, r in results.items():
            print(f"{name:<35} {r['accuracy']:>8.4f} {r['f1_macro']:>12.4f}")
    else:
        results = train_single(
            X_train, X_test, y_train, y_test,
            model_name=args.model,
            feature_type=args.features,
            save_dir=args.save_dir,
        )
        print_results(results)

    # Optionally save results to JSON
    if args.results_file:
        os.makedirs(os.path.dirname(args.results_file) or '.', exist_ok=True)
        with open(args.results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Results saved → {args.results_file}")

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
