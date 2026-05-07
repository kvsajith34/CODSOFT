"""
app.py
------
Flask web application for the Movie Genre Classifier.

Run:
    python app/app.py
    # Then open http://localhost:5000
"""

import sys
import os
import json
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify
from src.models import GenreClassifier
from src.data_preprocessing import TextPreprocessor

app = Flask(__name__)

# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent / 'models'
LOADED_MODELS = {}
preprocessor = TextPreprocessor()

GENRE_EMOJIS = {
    'Action': '💥',
    'Adventure': '🗺️',
    'Comedy': '😂',
    'Fantasy': '🧙',
    'Horror': '👻',
    'Romance': '❤️',
    'Science Fiction': '🚀',
    'Thriller': '🔪',
    'Drama': '🎭',
    'Mystery': '🔍',
    'Animation': '🎨',
    'Crime': '🕵️',
}

def load_all_models():
    """Load all available .pkl model files from the models directory."""
    pkl_files = list(MODELS_DIR.glob('*.pkl'))
    for pkl_file in pkl_files:
        model_key = pkl_file.stem  # e.g., "logistic_regression_tfidf"
        try:
            LOADED_MODELS[model_key] = GenreClassifier.load(str(pkl_file))
            print(f"  ✅ Loaded: {model_key}")
        except Exception as e:
            print(f"  ⚠️  Failed to load {pkl_file.name}: {e}")

    if not LOADED_MODELS:
        print("⚠️  No models found. Train models first: python src/train.py")


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route('/')
def index():
    """Main page."""
    available_models = list(LOADED_MODELS.keys())
    return render_template('index.html', models=available_models)


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for genre prediction.
    Expects JSON: { "text": "...", "model": "logistic_regression_tfidf", "top_k": 5 }
    Returns JSON: { "predictions": [{"genre": "...", "confidence": 0.87, "emoji": "🚀"}] }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    text = data.get('text', '').strip()
    model_key = data.get('model', '')
    top_k = int(data.get('top_k', 5))

    if not text:
        return jsonify({'error': 'Plot text is required'}), 400

    if not model_key or model_key not in LOADED_MODELS:
        if not LOADED_MODELS:
            return jsonify({'error': 'No models available. Run python src/train.py first.'}), 503
        model_key = list(LOADED_MODELS.keys())[0]

    # Preprocess & predict
    clean_text = preprocessor.clean_text(text)
    classifier = LOADED_MODELS[model_key]

    try:
        top_predictions = classifier.predict_top_k(clean_text, k=top_k)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    predictions = [
        {
            'genre': genre,
            'confidence': round(conf, 4),
            'emoji': GENRE_EMOJIS.get(genre, '🎬'),
        }
        for genre, conf in top_predictions
    ]

    return jsonify({
        'predictions': predictions,
        'model_used': model_key,
        'top_genre': predictions[0]['genre'] if predictions else None,
    })


@app.route('/models')
def list_models():
    """List available models."""
    return jsonify({'models': list(LOADED_MODELS.keys())})


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'models_loaded': len(LOADED_MODELS)})


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print("\n🎬 Movie Genre Classifier — Web App")
    print("=" * 40)
    print("Loading models...")
    load_all_models()
    print(f"\n🌐 Starting server at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
