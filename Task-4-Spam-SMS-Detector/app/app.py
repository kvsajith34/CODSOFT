"""
Flask web application for SMS Spam Detection.
Serves a clean UI and a REST API endpoint.

Run locally:
    python app/app.py

Deploy (Render / Railway / Heroku):
    gunicorn app.app:app
"""

import os
import sys
import json

from flask import Flask, request, jsonify, render_template_string

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.predict import load_model, predict_message

app = Flask(__name__)
model = None   # lazy-loaded on first request


def get_model():
    global model
    if model is None:
        model = load_model()
    return model


# ── HTML template ────────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SMS Spam Detector</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', Arial, sans-serif;
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      min-height: 100vh;
      display: flex; align-items: center; justify-content: center;
      padding: 20px;
    }
    .card {
      background: #fff;
      border-radius: 16px;
      padding: 40px;
      width: 100%; max-width: 600px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    }
    h1 { font-size: 1.8rem; color: #1a1a2e; margin-bottom: 6px; }
    .subtitle { color: #666; font-size: 0.9rem; margin-bottom: 28px; }
    label { display: block; font-weight: 600; color: #333; margin-bottom: 8px; }
    textarea {
      width: 100%; height: 120px; padding: 14px;
      border: 2px solid #ddd; border-radius: 10px;
      font-size: 1rem; resize: vertical;
      transition: border-color .2s;
    }
    textarea:focus { outline: none; border-color: #4361ee; }
    button {
      margin-top: 16px; width: 100%; padding: 14px;
      background: #4361ee; color: #fff;
      border: none; border-radius: 10px;
      font-size: 1rem; font-weight: 700;
      cursor: pointer; transition: background .2s;
    }
    button:hover { background: #3a0ca3; }
    button:disabled { background: #aaa; cursor: not-allowed; }
    .result {
      margin-top: 24px; padding: 18px 24px;
      border-radius: 10px; display: none;
    }
    .result.spam { background: #ffe0e0; border-left: 5px solid #e63946; }
    .result.ham  { background: #d8f3dc; border-left: 5px solid #2d6a4f; }
    .result-label { font-size: 1.4rem; font-weight: 800; margin-bottom: 4px; }
    .spam .result-label { color: #e63946; }
    .ham  .result-label { color: #2d6a4f; }
    .result-conf { color: #555; font-size: 0.95rem; }
    .error { margin-top: 16px; color: #e63946; font-size: 0.9rem; }
    .examples { margin-top: 28px; }
    .examples h3 { font-size: 0.85rem; color: #888; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
    .chip {
      display: inline-block; margin: 4px;
      padding: 6px 12px; background: #f0f0f0;
      border-radius: 20px; font-size: 0.82rem;
      cursor: pointer; transition: background .15s;
    }
    .chip:hover { background: #dce3ff; }
  </style>
</head>
<body>
<div class="card">
  <h1>📱 SMS Spam Detector</h1>
  <p class="subtitle">Powered by Machine Learning · TF-IDF + Classifier</p>

  <label for="sms">Enter SMS message</label>
  <textarea id="sms" placeholder="Paste or type an SMS message here…"></textarea>
  <button id="btn" onclick="classify()">Classify Message</button>

  <div class="result" id="result">
    <div class="result-label" id="result-label"></div>
    <div class="result-conf"  id="result-conf"></div>
  </div>
  <div class="error" id="error"></div>

  <div class="examples">
    <h3>Try an example</h3>
    {% for ex in examples %}
    <span class="chip" onclick="document.getElementById('sms').value=this.dataset.text" data-text="{{ ex }}">{{ ex[:50] }}…</span>
    {% endfor %}
  </div>
</div>

<script>
async function classify() {
  const text = document.getElementById('sms').value.trim();
  const btn  = document.getElementById('btn');
  const res  = document.getElementById('result');
  const err  = document.getElementById('error');

  err.textContent = '';
  res.style.display = 'none';

  if (!text) { err.textContent = 'Please enter an SMS message.'; return; }

  btn.disabled = true;
  btn.textContent = 'Classifying…';

  try {
    const r = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || 'Server error');

    res.className = 'result ' + data.prediction;
    document.getElementById('result-label').textContent =
      data.prediction === 'spam' ? '🚫 SPAM' : '✅ HAM (Legitimate)';
    document.getElementById('result-conf').textContent =
      `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    res.style.display = 'block';
  } catch(e) {
    err.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Classify Message';
  }
}

document.getElementById('sms').addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') classify();
});
</script>
</body>
</html>
"""

EXAMPLE_MESSAGES = [
    "Congratulations! You've won a £1,000 Tesco gift card. Click here to claim your prize now!",
    "Hey, are we still meeting for lunch tomorrow at noon?",
    "URGENT: Your account has been compromised. Call 0800-FREE now to secure your funds!",
    "Don't forget to pick up milk on your way home. Thanks!",
    "FREE entry to a weekly competition to win FA Cup tickets. Text FA to 87121 now!",
    "I'll be home by 7. Can you start dinner?",
]


@app.route("/")
def index():
    return render_template_string(HTML, examples=EXAMPLE_MESSAGES)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "JSON body must contain a 'text' field."}), 400

    text = str(data["text"]).strip()
    if not text:
        return jsonify({"error": "Text field is empty."}), 400

    try:
        result = predict_message(text, get_model())
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
