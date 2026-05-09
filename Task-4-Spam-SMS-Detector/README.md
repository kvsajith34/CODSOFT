# 📱 SMS Spam Detection

> A production-ready Machine Learning pipeline that classifies SMS messages as **spam** or **ham (legitimate)** — deployable as a Flask web app or REST API.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)
![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

- **Three classifiers** compared: Naive Bayes, Logistic Regression, Linear SVM
- **TF-IDF vectorization** with bigrams, sublinear term frequency, and min-DF filtering
- **End-to-end sklearn Pipeline** — preprocessing + vectorizing + classification in one call
- **5-fold cross-validation** during training; held-out test set evaluation
- **Flask web UI** with instant predictions and a REST API endpoint
- **16 automated tests** covering preprocessing, prediction, and the API
- **GitHub Actions CI** — tests run on every push
- One-command deploy to **Render / Railway / Heroku**

---

## 📊 Model Results (UCI SMS Spam Dataset — 5,571 messages)

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Naive Bayes         | 97.7%    | 0.97      | 0.84   | 0.90     | 0.992   |
| Logistic Regression | 97.5%    | 0.98      | 0.82   | 0.89     | 0.996   |
| **LinearSVC** ✅    | **98.2%**| **0.99**  | **0.86**| **0.92**| **0.996**|

> Best model: **Linear SVM** saved to `models/best_model.pkl`

---

## 🗂 Project Structure

```
spam_sms_detection/
├── data/
│   └── spam.csv              # UCI SMS Spam dataset
├── src/
│   ├── preprocess.py         # Text cleaning & feature prep
│   ├── train.py              # Training + evaluation + model saving
│   └── predict.py            # Inference (single message or batch CSV)
├── app/
│   └── app.py                # Flask web app + REST API
├── tests/
│   └── test_pipeline.py      # 16 pytest tests
├── models/                   # Saved .pkl pipelines (git-ignored)
├── reports/                  # Confusion matrices + comparison chart
├── .github/
│   └── workflows/ci.yml      # GitHub Actions CI
├── requirements.txt
├── setup.py
├── Procfile                  # For Heroku/Render/Railway
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/spam-sms-detection.git
cd spam-sms-detection
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the models

```bash
python src/train.py
# Trained models saved to models/
# Reports saved to reports/
```

### 3. Predict from the command line

```bash
# Single message
python src/predict.py "Congratulations! You have won a FREE iPhone. Call now!"

# Batch CSV
python src/predict.py --batch data/spam.csv
```

### 4. Start the web app

```bash
python app/app.py
# Open http://localhost:5000
```

---

## 🌐 REST API

**POST** `/predict`

```json
// Request
{ "text": "Your free prize is waiting. Click to claim!" }

// Response
{
  "text": "Your free prize is waiting. Click to claim!",
  "prediction": "spam",
  "confidence": 0.9874
}
```

**GET** `/health` → `{ "status": "ok" }`

---

## ☁️ Deploy to the Cloud

### Render (Free tier)

1. Push your repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set **Start Command** to:
   ```
   python src/train.py && gunicorn app.app:app
   ```
5. Click **Deploy**

### Railway

```bash
railway init
railway up
```

### Heroku

```bash
heroku create your-app-name
git push heroku main
heroku run python src/train.py
```

> ⚠️ The trained model is git-ignored (binary file). Your deploy command must run `python src/train.py` before starting the server.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- `TestCleanText` — URL removal, lowercasing, stopword removal, punctuation stripping
- `TestLoadAndPrepare` — column names, binary labels, no nulls
- `TestPredict` — spam/ham classification, result shape
- `TestFlaskApp` — index, health, `/predict` endpoint, error handling

---

## ⚙️ How It Works

```
Raw SMS
  │
  ▼
clean_text()
  ├── Lowercase
  ├── Remove URLs & phone numbers
  ├── Strip punctuation & digits
  ├── Remove stopwords
  └── Stem tokens
  │
  ▼
TF-IDF Vectorizer
  ├── max_features=10,000
  ├── ngram_range=(1,2)     ← bigrams capture "free prize", "call now"
  └── sublinear_tf=True
  │
  ▼
Classifier (LinearSVC / NB / LR)
  │
  ▼
"spam" or "ham"
```

---

## 📈 Extending the Project

| Idea | Where to change |
|------|----------------|
| Try Word2Vec / FastText embeddings | `src/preprocess.py` + `src/train.py` |
| Add a new classifier | `CLASSIFIERS` dict in `src/train.py` |
| Tune hyperparameters | `TFIDF_PARAMS` + classifier kwargs in `src/train.py` |
| Swap dataset | Replace `data/spam.csv` and update column names in `preprocess.py` |
| Add authentication to API | `app/app.py` |

---

## 📄 Dataset

[UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
— 5,574 English SMS messages tagged as spam or ham.

---

## 📝 License

MIT License. See `LICENSE` for details.
