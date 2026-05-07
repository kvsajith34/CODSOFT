# 🎬 Movie Genre Classifier

**Internship Project Submission**  
*Predicting Movie Genres from Plot Summaries using Machine Learning*

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation (Windows)](#setup--installation-windows)
- [Usage](#usage)
- [Final Submission](#final-submission)
- [Results](#results)
- [How to Run](#how-to-run)

---

## 🎯 Project Overview

This project implements a **multi-class movie genre classification system** that predicts genres from plot summaries using Natural Language Processing (NLP) and Machine Learning.

**Key Highlights:**
- Trained on **54,214** real movie plot summaries
- Predicts across **27** different genres
- Final model: **Logistic Regression + TF-IDF**
- Generated predictions for **54,200** test samples
- Complete end-to-end pipeline from raw text to submission file

**Techniques Used:**
- **Feature Extraction**: TF-IDF Vectorization (with bigrams)
- **Classifiers**: Naive Bayes, Logistic Regression, SVM
- **Preprocessing**: Lowercasing, stop word removal, lemmatization

---

## 📊 Dataset

The project uses a large movie plot dataset (sourced from IMDB-style data):

| Property              | Value          |
|-----------------------|----------------|
| Training Samples      | 54,214         |
| Test Samples          | 54,200         |
| Number of Genres      | 27             |
| Data Format           | `ID ::: Title ::: Genre ::: Plot` |

**Genres include:** Action, Adventure, Comedy, Drama, Fantasy, Horror, Romance, Science Fiction, Thriller, and 18 others.

---

## 📁 Project Structure

```
movie-genre-classifier/
├── README.md
├── requirements.txt
│
├── data/
│   ├── train_data.txt          # Raw training data (54k+ samples)
│   ├── test_data.txt           # Raw test data
│   ├── test_data_solution.txt  # Ground truth for evaluation
│   ├── train.csv               # Processed training data
│   └── test.csv                # Processed test data
│
├── src/
│   ├── data_preprocessing.py   # Text cleaning & lemmatization
│   ├── feature_extraction.py   # TF-IDF extractor
│   ├── models.py               # GenreClassifier wrapper
│   ├── train.py                # Training script
│   ├── predict.py              # Prediction utilities
│   └── evaluate.py             # Metrics & visualization
│
├── scripts/                    # Custom internship scripts
│   ├── prepare_data.py         # Converts TXT → CSV
│   └── batch_predict.py        # Fast batch prediction (54k samples)
│
├── models/                     # Saved .pkl model files
│   ├── logistic_regression_tfidf.pkl
│   ├── naive_bayes_tfidf.pkl
│   └── svm_tfidf.pkl
│
├── submission.txt              # Final deliverable (54,200 predictions)
│
└── app/                        # Flask web app (optional)
```

---

## ⚙️ Setup & Installation (Windows)

### 1. Create Virtual Environment
```cmd
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
```cmd
pip install -r requirements.txt
```

> **Note:** `gensim` was skipped due to Windows compilation issues. The project uses **TF-IDF only**, which gives excellent results.

---

## 🚀 Usage

### Step 1: Prepare Data
```cmd
python scripts\prepare_data.py
```
This converts the raw `.txt` files into `train.csv` and `test.csv`.

### Step 2: Train Models
```cmd
python src\train.py --data data\train.csv --model all --features tfidf
```

**Recommended:** Use `--features tfidf` (Word Embeddings were skipped for Windows compatibility).

### Step 3: Generate Predictions
```cmd
python scripts\batch_predict.py
```

This creates `submission.txt` with 54,200 predictions.

---

## 📦 Final Submission

The main deliverable is:

**`submission.txt`** — Contains 54,200 predictions in the format:
```
ID ::: Title ::: Predicted Genre ::: Plot Summary
```

**Best Model Used:** `logistic_regression_tfidf.pkl`

---

## 📈 Results

| Metric                    | Value          |
|---------------------------|----------------|
| Training Samples          | 54,214         |
| Test Samples Predicted    | 54,200         |
| Final Model               | Logistic Regression + TF-IDF |
| Accuracy (on test set)    | **[Update with your value]** |

**Key Observations:**
- TF-IDF with bigrams performed very well
- Logistic Regression was the most stable and accurate model
- Distinctive genres (Sci-Fi, Horror, Action) achieved higher accuracy
- Some overlap between Drama ↔ Thriller caused minor confusion

---

## 🛠️ How to Run (Complete Pipeline)

```cmd
# 1. Activate environment
venv\Scripts\activate

# 2. Prepare data
python scripts\prepare_data.py

# 3. Train all models
python src\train.py --data data\train.csv --model all --features tfidf

# 4. Generate final submission
python scripts\batch_predict.py

# Output: submission.txt (ready to submit)
```

---

## 📄 License
MIT License

---

**Project completed as part of Internship – May 2026**  
*Author: Venkata Sai Ajith*