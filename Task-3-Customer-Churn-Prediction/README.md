# Bank Churn Predictor

My version of the CodSoft ML Internship Task 3 - predicting which bank customers will leave.

I used the standard bank churn dataset (close enough to subscription churn since banking products are recurring). Trained Logistic Regression, Random Forest and Gradient Boosting just like the brief asked.

This is a cleaned up, more straightforward version of the usual template.

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Approach](#-approach)
- [Results](#-results)
- [Setup & Usage](#-setup--usage)
- [Web App](#-web-app)
- [Key Findings](#-key-findings)

---

## Problem Statement

The task was to build a model that predicts customer churn for a subscription-style business using historical data (demographics + usage patterns). I went with the popular bank customer dataset because it has exactly the features needed: age, gender, country, how long they've been with us, how much they have in account, how many products, if they're active, etc.

Used Logistic Regression (baseline), Random Forest and Gradient Boosting as requested.

---

## 📊 Dataset

**Source:** [Kaggle — Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)

| Feature | Description |
|---|---|
| `CreditScore` | Customer's credit score |
| `Geography` | Country (France / Spain / Germany) |
| `Gender` | Male / Female |
| `Age` | Customer age |
| `Tenure` | Years with the bank |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Has a credit card? (0/1) |
| `IsActiveMember` | Active in last period? (0/1) |
| `EstimatedSalary` | Estimated annual salary |
| `Exited` | **Target** — Did customer leave? (0/1) |

**10,000 customers · 14 columns · ~20% churn rate**

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── Churn_Modelling.csv      ← download from Kaggle (not committed)
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py            ← loading, encoding, scaling, splitting
│   ├── train.py                 ← model definitions + GridSearchCV
│   └── evaluate.py              ← metrics, plots (ROC, confusion, importance)
│
├── notebooks/
│   └── churn_analysis.ipynb     ← full end-to-end Jupyter walkthrough
│
├── models/                      ← saved .pkl files (created after training)
├── outputs/                     ← plots and charts (created after training)
│   └── eda/                     ← EDA plots
│
├── main.py                      ← full pipeline: load → train → evaluate
├── eda.py                       ← standalone EDA script
├── app.py                       ← Streamlit web app
├── requirements.txt
└── .gitignore
```

---

## 🧠 Approach

### 1. Exploratory Data Analysis
- Target class distribution and churn rate analysis
- Feature distributions split by churn status
- Categorical churn rates (geography, gender, products, etc.)
- Correlation heatmap

### 2. Preprocessing
- Drop irrelevant identifiers (`RowNumber`, `CustomerId`, `Surname`)
- Label-encode `Gender`
- One-hot encode `Geography`
- Stratified 80/20 train-test split
- `StandardScaler` normalisation

### 3. Models & Hyperparameter Tuning

| Model | Why it's here |
|---|---|
| **Logistic Regression** | Interpretable baseline; good for linearly separable patterns |
| **Random Forest** | Handles non-linearity; robust to outliers; built-in feature importance |
| **Gradient Boosting** | Often the strongest performer; sequential error correction |

All three use `GridSearchCV` with 5-fold stratified cross-validation, scored on **ROC-AUC**.

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- **ROC-AUC** (primary — appropriate for imbalanced data)
- Confusion matrices
- Churn probability distributions

---

## 📈 Results

> *(Run the pipeline on the full dataset to see exact numbers — these are representative)*

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.81 | ~0.57 | ~0.62 | ~0.59 | ~0.85 |
| Random Forest | ~0.86 | ~0.73 | ~0.48 | ~0.58 | ~0.87 |
| **Gradient Boosting** | **~0.87** | **~0.76** | **~0.52** | **~0.62** | **~0.88** |

**Gradient Boosting** consistently produces the best ROC-AUC and F1 score.

---

## 🚀 Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `Churn_Modelling.csv` from [Kaggle](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction) and place it at:
```
data/Churn_Modelling.csv
```

### 4. Run EDA (optional but recommended)
```bash
python eda.py
# → outputs/eda/ will be populated with plots
```

### 5. Train all models
```bash
python main.py
# Fast run (skip GridSearchCV):
python main.py --skip-tuning
```

### 6. Explore the notebook
```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

---

## 🌐 Web App

An interactive Streamlit app lets you adjust customer attributes with sliders and see the predicted churn probability in real time.

```bash
# Train models first, then:
streamlit run app.py
```

Features:
- Live churn probability gauge
- Risk level indicator (Low / Medium / High)
- Batch CSV upload with downloadable predictions
- Switch between models in the sidebar

---

## 🔍 Key Findings

- **Age** is the single strongest predictor of churn. Customers aged 40–60 are at significantly higher risk.
- **Inactive members** are roughly 2× more likely to churn than active ones.
- **German customers** churn at a noticeably higher rate (~32%) compared to French (~16%) and Spanish (~17%) customers.
- Customers with **1 or 3+ products** churn more than those with exactly 2.
- A **zero account balance** is a surprisingly strong churn signal — these customers may have already mentally left.

---

## 🛠 Tech Stack

- **Python 3.9+**
- `pandas` · `numpy` — data wrangling
- `scikit-learn` — modelling and evaluation
- `matplotlib` · `seaborn` — visualisation
- `streamlit` — web application
- `jupyter` — notebook exploration

---

## 👤 Author

Built as part of the **CodSoft Machine Learning Internship**.

---

## 📄 License

This project is open-source under the MIT License.
