# Credit Card Fraud Detection – Task 2

Hi! This is my submission for Task 2 of the Codesoft internship.

I built a full machine learning pipeline that can detect fraudulent credit card transactions. 
The dataset is super imbalanced (only about 0.5% fraud), so I spent a lot of time experimenting 
with different ways to handle that. I also engineered some new features, trained four different 
models (Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting), and tuned 
the decision threshold using a simple cost model instead of just using 0.5.

Everything is written in Python using scikit-learn and imbalanced-learn. I tried to keep the 
code readable and well-commented so it's easy to follow.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run training (uses synthetic data if you don't have the real CSV)
python train.py

# 3. Score new transactions
python predict.py --input my_new_transactions.csv --model random_forest
```

If you want to use the real Sparkov dataset, download it from Kaggle and put `fraudTrain.csv` 
(or any CSV with the same columns) into the `data/` folder as `fraud_data.csv`.

## What the pipeline does

- Loads the dataset (or creates synthetic data if the file is missing)
- Engineers useful features (log amount, time of day, distance, age, etc.)
- Handles class imbalance with SMOTE / undersampling / combined (inside CV folds only!)
- Trains and compares 4 models with 5-fold stratified cross-validation
- Finds the best decision threshold using a realistic cost model ($120 per missed fraud, $2 per false alarm)
- Saves trained models + nice diagnostic plots (ROC, PR curves, confusion matrices, threshold analysis)

## Models I trained

| Model           | Why I included it                  |
|-----------------|------------------------------------|
| Logistic Regression | Simple baseline, easy to interpret |
| Decision Tree       | Required by the task, good starting point |
| Random Forest       | Usually strong on tabular data     |
| Gradient Boosting   | Fast and often gives the best performance |

## Files in this project

- `train.py` – main training script (run this first)
- `predict.py` – use a saved model to score new data
- `src/data_loader.py` – loads real data or creates synthetic data
- `src/features.py` – all the feature engineering
- `src/models.py` – defines the four models
- `src/balancer.py` – SMOTE / undersampling helpers
- `src/evaluate.py` – metrics, plots, and threshold tuning
- `requirements.txt` – all the packages you need
- `data/` – put your CSV here (or leave empty for synthetic data)
- `outputs/` – trained models and plots get saved here

## Notes

- I used the Sparkov simulated fraud dataset (the big one with merchant, category, lat/long, etc.)
- If you don't want to download the huge CSV, the code will generate synthetic data that works fine for testing.
- All random seeds are fixed at 42 so results are reproducible.
- The cost model ($120 / $2) is just something I made up based on articles I read – feel free to change it in evaluate.py.

Thanks for checking out my project! If you have any questions or feedback, I'm happy to chat.

— Venkata Sai Ajith K | Codesoft Internship – Task 2