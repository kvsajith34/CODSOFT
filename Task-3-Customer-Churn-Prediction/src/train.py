"""
train.py

Trains the three models with a bit of hyperparam tuning using GridSearch.
I picked these three because the task asked for them.
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")


# small grids so it runs fast even on a laptop
PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs"],
        "max_iter": [500],
        "class_weight": ["balanced", None],
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [6, 10, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced", None],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
    },
}

BASE_MODELS = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}


def train_all(X_train, y_train, cv_folds=5, n_jobs=-1):
    # train each model with grid search, pick best by AUC
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    trained = {}

    for name, base_est in BASE_MODELS.items():
        print(f"Training {name}...")
        grid = GridSearchCV(
            estimator=base_est,
            param_grid=PARAM_GRIDS[name],
            scoring="roc_auc",  # AUC good for imbalanced data like this
            cv=cv,
            n_jobs=n_jobs,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        best = grid.best_estimator_
        print(f"  Best CV AUC: {grid.best_score_:.4f} with params {grid.best_params_}")
        trained[name] = best

    return trained


def save_models(models, output_dir="models"):
    # save the trained models as pickle files
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        path = os.path.join(output_dir, fname)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved model to {path}")


def load_model(name, model_dir="models"):
    fname = name.lower().replace(" ", "_") + ".pkl"
    path = os.path.join(model_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Can't find saved model at {path} - run training first")
    with open(path, "rb") as f:
        return pickle.load(f)
