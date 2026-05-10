"""
models.py

I wanted to try a few different approaches as per the internship task requirements 
(Logistic Regression, Decision Trees, Random Forests). I threw in HistGradientBoosting 
as a bonus because it usually performs well on tabular data like this and trains pretty fast.

Nothing too fancy with hyperparameter tuning – I just picked values that seemed reasonable 
after a few quick experiments. The goal was to show different model types rather than 
squeeze out the absolute best accuracy.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from .balancer import resample


def get_models() -> dict:
    """Returns the four models I'm using for this project.
    Each one has its own little personality.
    """
    return {
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.01,                    # I kept this small to prevent overfitting
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=10,                          # Shallow tree so it doesn't overfit too badly
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boost": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
        ),
    }


def cross_validate_model(
    model,
    X,
    y,
    resample_strategy: str = "combined",
    n_splits: int = 5,
) -> dict:
    """Runs stratified k-fold CV but resamples only on the training part of each fold.
    This is important – if you resample the whole dataset first you leak information.
    I learned that the hard way on an earlier project.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    roc_scores, pr_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        X_tr_res, y_tr_res = resample(X_tr, y_tr, strategy=resample_strategy)

        model.fit(X_tr_res, y_tr_res)
        probs = model.predict_proba(X_val)[:, 1]

        roc = roc_auc_score(y_val, probs)
        pr = average_precision_score(y_val, probs)
        roc_scores.append(roc)
        pr_scores.append(pr)

        print(
            f"  Fold {fold}/{n_splits}  |  "
            f"AUC-ROC: {roc:.4f}  |  AUC-PR: {pr:.4f}"
        )

    return {
        "auc_roc": np.array(roc_scores),
        "auc_pr": np.array(pr_scores),
    }


def train_final(model, X_train, y_train, resample_strategy: str = "combined"):
    """Resample the full training set and fit the model. Pretty straightforward."""
    X_res, y_res = resample(X_train, y_train, strategy=resample_strategy)
    model.fit(X_res, y_res)
    return model