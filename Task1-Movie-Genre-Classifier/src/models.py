"""
models.py
---------
Classifier wrappers, pipeline builders, and model registry.
"""

import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler


# ──────────────────────────────────────────────
# Model Definitions
# ──────────────────────────────────────────────

def get_naive_bayes(alpha: float = 0.5) -> ComplementNB:
    """
    Complement Naive Bayes — works better than MultinomialNB
    for imbalanced datasets.

    Args:
        alpha: Laplace smoothing parameter
    """
    return ComplementNB(alpha=alpha)


def get_logistic_regression(C: float = 5.0, max_iter: int = 1000) -> LogisticRegression:
    """
    Logistic Regression with L2 regularization.

    Args:
        C:        Inverse regularization strength (larger = less regularization)
        max_iter: Max solver iterations
    """
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver='lbfgs',
        n_jobs=-1,
        random_state=42,
    )


def get_svm(C: float = 1.0) -> CalibratedClassifierCV:
    """
    LinearSVC wrapped in CalibratedClassifierCV for probability estimates.

    Args:
        C: Regularization parameter
    """
    svc = LinearSVC(C=C, max_iter=2000, random_state=42)
    return CalibratedClassifierCV(svc, cv=3)


# ──────────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────────

MODEL_REGISTRY = {
    'naive_bayes': get_naive_bayes,
    'logistic_regression': get_logistic_regression,
    'svm': get_svm,
}


def get_model(model_name: str, **kwargs):
    """
    Factory function to instantiate a classifier by name.

    Args:
        model_name: One of 'naive_bayes', 'logistic_regression', 'svm'
        **kwargs:   Hyperparameters passed to the model

    Returns:
        sklearn-compatible classifier instance
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)


# ──────────────────────────────────────────────
# Genre Classifier (High-Level Wrapper)
# ──────────────────────────────────────────────

class GenreClassifier:
    """
    High-level wrapper combining a feature extractor and a classifier.

    Attributes:
        feature_type: 'tfidf' or 'embeddings'
        model_name:   Classifier name from MODEL_REGISTRY
        extractor:    Fitted feature extractor
        classifier:   Fitted sklearn classifier
        label_encoder: Maps integer predictions back to genre strings
    """

    def __init__(self, feature_type: str = 'tfidf', model_name: str = 'logistic_regression'):
        from src.feature_extraction import get_extractor
        self.feature_type = feature_type
        self.model_name = model_name
        self.extractor = get_extractor(feature_type)
        self.classifier = get_model(model_name)
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, X_train, y_train):
        """
        Fit the extractor and classifier.

        Args:
            X_train: Iterable of cleaned plot summaries
            y_train: Iterable of genre labels
        """
        print(f"\n🔧 Training {self.model_name.upper()} with {self.feature_type.upper()} features...")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_train)

        # Extract features
        X_features = self.extractor.fit_transform(list(X_train))

        # Scale if embeddings (dense matrix, can have negative values)
        if self.feature_type == 'embeddings':
            self.scaler = MaxAbsScaler()
            X_features = self.scaler.fit_transform(X_features)
        else:
            self.scaler = None

        # Fit classifier
        self.classifier.fit(X_features, y_encoded)
        self.is_fitted = True
        print(f"✅ Training complete!")
        return self

    def predict(self, texts):
        """
        Predict genres for new plot summaries.

        Args:
            texts: List of raw or cleaned plot summaries

        Returns:
            Array of predicted genre strings
        """
        X = self._prepare_features(texts)
        y_pred = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, texts):
        """
        Predict genre probability distributions.

        Args:
            texts: List of raw or cleaned plot summaries

        Returns:
            (classes array, probabilities 2D array)
        """
        X = self._prepare_features(texts)
        proba = self.classifier.predict_proba(X)
        classes = self.label_encoder.classes_
        return classes, proba

    def predict_top_k(self, text: str, k: int = 3):
        """
        Return top-k genre predictions with probabilities for a single text.

        Returns:
            List of (genre, probability) tuples, sorted descending
        """
        classes, proba = self.predict_proba([text])
        proba = proba[0]
        top_k_indices = np.argsort(proba)[::-1][:k]
        return [(classes[i], float(proba[i])) for i in top_k_indices]

    def _prepare_features(self, texts):
        """Internal: extract and optionally scale features."""
        if not self.is_fitted:
            raise RuntimeError("GenreClassifier must be fitted before prediction.")
        X = self.extractor.transform(list(texts))
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def save(self, path: str):
        """Serialize the entire classifier to disk."""
        joblib.dump(self, path)
        print(f"💾 Model saved → {path}")

    @classmethod
    def load(cls, path: str):
        """Load a saved GenreClassifier from disk."""
        model = joblib.load(path)
        print(f"📂 Model loaded ← {path}")
        return model

    def __repr__(self):
        return (
            f"GenreClassifier("
            f"model={self.model_name}, "
            f"features={self.feature_type}, "
            f"fitted={self.is_fitted})"
        )


if __name__ == "__main__":
    print("Available models:", list(MODEL_REGISTRY.keys()))
    for name in MODEL_REGISTRY:
        m = get_model(name)
        print(f"  {name}: {type(m).__name__}")
