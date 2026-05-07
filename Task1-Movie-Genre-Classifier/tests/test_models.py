"""
test_models.py
--------------
Unit tests for the Movie Genre Classifier project.

Run:
    pytest tests/test_models.py -v
"""

import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import TextPreprocessor, load_data, split_data
from src.feature_extraction import TFIDFExtractor, WordEmbeddingExtractor
from src.models import GenreClassifier, get_model, MODEL_REGISTRY


# ──────────────────────────────────────────────
# Sample Data Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_plots():
    return [
        "astronaut travel space wormhole new home humanity",
        "wizard discover magic school battle dark sorcerer",
        "detective investigate murder small town secret society",
        "young woman fall love charming stranger dangerous secret",
        "soldier rescue general behind enemy lines world war",
    ]

@pytest.fixture
def sample_genres():
    return [
        "Science Fiction",
        "Fantasy",
        "Thriller",
        "Romance",
        "Action",
    ]


# ──────────────────────────────────────────────
# Text Preprocessing Tests
# ──────────────────────────────────────────────

class TestTextPreprocessor:

    def test_lowercasing(self):
        tp = TextPreprocessor()
        result = tp.clean_text("Hello WORLD")
        assert result == result.lower()

    def test_removes_punctuation(self):
        tp = TextPreprocessor()
        result = tp.clean_text("Hello, World! How are you?")
        assert ',' not in result
        assert '!' not in result
        assert '?' not in result

    def test_removes_urls(self):
        tp = TextPreprocessor()
        result = tp.clean_text("Visit http://example.com for more info")
        assert 'http' not in result
        assert 'example.com' not in result

    def test_removes_html(self):
        tp = TextPreprocessor()
        result = tp.clean_text("<p>Hello <b>world</b></p>")
        assert '<p>' not in result
        assert '<b>' not in result

    def test_empty_string(self):
        tp = TextPreprocessor()
        result = tp.clean_text("")
        assert result == ""

    def test_none_input(self):
        tp = TextPreprocessor()
        result = tp.clean_text(None)
        assert result == ""

    def test_transform_list(self):
        tp = TextPreprocessor()
        texts = ["Hello World", "This is a test"]
        results = tp.transform(texts)
        assert len(results) == 2


# ──────────────────────────────────────────────
# Feature Extraction Tests
# ──────────────────────────────────────────────

class TestTFIDFExtractor:

    def test_fit_transform_shape(self, sample_plots):
        extractor = TFIDFExtractor(max_features=100)
        matrix = extractor.fit_transform(sample_plots)
        assert matrix.shape[0] == len(sample_plots)
        assert matrix.shape[1] <= 100

    def test_transform_before_fit_raises(self, sample_plots):
        extractor = TFIDFExtractor()
        with pytest.raises(RuntimeError):
            extractor.transform(sample_plots)

    def test_fit_returns_self(self, sample_plots):
        extractor = TFIDFExtractor()
        result = extractor.fit(sample_plots)
        assert result is extractor

    def test_is_fitted_flag(self, sample_plots):
        extractor = TFIDFExtractor()
        assert not extractor.is_fitted
        extractor.fit(sample_plots)
        assert extractor.is_fitted


class TestWordEmbeddingExtractor:

    def test_fit_transform_shape(self, sample_plots):
        extractor = WordEmbeddingExtractor(vector_size=50)
        vectors = extractor.fit_transform(sample_plots)
        assert vectors.shape == (len(sample_plots), 50)

    def test_zero_vector_for_unknown_words(self):
        extractor = WordEmbeddingExtractor(vector_size=50)
        extractor.fit(["hello world test"])
        # Text with completely unknown words
        vectors = extractor.transform(["zzzzxxxxxyyyyyyy"])
        assert vectors.shape == (1, 50)


# ──────────────────────────────────────────────
# Model Tests
# ──────────────────────────────────────────────

class TestGetModel:

    def test_all_models_available(self):
        for name in MODEL_REGISTRY:
            model = get_model(name)
            assert model is not None

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            get_model('unknown_model')


class TestGenreClassifier:

    @pytest.fixture
    def fitted_classifier(self, sample_plots, sample_genres):
        clf = GenreClassifier(feature_type='tfidf', model_name='logistic_regression')
        clf.fit(sample_plots, sample_genres)
        return clf

    def test_fit_sets_is_fitted(self, sample_plots, sample_genres):
        clf = GenreClassifier()
        assert not clf.is_fitted
        clf.fit(sample_plots, sample_genres)
        assert clf.is_fitted

    def test_predict_returns_correct_count(self, fitted_classifier, sample_plots):
        preds = fitted_classifier.predict(sample_plots)
        assert len(preds) == len(sample_plots)

    def test_predict_valid_genres(self, fitted_classifier, sample_plots, sample_genres):
        preds = fitted_classifier.predict(sample_plots)
        for p in preds:
            assert p in sample_genres

    def test_predict_proba_shape(self, fitted_classifier, sample_plots, sample_genres):
        classes, proba = fitted_classifier.predict_proba(sample_plots)
        assert proba.shape == (len(sample_plots), len(set(sample_genres)))
        assert len(classes) == len(set(sample_genres))

    def test_predict_proba_sums_to_one(self, fitted_classifier, sample_plots):
        _, proba = fitted_classifier.predict_proba(sample_plots)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(len(sample_plots)), atol=1e-5)

    def test_predict_top_k(self, fitted_classifier):
        result = fitted_classifier.predict_top_k("astronaut space mission", k=3)
        assert len(result) == 3
        genres = [r[0] for r in result]
        confs = [r[1] for r in result]
        # Confidence should be descending
        assert confs == sorted(confs, reverse=True)

    def test_predict_before_fit_raises(self):
        clf = GenreClassifier()
        with pytest.raises(RuntimeError):
            clf.predict(["some text"])

    def test_save_and_load(self, fitted_classifier, sample_plots, tmp_path):
        save_path = str(tmp_path / "test_model.pkl")
        fitted_classifier.save(save_path)
        loaded = GenreClassifier.load(save_path)
        original_preds = fitted_classifier.predict(sample_plots)
        loaded_preds = loaded.predict(sample_plots)
        assert list(original_preds) == list(loaded_preds)

    def test_repr(self, fitted_classifier):
        r = repr(fitted_classifier)
        assert 'GenreClassifier' in r
        assert 'fitted=True' in r


# ──────────────────────────────────────────────
# Data Loading Tests
# ──────────────────────────────────────────────

class TestDataLoading:

    def test_load_sample_data(self):
        df = load_data("data/sample_data.csv")
        assert 'plot' in df.columns
        assert 'genre' in df.columns
        assert 'clean_plot' in df.columns
        assert len(df) > 0

    def test_split_data_sizes(self):
        df = load_data("data/sample_data.csv")
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
        total = len(X_train) + len(X_test)
        assert total == len(df)
        assert len(X_test) / total == pytest.approx(0.2, abs=0.05)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
