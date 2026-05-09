"""
Unit & integration tests for the SMS Spam Detection pipeline.

Run with:
    pytest tests/ -v
"""

import os
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocess import clean_text, load_and_prepare


# ── preprocess tests ─────────────────────────────────────────────────────────

class TestCleanText:
    def test_lowercases(self):
        assert clean_text("HELLO WORLD") == clean_text("hello world")

    def test_removes_url(self):
        result = clean_text("Visit https://spam.com to claim your prize")
        assert "http" not in result and "spam.com" not in result

    def test_removes_punctuation(self):
        result = clean_text("Hello!!! How are you???")
        assert "!" not in result and "?" not in result

    def test_removes_stopwords(self):
        result = clean_text("this is a test message")
        # "this", "is", "a" are stopwords
        assert "this" not in result.split()
        assert "is"   not in result.split()

    def test_empty_string(self):
        result = clean_text("")
        assert result == ""

    def test_returns_string(self):
        assert isinstance(clean_text("some text"), str)


class TestLoadAndPrepare:
    DATA_PATH = os.path.join(ROOT, "data", "spam.csv")

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(ROOT, "data", "spam.csv")),
        reason="spam.csv not found"
    )
    def test_columns(self):
        df = load_and_prepare(self.DATA_PATH)
        for col in ["label", "text", "label_num", "clean_text"]:
            assert col in df.columns, f"Missing column: {col}"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(ROOT, "data", "spam.csv")),
        reason="spam.csv not found"
    )
    def test_labels_binary(self):
        df = load_and_prepare(self.DATA_PATH)
        assert set(df["label_num"].unique()).issubset({0, 1})

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(ROOT, "data", "spam.csv")),
        reason="spam.csv not found"
    )
    def test_no_nulls(self):
        df = load_and_prepare(self.DATA_PATH)
        assert df.isnull().sum().sum() == 0


# ── prediction tests (requires trained model) ────────────────────────────────

class TestPredict:
    MODEL_PATH = os.path.join(ROOT, "models", "best_model.pkl")

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(ROOT, "models", "best_model.pkl")),
        reason="Model not trained yet"
    )
    def test_spam_message(self):
        from src.predict import predict_message, load_model
        model = load_model(self.MODEL_PATH)
        result = predict_message(
            "Congratulations! You've won a FREE prize. Call now!", model
        )
        assert result["prediction"] == "spam"
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(ROOT, "models", "best_model.pkl")),
        reason="Model not trained yet"
    )
    def test_ham_message(self):
        from src.predict import predict_message, load_model
        model = load_model(self.MODEL_PATH)
        result = predict_message("See you at the meeting tomorrow morning.", model)
        assert result["prediction"] == "ham"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(ROOT, "models", "best_model.pkl")),
        reason="Model not trained yet"
    )
    def test_result_keys(self):
        from src.predict import predict_message, load_model
        model  = load_model(self.MODEL_PATH)
        result = predict_message("Test message", model)
        for key in ("text", "prediction", "confidence"):
            assert key in result


# ── Flask app tests ───────────────────────────────────────────────────────────

class TestFlaskApp:
    @pytest.fixture
    def client(self):
        sys.path.insert(0, ROOT)
        import importlib.util, importlib.machinery
        spec = importlib.util.spec_from_file_location(
            "flask_app", os.path.join(ROOT, "app", "app.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        flask_app = mod.app
        flask_app.config["TESTING"] = True
        with flask_app.test_client() as c:
            yield c

    def test_index(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.get_json()["status"] == "ok"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(ROOT, "models", "best_model.pkl")),
        reason="Model not trained yet"
    )
    def test_predict_endpoint(self, client):
        r = client.post(
            "/predict",
            json={"text": "You have won a free holiday!"},
            content_type="application/json",
        )
        assert r.status_code == 200
        data = r.get_json()
        assert "prediction" in data

    def test_predict_missing_text(self, client):
        r = client.post("/predict", json={}, content_type="application/json")
        assert r.status_code == 400
