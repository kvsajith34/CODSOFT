"""
feature_extraction.py
---------------------
Feature extraction using TF-IDF and Word Embeddings (Word2Vec).
"""

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


# ──────────────────────────────────────────────
# TF-IDF Feature Extractor
# ──────────────────────────────────────────────

class TFIDFExtractor:
    """
    Wraps sklearn's TfidfVectorizer with convenient fit/transform/save methods.

    Parameters:
        max_features:   Maximum vocabulary size
        ngram_range:    Tuple (min_n, max_n) for n-gram range
        sublinear_tf:   Apply sublinear TF scaling (log(1+tf))
    """

    def __init__(self, max_features: int = 10000, ngram_range=(1, 2), sublinear_tf: bool = True):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=1,
            max_df=0.95,
            strip_accents='unicode',
            analyzer='word',
        )
        self.is_fitted = False

    def fit(self, texts):
        """Fit the vectorizer on training texts."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        print(f"📝 TF-IDF fitted | Vocabulary size: {vocab_size:,}")
        return self

    def transform(self, texts):
        """Transform texts to TF-IDF feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("TFIDFExtractor must be fitted before transform.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def save(self, path: str):
        joblib.dump(self.vectorizer, path)
        print(f"💾 TF-IDF vectorizer saved → {path}")

    @classmethod
    def load(cls, path: str):
        extractor = cls.__new__(cls)
        extractor.vectorizer = joblib.load(path)
        extractor.is_fitted = True
        return extractor


# ──────────────────────────────────────────────
# Word Embedding Feature Extractor
# ──────────────────────────────────────────────

class WordEmbeddingExtractor:
    """
    Trains a Word2Vec model and represents documents as the
    mean vector of their word embeddings.

    Parameters:
        vector_size:   Dimensionality of word vectors
        window:        Context window size
        min_count:     Minimum word frequency
        workers:       Number of CPU cores to use
    """

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, workers: int = 4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        self.is_fitted = False

    def _tokenize(self, texts):
        return [text.split() for text in texts]

    def fit(self, texts):
        """Train Word2Vec on the provided texts."""
        try:
            from gensim.models import Word2Vec
        except ImportError:
            raise ImportError("gensim is required for Word Embeddings. Run: pip install gensim")

        tokenized = self._tokenize(texts)
        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=10,
        )
        self.is_fitted = True
        print(f"🧠 Word2Vec fitted | Vocab: {len(self.model.wv):,} | Dim: {self.vector_size}")
        return self

    def _document_vector(self, tokens):
        """Average word vectors for tokens in the model vocabulary."""
        vectors = [
            self.model.wv[word]
            for word in tokens
            if word in self.model.wv
        ]
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.vector_size)

    def transform(self, texts):
        """Transform texts into document-level embedding matrix."""
        if not self.is_fitted:
            raise RuntimeError("WordEmbeddingExtractor must be fitted before transform.")
        tokenized = self._tokenize(texts)
        return np.vstack([self._document_vector(tokens) for tokens in tokenized])

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: str):
        if self.model:
            self.model.save(path)
            print(f"💾 Word2Vec model saved → {path}")

    @classmethod
    def load(cls, path: str):
        from gensim.models import Word2Vec
        extractor = cls.__new__(cls)
        extractor.model = Word2Vec.load(path)
        extractor.vector_size = extractor.model.vector_size
        extractor.is_fitted = True
        return extractor


# ──────────────────────────────────────────────
# Feature Extractor Factory
# ──────────────────────────────────────────────

def get_extractor(feature_type: str = 'tfidf', **kwargs):
    """
    Factory function to get the correct feature extractor.

    Args:
        feature_type: 'tfidf' or 'embeddings'
        **kwargs: Additional arguments passed to the extractor

    Returns:
        An instance of TFIDFExtractor or WordEmbeddingExtractor
    """
    extractors = {
        'tfidf': TFIDFExtractor,
        'embeddings': WordEmbeddingExtractor,
    }
    if feature_type not in extractors:
        raise ValueError(f"Unknown feature type '{feature_type}'. Choose from: {list(extractors.keys())}")
    return extractors[feature_type](**kwargs)


if __name__ == "__main__":
    # Quick test
    sample_texts = [
        "astronaut travel wormhole saturn new home humanity",
        "young wizard discover magical power school sorcerer",
        "detective investigate murder small town dark ritual",
    ]

    print("=== TF-IDF Test ===")
    tfidf = TFIDFExtractor(max_features=100)
    matrix = tfidf.fit_transform(sample_texts)
    print(f"Matrix shape: {matrix.shape}")

    print("\n=== Word Embedding Test ===")
    emb = WordEmbeddingExtractor(vector_size=50)
    vectors = emb.fit_transform(sample_texts)
    print(f"Vectors shape: {vectors.shape}")
