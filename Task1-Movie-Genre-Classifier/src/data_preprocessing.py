"""
data_preprocessing.py
---------------------
Text cleaning and preprocessing utilities for movie plot summaries.
"""

import re
import string
import pandas as pd
import numpy as np
import nltk

# ──────────────────────────────────────────────
# Bundled English stop words (NLTK fallback)
# ──────────────────────────────────────────────
_BUNDLED_STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than',
    'too','very','s','t','can','will','just','don','should','now','d','ll',
    'm','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn',
    'weren','won','wouldn',
}

def _get_stopwords():
    try:
        from nltk.corpus import stopwords as nltk_sw
        return set(nltk_sw.words('english'))
    except Exception:
        return _BUNDLED_STOPWORDS

def _tokenize(text: str):
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    except Exception:
        return text.split()

def _lemmatize(token: str):
    try:
        from nltk.stem import WordNetLemmatizer
        return WordNetLemmatizer().lemmatize(token)
    except Exception:
        return token

# Download required NLTK resources (only runs once, fails silently)
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_resources()

# ──────────────────────────────────────────────
# Core Text Cleaner
# ──────────────────────────────────────────────

class TextPreprocessor:
    """
    Cleans and normalizes raw text for NLP tasks.

    Steps:
        1. Lowercase
        2. Remove URLs, HTML tags, special characters
        3. Remove punctuation
        4. Tokenize
        5. Remove stop words
        6. Lemmatize
    """

    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = _get_stopwords()

    def clean_text(self, text: str) -> str:
        """Full cleaning pipeline returning a cleaned string."""
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove non-alphabetic characters (keep spaces)
        text = re.sub(r'[^a-z\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        tokens = _tokenize(text)

        # Remove stop words
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        # Lemmatize
        if self.lemmatize:
            tokens = [_lemmatize(t) for t in tokens]

        return ' '.join(tokens)

    def transform(self, texts):
        """Apply clean_text to a list or Series of texts."""
        if isinstance(texts, pd.Series):
            return texts.apply(self.clean_text)
        return [self.clean_text(t) for t in texts]


# ──────────────────────────────────────────────
# Data Loading & Validation
# ──────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load a CSV dataset with 'plot' and 'genre' columns.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Cleaned DataFrame with 'plot', 'genre', and 'clean_plot' columns.
    """
    df = pd.read_csv(filepath)

    # Validate required columns
    required_cols = {'plot', 'genre'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Drop rows with missing values
    original_len = len(df)
    df = df.dropna(subset=['plot', 'genre']).reset_index(drop=True)
    dropped = original_len - len(df)
    if dropped > 0:
        print(f"⚠️  Dropped {dropped} rows with missing values.")

    # Preprocess text
    preprocessor = TextPreprocessor()
    df['clean_plot'] = preprocessor.transform(df['plot'])

    print(f"✅ Loaded {len(df)} samples across {df['genre'].nunique()} genres.")
    print(f"   Genres: {sorted(df['genre'].unique())}")
    return df


def get_genre_distribution(df: pd.DataFrame) -> pd.Series:
    """Return value counts of genres sorted descending."""
    return df['genre'].value_counts()


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train/test split.

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    X = df['clean_plot']
    y = df['genre']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/sample_data.csv"
    df = load_data(filepath)
    print("\nGenre Distribution:")
    print(get_genre_distribution(df))
