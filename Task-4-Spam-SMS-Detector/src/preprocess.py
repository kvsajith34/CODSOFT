"""
Text preprocessing utilities for SMS Spam Detection.
Handles cleaning, tokenization, and stopword removal.
Uses a built-in stopword list so no NLTK download is required.
"""

import re
import string
import pandas as pd

# Built-in English stopwords (no NLTK dependency)
STOP_WORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than","too",
    "very","s","t","can","will","just","don","should","now","d","ll","m","o",
    "re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn","haven",
    "isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won",
    "wouldn","u","ur","r","2","4","gt","lt",
}

# Minimal Porter stemmer (suffix stripping — no NLTK needed)
class _SimpleStemmer:
    _suffixes = ["ing","tion","ly","er","es","ed","s"]

    def stem(self, word: str) -> str:
        for suf in self._suffixes:
            if word.endswith(suf) and len(word) - len(suf) > 2:
                return word[: -len(suf)]
        return word


stemmer = _SimpleStemmer()


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline:
      1. Lowercase
      2. Remove URLs
      3. Remove phone numbers
      4. Remove punctuation & digits
      5. Strip extra whitespace
      6. Remove stopwords
      7. Apply stemming
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\b\d{7,}\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def load_and_prepare(filepath: str) -> pd.DataFrame:
    """
    Load the raw SMS spam CSV, keep only the label and text columns,
    rename them, encode labels (spam=1, ham=0), and apply text cleaning.
    Returns a clean DataFrame with columns: label, label_num, text, clean_text.
    """
    df = pd.read_csv(filepath, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["label_num"] = df["label"].map({"spam": 1, "ham": 0})
    df["clean_text"] = df["text"].apply(clean_text)
    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/spam.csv"
    df = load_and_prepare(path)
    print(df.head())
    print(f"\nShape : {df.shape}")
    print(f"Spam  : {df['label_num'].sum()}")
    print(f"Ham   : {(df['label_num'] == 0).sum()}")
