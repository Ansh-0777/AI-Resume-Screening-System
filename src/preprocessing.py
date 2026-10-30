"""
preprocessing.py
Text cleaning pipeline: lowercase, punctuation removal, stopword removal,
tokenization, and lemmatization. Uses NLTK; downloads required corpora on
first run (falls back gracefully if offline).
"""

import re
import string

import nltk

_REQUIRED_NLTK = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]


def ensure_nltk_data():
    for pkg in _REQUIRED_NLTK:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


ensure_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    STOPWORDS = set()

lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/digits/extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    try:
        return word_tokenize(text)
    except LookupError:
        return text.split()


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def lemmatize(tokens: list) -> list:
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str) -> str:
    """Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize -> rejoin."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)
