import re
import ssl
from typing import Optional, Set

import certifi
import nltk

from src.utils.logger import logger


def _create_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


ssl._create_default_https_context = _create_ssl_context  # type: ignore[assignment]  # noqa: E501

nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk import pos_tag  # noqa: E402
from nltk.corpus import stopwords, wordnet  # noqa: E402
from nltk.tokenize import word_tokenize  # noqa: E402


# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text: str) -> str:
    """
    Clean the input text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing numbers
    4. Removing extra whitespace
    """
    try:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        logger.error(f"Error cleaning text: {text} - {e}")
        return ""
    return text


# -----------------------------
# Text tokenizer
# -----------------------------


def tokenize_text(text: str) -> list:
    """
    Tokenize the input text.
    """
    try:
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]
    except Exception as e:
        logger.error(f"Error tokenizing text: {text} - {e}")
        tokens = []

    return tokens


# -----------------------------
# Stopword helper
# -----------------------------


def get_stopwords(extra_words: Optional[Set[str]] = None) -> Set[str]:
    """
    Returns a set of English stopwords with optional extra words.
    """
    stop_words = set(stopwords.words("english"))
    if extra_words:
        stop_words.update(extra_words)
    return stop_words


# -----------------------------
# Remove stopwords
# -----------------------------


def remove_stopwords(tokens: list, stop_words: Set[str]) -> list:
    """
    Remove stopwords from the list of tokens.
    """
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens


# -----------------------------
# POS Tagging helper
# -----------------------------


def _map_pos(tag: str) -> str:
    """
    Internal helper function.
    Maps NLTK POS tags to WordNet POS tags for lemmatization.
    """
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return tag_dict.get(tag[0].upper(), wordnet.NOUN)


# -----------------------------
# Text Lemmatizer
# -----------------------------


def lemmatize_tokens(tokens: list, lemmatizer) -> list:
    """
    Lemmatize the input tokens using POS tagging.
    """
    tagged = pos_tag(tokens)
    return [lemmatizer.lemmatize(word, _map_pos(tag)) for word, tag in tagged]


# -----------------------------
# Ensure NLTK resources
# -----------------------------


def ensure_nltk_resources() -> None:
    required_resources = [
        ("corpora", "stopwords"),
        ("tokenizers", "punkt"),
        ("corpora", "wordnet"),
        ("taggers", "averaged_perceptron_tagger"),
    ]
    for resource_type, resource_name in required_resources:
        try:
            nltk.data.find(f"{resource_type}/{resource_name}")
        except LookupError:
            nltk.download(resource_name, quiet=True)
