from nltk.stem import WordNetLemmatizer
import re
import ssl

import certifi
import nltk

from utils.logger import logger


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
    """Lowercase, remove punctuation, numbers and extra whitespace."""
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
    """Tokenizes the input text."""
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


def get_stopwords(extra_words: set[str] | None = None) -> set[str]:
    """Returns a set of English stopwords with optional extra words."""
    stop_words = set(stopwords.words("english"))
    if extra_words:
        stop_words.update(extra_words)
    return stop_words


# -----------------------------
# Remove stopwords
# -----------------------------


def remove_stopwords(tokens: list, stop_words: set[str]) -> list:
    """Remove stopwords from a list of tokens."""
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens


# -----------------------------
# POS Tagging helper
# -----------------------------


def _map_pos(tag: str) -> str:
    """Maps NLTK POS tags to WordNet POS tags for lemmatization."""
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


def lemmatize_tokens(tokens: list, lemmatizer: WordNetLemmatizer) -> list[str]:
    """Lemmatize input tokens using POS tagging."""
    tagged = pos_tag(tokens)
    return [lemmatizer.lemmatize(word, _map_pos(tag)) for word, tag in tagged]


# -----------------------------
# Ensure NLTK resources
# -----------------------------


def ensure_nltk_resources() -> None:
    """Checks whether NLTK resources are downloaded and download if required."""
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
