import re
from typing import Optional, Set

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize


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
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# Text tokenizer
# -----------------------------


def tokenize_text(text: str) -> list:
    """
    Tokenize the input text.
    """
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha()]
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
