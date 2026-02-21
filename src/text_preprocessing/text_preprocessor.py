from typing import Optional, Set

import nltk
from nltk.stem import WordNetLemmatizer

from .helpers import (
    clean_text,
    get_stopwords,
    lemmatize_tokens,
    remove_stopwords,
    tokenize_text,
)

REQUIRED_RESOURCES = [
    ("corpora", "stopwords"),
    ("tokenizers", "punkt"),
    ("corpora", "wordnet"),
    ("taggers", "averaged_perceptron_tagger"),
]

for resource_type, resource_name in REQUIRED_RESOURCES:
    try:
        nltk.data.find(f"{resource_type}/{resource_name}")
    except LookupError:
        nltk.download(resource_name, quiet=True)


class TextPreprocessor:
    """
    Class to handle text preprocessing steps for gym reviews analysis.
    """

    def __init__(self, extra_stop_words: Optional[Set[str]] = None):
        self.stop_words = get_stopwords(extra_stop_words)
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str) -> str:
        """
        Preprocess the input text by:
        1. Cleaning the text
        2. Tokenizing the text
        3. Removing stopwords
        4. Lemmatizing the tokens
        """
        cleaned_text = clean_text(text)
        tokens = tokenize_text(cleaned_text)
        filtered_tokens = remove_stopwords(tokens, self.stop_words)
        lemmas = lemmatize_tokens(filtered_tokens, self.lemmatizer)

        return " ".join(lemmas)
