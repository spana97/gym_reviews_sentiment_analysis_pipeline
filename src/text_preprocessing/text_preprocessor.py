from typing import Optional, Set

from nltk.stem import WordNetLemmatizer

from text_preprocessing.helpers import (
    clean_text,
    ensure_nltk_resources,
    get_stopwords,
    lemmatize_tokens,
    remove_stopwords,
    tokenize_text,
)
from utils.logger import logger


class TextPreprocessor:
    """
    Class to handle text preprocessing steps for gym reviews analysis.
    """

    def __init__(self, extra_stop_words: Optional[Set[str]] = None):
        logger.info("Initializing TextPreprocessor...")
        ensure_nltk_resources()
        self.stop_words = get_stopwords(extra_stop_words)
        self.lemmatizer = WordNetLemmatizer()
        logger.info("TextPreprocessor initialized successfully.")

    def preprocess(self, text: str) -> str:
        """
        Preprocess the input text by:
        1. Cleaning the text
        2. Tokenizing the text
        3. Removing stopwords
        4. Lemmatizing the tokens
        """
        try:
            cleaned_text = clean_text(text)
            tokens = tokenize_text(cleaned_text)
            filtered_tokens = remove_stopwords(tokens, self.stop_words)
            lemmas = lemmatize_tokens(filtered_tokens, self.lemmatizer)
        except Exception as e:
            logger.error(f"Error preprocessing text: {text} - {e}")
            return ""

        return " ".join(lemmas)
