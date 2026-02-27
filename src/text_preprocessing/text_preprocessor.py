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
    """Clean and preprocess text data for NLP tasks."""

    def __init__(self, extra_stopwords: set[str] | None = None):
        """
        Initializes the TextPreprocessor with optional extra stopwords.

        Args:
            extra_stopwords (set[str] | None): Set of additional stopwords.
                Defaults to None.
        """
        logger.info("Initializing TextPreprocessor...")
        ensure_nltk_resources()
        self.stop_words = get_stopwords(extra_stopwords)
        self.lemmatizer = WordNetLemmatizer()
        logger.info("TextPreprocessor initialized successfully.")

    def preprocess(self, text: str) -> str:
        """
        Cleans text, tokenizes, removes stopwords and lemmatizes text.

        Args:
            text (str): Text data to be cleaned.

        Returns:
            str: Preprocessed and lemmatized text as a single string.
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
