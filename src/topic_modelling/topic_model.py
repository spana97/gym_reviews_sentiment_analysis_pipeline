from pathlib import Path
import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from utils.logger import logger


class TopicModel:
    """Wraps BERTopic to fit, save, load and extract topic information."""

    def __init__(self, config: dict):
        """
        Initializes TopicModel with configuration.

        Args:
            config (dict): Configuration dictionary containing
                topic_model parameters, model and data save paths.
        """
        logger.info(f"Initializing BERTopic model with config: {config}")
        self.config = config

        try:
            self.embed_model = SentenceTransformer(self.config["embed_model"])
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
        logger.info("Embedding model initialized successfully.")

        try:
            self.topic_model = BERTopic(
                language=self.config["language"],
                calculate_probabilities=self.config["calculate_probabilities"],
                verbose=self.config["verbose"],
                embedding_model=self.embed_model,
                low_memory=self.config["low_memory"],
                nr_topics=self.config["nr_topics"],
            )
        except Exception as e:
            logger.error(f"Error initializing BERTopic model: {e}")
            raise

        logger.info("BERTopic model initialized successfully.")

    def fit(self, documents: list[str]) -> tuple[list, list]:
        """
        Fit the BERTopic model to the documents.

        Args:
            documents (list): List of documents for topic modelling.

        Returns:
            tuple[list, list]: Topic assignments and their probabilities.
        """
        logger.info("Fitting BERTopic model to documents...")

        try:
            topics, probabilities = self.topic_model.fit_transform(documents)
        except Exception as e:
            logger.error(f"Error fitting BERTopic model: {e}")
            raise

        logger.info("BERTopic model fitted successfully.")
        return topics, probabilities

    def save(self, path: str):
        """Save the BERTopic model to the specified path."""
        logger.info(f"Saving BERTopic model to {path}...")
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.topic_model.save(path)
        except Exception as e:
            logger.error(f"Error saving BERTopic model: {e}")
            raise
        logger.info(f"BERTopic model saved successfully to {path}.")

    def load(self, path: str):
        """Load the BERTopic model from the specified path."""
        logger.info(f"Loading BERTopic model from {path}...")
        try:
            self.topic_model = BERTopic.load(path)
        except Exception as e:
            logger.error(f"Error loading BERTopic model: {e}")
            raise
        logger.info(f"BERTopic model loaded successfully from {path}.")

    def get_topic_info(self) -> pd.DataFrame:
        """Get the topic information from the BERTopic model."""
        logger.info("Getting topic information from BERTopic model...")
        try:
            topic_info = self.topic_model.get_topic_info()
        except Exception as e:
            logger.error(f"Error getting topic information: {e}")
            raise
        logger.info("Topic information retrieved successfully.")
        return topic_info
