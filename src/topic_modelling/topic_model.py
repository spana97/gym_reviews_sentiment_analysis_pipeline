from pathlib import Path

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


class TopicModel:
    """
    BERTopic class for topic modelling.
    """

    def __init__(self, config: dict):
        print("Initializing BERTopic model...")
        self.config = config
        self.topic_model = BERTopic(
            language=self.config["language"],
            calculate_probabilities=self.config["calculate_probabilities"],
            verbose=self.config["verbose"],
            embedding_model=SentenceTransformer(self.config["embed_model"]),
            low_memory=self.config["low_memory"],
            nr_topics=self.config["nr_topics"],
        )
        print("BERTopic model initialized successfully.")

    def fit_transform(self, documents: list):
        """
        Fit the BERTopic model to the documents.
        Returns the topics and probabilities.
        """
        print("Fitting BERTopic model to documents...")
        topics, probabilities = self.topic_model.fit_transform(documents)
        print("BERTopic model fitted successfully.")
        return topics, probabilities

    def save(self, path: str):
        """
        Save the BERTopic model to the specified path.
        """
        print(f"Saving BERTopic model to {path}...")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.topic_model.save(path)
        print(f"BERTopic model saved successfully to {path}.")

    def load(self, path: str):
        """
        Load the BERTopic model from the specified path.
        """
        print(f"Loading BERTopic model from {path}...")
        self.topic_model = BERTopic.load(path)
        print(f"BERTopic model loaded successfully from {path}.")

    def get_topic_model(self):
        return self.topic_model
