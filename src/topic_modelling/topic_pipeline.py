import os

import pandas as pd

from text_preprocessing.text_preprocessor import TextPreprocessor
from topic_modelling.format_and_save_clusters import format_and_save_clusters
from topic_modelling.topic_model import TopicModel
from utils.logger import logger


def run_topic_model_pipeline(config: dict):
    logger.info("Starting topic model pipeline")

    try:
        df = pd.read_parquet(config["data"]["etl_output"])
    except FileNotFoundError as e:
        logger.error(f"Input data file not found: {config['data']['etl_output']} | {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        raise

    preprocessor = TextPreprocessor(
        extra_stopwords=config["text_preprocessing"]["extra_stopwords"]
    )
    topic_text = df["review"].apply(preprocessor.preprocess).tolist()

    try:
        model_path = config["topic_model"]["model_output"]

        if os.path.exists(model_path):
            logger.info(f"Loading existing model from {model_path}")
            topic_model = TopicModel(config["topic_model"])
            topic_model.load(model_path)
            topics, probs = topic_model.transform(topic_text)

        else:
            logger.info(f"No existing model found at {model_path} — fitting new model.")
            topic_model = TopicModel(config["topic_model"])
            topics, probs = topic_model.fit(topic_text)
            topic_model.save(model_path)

        topic_info = topic_model.get_topic_info()
        topic_info.to_csv(config["topic_model"]["topics_output"], index=False)
        logger.info(f"Topic info saved to {config['topic_model']['topics_output']}")

        top5_topics = topic_info[topic_info["Topic"] != -1]["Representative_Docs"].iloc[
            :5
        ]

        clusters = format_and_save_clusters(
            top5_topics, config["topic_model"]["representative_docs_output"]
        )

        if clusters is None:
            logger.error(
                "format_and_save_clusters returned None — check upstream topic_info."
            )
            raise ValueError("Failed to format representative documents into clusters.")

    except KeyError as e:
        logger.error(f"Missing config key — check your config dict: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during topic modeling pipeline: {e}")
        raise

    logger.info("Topic model pipeline executed successfully.")
