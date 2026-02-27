import json
import pandas as pd

from utils.logger import logger


def format_json(topic_info: pd.Series) -> str | None:
    """
    Format BERTopic topic information into a JSON structure.

    Args:
        topic_info (pd.Series): Representative documents for each topic.

    Returns:
        str | None: JSON string of the form {"cluster_1": [...], "cluster_2": [...]}
            Returns None if formatting fails.
    """
    logger.info("Formatting topic information into JSON structure...")
    try:
        topics_docs = topic_info.tolist()
    except AttributeError as e:
        logger.error(f"Error converting topic information to list: {e}")
        return None

    clusters = {f"cluster_{i + 1}": docs for i, docs in enumerate(topics_docs)}

    try:
        formatted_input = json.dumps(clusters, indent=2)
    except TypeError as e:
        logger.error(f"Error converting clusters to JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during JSON formatting: {e}")
        return None

    logger.info("Topic information formatted successfully.")
    return formatted_input
