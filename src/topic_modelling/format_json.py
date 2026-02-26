import json

from src.utils.logger import logger


def format_json(topic_info):
    """
    Format the topic modelling information from BERTopic into a JSON structure.
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
