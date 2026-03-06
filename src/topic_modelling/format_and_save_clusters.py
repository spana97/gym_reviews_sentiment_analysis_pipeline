import json
from pathlib import Path

import pandas as pd

from utils.logger import logger


def format_and_save_clusters(topic_info: pd.Series, output_path: str) -> dict | None:
    """
    Format BERTopic topic information into a JSON structure and save to disk.

    Args:
        topic_info (pd.Series): Representative documents for each topic (first 5 rows).
        output_path (str): File path to save the JSON output.

    Returns:
        dict | None: Cluster dictionary. Returns None if formatting fails.
    """
    logger.info("Formatting topic information into JSON structure...")

    try:
        topics_docs = topic_info.tolist()
    except AttributeError as e:
        logger.error(
            f"Failed to convert topic_info to list — expected a pd.Series, got {type(topic_info)}: {e}"
        )
        return None

    try:
        clusters = {f"cluster_{i + 1}": docs for i, docs in enumerate(topics_docs)}
    except Exception as e:
        logger.error(f"Unexpected error building clusters dict: {e}")
        return None

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(clusters, f, indent=2)
        logger.info(f"Clusters saved successfully to {output_path}.")
    except FileNotFoundError as e:
        logger.error(
            f"Output path not found — check that the directory exists: {output_path} | {e}"
        )
        raise

    return clusters
