import json

import pandas as pd

from src.utils.logger import logger


def parse_insights(raw_response: str) -> pd.DataFrame:
    """
    Parses the response from the OpenAI API and converts it into a DataFrame.
    """
    logger.info("Parsing insights from raw response.")

    try:
        cleaned = raw_response.strip().strip("```json").strip("```").strip()
    except Exception as e:
        logger.error(f"Error cleaning raw response: {e}")
        raise
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON: {e}")
        raise

    logger.info("Raw response successfully parsed into JSON.")
    return pd.DataFrame(data)
