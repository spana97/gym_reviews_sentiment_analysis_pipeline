import pandas as pd

from src.utils.logger import logger


def extract(path: str) -> pd.DataFrame:
    """
    Load CSV into a DataFrame
    """
    try:
        df = pd.read_csv(path)
        logger.info(
            f"Loaded {path}: {df.shape[0]} rows, {df.shape[1]} columns"
        )  # noqa: E501
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}", exc_info=True)
        raise
    except pd.errors.ParserError:
        logger.error(f"Error parsing CSV from path: {path}", exc_info=True)
        raise
    except Exception:
        logger.error(f"Error loading CSV from path: {path}", exc_info=True)
        raise
