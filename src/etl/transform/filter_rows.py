import pandas as pd

from utils.logger import logger


def filter_rows(df: pd.DataFrame, max_score: int) -> pd.DataFrame:
    """Filters df for rows with a score less than or equal to max_score."""
    logger.debug(f"Filtering rows with score <= {max_score}")

    if "score" not in df:
        logger.error("Column 'score' not found in DataFrame")
        raise KeyError("Column 'score' is required for filtering")

    before = len(df)
    df = df[df["score"] <= max_score].copy()
    after = len(df)

    logger.info(f"Filtered rows with score <= {max_score}: {before} -> {after} rows")  # noqa: E501

    if after == 0:
        logger.warning(f"No rows left after filtering with max_score={max_score}")  # noqa: E501

    return df
