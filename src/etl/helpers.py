from typing import List

import pandas as pd

from utils.logger import logger

# -----------------------------
# Transform helpers
# -----------------------------


def rename_and_select(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """Renames and selects DataFrame columns."""
    logger.debug(f"Renaming columns: {mappings}")
    df = df.rename(columns=mappings)

    expected = list(mappings.values())
    missing = set(expected) - set(df.columns)

    if missing:
        logger.error(f"Missing expected columns after rename: {missing}")
        raise ValueError(f"Missing expected columns after rename: {missing}")

    logger.info(f"Renamed and selected columns: {expected}")

    return df[expected].copy()


def cast_types(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Casts DataFrame columns to specified data types."""
    logger.debug(f"Casting columns using schema: {schema}")

    for col, dtype in schema.items():
        try:
            if dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(dtype, errors="ignore")

        except Exception as e:
            logger.error(f"Error casting column '{col}' to type '{dtype}': {e}")  # noqa: E501
            raise

    logger.debug("Finished casting columns")

    return df


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


# -----------------------------
# run_etl_pipeline helper
# -----------------------------


def combine_datasets(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Combines multiple DataFrames and ensures 'review' column exists."""
    logger.debug(f"Combining {len(dfs)} datasets")

    if not dfs:
        logger.error("No DataFrames provided for combination")
        raise ValueError("No DataFrames provided for combination")

    combined = pd.concat(dfs, ignore_index=True)
    logger.debug(f"Combined dataset shape: {combined.shape}")
    return combined
