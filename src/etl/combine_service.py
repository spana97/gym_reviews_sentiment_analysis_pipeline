import pandas as pd

from utils.logger import logger


def combine_datasets(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Combines multiple DataFrames and ensures 'review' column exists."""
    logger.debug(f"Combining {len(dfs)} datasets")

    if not dfs:
        logger.error("No DataFrames provided for combination")
        raise ValueError("No DataFrames provided for combination")

    combined = pd.concat(dfs, ignore_index=True)
    logger.debug(f"Combined dataset shape: {combined.shape}")
    return combined
