import pandas as pd
from utils.logger import logger


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
