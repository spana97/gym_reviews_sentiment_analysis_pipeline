import pandas as pd

from src.etl.helpers import cast_types, filter_rows, rename_and_select
from src.utils.logger import logger


def transform(df: pd.DataFrame, source: str, config: dict) -> pd.DataFrame:
    """
    Transform raw input DataFrame into clean standardized format.
    """
    logger.info(f"Transforming data from source: {source}")

    mappings = config["rename_mappings"].get(source)
    if not mappings:
        logger.error(
            f'No rename mappings defined for source "{source}"', exc_info=True
        )  # noqa: E501
        raise ValueError(f'No mappings defined for source "{source}"')

    schema = config.get("schema")
    if not schema:
        logger.error("Schema missing in configuration", exc_info=True)
        raise ValueError("Schema missing in configuration")

    max_rating = config.get("filters", {}).get("low_rating_max")
    if max_rating is None:
        logger.error("low_rating_max missing in configuration", exc_info=True)
        raise ValueError("low_rating_max missing in configuration")

    # 1. Rename + keep expected columns
    try:
        df = rename_and_select(df, mappings)
    except Exception:
        logger.error("rename_and_select failed", exc_info=True)
        raise

    # 2. Remove rows without reviews

    if "review" not in df.columns:
        logger.error(
            f"Column 'review' not found in DataFrame after rename for source {source}",  # noqa: E501
            exc_info=True,
        )
        raise ValueError(
            f"Column 'review' not found in DataFrame for source {source}"
        )  # noqa: E501

    df = df.dropna(subset=["review"])

    # 3. Type casting
    try:
        df = cast_types(df, schema)
    except Exception:
        logger.error("cast_types failed", exc_info=True)
        raise

    # 4. Drop duplicates
    df = df.drop_duplicates()

    # 5. Rating filtering
    try:
        df = filter_rows(df, max_rating)
    except Exception:
        logger.error("filter_rows failed", exc_info=True)
        raise

    logger.info(
        f"Transform completed for source {source}: {df.shape[0]} rows, {df.shape[1]} columns"  # noqa: E501
    )
    return df
