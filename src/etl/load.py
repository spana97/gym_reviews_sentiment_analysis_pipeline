from pathlib import Path

from utils.logger import logger


def load(df, output_path: str) -> Path:
    """
    Save dataframe to parquet.
    """
    logger.info(f"Loading data to {output_path}")

    try:
        output = Path(output_path)
    except Exception as e:
        logger.error(f"Failed to create output path: {e}")
        raise

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        raise

    df.to_parquet(output)

    logger.info(f"Load successful: Saved {len(df)} rows -> {output}")
    return output
