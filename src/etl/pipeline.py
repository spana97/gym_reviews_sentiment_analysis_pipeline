from src.etl.extract import extract
from src.etl.helpers import combine_datasets
from src.etl.load import load
from src.etl.transform import transform
from src.utils.config_loader import load_config
from src.utils.logger import logger


def run_etl_pipeline():
    """
    Runs ETL pipeline:
    1. Extracts raw Google and Trustpilot reviews.
    2. Transforms each dataset.
    3. Combines datasets.
    4. Saves as a parquet.
    """
    logger.info("Starting ETL pipeline")

    config = load_config()

    # Extract
    try:
        google_raw = extract(config["data"]["raw_google"])
        trust_raw = extract(config["data"]["raw_trustpilot"])
    except Exception as e:
        logger.error(f"Error extracting raw datasets: {e}")
        raise
    logger.info("Datasets extracted successfully")

    # Transform
    try:
        logger.info("Transforming datasets")
        google_clean = transform(google_raw, "google", config)
        trustpilot_clean = transform(trust_raw, "trustpilot", config)
    except Exception as e:
        logger.error(f"Error transforming datasets: {e}")
        raise
    logger.info("Datasets transformed successfully")

    # Combine datasets
    try:
        logger.info("Combining datasets")
        combined = combine_datasets([google_clean, trustpilot_clean])
    except Exception as e:
        logger.error(f"Error combining datasets: {e}")
        raise
    logger.info("Datasets combined successfully")

    # Load
    try:
        logger.info("Loading combined dataset")
        load(combined, config["data"]["processed_output"])
    except Exception as e:
        logger.error(f"Error loading combined dataset: {e}")
        raise

    logger.info("ETL pipeline completed successfully")
