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
    logger.info("Extracting raw datasets")
    google_raw = extract(config["data"]["raw_google"])
    trust_raw = extract(config["data"]["raw_trustpilot"])

    # Transform
    logger.info("Transforming datasets")
    google_clean = transform(google_raw, "google", config)
    trustpilot_clean = transform(trust_raw, "trustpilot", config)

    logger.info("Combining datasets")
    combined = combine_datasets([google_clean, trustpilot_clean])

    # Load
    logger.info("Loading combined dataset")
    load(combined, config["data"]["processed_output"])

    logger.info("ETL pipeline completed successfully")
