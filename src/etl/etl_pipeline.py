from etl.extract_service import extract_dataset
from etl.transform.transform_service import transform_dataset
from etl.combine_service import combine_datasets
from etl.load_service import load_dataset
from utils.config_loader import load_config
from utils.logger import logger


def run_etl_pipeline():
    """Orchestrate full ETL for reviews."""
    logger.info("Starting ETL pipeline")

    config = load_config()

    datasets = {}
    for source in ["google", "trustpilot"]:
        raw_path = config["data"][f"raw_{source}"]
        datasets[source] = extract_dataset(raw_path)

    for source in datasets:
        datasets[source] = transform_dataset(datasets[source], source, config)

    combined = combine_datasets(list(datasets.values()))

    load_dataset(combined, config["data"]["etl_output"])

    logger.info("ETL pipeline completed successfully")

    return combined
