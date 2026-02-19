from src.etl.extract import extract
from src.etl.helpers import combine_datasets
from src.etl.load import load
from src.etl.transform import transform
from src.utils.config_loader import load_config


def run_etl_pipeline():
    """
    Runs ETL pipeline:
    1. Extracts raw Google and Trustpilot reviews.
    2. Transforms each dataset.
    3. Combines datasets.
    4. Saves as a parquet.
    """
    print("Running ETL...")
    config = load_config()

    # Extract
    google_raw = extract(config["data"]["raw_google"])
    trust_raw = extract(config["data"]["raw_trustpilot"])

    # Transform
    google_clean = transform(google_raw, "google", config)
    trustpilot_clean = transform(trust_raw, "trustpilot", config)

    combined = combine_datasets([google_clean, trustpilot_clean])

    # Load
    load(combined, config["data"]["processed_output"])

    print("ETL pipeline finished successfully")


if __name__ == "__main__":
    run_etl_pipeline()
