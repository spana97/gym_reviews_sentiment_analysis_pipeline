import os

import pandas as pd

from etl.extract import extract
from etl.helpers import combine_datasets
from etl.load import load
from etl.transform import transform


def test_run_etl_pipeline(tmp_path, test_config):

    google_test_df = extract(test_config["data"]["test_google"])
    trustpilot_test_df = extract(test_config["data"]["test_trustpilot"])

    google_clean = transform(google_test_df, "google", test_config)
    trustpilot_clean = transform(trustpilot_test_df, "trustpilot", test_config)
    combined = combine_datasets([google_clean, trustpilot_clean])

    output_path = tmp_path / "combined_test.parquet"
    load(combined, output_path)

    expected_columns = [
        "source",
        "location",
        "date_created",
        "review",
        "score",
    ]

    assert not combined.empty
    assert len(combined) == 6
    assert combined.columns.tolist() == expected_columns
    assert combined["score"].dtype == "int64"
    assert combined["review"].dtype == "string"
    assert pd.api.types.is_datetime64_any_dtype(combined["date_created"])
    assert all(combined["score"] <= test_config["filters"]["low_rating_max"])
    assert (combined["review"].isnull().sum() == 0).all()
    assert combined.duplicated().sum() == 0
    assert os.path.exists(output_path)

    saved_df = pd.read_parquet(output_path)
    assert saved_df.shape == combined.shape
