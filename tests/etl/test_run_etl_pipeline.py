import pandas as pd
from src.etl.extract import extract
from src.etl.transform import transform
from src.etl.helpers import combine_datasets
from src.etl.load import load
from tests.test_config import TEST_CONFIG, GOOGLE_TEST_DF, TRUSTPILOT_TEST_DF
import os

def test_run_etl_pipeline(tmp_path):

    google_test_df = GOOGLE_TEST_DF.copy()
    trustpilot_test_df = TRUSTPILOT_TEST_DF.copy()

    google_clean = transform(google_test_df, 'google', TEST_CONFIG)
    trustpilot_clean = transform(trustpilot_test_df, 'trustpilot', TEST_CONFIG)
    combined = combine_datasets([google_clean, trustpilot_clean])

    output_path = tmp_path / "combined_test.parquet"
    load(combined, output_path)

    expected_columns = ['source', 'location', 'date_created', 'review', 'score']

    assert not combined.empty, 'Combined DataFrame should not be empty'
    assert len(combined) == 5, 'Combined DataFrame should have 5 rows after filtering'
    assert combined.columns.tolist() == expected_columns, f'Combined DataFrame should have correct columns: {expected_columns}'
    assert combined['score'].dtype == 'int64', 'Score column should be of integer type'
    assert combined['review'].dtype == 'string', 'Review column should be of string type'
    assert pd.api.types.is_datetime64_any_dtype(combined['date_created']), 'Date column should be of datetime type'
    assert all(combined['score'] <= TEST_CONFIG['filters']['low_rating_max']), 'All scores should be less than or equal to low_rating_max after filtering'
    assert (combined['review'].isnull().sum() == 0).all(), 'There should be no null values in the review column of the combined DataFrame'
    assert combined.duplicated().sum() == 0, 'There should be no duplicate rows in the combined DataFrame'
    assert os.path.exists(output_path), 'Output file should be created'

    saved_df = pd.read_parquet(output_path)

    assert saved_df.shape == combined.shape, 'Saved DataFrame should have the same shape as the combined DataFrame'