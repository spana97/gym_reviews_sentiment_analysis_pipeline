from unittest.mock import patch
from etl.etl_pipeline import run_etl_pipeline


def test_run_etl_pipeline(test_google_df, test_trustpilot_df, test_config):

    with (
        patch("etl.etl_pipeline.load_config", return_value=test_config),
        patch("etl.etl_pipeline.extract_dataset") as mock_extract,
        patch("etl.etl_pipeline.load_dataset") as mock_load,
    ):
        mock_extract.side_effect = [test_google_df, test_trustpilot_df]

        combined = run_etl_pipeline()

        expected_columns = ["source", "location", "date_created", "review", "score"]

        for col in expected_columns:
            assert col in combined.columns

        assert combined.duplicated().sum() == 0

        mock_load.assert_called_once_with(combined, test_config["data"]["etl_output"])
