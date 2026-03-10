import pytest
import json
from unittest.mock import patch, MagicMock, mock_open
from insights.insight_generator_pipeline import run_insight_generator_pipeline

GET_ENV_PATH = "insights.insight_generator_pipeline.os.getenv"
PATH_EXISITS_PATH = "insights.insight_generator_pipeline.os.path.exists"
JSON_DUMPS_PATH = "insights.insight_generator_pipeline.json.dumps"
INSIGHT_GENERATOR_PATH = "insights.insight_generator_pipeline.InsightGenerator"
GENERATE_INSIGHTS_PATH = (
    "insights.insight_generator_pipeline.InsightGenerator.generate_insights"
)
PARSE_INSIGHTS_PATH = "insights.insight_generator_pipeline.parse_insights"
TO_CSV_PATH = "insights.insight_generator_pipeline.to_csv"


def test_api_key_not_found_raise(test_config):
    with (
        patch(GET_ENV_PATH, return_value=None),
        patch(PATH_EXISITS_PATH, return_value=True),
    ):
        with pytest.raises(
            ValueError, match="OPENAI_API_KEY environment variable not set."
        ):
            run_insight_generator_pipeline(test_config)


def test_representative_docs_path_not_found_raises(test_config):
    test_config["topic_model"]["representative_docs_output"] = "nonexistent.json"
    with (
        patch(GET_ENV_PATH, return_value="mock-api-token"),  # pragma: allowlist secret
        patch(PATH_EXISITS_PATH, return_value=False),
    ):
        with pytest.raises(
            FileNotFoundError, match="Representative docs file not found"
        ):
            run_insight_generator_pipeline(test_config)


def test_pipeline_success(test_config):

    sample_representative_docs = {"cluster 1": ["dirty", "not clean"]}
    mock_insights = [{"topic": "example", "insight": "test"}]

    mock_file = mock_open(read_data=json.dumps(sample_representative_docs))

    with (
        patch(GET_ENV_PATH, return_value="mock-api-token"),  # pragma: allowlist secret
        patch(PATH_EXISITS_PATH, return_value=True),
        patch("builtins.open", mock_file),
        patch(INSIGHT_GENERATOR_PATH) as mock_generator,
        patch(PARSE_INSIGHTS_PATH) as mock_parse_insights,
    ):
        mock_generator_instance = MagicMock()
        mock_generator.return_value = mock_generator_instance
        mock_generator_instance.generate_insights.return_value = mock_insights

        mock_df = MagicMock()
        mock_parse_insights.return_value = mock_df

        run_insight_generator_pipeline(test_config)

        mock_generator.assert_called_once_with(
            test_config["insights_generator"],
            api_key="mock-api-token",  # pragma: allowlist secret
        )

        mock_generator_instance.generate_insights.assert_called_once()

        mock_parse_insights.assert_called_once_with(mock_insights)

        mock_df.to_csv.assert_called_once_with(
            test_config["insights_generator"]["insights_output"], index=False
        )
