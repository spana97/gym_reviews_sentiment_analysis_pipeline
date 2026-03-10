from unittest.mock import MagicMock, patch

import pytest

from insights.insight_generator import InsightGenerator

OPENAI_PATH = "insights.insight_generator.OpenAI"


@pytest.fixture
def insight_generator(test_config, test_api_key):
    with patch(OPENAI_PATH):
        yield InsightGenerator(test_config["insights_generator"], test_api_key)


def test_init(test_config, test_api_key):
    with patch(OPENAI_PATH) as mock_openai:
        InsightGenerator(test_config["insights_generator"], test_api_key)
        mock_openai.assert_called_once_with(api_key=test_api_key)


def test_build_user_prompt(insight_generator):
    result = insight_generator._build_user_prompt("cluster data")
    assert result == "Analyse cluster data"


def test_generate_insights(insight_generator):
    insight_generator.client.responses.create.return_value = MagicMock(
        output_text="some insights"
    )
    result = insight_generator.generate_insights("cluster data here")
    assert result == "some insights"
    insight_generator.client.responses.create.assert_called_once()
