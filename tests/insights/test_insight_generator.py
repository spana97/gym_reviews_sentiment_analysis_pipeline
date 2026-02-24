from unittest.mock import MagicMock, patch

import pytest

from src.insights.insight_generator import InsightGenerator

OPENAI_PATH = "src.insights.insight_generator.OpenAI"


@pytest.fixture
def mock_config():
    return {
        "max_output_tokens": 10,
        "developer_prompt": "You are a data scientist",
        "user_prompt": "Analyse {clusters}",
    }


@pytest.fixture
def insight_generator(mock_config):
    with patch(OPENAI_PATH):
        yield InsightGenerator(mock_config, api_key="test_key")


def test_init(mock_config):
    with patch(OPENAI_PATH) as mock_openai:
        InsightGenerator(mock_config, api_key="test_key")
        mock_openai.assert_called_once_with(api_key="test_key")


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
