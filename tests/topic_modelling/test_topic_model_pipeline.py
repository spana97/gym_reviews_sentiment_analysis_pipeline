import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from topic_modelling.topic_pipeline import run_topic_model_pipeline


READ_PARQUET_PATH = "topic_modelling.topic_pipeline.pd.read_parquet"
TEXT_PREPROCESSOR_PATH = "topic_modelling.topic_pipeline.TextPreprocessor"
PATH_EXISTS_PATH = "topic_modelling.topic_pipeline.os.path.exists"
TOPIC_MODEL_PATH = "topic_modelling.topic_pipeline.TopicModel"
TOPIC_MODEL_FIT_PATH = "topic_modelling.topic_pipeline.TopicModel.fit"
TOPIC_MODEL_SAVE_PATH = "topic_modelling.topic_pipeline.TopicModel.save"
TOPIC_MODEL_LOAD_PATH = "topic_modelling.topic_pipeline.TopicModel.load"
TOPIC_MODEL_GET_TOPIC_INFO_PATH = (
    "topic_modelling.topic_pipeline.TopicModel.get_topic_info"
)
FORMAT_AND_SAVE_CLUSTERS_PATH = (
    "topic_modelling.topic_pipeline.format_and_save_clusters"
)
TO_CSV_PATH = "topic_modelling.topic_pipeline.pd.DataFrame.to_csv"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    return pd.DataFrame({"review": ["good product", "bad quality", "loved it"]})


@pytest.fixture
def sample_topic_info():
    return pd.DataFrame(
        {
            "Topic": [0, 1, 2],
            "Representative_Docs": [
                ["doc1", "doc2"],
                ["doc3", "doc4"],
                ["doc5", "doc6"],
            ],
        }
    )


@pytest.fixture
def mock_preprocessor():
    class DummyPreprocessor:
        def preprocess(self, text):
            return text

    return DummyPreprocessor()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parquet_not_found_raises(test_config):
    """Pipeline should raise FileNotFoundError if the input file doesn't exist."""
    test_config["data"]["etl_output"] = "nonexistent.parquet"
    with pytest.raises(FileNotFoundError):
        run_topic_model_pipeline(test_config)


def test_pipeline_fits_model_when_no_saved_model_exists(
    test_config, sample_df, sample_topic_info, mock_preprocessor
):

    with (
        patch(READ_PARQUET_PATH, return_value=sample_df),
        patch(TEXT_PREPROCESSOR_PATH, return_value=mock_preprocessor),
        patch(PATH_EXISTS_PATH, return_value=False),
        patch(TOPIC_MODEL_PATH) as MockTopicModel,
        patch(TO_CSV_PATH) as mock_to_csv,
        patch(
            FORMAT_AND_SAVE_CLUSTERS_PATH, return_value={"cluster_1": ["doc1"]}
        ) as mock_format_save_clusters,
    ):
        mock_model = MockTopicModel.return_value
        mock_model.fit.return_value = (MagicMock(), MagicMock())
        mock_model.get_topic_info.return_value = sample_topic_info

        run_topic_model_pipeline(test_config)

        mock_model.fit.assert_called_once()
        mock_model.save.assert_called_once()
        mock_model.transform.assert_not_called()
        mock_model.get_topic_info.assert_called_once()
        mock_to_csv.assert_called_once()
        mock_format_save_clusters.assert_called_once()


def test_pipeline_loads_model_when_saved_model_exists(
    test_config, sample_df, sample_topic_info, mock_preprocessor
):
    """If a saved model exists, pipeline should call load() and skip fit()."""
    with (
        patch(READ_PARQUET_PATH, return_value=sample_df),
        patch(TEXT_PREPROCESSOR_PATH, return_value=mock_preprocessor),
        patch(PATH_EXISTS_PATH, return_value=True),
        patch(TOPIC_MODEL_PATH) as MockTopicModel,
        patch(TO_CSV_PATH) as mock_to_csv,
        patch(
            FORMAT_AND_SAVE_CLUSTERS_PATH, return_value={"cluster_1": ["doc1"]}
        ) as mock_format_save_clusters,
    ):
        mock_model = MockTopicModel.return_value
        mock_model.load.return_value = MagicMock()
        mock_model.transform.return_value = (MagicMock(), MagicMock())
        mock_model.get_topic_info.return_value = sample_topic_info

        run_topic_model_pipeline(test_config)

        mock_model.load.assert_called_once()
        mock_model.transform.assert_called_once()
        mock_model.fit.assert_not_called()
        mock_model.get_topic_info.assert_called_once()
        mock_to_csv.assert_called_once()
        mock_format_save_clusters.assert_called_once()


def test_missing_format_and_save_clusters_raises(
    test_config, sample_df, sample_topic_info, mock_preprocessor
):

    with (
        patch(READ_PARQUET_PATH, return_value=sample_df),
        patch(TEXT_PREPROCESSOR_PATH, return_value=mock_preprocessor),
        patch(PATH_EXISTS_PATH, return_value=False),
        patch(TOPIC_MODEL_PATH) as MockTopicModel,
        patch(TO_CSV_PATH),
        patch(FORMAT_AND_SAVE_CLUSTERS_PATH, return_value=None) as mock_clusters,
    ):
        mock_model = MockTopicModel.return_value
        mock_model.fit.return_value = (MagicMock(), MagicMock())
        mock_model.get_topic_info.return_value = sample_topic_info

        with pytest.raises(ValueError):
            run_topic_model_pipeline(test_config)

        mock_clusters.assert_called_once()
