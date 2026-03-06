import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from topic_modelling.topic_pipeline import run_topic_model_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path):
    return {
        "data": {
            "etl_output": str(tmp_path / "reviews.parquet"),
        },
        "text_preprocessing": {
            "extra_stopwords": [],
        },
        "topic_model": {
            "model_output": str(tmp_path / "model"),
            "topics_output": str(tmp_path / "topics.csv"),
            "representative_docs_output": str(tmp_path / "clusters.json"),
        },
    }


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
    preprocessor = MagicMock()
    preprocessor.preprocess.side_effect = lambda x: x
    return preprocessor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_raises_if_parquet_not_found(config):
    """Pipeline should raise FileNotFoundError if the input file doesn't exist."""
    config["data"]["etl_output"] = "nonexistent.parquet"
    with pytest.raises(FileNotFoundError):
        run_topic_model_pipeline(config)


def test_pipeline_fits_model_when_no_saved_model_exists(
    config, sample_df, sample_topic_info, mock_preprocessor
):
    """If no model is saved, pipeline should call fit()."""
    with (
        patch(
            "topic_modelling.topic_model_pipeline.pd.read_parquet",
            return_value=sample_df,
        ),
        patch(
            "topic_modelling.topic_model_pipeline.TextPreprocessor",
            return_value=mock_preprocessor,
        ),
        patch(
            "topic_modelling.topic_model_pipeline.os.path.exists", return_value=False
        ),
        patch(
            "topic_modelling.topic_model_pipeline.TopicModel.fit",
            return_value=(MagicMock(), MagicMock()),
        ) as mock_fit,
        patch("topic_modelling.topic_model_pipeline.TopicModel.save"),
        patch(
            "topic_modelling.topic_model_pipeline.TopicModel.get_topic_info",
            return_value=sample_topic_info,
        ),
        patch(
            "topic_modelling.topic_model_pipeline.format_and_save_clusters",
            return_value={"cluster_1": ["doc1"]},
        ),
        patch("pandas.DataFrame.to_csv"),
    ):
        run_topic_model_pipeline(config)
        mock_fit.assert_called_once()


def test_pipeline_loads_model_when_saved_model_exists(
    config, sample_df, sample_topic_info, mock_preprocessor
):
    """If a saved model exists, pipeline should call load() and skip fit()."""
    with (
        patch(
            "topic_modelling.topic_model_pipeline.pd.read_parquet",
            return_value=sample_df,
        ),
        patch(
            "topic_modelling.topic_model_pipeline.TextPreprocessor",
            return_value=mock_preprocessor,
        ),
        patch("topic_modelling.topic_model_pipeline.os.path.exists", return_value=True),
        patch("topic_modelling.topic_model_pipeline.TopicModel.load") as mock_load,
        patch("topic_modelling.topic_model_pipeline.TopicModel.fit") as mock_fit,
        patch(
            "topic_modelling.topic_model_pipeline.TopicModel.get_topic_info",
            return_value=sample_topic_info,
        ),
        patch(
            "topic_modelling.topic_model_pipeline.format_and_save_clusters",
            return_value={"cluster_1": ["doc1"]},
        ),
        patch("pandas.DataFrame.to_csv"),
    ):
        run_topic_model_pipeline(config)
        mock_load.assert_called_once()
        mock_fit.assert_not_called()


def test_pipeline_raises_if_clusters_returns_none(
    config, sample_df, sample_topic_info, mock_preprocessor
):
    """Pipeline should raise ValueError if format_and_save_clusters returns None."""
    with (
        patch(
            "topic_modelling.topic_model_pipeline.pd.read_parquet",
            return_value=sample_df,
        ),
        patch(
            "topic_modelling.topic_model_pipeline.TextPreprocessor",
            return_value=mock_preprocessor,
        ),
        patch(
            "topic_modelling.topic_model_pipeline.os.path.exists", return_value=False
        ),
        patch(
            "topic_modelling.topic_model_pipeline.TopicModel.fit",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch("topic_modelling.topic_model_pipeline.TopicModel.save"),
        patch(
            "topic_modelling.topic_model_pipeline.TopicModel.get_topic_info",
            return_value=sample_topic_info,
        ),
        patch(
            "topic_modelling.topic_model_pipeline.format_and_save_clusters",
            return_value=None,
        ),
        patch("pandas.DataFrame.to_csv"),
    ):
        with pytest.raises(ValueError):
            run_topic_model_pipeline(config)
