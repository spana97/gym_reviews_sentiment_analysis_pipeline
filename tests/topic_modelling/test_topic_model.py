import sys
from unittest.mock import MagicMock, patch

import pytest

sys.modules["bertopic"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
from topic_modelling.topic_model import TopicModel  # noqa: E402

BERTOPIC_PATH = "topic_modelling.topic_model.BERTopic"
SENTENCE_TRANSFORMER_PATH = "topic_modelling.topic_model.SentenceTransformer"


# @pytest.fixture
# def test_config():
#     return {
#         "language": "english",
#         "calculate_probabilities": True,
#         "verbose": False,
#         "embed_model": "all-MiniLM-L6-v2",
#         "low_memory": False,
#         "nr_topics": "auto",
#     }


@pytest.fixture
def topic_model(test_config):
    with (
        patch(BERTOPIC_PATH) as mock_bertopic,
        patch(SENTENCE_TRANSFORMER_PATH) as mock_st,
    ):
        mock_bertopic_instance = mock_bertopic.return_value
        mock_st_instance = mock_st.return_value

        model = TopicModel(test_config["topic_model"])

        model._mock_bertopic_instance = mock_bertopic_instance
        model._mock_st_instance = mock_st_instance
        yield model


def test_init(test_config):
    with (
        patch(BERTOPIC_PATH) as mock_bertopic,
        patch(SENTENCE_TRANSFORMER_PATH) as mock_st,
    ):
        TopicModel(test_config["topic_model"])
        mock_st.assert_called_once_with(test_config["topic_model"]["embed_model"])

        mock_bertopic.assert_called_once_with(
            language=test_config["topic_model"]["language"],
            calculate_probabilities=test_config["topic_model"][
                "calculate_probabilities"
            ],
            verbose=test_config["topic_model"]["verbose"],
            embedding_model=mock_st.return_value,
            low_memory=test_config["topic_model"]["low_memory"],
            nr_topics=test_config["topic_model"]["nr_topics"],
        )


def test_fit_transform(topic_model):
    documents = ["doc one", "doc two", "doc three"]
    expected_topics = [0, 1, 0]
    expected_probs = [0.9, 0.8, 0.7]

    topic_model._mock_bertopic_instance.fit_transform.return_value = (
        expected_topics,
        expected_probs,
    )

    topics, probs = topic_model.fit(documents)

    topic_model._mock_bertopic_instance.fit_transform.assert_called_once_with(documents)
    assert topics == expected_topics
    assert probs == expected_probs


def test_save(topic_model, tmp_path):
    save_path = str(tmp_path / "models" / "my_model")
    topic_model.save(save_path)
    topic_model._mock_bertopic_instance.save.assert_called_once_with(save_path)


def test_load(topic_model):
    load_path = "some/path/model"
    with patch(BERTOPIC_PATH + ".load") as mock_load:
        mock_load.return_value = MagicMock()
        topic_model.load(load_path)
        mock_load.assert_called_once_with(load_path)


def test_get_topic_info(topic_model):

    result = topic_model.get_topic_info()
    topic_model._mock_bertopic_instance.get_topic_info.assert_called_once()
    assert result == topic_model._mock_bertopic_instance.get_topic_info()
