from text_preprocessing.helpers import remove_stopwords


def test_remove_stopwords():
    tokens = ["this", "is", "a", "test", "sentence"]
    stop_words = {"is", "a"}

    filtered_tokens = remove_stopwords(tokens, stop_words)
    expected_tokens = ["this", "test", "sentence"]

    assert filtered_tokens == expected_tokens
    assert "is" not in filtered_tokens
    assert "a" not in filtered_tokens
