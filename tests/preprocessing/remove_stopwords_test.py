from src.preprocessing.helpers import remove_stopwords


def test_remove_stopwords():
    tokens = ["this", "is", "a", "test", "sentence"]
    stop_words = {"is", "a"}

    filtered_tokens = remove_stopwords(tokens, stop_words)

    assert "this" in filtered_tokens, "'this' should not be removed"
    assert "test" in filtered_tokens, "'test' should not be removed"
    assert "sentence" in filtered_tokens, "'sentence' should not be removed"
    assert "is" not in filtered_tokens, "'is' should be removed"
    assert "a" not in filtered_tokens, "'a' should be removed"
