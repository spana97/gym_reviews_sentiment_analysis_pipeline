from text_preprocessing.helpers import tokenize_text


def test_tokenize_text():
    text = "This is a test sentence, with punctuation! And numbers: 123."
    tokens = tokenize_text(text)

    expected = [
        "This",
        "is",
        "a",
        "test",
        "sentence",
        "with",
        "punctuation",
        "And",
        "numbers",
    ]

    assert tokens == expected
    assert len(tokens) == 9
