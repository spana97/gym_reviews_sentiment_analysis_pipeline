from src.preprocessing.helpers import tokenize_text


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

    assert tokens == expected, f"Expected tokens {expected} but got {tokens}"
    assert (
        len(tokens) == 9
    ), "There should be exactly 9 tokens after cleaning and tokenization"
