from nltk.stem import WordNetLemmatizer

from src.text_preprocessing.helpers import lemmatize_tokens


def test_lemmatize_tokens():
    lemmatizer = WordNetLemmatizer()

    # Test with simple nouns
    tokens = ["dogs", "cats", "children"]
    expected = ["dog", "cat", "child"]
    assert (
        lemmatize_tokens(tokens, lemmatizer) == expected
    ), f"Expected {expected} but got {lemmatize_tokens(tokens, lemmatizer)}"

    # Test with verbs
    tokens = ["running", "swimming", "eating"]
    expected = ["run", "swim", "eat"]
    assert (
        lemmatize_tokens(tokens, lemmatizer) == expected
    ), f"Expected {expected} but got {lemmatize_tokens(tokens, lemmatizer)}"

    # Test with adjectives
    tokens = ["better", "worse", "best"]
    expected = ["well", "bad", "best"]
    assert (
        lemmatize_tokens(tokens, lemmatizer) == expected
    ), f"Expected {expected} but got {lemmatize_tokens(tokens, lemmatizer)}"

    # Test with mixed POS
    tokens = ["running", "dogs", "better"]
    expected = ["run", "dog", "well"]
    assert (
        lemmatize_tokens(tokens, lemmatizer) == expected
    ), f"Expected {expected} but got {lemmatize_tokens(tokens, lemmatizer)}"
