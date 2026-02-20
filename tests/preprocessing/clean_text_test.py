from src.preprocessing.helpers import clean_text


def test_clean_text():
    assert (
        clean_text("HELLO WORLD") == "hello world"
    ), "Text should be converted to lowercase"
    assert (
        clean_text("Hello, World!") == "hello world"
    ), "Punctuation should be removed and text should be lowercase"
    assert (
        clean_text("This is a test. 123") == "this is a test"
    ), "Numbers should be removed"
    assert (
        clean_text("   Extra   whitespace   ") == "extra whitespace"
    ), "Extra whitespace should be removed"
