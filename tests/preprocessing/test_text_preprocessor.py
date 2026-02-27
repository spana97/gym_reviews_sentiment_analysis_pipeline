from text_preprocessing.text_preprocessor import TextPreprocessor


def test_text_preprocessor():
    preprocessor = TextPreprocessor(extra_stop_words=["gym", "workout"])
    input_text = "The gym was great! I had a fantastic workout. 10/10."
    expected = "great fantastic"

    output = preprocessor.preprocess(input_text)
    assert output == expected, f"Expected {expected}, got {output}"
    assert isinstance(output, str), "Output should be a string"
