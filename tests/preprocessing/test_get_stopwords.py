from text_preprocessing.helpers import get_stopwords


def test_get_stopwords():
    stopwords = get_stopwords()

    extra = ["gym", "workout"]
    stopwords_with_extra = get_stopwords(extra)

    assert "the" in stopwords, "'the' should be in the default stopwords set"
    assert "gym" in stopwords_with_extra, (
        "'gym' should be in the custom stopwords set when added"
    )
    assert "workout" in stopwords_with_extra, (
        "'workout' should be in the custom stopwords set when added"
    )
    assert "the" in stopwords_with_extra, (
        "'the' should still be in the custom stopwords set"
    )
