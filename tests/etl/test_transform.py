import pandas as pd

from etl.transform import transform


def test_transform_google(google_test_df, test_config):

    result = transform(google_test_df.copy(), "google", test_config)

    assert result.shape == (2, 5)
    assert list(result["score"]) == [1, 3]
    assert list(result["review"]) == ["Bad", "Okay"]
    assert pd.api.types.is_integer_dtype(result["score"])
    assert pd.api.types.is_string_dtype(result["review"])
    assert pd.api.types.is_datetime64_any_dtype(result["date_created"])
    assert result["score"].max() <= 3
    assert (result["review"].isnull().sum() == 0).all()
    assert result.duplicated().sum() == 0


def test_transform_trustpilot(trustpilot_test_df, test_config):
    result = transform(trustpilot_test_df.copy(), "trustpilot", test_config)

    assert result.shape == (3, 5)
    assert list(result["score"]) == [1, 3, 2]
    assert list(result["review"]) == [
        "Terrible",
        "Average",
        "Poor",
    ]
    assert pd.api.types.is_integer_dtype(result["score"])
    assert pd.api.types.is_string_dtype(result["review"])
    assert pd.api.types.is_datetime64_any_dtype(result["date_created"])
    assert result["score"].max() <= 3
    assert (result["review"].isnull().sum() == 0).all()
    assert result.duplicated().sum() == 0
