import pandas as pd

from src.etl.transform import transform


def test_transform_google(google_test_df, test_config):

    result = transform(google_test_df.copy(), "google", test_config)

    assert result.shape == (
        2,
        5,
    ), "Transformed DataFrame should have 2 rows and 5 columns"
    assert list(result["score"]) == [
        1,
        3,
    ], "Scores should be correctly transformed and filtered"
    assert list(result["review"]) == [
        "Bad",
        "Okay",
    ], "Reviews should be correctly transformed"
    assert pd.api.types.is_integer_dtype(result["score"]), (
        "Score column should be of integer type"
    )
    assert pd.api.types.is_string_dtype(result["review"]), (
        "Review column should be of string type"
    )
    assert pd.api.types.is_datetime64_any_dtype(result["date_created"]), (
        "Date column should be of datetime type"
    )
    assert result["score"].max() <= 3, (
        "All scores should be less than or equal to 3 after filtering"
    )
    assert (result["review"].isnull().sum() == 0).all(), (
        "There should be no null values in the review column"
    )
    assert result.duplicated().sum() == 0, (
        "There should be no duplicate rows in the transformed DataFrame"
    )


def test_transform_trustpilot(trustpilot_test_df, test_config):
    result = transform(trustpilot_test_df.copy(), "trustpilot", test_config)

    assert result.shape == (
        3,
        5,
    ), "Transformed DataFrame should have 3 rows and 5 columns"
    assert list(result["score"]) == [
        1,
        3,
        2,
    ], "Scores should be correctly transformed and filtered"
    assert list(result["review"]) == [
        "Terrible",
        "Average",
        "Poor",
    ], "Reviews should be correctly transformed"
    assert pd.api.types.is_integer_dtype(result["score"]), (
        "Score column should be of integer type"
    )
    assert pd.api.types.is_string_dtype(result["review"]), (
        "Review column should be of string type"
    )
    assert pd.api.types.is_datetime64_any_dtype(result["date_created"]), (
        "Date column should be of datetime type"
    )
    assert result["score"].max() <= 3, (
        "All scores should be less than or equal to 3 after filtering"
    )
    assert (result["review"].isnull().sum() == 0).all(), (
        "There should be no null values in the review column"
    )
    assert result.duplicated().sum() == 0, (
        "There should be no duplicate rows in the transformed DataFrame"
    )
