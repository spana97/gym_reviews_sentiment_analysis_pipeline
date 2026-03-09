import pytest
import pandas as pd
from etl.transform.transform_service import transform_dataset


def test_missing_mappings_raises(test_config):
    df = pd.DataFrame({"review": ["bad", "okay"]})
    with pytest.raises(ValueError, match="No mappings defined for source"):
        transform_dataset(df, "unknown_source", test_config)


def test_missing_schema_raises(test_config):
    df = pd.DataFrame({"review": ["bad", "okay"]})
    test_config = test_config.copy()
    test_config.pop("schema")
    with pytest.raises(ValueError, match="Schema missing in configuration"):
        transform_dataset(df, "google", test_config)


def test_missing_low_rating_max_raises(test_config):
    df = pd.DataFrame({"review": ["bad", "okay"]})
    test_config = test_config.copy()
    test_config["filters"].pop("low_rating_max")
    with pytest.raises(ValueError, match="low_rating_max missing in configuration"):
        transform_dataset(df, "google", test_config)


@pytest.mark.parametrize(
    "df_fixture,source",
    [("test_google_df", "google"), ("test_trustpilot_df", "trustpilot")],
    ids=["google", "trustpilot"],
)
def test_removes_nan_reviews(df_fixture, source, test_config, request):

    df = request.getfixturevalue(df_fixture)

    transformed = transform_dataset(df, source, test_config)
    assert transformed["review"].isna().sum() == 0


@pytest.mark.parametrize(
    "df_fixture,source",
    [("test_google_df", "google"), ("test_trustpilot_df", "trustpilot")],
    ids=["google", "trustpilot"],
)
def test_filters_high_scores(df_fixture, source, test_config, low_rating_max, request):
    df = request.getfixturevalue(df_fixture)
    transformed = transform_dataset(df, source, test_config)
    assert transformed["score"].max() <= low_rating_max


@pytest.mark.parametrize(
    "df_fixture,source",
    [("test_google_df", "google"), ("test_trustpilot_df", "trustpilot")],
    ids=["google", "trustpilot"],
)
def test_drops_duplicates(df_fixture, source, test_config, request):

    df = request.getfixturevalue(df_fixture)
    duplicated_df = pd.concat([df, df])
    transformed = transform_dataset(duplicated_df, source, test_config)
    assert transformed.duplicated().sum() == 0
