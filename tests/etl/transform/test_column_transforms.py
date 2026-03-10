import pytest
import pandas as pd
from etl.transform.column_transforms import rename_and_select, cast_types


@pytest.mark.parametrize(
    "df_fixture,source,df_renamed_fixture",
    [
        ("test_google_df", "google", "test_google_df_renamed"),
        ("test_trustpilot_df", "trustpilot", "test_trustpilot_df_renamed"),
    ],
    ids=["google", "trustpilot"],
)
def test_rename_and_select(
    df_fixture, source, df_renamed_fixture, test_config, request
):

    df = request.getfixturevalue(df_fixture)
    mapping = test_config["rename_mappings"][source]
    expected = request.getfixturevalue(df_renamed_fixture)

    result = rename_and_select(df, mapping)

    assert list(result.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    assert len(result) == len(expected)


@pytest.mark.parametrize(
    "df_renamed_fixture",
    [
        ("test_google_df_renamed"),
        ("test_trustpilot_df_renamed"),
    ],
    ids=["google", "trustpilot"],
)
def test_cast_types(df_renamed_fixture, request, test_config):

    df = request.getfixturevalue(df_renamed_fixture)

    result = cast_types(df, test_config["schema"])

    assert pd.api.types.is_string_dtype(result["source"])
    assert pd.api.types.is_string_dtype(result["location"])
    assert pd.api.types.is_datetime64_any_dtype(result["date_created"])
    assert pd.api.types.is_string_dtype(result["review"])
    assert pd.api.types.is_integer_dtype(result["score"])
