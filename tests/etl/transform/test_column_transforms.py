import pytest
import pandas as pd
from etl.transform.column_transforms import rename_and_select, cast_types


@pytest.mark.parametrize(
    "df_fixture,col_mapping,df_renamed_fixture",
    [
        ("test_google_df", "google_col_mapping", "test_google_df_renamed"),
        ("test_trustpilot_df", "trustpilot_col_mapping", "test_trustpilot_df_renamed"),
    ],
    ids=["google", "trustpilot"],
)
def test_rename_and_select(df_fixture, col_mapping, df_renamed_fixture, request):

    df = request.getfixturevalue(df_fixture)
    mapping = request.getfixturevalue(col_mapping)
    expected = request.getfixturevalue(df_renamed_fixture)

    result = rename_and_select(df, mapping)

    assert list(result.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    assert len(result) == len(expected)


@pytest.mark.parametrize(
    "df_renamed_fixture,schema_fixture",
    [
        ("test_google_df_renamed", "schema"),
        ("test_trustpilot_df_renamed", "schema"),
    ],
    ids=["google", "trustpilot"],
)
def test_cast_types(df_renamed_fixture, schema_fixture, request):

    df = request.getfixturevalue(df_renamed_fixture)
    schema = request.getfixturevalue(schema_fixture)

    result = cast_types(df, schema)

    assert pd.api.types.is_string_dtype(result["source"])
    assert pd.api.types.is_string_dtype(result["location"])
    assert pd.api.types.is_datetime64_any_dtype(result["date_created"])
    assert pd.api.types.is_string_dtype(result["review"])
    assert pd.api.types.is_integer_dtype(result["score"])
