import pandas as pd
import pytest
from etl.transform.filter_rows import filter_rows


def test_missing_score_column_raises(low_rating_max):

    df = pd.DataFrame({"review": ["bad", "good"]})
    with pytest.raises(KeyError):
        filter_rows(df, low_rating_max)


def test_filter_rows_above_max_score(low_rating_max):

    df = pd.DataFrame({"score": [1, 3, 5], "review": ["bad", "okay", "good"]})

    expected = pd.DataFrame({"score": [1, 3], "review": ["bad", "okay"]})

    result = filter_rows(df, low_rating_max)

    assert result.shape == expected.shape
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    assert len(result) == len(expected)
