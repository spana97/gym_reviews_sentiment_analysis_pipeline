import pandas as pd

from etl.extract_service import extract_dataset


def test_extract_dataset(tmp_path):
    test_file = tmp_path / "sample.csv"
    test_file.write_text("review,rating\nGreat gym!,5\nOkay gym,3")
    df = extract_dataset(test_file)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == [
        "review",
        "rating",
    ]
