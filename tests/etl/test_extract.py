from src.etl.extract import extract
import pandas as pd


def test_extract(tmp_path):
    test_file = tmp_path / "sample.csv"
    test_file.write_text("review,rating\nGreat gym!,5\nOkay gym,3")
    df = extract(test_file)

    assert isinstance(df, pd.DataFrame), "Extract should return a DataFrame"
    assert df.shape == (2, 2), "DataFrame should have 2 rows and 2 columns"
    assert list(df.columns) == [
        "review",
        "rating",
    ], "DataFrame should have correct columns"
