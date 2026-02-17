import pytest
import pandas as pd
from src.etl.extract import extract

def test_extract(tmp_path):

    test_file = tmp_path / 'sample.csv'
    test_file.write_text("review,rating\nGreat gym!,5\nOkay gym,3")

    df = extract(str(test_file))

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2,2)
    assert list(df.columns) == ['review', 'rating']
    assert df['rating'].dtype == int or df['rating'].dtype == float