import pandas as pd 
from src.etl.transform import transform
from tests.test_config import TEST_CONFIG, GOOGLE_TEST_DF, TRUSTPILOT_TEST_DF

def test_transform_google():
    
    result = transform(GOOGLE_TEST_DF.copy(), 'google', TEST_CONFIG) 

    assert result.shape == (2, 5), 'Transformed DataFrame should have 2 rows and 5 columns'
    assert list(result['score']) == [1, 3], 'Scores should be correctly transformed and filtered'
    assert list(result['review']) == ['Bad', 'Okay'], 'Reviews should be correctly transformed'
    assert pd.api.types.is_integer_dtype(result['score']), 'Score column should be of integer type'
    assert pd.api.types.is_string_dtype(result['review']), 'Review column should be of string type'
    assert pd.api.types.is_datetime64_any_dtype(result['date_created']), 'Date column should be of datetime type'
    assert result['score'].max() <= 3, 'All scores should be less than or equal to 3 after filtering'
    assert (result['review'].isnull().sum() == 0).all(), 'There should be no null values in the review column'
    assert result.duplicated().sum() == 0, 'There should be no duplicate rows in the transformed DataFrame'

def test_transform_trustpilot():
    result = transform(TRUSTPILOT_TEST_DF.copy(), 'trustpilot', TEST_CONFIG) 

    assert result.shape == (3, 5), 'Transformed DataFrame should have 3 rows and 6 columns'
    assert list(result['score']) == [1, 3, 2], 'Scores should be correctly transformed and filtered'
    assert list(result['review']) == ['Terrible', 'Average', 'Poor'], 'Reviews should be correctly transformed'
    assert pd.api.types.is_integer_dtype(result['score']), 'Score column should be of integer type'
    assert pd.api.types.is_string_dtype(result['review']), 'Review column should be of string type'
    assert pd.api.types.is_datetime64_any_dtype(result['date_created']), 'Date column should be of datetime type'
    assert result['score'].max() <= 3, 'All scores should be less than or equal to 3 after filtering'
    assert (result['review'].isnull().sum() == 0).all(), 'There should be no null values in the review column'
    assert result.duplicated().sum() == 0, 'There should be no duplicate rows in the transformed DataFrame'
