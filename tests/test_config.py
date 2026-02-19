import pandas as pd
import numpy as np

TEST_CONFIG = {
    'rename_mappings': {
        'google': {
            'Social Media Source': 'source',
            "Club's Name": 'location',
            'Creation Date': 'date_created',
            'Comment': 'review',
            'Overall Score': 'score'
        },
        'trustpilot': {
            'Source Of Review': 'source',
            'Location Name': 'location',
            'Review Created (UTC)': 'date_created',
            'Review Content': 'review',
            'Review Stars': 'score'
        },
    },
    'schema': {
        'source': 'string',
        'location': 'string',
        'date_created': 'datetime',
        'review': 'string',
        'score': 'int64'
        
    },
    'filters': {
        'low_rating_max': 3
    },
    'data': {
        'test_google': 'tests/test_data/google_reviews_test.csv',
        'test_trustpilot': 'tests/test_data/trustpilot_reviews_test.csv',
    }
}

GOOGLE_TEST_DF = pd.DataFrame({
    'Overall Score': [1, 3, 5, 3, 4],
    'Comment': ['Bad', 'Okay', 'Great', 'Okay', np.nan],  
    'Creation Date': ['2022-01-01', '2023-03-15', '2024-02-10', '2023-03-15', '2024-01-20'],  
    'Social Media Source': ['Google'] * 5,
    "Club's Name": ['Club A', 'Club B', 'Club C', 'Club B', 'Club D'],
    'user_id': ['u1', 'u2', 'u3', 'u2', 'u5']
})

TRUSTPILOT_TEST_DF = pd.DataFrame({
    'Review Stars': [1, 3, 5, 2, 1],
    'Review Content': ['Terrible', 'Average', 'Excellent', 'Poor', np.nan],
    'Review Created (UTC)': ['2023-06-01', '2023-07-01', '2023-08-01', '2023-06-15', '2023-09-01'],
    'Source Of Review': ['Trustpilot'] * 5,
    'Location Name': ['Club X', 'Club Y', 'Club Z', 'Club X', 'Club W'],
    'user_id': ['t1', 't2', 't3', 't1', 't5']
})