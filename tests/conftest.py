import pytest
import pandas as pd
import numpy as np

# Global


@pytest.fixture
def schema():
    return {
        "source": "string",
        "location": "string",
        "date_created": "datetime",
        "review": "string",
        "score": "int64",
    }


@pytest.fixture
def low_rating_max():
    return 3


@pytest.fixture
def google_col_mapping():
    return {
        "Social Media Source": "source",
        "Club's Name": "location",
        "Creation Date": "date_created",
        "Comment": "review",
        "Overall Score": "score",
    }


@pytest.fixture
def test_google_df():
    return pd.DataFrame(
        {
            "Overall Score": [1, 3, 5, 3, 4],
            "Comment": ["Bad", "Okay", "Great", "Okay", np.nan],
            "Creation Date": [
                "2022-01-01",
                "2023-03-15",
                "2024-02-10",
                "2023-03-15",
                "2024-01-20",
            ],
            "Social Media Source": ["Google"] * 5,
            "Club's Name": ["Club A", "Club B", "Club C", "Club B", "Club D"],
            "user_id": ["u1", "u2", "u3", "u2", "u5"],
        }
    )


@pytest.fixture
def test_google_df_renamed():
    return pd.DataFrame(
        {
            "source": ["Google"] * 5,
            "location": ["Club A", "Club B", "Club C", "Club B", "Club D"],
            "date_created": [
                "2022-01-01",
                "2023-03-15",
                "2024-02-10",
                "2023-03-15",
                "2024-01-20",
            ],
            "review": ["Bad", "Okay", "Great", "Okay", np.nan],
            "score": [1, 3, 5, 3, 4],
        }
    )


# Trustpilot


@pytest.fixture
def trustpilot_col_mapping():
    return {
        "Source Of Review": "source",
        "Location Name": "location",
        "Review Created (UTC)": "date_created",
        "Review Content": "review",
        "Review Stars": "score",
    }


@pytest.fixture
def test_trustpilot_df():
    return pd.DataFrame(
        {
            "Review Stars": [1, 3, 5, 2, 1],
            "Review Content": ["Terrible", "Average", "Excellent", "Poor", np.nan],
            "Review Created (UTC)": [
                "2023-06-01",
                "2023-07-01",
                "2023-08-01",
                "2023-06-15",
                "2023-09-01",
            ],
            "Source Of Review": ["Trustpilot"] * 5,
            "Location Name": ["Club X", "Club Y", "Club Z", "Club X", "Club W"],
            "user_id": ["t1", "t2", "t3", "t1", "t5"],
        }
    )


@pytest.fixture
def test_trustpilot_df_renamed():
    return pd.DataFrame(
        {
            "source": ["Trustpilot"] * 5,
            "location": ["Club X", "Club Y", "Club Z", "Club X", "Club W"],
            "date_created": [
                "2023-06-01",
                "2023-07-01",
                "2023-08-01",
                "2023-06-15",
                "2023-09-01",
            ],
            "review": ["Terrible", "Average", "Excellent", "Poor", np.nan],
            "score": [1, 3, 5, 2, 1],
        }
    )


# test_config


@pytest.fixture
def test_config(schema, low_rating_max, google_col_mapping, trustpilot_col_mapping):
    return {
        "rename_mappings": {
            "google": google_col_mapping,
            "trustpilot": trustpilot_col_mapping,
        },
        "schema": schema,
        "filters": {"low_rating_max": low_rating_max},
        "data": {
            "raw_google": "path/to/google.csv",
            "raw_trustpilot": "path/to/trustpilot.csv",
            "etl_output": "path/to/output.csv",
        },
    }
