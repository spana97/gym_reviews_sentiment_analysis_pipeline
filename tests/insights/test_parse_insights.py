import json

import pandas as pd

from src.insights.format_insights import parse_insights


def test_topic_model():

    data = json.dumps(
        [
            {"id": 0, "test_num": 1, "result": "Passed"},
            {"id": 1, "test_num": 2, "result": "Failed"},
        ]
    )

    df = parse_insights(data)

    assert isinstance(df, pd.DataFrame), "Output should be a DataFrame"
    assert df.shape == (2, 3), "DataFrame should have 2 rows and 3 columns"
    assert list(df.columns) == [
        "id",
        "test_num",
        "result",
    ], "DataFrame should have the correct columns"
