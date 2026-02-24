import json

import pandas as pd

from src.topic_modelling.format_json import format_json


def test_format_json():

    topic_info = pd.Series([[0, 1, 2], [3, 4, 5]])
    expected_output = {"cluster_1": [0, 1, 2], "cluster_2": [3, 4, 5]}

    json_string = format_json(topic_info)
    assert isinstance(json_string, str)
    assert json.loads(json_string) == expected_output
