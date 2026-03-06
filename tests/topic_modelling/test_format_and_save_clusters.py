import json

import pandas as pd

from topic_modelling.format_and_save_clusters import format_and_save_clusters


def test_format_and_save_clusters(tmp_path):

    topic_info = pd.Series([[0, 1, 2], [3, 4, 5]])
    expected_output = {"cluster_1": [0, 1, 2], "cluster_2": [3, 4, 5]}
    output_path = tmp_path / "clusters.json"

    result = format_and_save_clusters(topic_info, str(output_path))

    assert isinstance(result, dict)
    assert result == expected_output

    assert output_path.exists()
    with open(output_path) as f:
        assert json.load(f) == result
