import json

import pandas as pd


def parse_insights(raw_response: str) -> pd.DataFrame:
    """
    Parses the response from the OpenAI API and converts it into a DataFrame.
    """
    cleaned = raw_response.strip().strip("```json").strip("```").strip()
    data = json.loads(cleaned)
    return pd.DataFrame(data)
