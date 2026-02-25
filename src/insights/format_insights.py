import json

import pandas as pd


def parse_insights(raw_response: str) -> pd.DataFrame:
    """
    Parses the response from the OpenAI API and converts it into a DataFrame.
    """
    print(f"Raw response: {raw_response}")  # add this
    cleaned = raw_response.strip().strip("```json").strip("```").strip()
    print(f"Cleaned response: {cleaned}")  # add this
    data = json.loads(cleaned)
    return pd.DataFrame(data)
