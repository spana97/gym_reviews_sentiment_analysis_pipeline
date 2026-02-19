import pandas as pd
from src.etl.helpers import rename_and_select, cast_types, filter_rows


def transform(df: pd.DataFrame, source: str, config: dict) -> pd.DataFrame:
    """
    Transform raw input DataFrame into clean standardized format.
    """
    mappings = config["rename_mappings"].get(source)
    if not mappings:
        raise ValueError(f'No mappings defined for source "{source}"')

    schema = config["schema"]

    # 1. Rename + keep expected columns
    df = rename_and_select(df, mappings)

    # 2. Remove rows without reviews
    df = df.dropna(subset=["review"])

    # 3. Type casting
    df = cast_types(df, schema)

    # 4. Drop duplicates
    df = df.drop_duplicates()

    # 5. Rating filtering
    max_rating = config["filters"]["low_rating_max"]
    df = filter_rows(df, max_rating)

    print(f"{source}: {df.shape[0]} rows after cleaning & filtering")
    return df
