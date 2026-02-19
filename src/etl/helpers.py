import pandas as pd
from typing import List

# Tranform helpers

def rename_and_select(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """
    Renames and selects DataFrame columns
    """
    df = df.rename(columns=mappings)

    expected = list(mappings.values())
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f'Missing expected columns: {missing}')

    return df[expected].copy()


def cast_types(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Updates column data types according to set mapping.
    """
    for col, dtype in schema.items():
        if dtype == 'datetime':
            df[col] = pd.to_datetime(df[col], errors='coerce')
        else:
            df[col] = df[col].astype(dtype, errors='ignore')
    return df


def filter_rows(df: pd.DataFrame, max_score: int) -> pd.DataFrame:
    """
    Filters df for rows with a score less than or equal to a max_score.
    """
    return df[df['score'] <= max_score].copy()


def transform(df: pd.DataFrame, source: str, config: dict) -> pd.DataFrame:
    """
    Transform raw input DataFrame into clean standardised format.
    """
    mappings = config['rename_mappings'].get(source)
    if not mappings:
        raise ValueError(f'No mappings defined for source "{source}"')

    schema = config['schema']

    # 1. Rename + keep expected columns
    df = rename_and_select(df, mappings)

    # 2. Remove rows without reviews
    df = df.dropna(subset=['review'])

    # 3. Type casting
    df = cast_types(df, schema)

    # 4. Drop duplicates
    df = df.drop_duplicates()

    # 5. Rating filtering
    max_rating = config['filters']['low_rating_max']
    df = filter_rows(df, max_rating)

    print(f'{source}: {df.shape[0]} rows after cleaning & filtering')
    return df


# Pipeline helpers

def combine_datasets(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combines multiple cleaned DataFrames into one.
    """
    if not dfs:
        raise ValueError('No DataFrames provided to combine. "dfs" list is empty.')
    combined = pd.concat(dfs, ignore_index=True)
    print(f'Combined dataset shape: {combined.shape}')
    return combined