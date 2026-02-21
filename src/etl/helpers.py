from typing import List

import pandas as pd

# -----------------------------
# transform helpers
# -----------------------------


def rename_and_select(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """
    Renames and selects DataFrame columns
    """
    df = df.rename(columns=mappings)

    expected = list(mappings.values())
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df[expected].copy()


def cast_types(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Updates column data types according to set mapping.
    """
    for col, dtype in schema.items():
        if dtype == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(dtype, errors="ignore")
    return df


def filter_rows(df: pd.DataFrame, max_score: int) -> pd.DataFrame:
    """
    Filters df for rows with a score less than or equal to a max_score.
    """
    return df[df["score"] <= max_score].copy()


# -----------------------------
# run_etl_pipeline helper
# -----------------------------


def combine_datasets(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combines multiple cleaned DataFrames into one.
    """
    if not dfs:
        raise ValueError(("No DataFrames provided to combine."))
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset shape: {combined.shape}")
    return combined
