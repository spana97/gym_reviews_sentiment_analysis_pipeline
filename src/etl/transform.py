import pandas as pd
from src.utils.config_loader import load_config

config = load_config()


def transform(df: pd.DataFrame, source: str) -> pd.DataFrame:

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'Expected pd.DataFrame, got {type(df).__name__}')

    mappings = config.get('rename_mappings', {}).get(source)
    if not mappings:
        raise ValueError(f'No rename mappings found for source: "{source}"')

    try:
        df = df.rename(columns=mappings)

        columns_to_keep = list(mappings.values())
        df = df[columns_to_keep]

        max_rating = config['filters']['low_rating_max']
        return df[df['score'] <= max_rating].copy()

    except KeyError as e:
        raise KeyError(f'Mapping error for "{source}": Column {e} not found in input data.') from e
    except Exception as e:
        raise RuntimeError(f'Transformation failed for {source}: {e}')