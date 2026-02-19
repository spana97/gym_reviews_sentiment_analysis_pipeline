import pandas as pd

def extract(path: str) -> pd.DataFrame:
    """
    Load CSV into a DataFrame
    """
    df = pd.read_csv(path)
    print(f'Loaded {path}: {df.shape[0]} rows, {df.shape[1]} columns')
    return df