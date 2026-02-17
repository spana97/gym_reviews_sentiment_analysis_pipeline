import pandas as pd

def extract(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file and returns a pandas DataFrane.
    """
    if not isinstance(file_path, str):
        raise TypeError(f'file_path must be string, got {type(file_path).__name__}')

    try:
        df = pd.read_csv(file_path)

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise

    except pd.errors.ParserError:
        print(f"Error parsing CSV file: {file_path}")
        raise

    except Exception as e:
        print(f"Unexpected error reading {file_path}: {e}")
        raise

    else:
        print(f"Loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df