from pathlib import Path


def load(df, output_path: str) -> Path:
    """
    Save dataframe to parquet.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output)
    print(f"Saved {len(df)} rows → {output}")

    return output
