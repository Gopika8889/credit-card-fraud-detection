"""Data loading and validation utilities for the fraud detection project."""

from pathlib import Path
from typing import Iterable, Union

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

TARGET_COLUMN = "Class"
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
EXPECTED_COLUMNS = ["Time", *PCA_COLUMNS, "Amount", TARGET_COLUMN]


def validate_columns(
    df: pd.DataFrame, expected_columns: Iterable[str] = EXPECTED_COLUMNS
) -> None:
    """Check that all expected columns exist in the DataFrame.

    Args:
        df: DataFrame to validate.
        expected_columns: Column names that must be present.

    Raises:
        ValueError: If any expected column is missing, or if the target
            column contains values other than 0 and 1.
    """
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    if TARGET_COLUMN in df.columns:
        invalid = set(df[TARGET_COLUMN].dropna().unique()) - {0, 1}
        if invalid:
            raise ValueError(
                f"'{TARGET_COLUMN}' must contain only 0/1, found: {invalid}"
            )


def print_basic_info(df: pd.DataFrame) -> None:
    """Print shape, dtypes and memory usage of a DataFrame.

    Args:
        df: DataFrame to describe.
    """
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory usage: {memory_mb:.2f} MB")
    print("\nData types:")
    print(df.dtypes.to_string())


def load_data(
    path: Union[str, Path] = DEFAULT_DATA_PATH, verbose: bool = True
) -> pd.DataFrame:
    """Load the credit card fraud CSV and validate its schema.

    Args:
        path: Path to creditcard.csv.
        verbose: If True, print shape, dtypes and memory usage.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the CSV does not exist at ``path``.
        ValueError: If expected columns are missing or the target is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from Kaggle and place "
            "creditcard.csv in data/raw/ (see README)."
        )

    df = pd.read_csv(path)
    validate_columns(df)

    if verbose:
        print_basic_info(df)
    return df


if __name__ == "__main__":
    load_data()