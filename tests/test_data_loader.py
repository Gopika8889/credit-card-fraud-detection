"""Unit tests for src.data_loader."""

import pandas as pd
import pytest

from src.data_loader import EXPECTED_COLUMNS, load_data


def _write_csv(path, columns):
    pd.DataFrame([[0] * len(columns)], columns=columns).to_csv(path, index=False)


def test_load_valid_file(tmp_path):
    csv = tmp_path / "data.csv"
    _write_csv(csv, EXPECTED_COLUMNS)
    df = load_data(csv, verbose=False)
    assert list(df.columns) == EXPECTED_COLUMNS


def test_missing_columns_raises(tmp_path):
    csv = tmp_path / "data.csv"
    _write_csv(csv, EXPECTED_COLUMNS[:-1])  # drop 'Class'
    with pytest.raises(ValueError):
        load_data(csv, verbose=False)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nope.csv", verbose=False)