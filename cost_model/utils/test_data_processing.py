# utils/test_data_processing.py
import pytest
import os
import pandas as pd
from datetime import datetime

from utils.data_processing import _infer_plan_year_end, load_and_clean_census

@pytest.mark.parametrize("fname,expected_year", [
    ("foo_2025.csv", 2025),
    ("foo_1970.csv", datetime.now().year - 1),
    ("foo_bad.csv", datetime.now().year - 1),
])
def test_infer_plan_year_end(fname, expected_year):
    ts = _infer_plan_year_end(fname)
    assert ts == pd.Timestamp(f"{expected_year}-12-31")


def test_load_and_clean_census_missing_required(tmp_path):
    # Create a simple CSV with only one column
    df = pd.DataFrame({"col1": [1, 2, 3]})
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)
    expected_cols = {"required": ["missing_col"]}
    result = load_and_clean_census(str(file_path), expected_cols)
    assert result is None
