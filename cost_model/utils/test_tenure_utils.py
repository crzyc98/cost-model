import pandas as pd
import pytest

from cost_model.utils.tenure_utils import standardize_tenure_band

@pytest.mark.parametrize(
    "inp,expected",
    [
        (" 5+ Yr ", "15+"),
        ("1-3 YEARS", "1-3"),
        ("<1", "<1"),
        (pd.NA, pd.NA),
    ],
)
def test_standardize_tenure_band_case_whitespace(inp, expected):
    result = standardize_tenure_band(inp)
    if pd.isna(expected):
        assert pd.isna(result)
    else:
        assert result == expected
