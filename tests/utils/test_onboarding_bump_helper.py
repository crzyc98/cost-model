import pandas as pd
import pytest
from utils.projection_utils import apply_onboarding_bump

def test_apply_onboarding_flat_rate():
    df = pd.DataFrame({'gross_compensation': [100.0, 200.0]})
    cfg = {'enabled': True, 'method': 'flat_rate', 'flat_rate': 0.1}
    out = apply_onboarding_bump(df, 'gross_compensation', cfg)
    # use approx to avoid floating point artifacts
    assert out['gross_compensation'].tolist() == pytest.approx([110.0, 220.0])

def test_no_bump_when_disabled():
    df = pd.DataFrame({'gross_compensation': [100.0, 200.0]})
    cfg = {'enabled': False, 'method': 'flat_rate', 'flat_rate': 0.1}
    out = apply_onboarding_bump(df, 'gross_compensation', cfg)
    assert out['gross_compensation'].tolist() == pytest.approx([100.0, 200.0])
