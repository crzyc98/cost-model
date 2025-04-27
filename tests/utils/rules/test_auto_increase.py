import pandas as pd
import numpy as np
import pytest
from utils.rules.auto_increase import apply

# Helper to build DataFrame
def make_df(deferral_rate, ai_enrolled=None, ai_opted_out=None):
    data = {'deferral_rate': [deferral_rate]}
    if ai_enrolled is not None:
        data['ai_enrolled'] = [ai_enrolled]
    if ai_opted_out is not None:
        data['ai_opted_out'] = [ai_opted_out]
    return pd.DataFrame(data)

YEAR = 2025

def test_ai_disabled():
    df = make_df(0.02)
    rules = {'auto_increase': {'enabled': False}}
    out = apply(df.copy(), rules, YEAR)
    pd.testing.assert_frame_equal(out, df)

def test_flag_initialization_no_changes():
    df = make_df(0.03)
    rules = {'auto_increase': {'enabled': True}}
    out = apply(df.copy(), rules, YEAR)
    # Flags should be added and False
    assert 'ai_opted_out' in out.columns and not out['ai_opted_out'].iloc[0]
    assert 'ai_enrolled' in out.columns and not out['ai_enrolled'].iloc[0]
    # Rate unchanged
    assert out['deferral_rate'].iloc[0] == pytest.approx(0.03)

def test_auto_increase_applied():
    df = make_df(deferral_rate=0.02, ai_enrolled=True, ai_opted_out=False)
    rules = {'auto_increase': {'enabled': True, 'increase_rate': 0.03, 'cap_rate': 0.10}}
    out = apply(df.copy(), rules, YEAR)
    assert out['deferral_rate'].iloc[0] == pytest.approx(0.05)
    assert out['ai_enrolled'].iloc[0]

def test_opted_out_skips():
    df = make_df(deferral_rate=0.02, ai_enrolled=True, ai_opted_out=True)
    rules = {'auto_increase': {'enabled': True, 'increase_rate': 0.03, 'cap_rate': 0.10}}
    out = apply(df.copy(), rules, YEAR)
    assert out['deferral_rate'].iloc[0] == pytest.approx(0.02)

def test_cap_rate_respected():
    df = make_df(deferral_rate=0.09, ai_enrolled=True, ai_opted_out=False)
    rules = {'auto_increase': {'enabled': True, 'increase_rate': 0.03, 'cap_rate': 0.10}}
    out = apply(df.copy(), rules, YEAR)
    assert out['deferral_rate'].iloc[0] == pytest.approx(0.10)
