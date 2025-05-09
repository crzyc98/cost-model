import pytest
from utils.rules.formula_parsers import parse_match_formula, parse_match_tiers

# parse_match_formula tests


def test_parse_match_formula_empty_and_none():
    assert parse_match_formula("") == (0.0, 0.0)
    assert parse_match_formula(None) == (0.0, 0.0)


def test_parse_match_formula_simple_percent():
    rate, cap = parse_match_formula("5%")
    assert rate == pytest.approx(0.05)
    assert cap == pytest.approx(1.0)


def test_parse_match_formula_standard_case():
    rate, cap = parse_match_formula("50% up to 6%")
    assert rate == pytest.approx(0.50)
    assert cap == pytest.approx(0.06)
    # Whitespace and case-insensitive
    rate2, cap2 = parse_match_formula(" 12.5%   Up To 2.5% ")
    assert rate2 == pytest.approx(0.125)
    assert cap2 == pytest.approx(0.025)


def test_parse_match_formula_invalid():
    # Missing percent
    assert parse_match_formula("up to 5%") == (0.0, 0.0)
    assert parse_match_formula("abc") == (0.0, 0.0)


# parse_match_tiers tests


def test_parse_match_tiers_empty_and_none():
    assert parse_match_tiers("") == []
    assert parse_match_tiers(None) == []


def test_parse_match_tiers_single_tier():
    tiers = parse_match_tiers("100% up to 3%")
    assert isinstance(tiers, list) and len(tiers) == 1
    assert tiers[0] == {
        "match_pct": pytest.approx(1.0),
        "deferral_cap_pct": pytest.approx(0.03),
    }


def test_parse_match_tiers_multiple_tiers():
    formula = "100% up to 3%, 50% up to 5%"
    tiers = parse_match_tiers(formula)
    # Should sort by deferral_cap_pct ascending
    assert tiers[0]["deferral_cap_pct"] < tiers[1]["deferral_cap_pct"]
    assert tiers == [
        {"match_pct": pytest.approx(1.0), "deferral_cap_pct": pytest.approx(0.03)},
        {"match_pct": pytest.approx(0.50), "deferral_cap_pct": pytest.approx(0.05)},
    ]


def test_parse_match_tiers_partial_invalid():
    # Only second part valid
    tiers = parse_match_tiers("foo, 75% up to 4%")
    assert len(tiers) == 1
    assert tiers[0] == {
        "match_pct": pytest.approx(0.75),
        "deferral_cap_pct": pytest.approx(0.04),
    }
