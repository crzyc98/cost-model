import pytest
from decimal import Decimal

from agents.contributions import ContributionsMixin, ZERO_DECIMAL


# Dummy class to test IRS limits loading
class DummyIRS:
    # bring in IRS key map and load method
    IRS_KEY_MAP = ContributionsMixin.IRS_KEY_MAP
    _load_irs_limits = ContributionsMixin._load_irs_limits

    def _safe_decimal(self, val, name):
        return Decimal(val)

    def __init__(self, limits, year):
        self.model = type(
            "M", (), {"year": year, "scenario_config": {"irs_limits": limits}}
        )


def test_load_irs_limits_canonical_and_legacy_keys():
    limits_cfg = {
        2025: {"deferral": 15000, "catch_up": 5000, "compensation_limit": 500000}
    }
    dummy = DummyIRS(limits_cfg, 2025)
    out = dummy._load_irs_limits()
    assert isinstance(out["deferral_limit"], Decimal)
    assert out["deferral_limit"] == Decimal("15000")
    assert out["catchup_limit"] == Decimal("5000")
    assert out["compensation_limit"] == Decimal("500000")


def test_load_irs_defaults_when_missing():
    dummy = DummyIRS({}, 2025)
    out = dummy._load_irs_limits()
    # Should return defaults
    assert out["deferral_limit"] == ZERO_DECIMAL + Decimal("23000")


# Dummy class to test employer match logic
tier_rule = {
    "tiers": [
        {"match_rate": 0.5, "cap_deferral_pct": 0.1},  # 50% up to 10%
        {"match_rate": 0.25, "cap_deferral_pct": 0.1, "min_tenure_months": 12},
    ],
    "dollar_cap": None,
}


class DummyMatch(ContributionsMixin):
    def __init__(self, comp, def_rate, tenure, rule):
        self.gross_compensation = Decimal(comp)
        self.deferral_rate = Decimal(def_rate)
        # set tenure in months for testing
        self._tenure_months = tenure
        self.total_employee = Decimal("0")
        self.contributions_current_year = {
            "total_employee": ZERO_DECIMAL,
            "employee_pretax": ZERO_DECIMAL,
            "employee_catchup": ZERO_DECIMAL,
            "employer_match": ZERO_DECIMAL,
            "employer_nec": ZERO_DECIMAL,
        }
        self.model = type(
            "M",
            (),
            {"year": 2025, "scenario_config": {"plan_rules": {"employer_match": rule}}},
        )
        self.participation_date = None
        self.is_active = True
        self.is_participating = True
        self._calculate_contributions()

    def _calculate_tenure_months(self, as_of):
        return self._tenure_months


@pytest.mark.parametrize(
    "tenure,expected_match",
    [
        (
            24,
            Decimal("45.00"),
        ),  # 20% deferral yields 120; tier1 60@50%=30 + tier2 60@25%=15 =>45
        (6, Decimal("30.00")),  # only tier1: 60@50% =30
    ],
)
def test_tiered_match_with_tenure(tenure, expected_match):
    # deferral_rate 0.2 on comp 600 -> 120 total_emp
    dm = DummyMatch(comp="600", def_rate="0.2", tenure=tenure, rule=tier_rule)
    # match contributions stored in employer_match key
    assert dm.contributions_current_year["employer_match"] == expected_match


def test_dollar_cap_enforcement():
    cap_rule = {
        "tiers": [{"match_rate": 1.0, "cap_deferral_pct": 1.0}],  # 100% up to 100%
        "dollar_cap": 50,
    }
    # comp 100, deferral_rate 0.6 -> total_emp 60; cap_base=100 -> 60 cap, but dollar_cap=50
    dm = DummyMatch(comp="100", def_rate="0.6", tenure=0, rule=cap_rule)
    assert dm.contributions_current_year["employer_match"] == Decimal("50.00")


def test_invalid_match_config_is_skipped(caplog):
    bad_rule = {"tiers": [], "dollar_cap": None}
    dm = DummyMatch(comp="1000", def_rate="0.1", tenure=0, rule=bad_rule)
    # Should log a warning and skip match (employer_match remains 0)
    assert dm.contributions_current_year["employer_match"] == ZERO_DECIMAL
    assert "Invalid employer_match config" in caplog.text
