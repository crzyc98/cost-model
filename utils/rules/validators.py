"""
rules/validators.py
Validators for plan rules.
"""

from pydantic import BaseModel, confloat, conint, conlist, ValidationError, Field, root_validator
from typing import Optional, Dict, Any

class Tier(BaseModel):
    match_rate: confloat(ge=0, le=1)
    cap_deferral_pct: confloat(ge=0, le=1)
    min_tenure_months: Optional[conint(ge=0)] = Field(None, alias="min_service_months")  # optional tenure gating

class MatchRule(BaseModel):
    tiers: conlist(Tier, min_items=1)
    dollar_cap: Optional[confloat(ge=0)] = None

    @root_validator
    def check_tiers_increasing(cls, values):
        caps = [t.cap_deferral_pct for t in values.get('tiers', [])]
        if any(c2 <= c1 for c1, c2 in zip(caps, caps[1:])):
            raise ValueError("cap_deferral_pct must be strictly increasing across tiers")
        return values

def load_match_rule(raw: dict) -> MatchRule:
    try:
        return MatchRule(**raw)
    except ValidationError as e:
        raise ValueError(f"Invalid employer_match config (keys={list(raw.keys())}): {e}")

class EligibilityRule(BaseModel):
    min_age: conint(ge=0) = 21
    min_service_months: conint(ge=0) = 0
    min_hours_worked: Optional[conint(ge=0)] = None

class AutoEnrollmentRule(BaseModel):
    enabled: bool
    default_rate: confloat(ge=0, le=1)
    proactive_enrollment_probability: confloat(ge=0, le=1)
    re_enroll_existing: Optional[bool] = False
    outcome_distribution: Optional[Dict[str, float]] = None
    window_days: Optional[int] = None
    proactive_rate_range: Optional[tuple] = None

class AutoIncreaseRule(BaseModel):
    enabled: bool
    increase_rate: confloat(ge=0, le=1)
    cap_rate: confloat(ge=0, le=1)
    apply_to_new_hires_only: bool = False
    re_enroll_existing_below_cap: bool = False

class ContributionsRule(BaseModel):
    enabled: bool

class NonElectiveRule(BaseModel):
    rate: confloat(ge=0)

class PlanRules(BaseModel):
    eligibility: EligibilityRule
    auto_enrollment: AutoEnrollmentRule
    auto_increase: AutoIncreaseRule
    contributions: ContributionsRule
    employer_match: MatchRule
    employer_nec: NonElectiveRule
    irs_limits: Dict[int, Dict[str, float]]
