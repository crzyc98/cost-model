from pydantic import BaseModel, confloat, conint, conlist, ValidationError
from typing import Optional

class Tier(BaseModel):
    match_rate: confloat(ge=0, le=1)
    cap_deferral_pct: confloat(ge=0, le=1)
    min_tenure_months: Optional[conint(ge=0)] = None  # optional tenure gating

class MatchRule(BaseModel):
    tiers: conlist(Tier, min_items=1)
    dollar_cap: Optional[float] = None


def load_match_rule(raw: dict) -> MatchRule:
    try:
        return MatchRule(**raw)
    except ValidationError as e:
        raise ValueError(f"Invalid employer_match config: {e}")
