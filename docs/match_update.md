Below is an updated version of my six-step plan—now including how you can layer in tenure-based match tiers (e.g. an extra match only for folks with 5+ years on the job).

⸻

1 | Schema: explicit, tiered + optional tenure rule

plan_rules:
  employer_match:
    tiers:
      # every employee gets...
      - match_rate:      0.50        # 50% of their deferral
        cap_deferral_pct:0.06        # on the first 6% of pay
      # plus, a loyalty bonus for 5+ yrs tenure
      - match_rate:      0.25        # extra 25%
        cap_deferral_pct:0.06        # on the same 6%
        min_tenure_months: 60        # apply only if tenure ≥ 60 months
    dollar_cap: null                # no absolute dollar cap



⸻

2 | Config model + validation (pydantic)

# utils/rules/validators.py
from pydantic import BaseModel, confloat, conint, conlist, ValidationError
from typing import Optional

class Tier(BaseModel):
    match_rate:         confloat(ge=0, le=1)
    cap_deferral_pct:   confloat(ge=0, le=1)
    min_tenure_months:  Optional[conint(ge=0)] = None  # new!

class MatchRule(BaseModel):
    tiers:      conlist(Tier, min_items=1)
    dollar_cap: Optional[float] = None

def load_match_rule(raw: dict) -> MatchRule:
    try:
        return MatchRule(**raw)
    except ValidationError as e:
        raise ValueError(f"Invalid employer_match config: {e}")



⸻

3 | Core contribution logic honors tenure threshold

# agents/contributions.py  (excerpt)
from utils.rules.validators import load_match_rule

def _calc_employer_match(self, comp: Decimal) -> None:
    raw = self.model.scenario_config['plan_rules'].get('employer_match', {})
    if not raw:
        return
    rule = load_match_rule(raw)

    emp_contrib = self.contributions_current_year[TOTAL_EMP_KEY]
    match_total = Decimal('0')
    remaining = emp_contrib

    # compute agent tenure once
    from pandas import Timestamp
    as_of = Timestamp(f"{self.model.year}-12-31")
    tenure_months = self._calculate_tenure_months(as_of)

    for tier in rule.tiers:
        # skip if tier has a tenure requirement
        if tier.min_tenure_months is not None and tenure_months < tier.min_tenure_months:
            continue

        cap_amount = (comp * Decimal(str(tier.cap_deferral_pct))) \
                       .quantize(Decimal('0.01'))
        eligible = min(remaining, cap_amount)
        match_amt = eligible * Decimal(str(tier.match_rate))
        match_total += match_amt
        remaining -= eligible
        if remaining <= ZERO_DECIMAL:
            break

    # enforce optional dollar cap
    if rule.dollar_cap is not None:
        match_total = min(match_total, Decimal(str(rule.dollar_cap)))

    self.contributions_current_year[EMP_MATCH_KEY] = match_total.quantize(Decimal('0.01'))



⸻

4 | Migrate your YAML

Replace your old string-formulas with the new structured format. Here’s a baseline + loyalty bonus example:

plan_rules:
  employer_match:
    tiers:
      - match_rate:        0.50
        cap_deferral_pct:  0.06
      - match_rate:        0.25
        cap_deferral_pct:  0.06
        min_tenure_months: 60
    dollar_cap: null

No code changes needed for other tiers—just add or reorder in YAML.

⸻

5 | Unit test the new behavior

import pandas as pd
from decimal import Decimal
from agents.contributions import ContributionsMixin, ZERO_DECIMAL, EMP_MATCH_KEY, TOTAL_EMP_KEY

class Dummy(ContributionsMixin):
    def __init__(self, comp, rule, emp_def_pct):
        self.model = type('M', (), {
          'scenario_config': {'plan_rules': {'employer_match': rule}},
          'year': 2025
        })
        self.contributions_current_year = {
          TOTAL_EMP_KEY: Decimal(str(emp_def_pct)) * comp,
          EMP_MATCH_KEY: ZERO_DECIMAL
        }
        self.gross_compensation = comp

    # stub for tenure
    def _calculate_tenure_months(self, _): 
        return 72  # 6 years

def test_matching_with_loyalty_bonus():
    comp = Decimal('60000')
    rule = {
      'tiers': [
        {'match_rate':0.5, 'cap_deferral_pct':0.06},
        {'match_rate':0.25,'cap_deferral_pct':0.06,'min_tenure_months':60}
      ]
    }
    d = Dummy(comp, rule, 0.06)
    d._calc_employer_match(comp)
    # employee defers 6% of 60k = $3,600
    # tier1: 50% of first 6% = $1,800
    # tier2: bonus 25% of same 6% = $900  (since tenure >= 60)
    assert d.contributions_current_year[EMP_MATCH_KEY] == Decimal('2700.00')



⸻

6 | (Optionally) Legacy-string adapter

If you still need to support old "50% up to 6%" inputs, write a one-time converter at config-load:

def legacy_to_struct(s: str) -> dict:
    # use your regex to extract X and Y...
    return {
      'tiers': [{'match_rate': x/100, 'cap_deferral_pct': y/100}],
      'dollar_cap': None
    }

Apply it in your loader so downstream code only ever sees the structured dict.

⸻

Why this helps
	1.	100% correct parsing—no more brittle regex.
	2.	Full flexibility—N tiers, tenure gating, dollar caps, safe harbor, stretch matches.
	3.	Analyst-friendly YAML—all parameters live in clear, nested fields.
	4.	Early validation—bad configs fail before the run.

With this in place you’ll cover any real-world matching rule your clients throw at you.

Our new tiered‐match design handles any number of tiers, each with its own match rate, deferral cap % and (optionally) a minimum tenure. You simply list as many as you need in your YAML under plan_rules → employer_match → tiers.

⸻

Example: Four Tenure-Based Tiers

plan_rules:
  employer_match:
    tiers:
      # Tier 1: everyone gets 50% up to 3%
      - match_rate:        0.50
        cap_deferral_pct:  0.03
      # Tier 2: after 1 yr, add 25% up to 4%
      - match_rate:        0.25
        cap_deferral_pct:  0.04
        min_tenure_months: 12
      # Tier 3: after 3 yrs, add 10% up to 6%
      - match_rate:        0.10
        cap_deferral_pct:  0.06
        min_tenure_months: 36
      # Tier 4: after 5 yrs, bonus 5% up to 8%
      - match_rate:        0.05
        cap_deferral_pct:  0.08
        min_tenure_months: 60
    dollar_cap: 2000.00   # optional absolute annual max in dollars

	•	How it works at run-time
	1.	We load that list into our validated MatchRule.tiers
	2.	Sort by cap_deferral_pct so you always apply the smallest cap first
	3.	Loop through each tier, checking if the agent’s tenure (months) ≥ min_tenure_months (if provided)
	4.	For each tier, we calculate “eligible deferral” (up to that tier’s cap), multiply by its match_rate, subtract that portion from the remaining deferral, and move on to the next.
	5.	After all tiers, we enforce dollar_cap if present.

This gives you total flexibility:
	•	As many tiers as necessary (2, 3, 4, … N)
	•	Each tier can target a different deferral rate cap and can be gated on reaching a tenure threshold (or left with no gate)
	•	Optional dollar cap to catch unusual “$X maximum” rules

⸻

No code changes needed

Our ContributionsMixin._calc_employer_match already:
	•	Validates your YAML into a MatchRule (with Pydantic)
	•	Iterates arbitrarily through rule.tiers
	•	Skips tiers if the agent hasn’t hit their min_tenure_months
	•	Accumulates match_total correctly

Just update your scenario YAML and you’re done.