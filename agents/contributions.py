# agents/contributions.py
import logging
import re
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import pandas as pd
from utils.decimal_helpers import ZERO_DECIMAL  # Shared decimal helper
from utils.rules.validators import load_match_rule
from typing import Dict, Union

logger = logging.getLogger(__name__)

# Constants for contribution keys
EMP_PRETAX_KEY = 'employee_pretax'
EMP_CATCHUP_KEY = 'employee_catchup'
EMP_MATCH_KEY = 'employer_match'
EMP_NEC_KEY = 'employer_nec'
TOTAL_EMP_KEY = 'total_employee'
TOTAL_ER_KEY = 'total_employer'

# Default IRS limits (used if scenario_config missing)
DEFAULT_IRS_LIMITS = {
    'compensation_limit': Decimal('345000'),
    'deferral_limit': Decimal('23000'),
    'catchup_limit': Decimal('7500'),
    'catchup_eligibility_age': 50
}

class ContributionsMixin:
    """
    Mixin providing precise, configurable contribution calculation logic.

    Attributes required on self:
      - is_active (bool)
      - is_participating (bool)
      - gross_compensation (Decimal)
      - deferral_rate (Decimal)
      - participation_date (datetime-like or None)
      - model.year (int)
      - model.scenario_config['plan_rules']['employer_match']: dict with:
          tiers: List[{
            match_rate: float,
            cap_deferral_pct: float,
            min_tenure_months?: int
          }]
          dollar_cap?: float

    After _calculate_contributions(), sets self.contributions_current_year keys:
      employee_pretax, employee_catchup, employer_match,
      employer_nec, total_employee, total_employer
    """

    def _calculate_contributions(self):
        """Orchestrate all contribution steps for the year."""
        # Initialize output structure
        self.contributions_current_year = {
            EMP_PRETAX_KEY: ZERO_DECIMAL,
            EMP_CATCHUP_KEY: ZERO_DECIMAL,
            EMP_MATCH_KEY: ZERO_DECIMAL,
            EMP_NEC_KEY: ZERO_DECIMAL,
            TOTAL_EMP_KEY: ZERO_DECIMAL,
            TOTAL_ER_KEY: ZERO_DECIMAL,
        }

        if not getattr(self, 'is_active', False) or not getattr(self, 'is_participating', False):
            return

        gross_comp = self._safe_decimal(self.gross_compensation, 'gross_compensation')
        irs = self._load_irs_limits()
        eligible_comp = min(gross_comp, irs['compensation_limit'])
        if eligible_comp <= ZERO_DECIMAL:
            return

        # 1. Employee pre-tax
        self._calc_employee_deferral(eligible_comp, irs['deferral_limit'])
        # 2. Catch-up
        self._calc_catchup(eligible_comp, irs)
        # 3. Employer match
        self._calc_employer_match(eligible_comp)
        # 4. Employer NEC
        self._calc_employer_nec(eligible_comp)
        # 5. Totals
        self._finalize_totals()
        # 6. Proration if mid-year participation
        self._prorate_contributions()

    IRS_KEY_MAP = {
        # canonical: [legacy, ...]
        'compensation_limit': ['comp_limit', 'compensation_limit'],
        'deferral_limit': ['deferral_limit', 'deferral'],
        'catchup_limit': ['catchup_limit', 'catch_up', 'catchup'],
        'catchup_eligibility_age': ['catchup_eligibility_age', 'catch_up_age', 'catchup_age'],
    }
    def _load_irs_limits(self) -> Dict[str, Union[Decimal, int]]:
        """Retrieve and convert IRS limits for the current year, supporting legacy and canonical keys."""
        cfg = self.model.scenario_config.get('irs_limits', {})
        limits = cfg.get(self.model.year, {})
        if not limits:
            return DEFAULT_IRS_LIMITS.copy()
        out = {}
        for canon, legacy_keys in ContributionsMixin.IRS_KEY_MAP.items():
            val = None
            for k in legacy_keys:
                if k in limits:
                    val = self._safe_decimal(limits[k], canon)
                    break
            if val is None:
                val = DEFAULT_IRS_LIMITS[canon]
            out[canon] = val
        return out


    def _calc_employee_deferral(self, comp, def_limit):
        """Calculate pre-tax deferral up to IRS limit."""
        rate = self._safe_decimal(self.deferral_rate, 'deferral_rate')
        potential = (comp * rate).quantize(Decimal('0.01'), ROUND_HALF_UP)
        actual = min(potential, def_limit)
        self.contributions_current_year[EMP_PRETAX_KEY] = actual

    def _calc_catchup(self, comp, irs):
        """Calculate catch-up contributions for eligible age."""
        try:
            age = self._calculate_age(pd.Timestamp(year=self.model.year, month=12, day=31))
        except Exception:
            return
        if age >= irs['catchup_eligibility_age']:
            pretax = self.contributions_current_year[EMP_PRETAX_KEY]
            remaining = max(ZERO_DECIMAL, (comp * self._safe_decimal(self.deferral_rate, 'deferral_rate')) - pretax)
            catchup = min(remaining, irs['catchup_limit'])
            self.contributions_current_year[EMP_CATCHUP_KEY] = catchup.quantize(Decimal('0.01'), ROUND_HALF_UP)

    def _calc_employer_match(self, comp: Decimal) -> None:
        """Apply structured tiered employer match including tenure gating and dollar cap."""
        raw = self.model.scenario_config.get('plan_rules', {}).get('employer_match', {})
        if not raw:
            return
        try:
            rule = load_match_rule(raw)
        except ValueError as e:
            logger.warning("%r: Invalid employer_match config: %s", self, e)
            return

        # compute total employee deferrals
        emp_contrib = self.contributions_current_year[EMP_PRETAX_KEY] + self.contributions_current_year[EMP_CATCHUP_KEY]
        match_total = Decimal('0')
        remaining = emp_contrib

        # compute tenure in months as of year-end
        as_of = pd.Timestamp(f"{self.model.year}-12-31")
        tenure_months = self._calculate_tenure_months(as_of)

        for tier in rule.tiers:
            if tier.min_tenure_months is not None and tenure_months < tier.min_tenure_months:
                continue
            cap_amount = (comp * Decimal(str(tier.cap_deferral_pct))).quantize(Decimal('0.01'))
            eligible = min(remaining, cap_amount)
            match_amt = eligible * Decimal(str(tier.match_rate))
            match_total += match_amt
            remaining -= eligible
            if remaining <= ZERO_DECIMAL:
                break

        if rule.dollar_cap is not None:
            match_total = min(match_total, Decimal(str(rule.dollar_cap)))

        self.contributions_current_year[EMP_MATCH_KEY] = match_total.quantize(Decimal('0.01'))

    def _calc_employer_nec(self, comp):
        """Parse and apply non-elective contribution formula."""
        fmt = self.model.scenario_config.get('employer_nec_formula', '')
        if not fmt:
            return
        m = re.match(r"(?P<rate>[0-9.]+)_pct", fmt)
        if not m:
            logger.warning("%r: Unexpected NEC formula '%s'", self, fmt)
            return
        try:
            pct = Decimal(m.group('rate')) / Decimal('100')
            nec_amt = (comp * pct).quantize(Decimal('0.01'), ROUND_HALF_UP)
            self.contributions_current_year[EMP_NEC_KEY] = nec_amt
        except (InvalidOperation, ValueError) as e:
            logger.warning("%r: Error computing NEC from '%s': %s", self, fmt, e)

    def _finalize_totals(self):
        """Sum up employee and employer contributions."""
        emp = self.contributions_current_year[EMP_PRETAX_KEY] + self.contributions_current_year[EMP_CATCHUP_KEY]
        er = self.contributions_current_year[EMP_MATCH_KEY] + self.contributions_current_year[EMP_NEC_KEY]
        self.contributions_current_year[TOTAL_EMP_KEY] = emp
        self.contributions_current_year[TOTAL_ER_KEY] = er

    def _prorate_contributions(self):
        """Prorate all contributions based on mid-year participation_date."""
        pd_date = getattr(self, 'participation_date', None)
        if pd_date is None or pd.isna(pd_date):
            return
        start = pd.Timestamp(year=self.model.year, month=1, day=1)
        end = pd.Timestamp(year=self.model.year, month=12, day=31)
        part = pd.to_datetime(pd_date)
        if part > end:
            return
        part_start = max(part, start)
        days_active = (end - part_start).days + 1
        days_year = (end - start).days + 1
        frac = Decimal(days_active) / Decimal(days_year)
        for key, val in self.contributions_current_year.items():
            prorated = (val * frac).quantize(Decimal('0.01'), ROUND_HALF_UP)
            self.contributions_current_year[key] = prorated

    def _safe_decimal(self, value, name):
        """Convert a value to Decimal, logging if invalid."""
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError) as e:
            logger.warning("%r: Invalid decimal for %s: %s", self, name, e)
            return ZERO_DECIMAL