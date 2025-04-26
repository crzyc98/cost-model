# agents/contributions.py
import logging
import re
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import pandas as pd
from utils.decimal_helpers import ZERO_DECIMAL  # Shared decimal helper

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
    """Mixin providing precise, configurable contribution calculation logic.

    Attributes required on self:
      - self.is_active (bool)
      - self.is_participating (bool)
      - self.gross_compensation (Decimal or numeric)
      - self.deferral_rate (Decimal or numeric)
      - self.participation_date (datetime-like or None)
      - self.model.year (int)
      - self.model.scenario_config (dict) with keys:
          - 'irs_limits': {year: {lim keys}}
          - 'employer_match_formula': str, e.g. "1.0_of_1.0_up_to_6.0_pct"
          - 'employer_nec_formula': str, e.g. "3.0_pct"

    After running _calculate_contributions(), the agent will have:
      self.contributions_current_year: dict with keys:
        EMP_PRETAX_KEY, EMP_CATCHUP_KEY, EMP_MATCH_KEY,
        EMP_NEC_KEY, TOTAL_EMP_KEY, TOTAL_ER_KEY
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
    def _load_irs_limits(self):
        """Retrieve and convert IRS limits for the current year, supporting legacy and canonical keys."""
        cfg = self.model.scenario_config.get('irs_limits', {})
        limits = cfg.get(self.model.year, {})
        if not limits:
            return DEFAULT_IRS_LIMITS.copy()
        out = {}
        for canon, legacy_keys in self.IRS_KEY_MAP.items():
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

    def _calc_employer_match(self, comp):
        """Parse and apply employer match formula, supporting canonical and legacy patterns."""
        fmt = self.model.scenario_config.get('employer_match_formula', '')
        total_emp = self.contributions_current_year[EMP_PRETAX_KEY] + self.contributions_current_year[EMP_CATCHUP_KEY]
        if not fmt or total_emp <= ZERO_DECIMAL:
            return
        # Canonical: "1.0_of_1.0_up_to_6.0_pct"
        # Legacy: "50% up to 6%" or "50pct up to 6pct"
        m = re.match(r"(?P<rate>[0-9.]+)_of_(?P<on>[0-9.]+)_up_to_(?P<cap>[0-9.]+)_pct", fmt)
        if not m:
            m = re.match(r"(?P<rate>[0-9.]+)[%p][c]*\s*up to\s*(?P<cap>[0-9.]+)[%p][c]*", fmt, re.IGNORECASE)
            if m:
                # Legacy: treat as match_rate% up to cap% of comp
                try:
                    rate = Decimal(m.group('rate')) / Decimal('100')
                    on = Decimal('1.0')  # Assume 100% of comp
                    cap_pct = Decimal(m.group('cap')) / Decimal('100')
                    cap_base = (comp * cap_pct).quantize(Decimal('0.01'), ROUND_HALF_UP)
                    subj = min(total_emp, cap_base)
                    match_amt = (subj * rate * on).quantize(Decimal('0.01'), ROUND_HALF_UP)
                    self.contributions_current_year[EMP_MATCH_KEY] = match_amt
                except (InvalidOperation, ValueError) as e:
                    logger.warning("%r: Error computing legacy match from '%s': %s", self, fmt, e)
                return
            logger.warning("%r: Unexpected match formula '%s'", self, fmt)
            return
        try:
            rate = Decimal(m.group('rate'))
            on = Decimal(m.group('on'))
            cap_pct = Decimal(m.group('cap')) / Decimal('100')
            cap_base = (comp * cap_pct).quantize(Decimal('0.01'), ROUND_HALF_UP)
            subj = min(total_emp, cap_base)
            match_amt = (subj * rate * on).quantize(Decimal('0.01'), ROUND_HALF_UP)
            self.contributions_current_year[EMP_MATCH_KEY] = match_amt
        except (InvalidOperation, ValueError) as e:
            logger.warning("%r: Error computing match from '%s': %s", self, fmt, e)

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