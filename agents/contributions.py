from decimal import Decimal, ROUND_HALF_UP
import pandas as pd

from utils.decimal_helpers import ZERO_DECIMAL  # Shared decimal helper

class ContributionsMixin:
    """Mixin providing contribution calculation logic."""

    def _calculate_contributions(self):
        """Calculates employee and employer contributions for the year using Decimal."""
        # Reset contributions for the year
        self.contributions_current_year = {
            'employee_pretax': ZERO_DECIMAL,
            'employee_catchup': ZERO_DECIMAL,
            'employer_match': ZERO_DECIMAL,
            'employer_nec': ZERO_DECIMAL,
            'total_employee': ZERO_DECIMAL,
            'total_employer': ZERO_DECIMAL,
        }

        # No contributions if inactive or not participating
        if not self.is_active or not self.is_participating:
            return

        gross_comp = Decimal(str(self.gross_compensation))
        config = self.model.scenario_config
        irs_limits = config.get('irs_limits', {}).get(self.model.year, {})
        if not irs_limits:
            irs_limits = {'compensation_limit': Decimal('345000'),
                          'deferral_limit': Decimal('23000'),
                          'catchup_limit': Decimal('7500'),
                          'catchup_eligibility_age': 50}
        else:
            # Convert numeric limits to Decimal where appropriate
            irs_limits = {k: Decimal(str(v)) if k != 'catchup_eligibility_age' else v for k, v in irs_limits.items()}

        comp_limit = irs_limits.get('compensation_limit')
        deferral_limit = irs_limits.get('deferral_limit')
        catchup_limit = irs_limits.get('catchup_limit')
        catchup_age = irs_limits.get('catchup_eligibility_age')

        eligible_comp = min(gross_comp, comp_limit)
        if eligible_comp <= ZERO_DECIMAL:
            return

        rate = Decimal(str(self.deferral_rate))
        potential_deferral = (eligible_comp * rate).quantize(Decimal('0.01'), ROUND_HALF_UP)
        actual_deferral = min(potential_deferral, deferral_limit)
        self.contributions_current_year['employee_pretax'] = actual_deferral

        # Catch-up
        try:
            age = self._calculate_age(pd.Timestamp(f"{self.model.year}-12-31"))
            if age >= catchup_age:
                remaining = max(ZERO_DECIMAL, potential_deferral - actual_deferral)
                catchup_amt = min(remaining, catchup_limit)
                self.contributions_current_year['employee_catchup'] = catchup_amt.quantize(Decimal('0.01'), ROUND_HALF_UP)
        except:
            pass

        # Total employee
        total_emp = (self.contributions_current_year['employee_pretax'] +
                     self.contributions_current_year['employee_catchup'])
        self.contributions_current_year['total_employee'] = total_emp

        # Employer match
        match_formula = config.get('employer_match_formula', '')
        if match_formula and total_emp > ZERO_DECIMAL:
            try:
                parts = match_formula.split('_')
                match_rate = Decimal(parts[0])
                match_on = Decimal(parts[2])
                cap_pct = Decimal(parts[5]) / Decimal('100')
                cap_base = (eligible_comp * cap_pct).quantize(Decimal('0.01'), ROUND_HALF_UP)
                subj = min(total_emp, cap_base)
                self.contributions_current_year['employer_match'] = (subj * match_rate * match_on).quantize(Decimal('0.01'), ROUND_HALF_UP)
            except:
                pass

        # Employer NEC
        nec = config.get('employer_nec_formula', '')
        if nec:
            try:
                rate_pct = Decimal(nec.split('_')[0]) / Decimal('100')
                self.contributions_current_year['employer_nec'] = (eligible_comp * rate_pct).quantize(Decimal('0.01'), ROUND_HALF_UP)
            except:
                pass

        # Total employer
        self.contributions_current_year['total_employer'] = (
            self.contributions_current_year['employer_match'] +
            self.contributions_current_year['employer_nec']
        )

        # Prorate contributions for mid-year participation
        start_of_year = pd.Timestamp(year=self.model.year, month=1, day=1)
        end_of_year = pd.Timestamp(year=self.model.year, month=12, day=31)
        pd_date = self.participation_date
        if pd_date and not pd.isna(pd_date):
            part_start = pd.to_datetime(pd_date)
            if part_start <= end_of_year:
                part_start = max(part_start, start_of_year)
                days_active = (end_of_year - part_start).days + 1
                days_year = (end_of_year - start_of_year).days + 1
                fraction = Decimal(days_active) / Decimal(days_year)
                for key, val in self.contributions_current_year.items():
                    self.contributions_current_year[key] = (val * fraction).quantize(Decimal('0.01'), ROUND_HALF_UP)
