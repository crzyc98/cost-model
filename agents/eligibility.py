import pandas as pd
from dateutil.relativedelta import relativedelta


class EligibilityMixin:
    """Mixin providing eligibility and age/tenure logic."""

    def _calculate_age(self, current_date: pd.Timestamp) -> int:
        """Calculates the agent's age as of a given date."""
        if pd.isnull(self.birth_date):
            return 0
        return relativedelta(current_date, self.birth_date).years

    def _calculate_tenure_months(self, current_date) -> int:
        """Calculates the agent's tenure in months as of a given date."""
        if pd.isna(self.hire_date):
            return 0
        hire_date_ts = pd.Timestamp(self.hire_date)
        delta_years = current_date.year - hire_date_ts.year
        delta_months = current_date.month - hire_date_ts.month
        total_months = delta_years * 12 + delta_months
        if current_date.day < hire_date_ts.day:
            total_months -= 1
        return max(0, total_months)

    def _update_eligibility(self):
        """Updates the agent's eligibility status based on plan rules."""
        # Skip if already eligible (e.g., grandfathered)
        if self.is_eligible:
            return

        rules = self.model.plan_rules.get('eligibility', {})
        min_age = rules.get('min_age', 0)
        min_service = rules.get('min_service_months', 0)

        # Service requirement
        if min_service == 0:
            meets_service = True
        else:
            end_of_year = pd.Timestamp(f"{self.model.year}-12-31")
            meets_service = (self._calculate_tenure_months(end_of_year) >= min_service)

        # Age requirement
        end_of_year = pd.Timestamp(f"{self.model.year}-12-31")
        age = self._calculate_age(end_of_year)
        meets_age = (age >= min_age)

        self.is_eligible = meets_age and meets_service
