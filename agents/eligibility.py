# agents/eligibility.py
import logging
from typing import Any, Callable, Optional

import pandas as pd
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)


class EligibilityMixin:
    """Mixin providing eligibility and age/tenure logic for agents.

    Attributes required on self:
      - self.birth_date: pd.Timestamp or None
      - self.hire_date: pd.Timestamp or None
      - self.is_eligible: bool
      - self.model.year: int
      - self.plan_rules: dict containing 'eligibility' key.

    """

    @staticmethod
    def custom_eligibility_checker(
        agent: Any,
        as_of_date: pd.Timestamp
    ) -> bool:
        """
        Stub for custom eligibility logic. Override as needed.

        Args:
            agent: The agent instance.
            as_of_date: The date for eligibility check.

        Returns:
            bool: Eligibility status.
        """
        # Example: Always ineligible (override with your own logic)
        return False

    @property
    def plan_rules(self) -> dict:
        """Return plan rules dict or {}."""
        return getattr(self.model, "scenario_config", {}).get("plan_rules", {})

    def _calculate_age(self, current_date: pd.Timestamp) -> int:
        """Calculate age in years as of the given current_date."""
        birth = getattr(self, 'birth_date', None)
        if not isinstance(birth, pd.Timestamp) or pd.isna(birth):
            return 0
        delta = relativedelta(current_date, birth)
        return max(0, delta.years)

    def _calculate_tenure_months(self, current_date: pd.Timestamp) -> int:
        """Calculate tenure in whole months as of the given current_date."""
        hire = getattr(self, 'hire_date', None)
        if not isinstance(hire, pd.Timestamp) or pd.isna(hire):
            return 0
        delta_years = current_date.year - hire.year
        delta_months = current_date.month - hire.month
        months = delta_years * 12 + delta_months
        if current_date.day < hire.day:
            months -= 1
        return max(0, months)

    def _update_eligibility(self) -> None:
        """Update eligibility based on plan rules or custom checker."""
        prior = getattr(self, 'is_eligible', False)
        rules = self.plan_rules.get('eligibility', {})
        min_age = rules.get('min_age', 0)
        min_service = rules.get('min_service_months', 0)
        custom_checker: Optional[
            Callable[[Any, pd.Timestamp], bool]
        ] = rules.get('custom_checker')
        end_of_year = pd.Timestamp(
            f"{self.model.year}-12-31"
        )
        if min_service:
            meets_service = (
                self._calculate_tenure_months(end_of_year)
                >= min_service
            )
        else:
            meets_service = True
        meets_age = (
            self._calculate_age(end_of_year)
            >= min_age
        )
        if callable(custom_checker):
            new_status = custom_checker(self, end_of_year)
        else:
            new_status = meets_age and meets_service
        if new_status != prior:
            logger.debug(
                "%r eligibility changed from %s to %s "
                "(age>=%s:%s, tenure>=%s:%s)",
                self,
                prior,
                new_status,
                min_age,
                meets_age,
                min_service,
                meets_service,
            )
        self.is_eligible = new_status
