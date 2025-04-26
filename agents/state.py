# agents/state.py
import pandas as pd  # type: ignore[import-untyped]
import logging
from dateutil.relativedelta import relativedelta
from typing import Any, Optional
from utils.status_enums import EnrollmentMethod, EmploymentStatus

logger = logging.getLogger(__name__)


class StateMixin:
    """Mixin providing employment status logic and age/tenure helpers."""
    model: Any  # Provided by host simulation model
    birth_date: Optional[pd.Timestamp]
    hire_date: Optional[pd.Timestamp]
    termination_date: Optional[pd.Timestamp]
    is_active: bool
    employment_status: str
    is_new_hire: bool

    # Enrollment method constants
    ENROLL_METHOD_AE = EnrollmentMethod.AE.value
    ENROLL_METHOD_MANUAL = EnrollmentMethod.MANUAL.value
    ENROLL_METHOD_NONE = EnrollmentMethod.NONE.value

    # Employment status constants
    STATUS_PREV_TERMINATED = EmploymentStatus.PREV_TERMINATED.value
    STATUS_NEW_HIRE = EmploymentStatus.NEW_HIRE.value
    STATUS_ACTIVE_CONTINUOUS = EmploymentStatus.ACTIVE_CONTINUOUS.value
    STATUS_NOT_HIRED = EmploymentStatus.NOT_HIRED.value
    STATUS_UNKNOWN = EmploymentStatus.UNKNOWN.value

    def _calculate_age(self, current_date: pd.Timestamp) -> int:
        """Return age in years as of current_date."""
        birth = getattr(self, 'birth_date', None)
        if pd.isna(birth):
            return 0
        return relativedelta(current_date, pd.to_datetime(birth)).years

    def _calculate_tenure_months(self, current_date: pd.Timestamp) -> int:
        """Return tenure in whole months as of current_date."""
        hire = getattr(self, 'hire_date', None)
        if pd.isna(hire):
            return 0
        hire_ts = pd.to_datetime(hire)
        years = current_date.year - hire_ts.year
        months = current_date.month - hire_ts.month
        total = years * 12 + months
        if current_date.day < hire_ts.day:
            total -= 1
        return max(0, total)

    def _initialize_employment_status(self) -> None:
        """Set is_active and employment_status based on hire/term dates."""
        term_val = getattr(self, 'termination_date', None)
        tdate = pd.to_datetime(term_val)
        self.is_active = pd.isna(tdate) or tdate.year >= self.model.start_year
        logger.debug(
            "%r _initialize_employment_status: is_active=%s",
            self, self.is_active,
        )

        if not self.is_active:
            self.employment_status = self.STATUS_PREV_TERMINATED
            logger.debug(
                "%r status set to PREV_TERMINATED", self
            )
            return

        hire = getattr(self, 'hire_date', None)
        if pd.isna(hire):
            self.employment_status = self.STATUS_UNKNOWN
            logger.debug(
                "%r status set to UNKNOWN", self
            )
            return

        year = pd.to_datetime(hire).year
        if year > self.model.start_year:
            self.employment_status = self.STATUS_NOT_HIRED
        elif year == self.model.start_year:
            self.employment_status = self.STATUS_NEW_HIRE
        else:
            self.employment_status = self.STATUS_ACTIVE_CONTINUOUS
        logger.debug(
            "%r status initialized to %s", self, self.employment_status
        )

    def _determine_status_for_year(self) -> None:
        """Update employment_status for agents post-hire year."""
        if getattr(self, 'is_new_hire', False):
            self.employment_status = self.STATUS_NEW_HIRE
        else:
            self.employment_status = self.STATUS_ACTIVE_CONTINUOUS
