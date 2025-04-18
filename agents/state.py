from pandas import Timestamp
import pandas as pd
from dateutil.relativedelta import relativedelta
from utils.decimal_helpers import ZERO_DECIMAL  # Moved to utils
from utils.status_enums import EnrollmentMethod, EmploymentStatus  # Moved status enums to utils

class StateMixin:
    """Mixin providing status constants and age/tenure logic."""

    # Enrollment method constants
    ENROLL_METHOD_AE = EnrollmentMethod.AE.value
    ENROLL_METHOD_MANUAL = EnrollmentMethod.MANUAL.value
    ENROLL_METHOD_NONE = EnrollmentMethod.NONE.value

    # Employment status constants
    STATUS_PREV_TERMINATED = EmploymentStatus.PREV_TERMINATED.value
    STATUS_NEW_HIRE = EmploymentStatus.NEW_HIRE.value
    STATUS_ACTIVE_INITIAL = EmploymentStatus.ACTIVE_INITIAL.value
    STATUS_ACTIVE_CONTINUOUS = EmploymentStatus.ACTIVE_CONTINUOUS.value
    STATUS_NOT_HIRED = EmploymentStatus.NOT_HIRED.value
    STATUS_UNKNOWN = EmploymentStatus.UNKNOWN.value

    def _calculate_age(self, current_date: pd.Timestamp) -> int:
        """Calculates the agent's age as of a given date."""
        if pd.isnull(self.birth_date):
            return 0
        return relativedelta(current_date, self.birth_date).years

    def _calculate_tenure_months(self, current_date: pd.Timestamp) -> int:
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

    def _initialize_employment_status(self) -> None:
        """Determine initial employment status and is_active flag based on hire/termination dates and model start year."""
        term_date = self.termination_date
        if pd.isna(term_date):
            self.is_active = True
        else:
            self.is_active = (term_date.year >= self.model.start_year)
        if not self.is_active:
            self.employment_status = self.STATUS_PREV_TERMINATED
        else:
            if self.hire_date and pd.to_datetime(self.hire_date).year == self.model.start_year:
                self.employment_status = self.STATUS_NEW_HIRE
            elif self.hire_date and self.hire_year < self.model.start_year:
                self.employment_status = self.STATUS_ACTIVE_CONTINUOUS
            else:
                self.employment_status = self.STATUS_ACTIVE_INITIAL
        if self.hire_date:
            if self.hire_year < self.model.start_year:
                self.employment_status = self.STATUS_ACTIVE_CONTINUOUS
            elif self.hire_year == self.model.start_year:
                self.employment_status = self.STATUS_NEW_HIRE
            else:
                self.employment_status = self.STATUS_NOT_HIRED
        else:
            self.employment_status = self.STATUS_UNKNOWN

    def _determine_status_for_year(self) -> None:
        """Update employment_status at end of year based on new hire flag."""
        if getattr(self, 'is_new_hire', False):
            self.employment_status = self.STATUS_NEW_HIRE
        else:
            self.employment_status = self.STATUS_ACTIVE_CONTINUOUS
