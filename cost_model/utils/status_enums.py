# utils/status_enums.py

from enum import Enum


class EnrollmentMethod(Enum):
    """Enumeration of enrollment methods."""

    AE = "AE"
    MANUAL = "Manual"
    NONE = "None"


class EmploymentStatus(Enum):
    """Enumeration of employment statuses."""

    PREV_TERMINATED = "Previously Terminated"
    TERMINATED = "Terminated"
    NEW_HIRE = "New Hire"
    ACTIVE_INITIAL = "Active Initial"
    ACTIVE_CONTINUOUS = "Active Continuous"
    NOT_HIRED = "Not Hired"
    INACTIVE = "Inactive"


# Explicit exports
__all__ = ["EnrollmentMethod", "EmploymentStatus"]
