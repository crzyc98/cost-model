"""
utils/constants.py

Project-wide constants for employment status and other magic values.
"""

from utils.status_enums import EmploymentStatus

## Statuses considered "currently active" in plan logic.
ACTIVE_STATUSES = {
    EmploymentStatus.NEW_HIRE.value,
    EmploymentStatus.ACTIVE_INITIAL.value,
    EmploymentStatus.ACTIVE_CONTINUOUS.value,
}

# Raw string constants for backward compatibility
ACTIVE_STATUS       = EmploymentStatus.ACTIVE_INITIAL.value
NEW_HIRE_STATUS     = EmploymentStatus.NEW_HIRE.value
TERMINATED_STATUS   = EmploymentStatus.TERMINATED.value
INACTIVE_STATUS     = EmploymentStatus.INACTIVE.value
NOT_HIRED_STATUS    = EmploymentStatus.NOT_HIRED.value

__all__ = [
    'EmploymentStatus', 'ACTIVE_STATUSES',
    'ACTIVE_STATUS','NEW_HIRE_STATUS','TERMINATED_STATUS',
    'INACTIVE_STATUS','NOT_HIRED_STATUS'
]
