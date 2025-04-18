from enum import Enum

class EnrollmentMethod(Enum):
    """Enumeration of enrollment methods."""
    AE = 'AE'
    MANUAL = 'Manual'
    NONE = 'None'

class EmploymentStatus(Enum):
    """Enumeration of employment statuses."""
    PREV_TERMINATED = 'Previously Terminated'
    NEW_HIRE = 'New Hire Active'
    ACTIVE_INITIAL = 'Active Initial'
    ACTIVE_CONTINUOUS = 'Active Continuous'
    NOT_HIRED = 'Not Hired'
    UNKNOWN = 'Unknown'
