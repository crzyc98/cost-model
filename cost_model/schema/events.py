"""
Event type definitions and constants.

This module defines all event types used in the cost model simulation
and provides utilities for event classification and validation.
"""

from enum import Enum
from typing import Dict, List, Optional, Set


class EventTypes(str, Enum):
    """Enumeration of all event types in the system."""

    # Core employment events
    HIRE = "hire"
    TERMINATION = "termination"
    NEW_HIRE_TERMINATION = "new_hire_termination"

    # Compensation events
    COMPENSATION = "compensation"
    COLA = "cola"
    RAISE = "raise"
    PROMOTION = "promotion"

    # Plan participation events
    CONTRIBUTION = "contribution"
    ENROLLMENT = "enrollment"
    DEFERRAL_CHANGE = "deferral_change"

    # Administrative events
    DATA_CORRECTION = "data_correction"
    REHIRE = "rehire"
    STATUS_CHANGE = "status_change"

    # Plan rule events
    AUTO_ENROLLMENT = "auto_enrollment"
    AUTO_INCREASE = "auto_increase"
    ELIGIBILITY_CHANGE = "eligibility_change"

    # System events
    SIMULATION_START = "simulation_start"
    SIMULATION_END = "simulation_end"
    YEAR_END_PROCESSING = "year_end_processing"


class EventStatus(str, Enum):
    """Status of events in the system."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


class EventCategories:
    """Categories of events for processing and validation."""

    # Events that affect employment status
    EMPLOYMENT_EVENTS: Set[str] = {
        EventTypes.HIRE,
        EventTypes.TERMINATION,
        EventTypes.NEW_HIRE_TERMINATION,
        EventTypes.REHIRE,
        EventTypes.STATUS_CHANGE,
    }

    # Events that affect compensation
    COMPENSATION_EVENTS: Set[str] = {
        EventTypes.COMPENSATION,
        EventTypes.COLA,
        EventTypes.RAISE,
        EventTypes.PROMOTION,
    }

    # Events that affect plan participation
    PLAN_EVENTS: Set[str] = {
        EventTypes.CONTRIBUTION,
        EventTypes.ENROLLMENT,
        EventTypes.DEFERRAL_CHANGE,
        EventTypes.AUTO_ENROLLMENT,
        EventTypes.AUTO_INCREASE,
        EventTypes.ELIGIBILITY_CHANGE,
    }

    # Events that create new employees
    EMPLOYEE_CREATION_EVENTS: Set[str] = {
        EventTypes.HIRE,
        EventTypes.REHIRE,
    }

    # Events that remove employees
    EMPLOYEE_REMOVAL_EVENTS: Set[str] = {
        EventTypes.TERMINATION,
        EventTypes.NEW_HIRE_TERMINATION,
    }

    # Events that modify existing employee data
    EMPLOYEE_MODIFICATION_EVENTS: Set[str] = {
        EventTypes.COMPENSATION,
        EventTypes.COLA,
        EventTypes.RAISE,
        EventTypes.PROMOTION,
        EventTypes.CONTRIBUTION,
        EventTypes.DEFERRAL_CHANGE,
        EventTypes.STATUS_CHANGE,
        EventTypes.AUTO_ENROLLMENT,
        EventTypes.AUTO_INCREASE,
        EventTypes.ELIGIBILITY_CHANGE,
        EventTypes.DATA_CORRECTION,
    }

    # System/administrative events
    SYSTEM_EVENTS: Set[str] = {
        EventTypes.SIMULATION_START,
        EventTypes.SIMULATION_END,
        EventTypes.YEAR_END_PROCESSING,
    }

    # Events that require special validation
    HIGH_PRIORITY_EVENTS: Set[str] = {
        EventTypes.HIRE,
        EventTypes.TERMINATION,
        EventTypes.COMPENSATION,
        EventTypes.PROMOTION,
    }


class EventProcessingOrder:
    """Defines the order in which events should be processed."""

    # Processing order within a single day/time period
    PROCESSING_ORDER: List[str] = [
        # System events first
        EventTypes.SIMULATION_START,
        # Employment status changes
        EventTypes.HIRE,
        EventTypes.REHIRE,
        EventTypes.STATUS_CHANGE,
        # Compensation changes
        EventTypes.PROMOTION,  # Promotion before other comp changes
        EventTypes.COMPENSATION,
        EventTypes.COLA,
        EventTypes.RAISE,
        # Plan participation changes
        EventTypes.ENROLLMENT,
        EventTypes.AUTO_ENROLLMENT,
        EventTypes.DEFERRAL_CHANGE,
        EventTypes.AUTO_INCREASE,
        EventTypes.CONTRIBUTION,
        EventTypes.ELIGIBILITY_CHANGE,
        # Terminations last (except new hire terminations)
        EventTypes.NEW_HIRE_TERMINATION,
        EventTypes.TERMINATION,
        # Administrative/correction events
        EventTypes.DATA_CORRECTION,
        # System events last
        EventTypes.YEAR_END_PROCESSING,
        EventTypes.SIMULATION_END,
    ]

    @classmethod
    def get_processing_priority(cls, event_type: str) -> int:
        """Get the processing priority for an event type.

        Args:
            event_type: The event type to get priority for

        Returns:
            Priority number (lower = higher priority)
        """
        try:
            return cls.PROCESSING_ORDER.index(event_type)
        except ValueError:
            # Unknown event types get lowest priority
            return len(cls.PROCESSING_ORDER)


class EventValidationRules:
    """Validation rules for different event types."""

    # Required fields for each event type
    REQUIRED_FIELDS: Dict[str, List[str]] = {
        EventTypes.HIRE: [
            "employee_id",
            "event_date",
            "gross_compensation",
        ],
        EventTypes.TERMINATION: [
            "employee_id",
            "event_date",
        ],
        EventTypes.NEW_HIRE_TERMINATION: [
            "employee_id",
            "event_date",
            "gross_compensation",  # Need comp for reconstruction
        ],
        EventTypes.COMPENSATION: [
            "employee_id",
            "event_date",
            "gross_compensation",
        ],
        EventTypes.PROMOTION: [
            "employee_id",
            "event_date",
            "gross_compensation",
            "job_level",
        ],
        EventTypes.CONTRIBUTION: [
            "employee_id",
            "event_date",
            "deferral_rate",
        ],
        EventTypes.ENROLLMENT: [
            "employee_id",
            "event_date",
        ],
    }

    # Optional fields for each event type
    OPTIONAL_FIELDS: Dict[str, List[str]] = {
        EventTypes.HIRE: [
            "job_level",
            "deferral_rate",
            "employee_birth_date",
        ],
        EventTypes.TERMINATION: [
            "termination_reason",
        ],
        EventTypes.COMPENSATION: [
            "job_level",
        ],
    }

    @classmethod
    def get_required_fields(cls, event_type: str) -> List[str]:
        """Get required fields for an event type.

        Args:
            event_type: The event type

        Returns:
            List of required field names
        """
        return cls.REQUIRED_FIELDS.get(event_type, ["employee_id", "event_date"])

    @classmethod
    def get_optional_fields(cls, event_type: str) -> List[str]:
        """Get optional fields for an event type.

        Args:
            event_type: The event type

        Returns:
            List of optional field names
        """
        return cls.OPTIONAL_FIELDS.get(event_type, [])


def is_employment_event(event_type: str) -> bool:
    """Check if an event affects employment status."""
    return event_type in EventCategories.EMPLOYMENT_EVENTS


def is_compensation_event(event_type: str) -> bool:
    """Check if an event affects compensation."""
    return event_type in EventCategories.COMPENSATION_EVENTS


def is_plan_event(event_type: str) -> bool:
    """Check if an event affects plan participation."""
    return event_type in EventCategories.PLAN_EVENTS


def creates_employee(event_type: str) -> bool:
    """Check if an event creates a new employee."""
    return event_type in EventCategories.EMPLOYEE_CREATION_EVENTS


def removes_employee(event_type: str) -> bool:
    """Check if an event removes an employee."""
    return event_type in EventCategories.EMPLOYEE_REMOVAL_EVENTS


def get_event_category(event_type: str) -> Optional[str]:
    """Get the primary category for an event type.

    Args:
        event_type: The event type to categorize

    Returns:
        Category name or None if not found
    """
    if is_employment_event(event_type):
        return "employment"
    elif is_compensation_event(event_type):
        return "compensation"
    elif is_plan_event(event_type):
        return "plan"
    elif event_type in EventCategories.SYSTEM_EVENTS:
        return "system"
    else:
        return None


def validate_event_type(event_type: str) -> bool:
    """Validate that an event type is recognized.

    Args:
        event_type: The event type to validate

    Returns:
        True if valid, False otherwise
    """
    return event_type in [et.value for et in EventTypes]
