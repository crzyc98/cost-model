"""
Centralized column definitions for all data schemas.

This module defines all column names used throughout the cost model in a 
centralized, type-safe manner. It provides enums for different data types
and ensures consistency across all modules.
"""

from enum import Enum
from typing import List, Set, Dict, Any


class SnapshotColumns(str, Enum):
    """Column definitions for employee snapshot data."""
    
    # Core employee identifiers
    EMP_ID = "employee_id"
    SIMULATION_YEAR = "simulation_year"
    
    # Personal information
    EMP_HIRE_DATE = "employee_hire_date"
    EMP_BIRTH_DATE = "employee_birth_date"
    EMP_TERM_DATE = "employee_termination_date"
    
    # Employment status
    EMP_ACTIVE = "active"
    EMP_EXITED = "exited"
    EMP_STATUS_EOY = "employee_status_eoy"
    
    # Compensation
    EMP_GROSS_COMP = "employee_gross_compensation"
    EMP_DEFERRAL_RATE = "employee_deferral_rate"
    
    # Calculated fields
    EMP_TENURE = "employee_tenure"
    EMP_TENURE_BAND = "employee_tenure_band"
    EMP_AGE = "employee_age"
    EMP_AGE_BAND = "employee_age_band"
    
    # Job information
    EMP_LEVEL = "employee_level"
    EMP_LEVEL_SOURCE = "job_level_source"
    
    # Plan contributions
    EMP_CONTRIBUTION = "employee_contribution"
    EMPLOYER_CORE_CONTRIBUTION = "employer_core_contribution"
    EMPLOYER_MATCH_CONTRIBUTION = "employer_match_contribution"
    
    # Eligibility and enrollment
    IS_ELIGIBLE = "is_eligible"
    IS_ENROLLED = "is_enrolled"
    
    # Risk and modeling
    TERM_RATE = "term_rate"
    PROMOTION_RATE = "promotion_rate"


class EventColumns(str, Enum):
    """Column definitions for event data."""
    
    # Core event identifiers
    EVENT_ID = "event_id"
    EVENT_TYPE = "event_type"
    EVENT_DATE = "event_date"
    EVENT_STATUS = "event_status"
    
    # Employee reference
    EMP_ID = "employee_id"
    SIMULATION_YEAR = "simulation_year"
    
    # Event payload fields
    EVENT_PAYLOAD = "event_payload"
    
    # Common event data
    GROSS_COMPENSATION = "gross_compensation"
    DEFERRAL_RATE = "deferral_rate"
    JOB_LEVEL = "job_level"
    TERMINATION_REASON = "termination_reason"
    
    # Metadata
    CREATED_BY = "created_by"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"


class PlanRuleColumns(str, Enum):
    """Column definitions for plan rule data."""
    
    # Rule identifiers
    RULE_ID = "rule_id"
    RULE_TYPE = "rule_type"
    RULE_NAME = "rule_name"
    
    # Employee reference
    EMP_ID = "employee_id"
    
    # Rule configuration
    RULE_CONFIG = "rule_config"
    EFFECTIVE_DATE = "effective_date"
    EXPIRATION_DATE = "expiration_date"
    
    # Results
    RULE_RESULT = "rule_result"
    APPLIED_DATE = "applied_date"


class ColumnGroups:
    """Predefined groups of columns for common operations."""
    
    # Core employee data
    EMPLOYEE_CORE: List[str] = [
        SnapshotColumns.EMP_ID,
        SnapshotColumns.EMP_HIRE_DATE,
        SnapshotColumns.EMP_GROSS_COMP,
        SnapshotColumns.EMP_ACTIVE,
    ]
    
    # All personal information
    EMPLOYEE_PERSONAL: List[str] = [
        SnapshotColumns.EMP_ID,
        SnapshotColumns.EMP_HIRE_DATE,
        SnapshotColumns.EMP_BIRTH_DATE,
        SnapshotColumns.EMP_TERM_DATE,
    ]
    
    # Employment status fields
    EMPLOYMENT_STATUS: List[str] = [
        SnapshotColumns.EMP_ACTIVE,
        SnapshotColumns.EMP_EXITED,
        SnapshotColumns.EMP_STATUS_EOY,
    ]
    
    # Compensation fields
    COMPENSATION: List[str] = [
        SnapshotColumns.EMP_GROSS_COMP,
        SnapshotColumns.EMP_DEFERRAL_RATE,
    ]
    
    # Calculated demographics
    DEMOGRAPHICS: List[str] = [
        SnapshotColumns.EMP_TENURE,
        SnapshotColumns.EMP_TENURE_BAND,
        SnapshotColumns.EMP_AGE,
        SnapshotColumns.EMP_AGE_BAND,
    ]
    
    # Job-related fields
    JOB_INFO: List[str] = [
        SnapshotColumns.EMP_LEVEL,
        SnapshotColumns.EMP_LEVEL_SOURCE,
    ]
    
    # Plan contribution fields
    CONTRIBUTIONS: List[str] = [
        SnapshotColumns.EMP_CONTRIBUTION,
        SnapshotColumns.EMPLOYER_CORE_CONTRIBUTION,
        SnapshotColumns.EMPLOYER_MATCH_CONTRIBUTION,
    ]
    
    # Eligibility fields
    ELIGIBILITY: List[str] = [
        SnapshotColumns.IS_ELIGIBLE,
        SnapshotColumns.IS_ENROLLED,
    ]
    
    # Required fields for minimal snapshot
    REQUIRED_SNAPSHOT: List[str] = [
        SnapshotColumns.EMP_ID,
        SnapshotColumns.EMP_HIRE_DATE,
        SnapshotColumns.EMP_GROSS_COMP,
        SnapshotColumns.EMP_ACTIVE,
        SnapshotColumns.SIMULATION_YEAR,
    ]
    
    # All snapshot columns in logical order
    ALL_SNAPSHOT: List[str] = [
        SnapshotColumns.EMP_ID,
        SnapshotColumns.SIMULATION_YEAR,
        SnapshotColumns.EMP_HIRE_DATE,
        SnapshotColumns.EMP_BIRTH_DATE,
        SnapshotColumns.EMP_TERM_DATE,
        SnapshotColumns.EMP_ACTIVE,
        SnapshotColumns.EMP_EXITED,
        SnapshotColumns.EMP_STATUS_EOY,
        SnapshotColumns.EMP_GROSS_COMP,
        SnapshotColumns.EMP_DEFERRAL_RATE,
        SnapshotColumns.EMP_TENURE,
        SnapshotColumns.EMP_TENURE_BAND,
        SnapshotColumns.EMP_AGE,
        SnapshotColumns.EMP_AGE_BAND,
        SnapshotColumns.EMP_LEVEL,
        SnapshotColumns.EMP_LEVEL_SOURCE,
        SnapshotColumns.EMP_CONTRIBUTION,
        SnapshotColumns.EMPLOYER_CORE_CONTRIBUTION,
        SnapshotColumns.EMPLOYER_MATCH_CONTRIBUTION,
        SnapshotColumns.IS_ELIGIBLE,
        SnapshotColumns.IS_ENROLLED,
        SnapshotColumns.TERM_RATE,
        SnapshotColumns.PROMOTION_RATE,
    ]
    
    # Core event columns
    EVENT_CORE: List[str] = [
        EventColumns.EVENT_ID,
        EventColumns.EVENT_TYPE,
        EventColumns.EVENT_DATE,
        EventColumns.EMP_ID,
    ]
    
    # All event columns
    ALL_EVENT: List[str] = [
        EventColumns.EVENT_ID,
        EventColumns.EVENT_TYPE,
        EventColumns.EVENT_DATE,
        EventColumns.EVENT_STATUS,
        EventColumns.EMP_ID,
        EventColumns.SIMULATION_YEAR,
        EventColumns.EVENT_PAYLOAD,
        EventColumns.GROSS_COMPENSATION,
        EventColumns.DEFERRAL_RATE,
        EventColumns.JOB_LEVEL,
        EventColumns.TERMINATION_REASON,
        EventColumns.CREATED_BY,
        EventColumns.CREATED_AT,
        EventColumns.UPDATED_AT,
    ]


def get_column_group(group_name: str) -> List[str]:
    """Get a predefined column group by name.
    
    Args:
        group_name: Name of the column group
        
    Returns:
        List of column names in the group
        
    Raises:
        ValueError: If group_name is not found
    """
    if not hasattr(ColumnGroups, group_name):
        available_groups = [attr for attr in dir(ColumnGroups) if not attr.startswith('_')]
        raise ValueError(f"Unknown column group '{group_name}'. Available groups: {available_groups}")
    
    return getattr(ColumnGroups, group_name)


def validate_columns_exist(df_columns: List[str], required_columns: List[str]) -> List[str]:
    """Validate that required columns exist in a DataFrame.
    
    Args:
        df_columns: List of columns in the DataFrame
        required_columns: List of required column names
        
    Returns:
        List of missing column names
    """
    df_columns_set = set(df_columns)
    required_columns_set = set(required_columns)
    missing_columns = required_columns_set - df_columns_set
    return list(missing_columns)


def get_column_mapping() -> Dict[str, str]:
    """Get mapping from enum values to string values for all columns.
    
    Returns:
        Dictionary mapping enum names to column values
    """
    mapping = {}
    
    # Add snapshot columns
    for col in SnapshotColumns:
        mapping[col.name] = col.value
    
    # Add event columns
    for col in EventColumns:
        mapping[col.name] = col.value
    
    # Add plan rule columns
    for col in PlanRuleColumns:
        mapping[col.name] = col.value
    
    return mapping