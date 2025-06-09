"""
Constants and default values for snapshot processing.

This module centralizes all magic numbers, default values, and configuration
constants used throughout the snapshot creation and update process.
"""

from typing import Dict, Any

# Default compensation values
DEFAULT_COMPENSATION = 50000.0

# Level-based default compensation
LEVEL_BASED_DEFAULTS: Dict[str, float] = {
    'BAND_1': 45000.0,
    'BAND_2': 55000.0, 
    'BAND_3': 65000.0,
    'BAND_4': 75000.0,
    'BAND_5': 85000.0,
    'EXECUTIVE': 120000.0
}

# Census data column mappings
CENSUS_COLUMN_MAPPINGS: Dict[str, str] = {
    'employee_id': 'EMP_ID',
    'emp_id': 'EMP_ID',
    'id': 'EMP_ID',
    'employee_hire_date': 'EMP_HIRE_DATE',
    'hire_date': 'EMP_HIRE_DATE',
    'employee_birth_date': 'EMP_BIRTH_DATE',
    'birth_date': 'EMP_BIRTH_DATE',
    'employee_gross_compensation': 'EMP_GROSS_COMP',
    'gross_compensation': 'EMP_GROSS_COMP',
    'compensation': 'EMP_GROSS_COMP',
    'annual_comp': 'EMP_GROSS_COMP',
    'employee_deferral_rate': 'EMP_DEFERRAL_RATE',
    'deferral_rate': 'EMP_DEFERRAL_RATE',
    'employee_termination_date': 'EMP_TERM_DATE',
    'termination_date': 'EMP_TERM_DATE',
    'term_date': 'EMP_TERM_DATE',
    'active': 'EMP_ACTIVE',
    'employee_level': 'EMP_LEVEL',
    'job_level': 'EMP_LEVEL',
    'level': 'EMP_LEVEL'
}

# Required columns for initial snapshot creation
REQUIRED_CENSUS_COLUMNS = [
    'EMP_ID',
    'EMP_HIRE_DATE'
]

# Optional columns that will be inferred if missing
OPTIONAL_CENSUS_COLUMNS = [
    'EMP_BIRTH_DATE',
    'EMP_GROSS_COMP',
    'EMP_DEFERRAL_RATE',
    'EMP_TERM_DATE',
    'EMP_ACTIVE',
    'EMP_LEVEL'
]

# Tenure band definitions (in years)
TENURE_BANDS: Dict[str, tuple] = {
    'NEW_HIRE': (0, 1),
    'EARLY_CAREER': (1, 5),
    'MID_CAREER': (5, 15),
    'SENIOR': (15, 25),
    'VETERAN': (25, float('inf'))
}

# Age band definitions (in years)
AGE_BANDS: Dict[str, tuple] = {
    'YOUNG': (18, 30),
    'EARLY_CAREER': (30, 40),
    'MID_CAREER': (40, 50),
    'SENIOR': (50, 60),
    'PRE_RETIREMENT': (60, 67),
    'POST_RETIREMENT': (67, float('inf'))
}

# Data validation thresholds
VALIDATION_THRESHOLDS = {
    'min_compensation': 15000.0,
    'max_compensation': 500000.0,
    'min_age': 16,
    'max_age': 80,
    'max_tenure': 60,
    'max_deferral_rate': 1.0
}

# Compensation extraction priorities
COMPENSATION_EXTRACTION_PRIORITIES = [
    'hire_events',
    'promotion_events', 
    'compensation_events',
    'default_by_level',
    'global_default'
]

# File format support
SUPPORTED_FILE_FORMATS = ['.parquet', '.csv']

# Logging configuration
LOGGING_CONFIG = {
    'enable_timing': True,
    'enable_memory_tracking': False,
    'log_level': 'INFO'
}