"""
Data validation functions for snapshot processing.

Provides comprehensive validation for census data, snapshot data,
and various data quality checks throughout the processing pipeline.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from .models import ValidationResult, SnapshotConfig
from .constants import (
    REQUIRED_CENSUS_COLUMNS, 
    VALIDATION_THRESHOLDS,
    SUPPORTED_FILE_FORMATS
)
from .exceptions import ValidationError
from .types import (
    FilePath, CompensationAmount, AgeYears, TenureYears, ColumnName,
    ValidatorConfig, ValidationResult as ValidationResultType
)

logger = logging.getLogger(__name__)


class SnapshotValidator:
    """Comprehensive data validator for snapshot processing.
    
    This class provides extensive validation capabilities for census data,
    snapshot data, and various data quality checks throughout the processing pipeline.
    """
    
    def __init__(self, config: SnapshotConfig) -> None:
        """Initialize the validator.
        
        Args:
            config: Snapshot configuration containing validation parameters.
        """
        self.config = config
        self.thresholds = VALIDATION_THRESHOLDS
    
    def validate_census_file(self, file_path: FilePath) -> ValidationResult:
        """Validate census file exists and has supported format."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check file exists
        from pathlib import Path
        path = Path(file_path)
        if not path.exists():
            result.add_error(f"Census file not found: {file_path}")
            return result
        
        # Check file format
        if path.suffix.lower() not in SUPPORTED_FILE_FORMATS:
            result.add_error(f"Unsupported file format: {path.suffix}. "
                           f"Supported formats: {SUPPORTED_FILE_FORMATS}")
        
        return result
    
    def validate_census_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate census data structure and content."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check not empty
        if df.empty:
            result.add_error("Census data is empty")
            return result
        
        # Check required columns
        missing_required = set(self.config.required_columns) - set(df.columns)
        if missing_required:
            result.add_error(f"Missing required columns: {missing_required}")
        
        # Check for duplicate employee IDs
        if 'EMP_ID' in df.columns:
            duplicates = df['EMP_ID'].duplicated().sum()
            if duplicates > 0:
                result.add_error(f"Found {duplicates} duplicate employee IDs")
        
        # Validate data types and ranges
        self._validate_data_ranges(df, result)
        
        return result
    
    def validate_snapshot_completeness(self, df: pd.DataFrame) -> ValidationResult:
        """Validate snapshot has all required data for processing."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if df.empty:
            result.add_error("Snapshot is empty")
            return result
        
        # Check critical columns exist  
        critical_columns = ['EMP_ID', 'EMP_HIRE_DATE', 'EMP_ACTIVE']
        
        missing_critical = [col for col in critical_columns if col not in df.columns]
        if missing_critical:
            result.add_error(f"Missing critical snapshot columns: {missing_critical}")
        
        # Check for null values in critical columns
        for col in critical_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    result.add_warning(f"Column {col} has {null_count} null values")
        
        return result
    
    def validate_compensation_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate compensation data ranges and consistency."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        EMP_GROSS_COMP = 'EMP_GROSS_COMP'
        
        if EMP_GROSS_COMP not in df.columns:
            result.add_warning("No compensation column found")
            return result
        
        comp_data = df[EMP_GROSS_COMP].dropna()
        if comp_data.empty:
            result.add_warning("No compensation data available")
            return result
        
        # Check ranges
        min_comp = self.thresholds['min_compensation']
        max_comp = self.thresholds['max_compensation']
        
        too_low = (comp_data < min_comp).sum()
        too_high = (comp_data > max_comp).sum()
        
        if too_low > 0:
            result.add_warning(f"{too_low} employees have compensation below ${min_comp:,.0f}")
        
        if too_high > 0:
            result.add_warning(f"{too_high} employees have compensation above ${max_comp:,.0f}")
        
        # Check for unrealistic values (zeros, negatives)
        zero_or_negative = (comp_data <= 0).sum()
        if zero_or_negative > 0:
            result.add_error(f"{zero_or_negative} employees have zero or negative compensation")
        
        return result
    
    def _validate_data_ranges(self, df: pd.DataFrame, result: ValidationResult):
        """Validate data ranges for various columns."""
        
        # Age validation
        EMP_AGE = 'EMP_AGE'
        if EMP_AGE in df.columns:
            age_data = df[EMP_AGE].dropna()
            if not age_data.empty:
                min_age = self.thresholds['min_age']
                max_age = self.thresholds['max_age']
                
                invalid_ages = ((age_data < min_age) | (age_data > max_age)).sum()
                if invalid_ages > 0:
                    result.add_warning(f"{invalid_ages} employees have ages outside valid range "
                                     f"({min_age}-{max_age})")
        
        # Tenure validation  
        EMP_TENURE = 'EMP_TENURE'
        if EMP_TENURE in df.columns:
            tenure_data = df[EMP_TENURE].dropna()
            if not tenure_data.empty:
                max_tenure = self.thresholds['max_tenure']
                invalid_tenure = (tenure_data > max_tenure).sum()
                if invalid_tenure > 0:
                    result.add_warning(f"{invalid_tenure} employees have tenure > {max_tenure} years")
        
        # Deferral rate validation
        EMP_DEFERRAL_RATE = 'EMP_DEFERRAL_RATE'
        if EMP_DEFERRAL_RATE in df.columns:
            deferral_data = df[EMP_DEFERRAL_RATE].dropna()
            if not deferral_data.empty:
                max_deferral = self.thresholds['max_deferral_rate']
                invalid_deferral = ((deferral_data < 0) | (deferral_data > max_deferral)).sum()
                if invalid_deferral > 0:
                    result.add_warning(f"{invalid_deferral} employees have invalid deferral rates "
                                     f"(should be 0-{max_deferral})")


def validate_and_log_results(result: ValidationResult, operation_name: str):
    """Log validation results and raise exception if critical errors found."""
    
    if result.errors:
        logger.error(f"Validation errors in {operation_name}:")
        for error in result.errors:
            logger.error(f"  - {error}")
    
    if result.warnings:
        logger.warning(f"Validation warnings in {operation_name}:")
        for warning in result.warnings:
            logger.warning(f"  - {warning}")
    
    if not result.is_valid:
        raise ValidationError(f"Validation failed for {operation_name}: {result.errors}")
    
    if result.warnings:
        logger.info(f"Validation completed for {operation_name} with {len(result.warnings)} warnings")
    else:
        logger.info(f"Validation passed for {operation_name}")