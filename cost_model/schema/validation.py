"""
Schema validation utilities for snapshots and events.

This module provides comprehensive validation capabilities for data schemas,
ensuring data quality and consistency throughout the cost model.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

from .columns import SnapshotColumns, EventColumns, ColumnGroups
from .events import EventTypes, EventValidationRules
from .dtypes import SnapshotDTypes, EventDTypes, DataTypeValidator


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Add an error to the validation result."""
        self.errors.append(message)
        self.is_valid = False
        if context:
            self.metadata.setdefault("error_contexts", []).append(context)
    
    def add_warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Add a warning to the validation result."""
        self.warnings.append(message)
        if context:
            self.metadata.setdefault("warning_contexts", []).append(context)
    
    def has_issues(self) -> bool:
        """Check if there are any errors or warnings."""
        return bool(self.errors or self.warnings)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "has_issues": self.has_issues()
        }


class SchemaValidator:
    """Comprehensive schema validator for snapshots and events."""
    
    def __init__(self, strict_mode: bool = False):
        """Initialize the schema validator.
        
        Args:
            strict_mode: If True, treat warnings as errors
        """
        self.strict_mode = strict_mode
    
    def validate_snapshot_schema(self, df: pd.DataFrame, 
                                check_data_quality: bool = True) -> ValidationResult:
        """Validate a snapshot DataFrame against the schema.
        
        Args:
            df: DataFrame to validate
            check_data_quality: Whether to perform data quality checks
            
        Returns:
            Validation result with any issues found
        """
        result = ValidationResult(is_valid=True)
        
        # Basic structure validation
        self._validate_basic_structure(df, "snapshot", result)
        
        # Required columns validation
        self._validate_required_columns(
            df, ColumnGroups.REQUIRED_SNAPSHOT, result
        )
        
        # Data type validation
        self._validate_snapshot_dtypes(df, result)
        
        # Data quality validation
        if check_data_quality:
            self._validate_snapshot_data_quality(df, result)
        
        # Business rule validation
        self._validate_snapshot_business_rules(df, result)
        
        return result
    
    def validate_event_schema(self, df: pd.DataFrame,
                            check_data_quality: bool = True) -> ValidationResult:
        """Validate an event DataFrame against the schema.
        
        Args:
            df: DataFrame to validate
            check_data_quality: Whether to perform data quality checks
            
        Returns:
            Validation result with any issues found
        """
        result = ValidationResult(is_valid=True)
        
        # Basic structure validation
        self._validate_basic_structure(df, "event", result)
        
        # Required columns validation
        self._validate_required_columns(
            df, ColumnGroups.EVENT_CORE, result
        )
        
        # Data type validation
        self._validate_event_dtypes(df, result)
        
        # Event-specific validation
        if check_data_quality:
            self._validate_event_data_quality(df, result)
        
        # Event business rules validation
        self._validate_event_business_rules(df, result)
        
        return result
    
    def _validate_basic_structure(self, df: pd.DataFrame, 
                                 schema_type: str, 
                                 result: ValidationResult):
        """Validate basic DataFrame structure."""
        if df is None:
            result.add_error(f"{schema_type} DataFrame is None")
            return
        
        if df.empty:
            result.add_warning(f"{schema_type} DataFrame is empty")
            return
        
        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            duplicates = [col for col in df.columns if list(df.columns).count(col) > 1]
            result.add_error(f"Duplicate column names found: {set(duplicates)}")
        
        # Check for unnamed columns
        unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
        if unnamed_cols:
            result.add_warning(f"Unnamed columns found: {unnamed_cols}")
    
    def _validate_required_columns(self, df: pd.DataFrame,
                                  required_columns: List[str],
                                  result: ValidationResult):
        """Validate that required columns are present."""
        missing_columns = []
        for col in required_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            result.add_error(
                f"Missing required columns: {missing_columns}",
                {"available_columns": list(df.columns)}
            )
    
    def _validate_snapshot_dtypes(self, df: pd.DataFrame, 
                                 result: ValidationResult):
        """Validate snapshot data types."""
        dtype_validation = DataTypeValidator.validate_snapshot_dtypes(df)
        
        for column, conversion_info in dtype_validation["conversions_needed"].items():
            if self.strict_mode:
                result.add_error(
                    f"Column {column} has incorrect type {conversion_info['current']}, "
                    f"expected {conversion_info['expected']}"
                )
            else:
                result.add_warning(
                    f"Column {column} should be converted from {conversion_info['current']} "
                    f"to {conversion_info['expected']}"
                )
    
    def _validate_event_dtypes(self, df: pd.DataFrame, 
                              result: ValidationResult):
        """Validate event data types."""
        # Similar to snapshot validation but for events
        for column in df.columns:
            if column in EventDTypes.DTYPES:
                expected_dtype = EventDTypes.DTYPES[column]
                actual_dtype = str(df[column].dtype)
                
                if not DataTypeValidator._dtypes_compatible(actual_dtype, expected_dtype):
                    if self.strict_mode:
                        result.add_error(
                            f"Event column {column} has incorrect type {actual_dtype}, "
                            f"expected {expected_dtype}"
                        )
                    else:
                        result.add_warning(
                            f"Event column {column} should be {expected_dtype}, "
                            f"currently {actual_dtype}"
                        )
    
    def _validate_snapshot_data_quality(self, df: pd.DataFrame, 
                                       result: ValidationResult):
        """Validate snapshot data quality."""
        # Check for null values in non-nullable columns
        for column in SnapshotDTypes.NON_NULLABLE_COLUMNS:
            if column in df.columns:
                null_count = df[column].isna().sum()
                if null_count > 0:
                    result.add_error(
                        f"Column {column} has {null_count} null values but should not be nullable"
                    )
        
        # Check employee ID format and uniqueness
        if SnapshotColumns.EMP_ID in df.columns:
            emp_ids = df[SnapshotColumns.EMP_ID]
            
            # Check for duplicates
            duplicate_count = emp_ids.duplicated().sum()
            if duplicate_count > 0:
                result.add_error(f"Found {duplicate_count} duplicate employee IDs")
            
            # Check for empty/null IDs
            invalid_ids = emp_ids.isna() | (emp_ids == "")
            if invalid_ids.any():
                result.add_error(f"Found {invalid_ids.sum()} invalid employee IDs")
        
        # Validate compensation values
        if SnapshotColumns.EMP_GROSS_COMP in df.columns:
            compensation = df[SnapshotColumns.EMP_GROSS_COMP]
            
            # Check for negative compensation
            negative_comp = compensation < 0
            if negative_comp.any():
                result.add_warning(f"Found {negative_comp.sum()} employees with negative compensation")
            
            # Check for unrealistic compensation values
            very_high_comp = compensation > 1000000  # $1M threshold
            if very_high_comp.any():
                result.add_warning(f"Found {very_high_comp.sum()} employees with very high compensation (>$1M)")
        
        # Validate dates
        self._validate_dates(df, result)
        
        # Validate percentages/rates
        self._validate_rates(df, result)
    
    def _validate_event_data_quality(self, df: pd.DataFrame, 
                                    result: ValidationResult):
        """Validate event data quality."""
        # Validate event types
        if EventColumns.EVENT_TYPE in df.columns:
            event_types = df[EventColumns.EVENT_TYPE]
            invalid_types = ~event_types.isin([et.value for et in EventTypes])
            if invalid_types.any():
                invalid_values = event_types[invalid_types].unique()
                result.add_error(f"Invalid event types found: {list(invalid_values)}")
        
        # Validate event dates
        if EventColumns.EVENT_DATE in df.columns:
            event_dates = pd.to_datetime(df[EventColumns.EVENT_DATE], errors='coerce')
            invalid_dates = event_dates.isna()
            if invalid_dates.any():
                result.add_error(f"Found {invalid_dates.sum()} invalid event dates")
        
        # Validate event-specific fields
        self._validate_event_specific_fields(df, result)
    
    def _validate_snapshot_business_rules(self, df: pd.DataFrame, 
                                         result: ValidationResult):
        """Validate business rules for snapshots."""
        # Active employees should not have termination dates
        if all(col in df.columns for col in [SnapshotColumns.EMP_ACTIVE, SnapshotColumns.EMP_TERM_DATE]):
            active_with_term_date = (
                (df[SnapshotColumns.EMP_ACTIVE] == True) & 
                df[SnapshotColumns.EMP_TERM_DATE].notna()
            )
            if active_with_term_date.any():
                result.add_warning(
                    f"Found {active_with_term_date.sum()} active employees with termination dates"
                )
        
        # Terminated employees should have termination dates
        if all(col in df.columns for col in [SnapshotColumns.EMP_ACTIVE, SnapshotColumns.EMP_TERM_DATE]):
            inactive_without_term_date = (
                (df[SnapshotColumns.EMP_ACTIVE] == False) & 
                df[SnapshotColumns.EMP_TERM_DATE].isna()
            )
            if inactive_without_term_date.any():
                result.add_warning(
                    f"Found {inactive_without_term_date.sum()} inactive employees without termination dates"
                )
        
        # Tenure should be non-negative
        if SnapshotColumns.EMP_TENURE in df.columns:
            negative_tenure = df[SnapshotColumns.EMP_TENURE] < 0
            if negative_tenure.any():
                result.add_error(f"Found {negative_tenure.sum()} employees with negative tenure")
        
        # Age should be reasonable
        if SnapshotColumns.EMP_AGE in df.columns:
            unreasonable_age = (df[SnapshotColumns.EMP_AGE] < 16) | (df[SnapshotColumns.EMP_AGE] > 100)
            if unreasonable_age.any():
                result.add_warning(f"Found {unreasonable_age.sum()} employees with unreasonable ages")
    
    def _validate_event_business_rules(self, df: pd.DataFrame, 
                                      result: ValidationResult):
        """Validate business rules for events."""
        # Group by event type and validate specific rules
        if EventColumns.EVENT_TYPE in df.columns:
            for event_type in df[EventColumns.EVENT_TYPE].unique():
                event_subset = df[df[EventColumns.EVENT_TYPE] == event_type]
                self._validate_event_type_rules(event_type, event_subset, result)
    
    def _validate_dates(self, df: pd.DataFrame, result: ValidationResult):
        """Validate date columns."""
        date_columns = [col for col in df.columns if "date" in col.lower()]
        
        for col in date_columns:
            if col in df.columns:
                try:
                    dates = pd.to_datetime(df[col], errors='coerce')
                    invalid_dates = dates.isna() & df[col].notna()
                    if invalid_dates.any():
                        result.add_error(f"Found {invalid_dates.sum()} invalid dates in {col}")
                    
                    # Check for future dates where inappropriate
                    if col in [SnapshotColumns.EMP_HIRE_DATE, SnapshotColumns.EMP_BIRTH_DATE]:
                        future_dates = dates > pd.Timestamp.now()
                        if future_dates.any():
                            result.add_warning(f"Found {future_dates.sum()} future dates in {col}")
                            
                except Exception as e:
                    result.add_error(f"Error validating dates in {col}: {str(e)}")
    
    def _validate_rates(self, df: pd.DataFrame, result: ValidationResult):
        """Validate rate/percentage columns."""
        rate_columns = [SnapshotColumns.EMP_DEFERRAL_RATE, SnapshotColumns.TERM_RATE, 
                       SnapshotColumns.PROMOTION_RATE]
        
        for col in rate_columns:
            if col in df.columns:
                rates = df[col]
                
                # Check for values outside [0, 1] range
                invalid_rates = (rates < 0) | (rates > 1)
                if invalid_rates.any():
                    result.add_warning(f"Found {invalid_rates.sum()} invalid rates in {col} (outside 0-1 range)")
    
    def _validate_event_specific_fields(self, df: pd.DataFrame, result: ValidationResult):
        """Validate event-specific required fields."""
        if EventColumns.EVENT_TYPE not in df.columns:
            return
        
        for event_type in df[EventColumns.EVENT_TYPE].unique():
            event_subset = df[df[EventColumns.EVENT_TYPE] == event_type]
            required_fields = EventValidationRules.get_required_fields(event_type)
            
            for field in required_fields:
                if field not in event_subset.columns:
                    result.add_error(f"Event type {event_type} missing required field: {field}")
                elif event_subset[field].isna().any():
                    null_count = event_subset[field].isna().sum()
                    result.add_error(
                        f"Event type {event_type} has {null_count} null values in required field: {field}"
                    )
    
    def _validate_event_type_rules(self, event_type: str, events: pd.DataFrame, 
                                  result: ValidationResult):
        """Validate rules specific to each event type."""
        # Hire events should have positive compensation
        if event_type == EventTypes.HIRE and EventColumns.GROSS_COMPENSATION in events.columns:
            zero_comp = events[EventColumns.GROSS_COMPENSATION] <= 0
            if zero_comp.any():
                result.add_warning(
                    f"Found {zero_comp.sum()} hire events with zero or negative compensation"
                )
        
        # Termination events should not have future dates
        if event_type in [EventTypes.TERMINATION, EventTypes.NEW_HIRE_TERMINATION]:
            if EventColumns.EVENT_DATE in events.columns:
                event_dates = pd.to_datetime(events[EventColumns.EVENT_DATE])
                future_terms = event_dates > pd.Timestamp.now()
                if future_terms.any():
                    result.add_warning(
                        f"Found {future_terms.sum()} termination events with future dates"
                    )


def validate_snapshot_schema(df: pd.DataFrame, 
                           strict_mode: bool = False,
                           check_data_quality: bool = True) -> ValidationResult:
    """Convenience function to validate a snapshot schema.
    
    Args:
        df: DataFrame to validate
        strict_mode: If True, treat warnings as errors
        check_data_quality: Whether to perform data quality checks
        
    Returns:
        Validation result
    """
    validator = SchemaValidator(strict_mode=strict_mode)
    return validator.validate_snapshot_schema(df, check_data_quality=check_data_quality)


def validate_event_schema(df: pd.DataFrame,
                         strict_mode: bool = False,
                         check_data_quality: bool = True) -> ValidationResult:
    """Convenience function to validate an event schema.
    
    Args:
        df: DataFrame to validate
        strict_mode: If True, treat warnings as errors
        check_data_quality: Whether to perform data quality checks
        
    Returns:
        Validation result
    """
    validator = SchemaValidator(strict_mode=strict_mode)
    return validator.validate_event_schema(df, check_data_quality=check_data_quality)