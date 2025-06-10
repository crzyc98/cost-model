"""
Data type definitions for schema validation and optimization.

This module defines the expected data types for all columns in snapshots
and events, providing both validation and memory optimization capabilities.
"""

from typing import Dict, Any, Union, Optional
import pandas as pd
import numpy as np
from .columns import SnapshotColumns, EventColumns


class SnapshotDTypes:
    """Data type definitions for snapshot columns."""
    
    # Core data types mapping
    DTYPES: Dict[str, str] = {
        # Identifiers
        SnapshotColumns.EMP_ID: "string",
        SnapshotColumns.SIMULATION_YEAR: "Int64",
        
        # Dates
        SnapshotColumns.EMP_HIRE_DATE: "datetime64[ns]",
        SnapshotColumns.EMP_BIRTH_DATE: "datetime64[ns]",
        SnapshotColumns.EMP_TERM_DATE: "datetime64[ns]",
        
        # Boolean fields
        SnapshotColumns.EMP_ACTIVE: "boolean",
        SnapshotColumns.EMP_EXITED: "boolean",
        SnapshotColumns.IS_ELIGIBLE: "boolean",
        SnapshotColumns.IS_ENROLLED: "boolean",
        
        # Categorical fields (memory efficient for repeated values)
        SnapshotColumns.EMP_STATUS_EOY: "category",
        SnapshotColumns.EMP_TENURE_BAND: "category",
        SnapshotColumns.EMP_AGE_BAND: "category",
        SnapshotColumns.EMP_LEVEL: "category",
        SnapshotColumns.EMP_LEVEL_SOURCE: "category",
        
        # Numeric fields
        SnapshotColumns.EMP_GROSS_COMP: "float64",
        SnapshotColumns.EMP_DEFERRAL_RATE: "float64",
        SnapshotColumns.EMP_TENURE: "float64",
        SnapshotColumns.EMP_AGE: "float64",
        SnapshotColumns.EMP_CONTRIBUTION: "float64",
        SnapshotColumns.EMPLOYER_CORE_CONTRIBUTION: "float64",
        SnapshotColumns.EMPLOYER_MATCH_CONTRIBUTION: "float64",
        SnapshotColumns.TERM_RATE: "float64",
        SnapshotColumns.PROMOTION_RATE: "float64",
    }
    
    # Categorical values for category columns
    CATEGORICAL_VALUES: Dict[str, list] = {
        SnapshotColumns.EMP_STATUS_EOY: [
            "ACTIVE", "TERMINATED", "RETIRED", "DISABLED", "DECEASED", "UNKNOWN"
        ],
        SnapshotColumns.EMP_TENURE_BAND: [
            "NEW_HIRE", "EARLY_CAREER", "MID_CAREER", "SENIOR", "VETERAN", "UNKNOWN"
        ],
        SnapshotColumns.EMP_AGE_BAND: [
            "YOUNG", "EARLY_CAREER", "MID_CAREER", "SENIOR", "PRE_RETIREMENT", 
            "POST_RETIREMENT", "UNKNOWN"
        ],
        SnapshotColumns.EMP_LEVEL_SOURCE: [
            "PROVIDED", "INFERRED_FROM_COMPENSATION", "DEFAULT", "NO_COMPENSATION_DATA"
        ],
    }
    
    # Nullable columns (can contain NaN values)
    NULLABLE_COLUMNS: set = {
        SnapshotColumns.EMP_BIRTH_DATE,
        SnapshotColumns.EMP_TERM_DATE,
        SnapshotColumns.EMP_AGE,
        SnapshotColumns.EMP_AGE_BAND,
        SnapshotColumns.TERM_RATE,
        SnapshotColumns.PROMOTION_RATE,
    }
    
    # Columns that should never be null
    NON_NULLABLE_COLUMNS: set = {
        SnapshotColumns.EMP_ID,
        SnapshotColumns.SIMULATION_YEAR,
        SnapshotColumns.EMP_HIRE_DATE,
        SnapshotColumns.EMP_ACTIVE,
        SnapshotColumns.EMP_GROSS_COMP,
    }


class EventDTypes:
    """Data type definitions for event columns."""
    
    DTYPES: Dict[str, str] = {
        # Identifiers
        EventColumns.EVENT_ID: "string",
        EventColumns.EVENT_TYPE: "category",
        EventColumns.EVENT_STATUS: "category",
        EventColumns.EMP_ID: "string",
        EventColumns.SIMULATION_YEAR: "Int64",
        
        # Dates
        EventColumns.EVENT_DATE: "datetime64[ns]",
        EventColumns.CREATED_AT: "datetime64[ns]",
        EventColumns.UPDATED_AT: "datetime64[ns]",
        
        # Numeric fields
        EventColumns.GROSS_COMPENSATION: "float64",
        EventColumns.DEFERRAL_RATE: "float64",
        
        # Categorical fields
        EventColumns.JOB_LEVEL: "category",
        EventColumns.TERMINATION_REASON: "category",
        EventColumns.CREATED_BY: "category",
        
        # JSON/object fields
        EventColumns.EVENT_PAYLOAD: "object",
    }
    
    CATEGORICAL_VALUES: Dict[str, list] = {
        EventColumns.EVENT_TYPE: [
            "hire", "termination", "new_hire_termination", "compensation", 
            "cola", "raise", "promotion", "contribution", "enrollment"
        ],
        EventColumns.EVENT_STATUS: [
            "pending", "processing", "completed", "failed", "cancelled", "rolled_back"
        ],
        EventColumns.TERMINATION_REASON: [
            "voluntary", "involuntary", "retirement", "death", "disability", "unknown"
        ],
        EventColumns.CREATED_BY: [
            "system", "user", "import", "simulation", "auto_rule"
        ],
    }


class DataTypeValidator:
    """Utilities for validating and converting data types."""
    
    @staticmethod
    def validate_snapshot_dtypes(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data types in a snapshot DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "conversions_needed": {}
        }
        
        for column, expected_dtype in SnapshotDTypes.DTYPES.items():
            if column not in df.columns:
                continue
                
            actual_dtype = str(df[column].dtype)
            
            # Check if conversion is needed
            if not DataTypeValidator._dtypes_compatible(actual_dtype, expected_dtype):
                results["conversions_needed"][column] = {
                    "current": actual_dtype,
                    "expected": expected_dtype
                }
        
        return results
    
    @staticmethod
    def convert_snapshot_dtypes(df: pd.DataFrame, 
                              strict: bool = False) -> pd.DataFrame:
        """Convert DataFrame to proper snapshot data types.
        
        Args:
            df: DataFrame to convert
            strict: If True, raise errors on conversion failures
            
        Returns:
            DataFrame with converted data types
        """
        df_converted = df.copy()
        conversion_errors = []
        
        for column, target_dtype in SnapshotDTypes.DTYPES.items():
            if column not in df_converted.columns:
                continue
                
            try:
                if target_dtype == "category":
                    # Handle categorical conversion with predefined categories
                    categories = SnapshotDTypes.CATEGORICAL_VALUES.get(column)
                    if categories:
                        df_converted[column] = pd.Categorical(
                            df_converted[column], 
                            categories=categories
                        )
                    else:
                        df_converted[column] = df_converted[column].astype("category")
                
                elif target_dtype == "datetime64[ns]":
                    df_converted[column] = pd.to_datetime(df_converted[column])
                
                elif target_dtype in ["Int64", "boolean"]:
                    # Use nullable integer/boolean types
                    df_converted[column] = df_converted[column].astype(target_dtype)
                
                else:
                    df_converted[column] = df_converted[column].astype(target_dtype)
                    
            except Exception as e:
                error_msg = f"Failed to convert {column} to {target_dtype}: {str(e)}"
                conversion_errors.append(error_msg)
                
                if strict:
                    raise ValueError(error_msg)
        
        if conversion_errors and not strict:
            import logging
            logger = logging.getLogger(__name__)
            for error in conversion_errors:
                logger.warning(error)
        
        return df_converted
    
    @staticmethod
    def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage through intelligent type conversion.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Memory-optimized DataFrame
        """
        df_optimized = df.copy()
        
        for column in df_optimized.columns:
            col_type = df_optimized[column].dtype
            
            if col_type == "object":
                # Convert to category if few unique values
                unique_ratio = df_optimized[column].nunique() / len(df_optimized)
                if unique_ratio < 0.5:  # Less than 50% unique
                    df_optimized[column] = df_optimized[column].astype("category")
            
            elif col_type == "int64":
                # Downcast integers
                df_optimized[column] = pd.to_numeric(
                    df_optimized[column], downcast="integer"
                )
            
            elif col_type == "float64":
                # Downcast floats
                df_optimized[column] = pd.to_numeric(
                    df_optimized[column], downcast="float"
                )
        
        return df_optimized
    
    @staticmethod
    def _dtypes_compatible(actual: str, expected: str) -> bool:
        """Check if two data types are compatible."""
        # Exact match
        if actual == expected:
            return True
        
        # Compatible numeric types
        numeric_types = ["int64", "Int64", "float64", "float32"]
        if actual in numeric_types and expected in numeric_types:
            return True
        
        # String compatibility
        if actual in ["object", "string"] and expected in ["object", "string"]:
            return True
        
        # Category compatibility
        if actual == "object" and expected == "category":
            return True
        
        return False
    
    @staticmethod
    def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed memory usage information for a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with memory usage details
        """
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        return {
            "total_memory_mb": total_memory / 1024 / 1024,
            "memory_per_column": (memory_usage / 1024 / 1024).to_dict(),
            "dtypes": df.dtypes.to_dict(),
            "shape": df.shape,
            "memory_per_row_bytes": total_memory / len(df) if len(df) > 0 else 0,
        }


def get_optimal_dtypes_for_snapshot() -> Dict[str, str]:
    """Get the optimal data types for snapshot columns.
    
    Returns:
        Dictionary mapping column names to optimal data types
    """
    return SnapshotDTypes.DTYPES.copy()


def get_optimal_dtypes_for_events() -> Dict[str, str]:
    """Get the optimal data types for event columns.
    
    Returns:
        Dictionary mapping column names to optimal data types
    """
    return EventDTypes.DTYPES.copy()