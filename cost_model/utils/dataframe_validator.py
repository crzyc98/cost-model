# cost_model/utils/dataframe_validator.py
"""
DataFrame validation and sanitization utilities for preventing duplicate columns
and ensuring data integrity throughout the pipeline.
"""

import logging
from functools import wraps
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataFrameValidator:
    """Centralized DataFrame validation and sanitization."""

    @staticmethod
    def validate_no_duplicates(df: pd.DataFrame, context: str = "") -> Tuple[bool, List[str]]:
        """
        Check if DataFrame has duplicate columns.

        Args:
            df: DataFrame to validate
            context: Context string for better error messages

        Returns:
            Tuple of (is_valid, list_of_duplicate_columns)
        """
        duplicated_mask = df.columns.duplicated()
        if duplicated_mask.any():
            duplicates = df.columns[duplicated_mask].tolist()
            logger.error(
                f"Duplicate columns found{' in ' + context if context else ''}: {duplicates}"
            )
            return False, duplicates
        return True, []

    @staticmethod
    def deduplicate_columns(
        df: pd.DataFrame,
        strategy: str = "keep_first",
        suffix_pattern: str = "_{n}",
        context: str = "",
    ) -> pd.DataFrame:
        """
        Remove or rename duplicate columns based on strategy.

        Args:
            df: DataFrame with potential duplicate columns
            strategy: How to handle duplicates
                - "keep_first": Keep first occurrence, drop others
                - "keep_last": Keep last occurrence, drop others
                - "rename": Rename duplicates with suffixes
            suffix_pattern: Pattern for renaming (used if strategy="rename")
            context: Context string for logging

        Returns:
            DataFrame with unique column names
        """
        if not df.columns.duplicated().any():
            return df

        # Log the duplicates found
        duplicates = df.columns[df.columns.duplicated(keep=False)].unique().tolist()
        logger.warning(f"Deduplicating columns{' in ' + context if context else ''}: {duplicates}")

        if strategy == "keep_first":
            return df.loc[:, ~df.columns.duplicated(keep="first")]

        elif strategy == "keep_last":
            return df.loc[:, ~df.columns.duplicated(keep="last")]

        elif strategy == "rename":
            # Create new column names with suffixes for duplicates
            new_columns = []
            column_counts = {}

            for col in df.columns:
                if col in column_counts:
                    column_counts[col] += 1
                    new_name = f"{col}{suffix_pattern.format(n=column_counts[col])}"
                    new_columns.append(new_name)
                else:
                    column_counts[col] = 0
                    new_columns.append(col)

            df_copy = df.copy()
            df_copy.columns = new_columns
            return df_copy

        else:
            raise ValueError(f"Unknown deduplication strategy: {strategy}")

    @staticmethod
    def merge_duplicate_columns(
        df: pd.DataFrame, merge_func: callable = None, context: str = ""
    ) -> pd.DataFrame:
        """
        Merge duplicate columns using a custom function.

        Args:
            df: DataFrame with duplicate columns
            merge_func: Function to merge duplicate columns (default: take first non-null)
            context: Context string for logging

        Returns:
            DataFrame with merged duplicate columns
        """
        if not df.columns.duplicated().any():
            return df

        if merge_func is None:
            # Default: take first non-null value
            def merge_func(series_list):
                return pd.concat(series_list, axis=1).bfill(axis=1).iloc[:, 0]

        # Group duplicate columns
        result_dict = {}
        processed = set()

        for col in df.columns:
            if col not in processed:
                # Get all columns with this name
                mask = df.columns == col
                if mask.sum() > 1:
                    # Multiple columns with same name - merge them
                    duplicate_cols = df.loc[:, mask]
                    merged = merge_func(
                        [duplicate_cols.iloc[:, i] for i in range(duplicate_cols.shape[1])]
                    )
                    result_dict[col] = merged
                    logger.info(
                        f"Merged {mask.sum()} duplicate '{col}' columns{' in ' + context if context else ''}"
                    )
                else:
                    # Single column - keep as is
                    result_dict[col] = df[col]
                processed.add(col)

        return pd.DataFrame(result_dict)


def validate_dataframe(require_unique_columns: bool = True, context: str = ""):
    """
    Decorator to validate DataFrames returned by functions.

    Args:
        require_unique_columns: Whether to enforce unique column names
        context: Context string for error messages
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if isinstance(result, pd.DataFrame) and require_unique_columns:
                validator = DataFrameValidator()
                is_valid, duplicates = validator.validate_no_duplicates(
                    result, context or func.__name__
                )

                if not is_valid:
                    # Auto-fix by deduplicating
                    logger.warning(f"Auto-fixing duplicate columns in {context or func.__name__}")
                    result = validator.deduplicate_columns(
                        result, strategy="keep_first", context=context or func.__name__
                    )

            return result

        return wrapper

    return decorator
