# cost_model/utils/column_schema_enforcer.py
"""
Schema enforcement to prevent duplicate columns at the source.
"""

from typing import Dict, List, Optional, Set
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ColumnSchemaEnforcer:
    """Enforce column schema and prevent duplicates."""
    
    def __init__(self, expected_columns: List[str], strict: bool = True):
        """
        Initialize schema enforcer.
        
        Args:
            expected_columns: List of expected column names
            strict: If True, raise errors; if False, log warnings
        """
        self.expected_columns = expected_columns
        self.expected_set = set(expected_columns)
        self.strict = strict
    
    def validate_and_clean(self, df: pd.DataFrame, source: str = "") -> pd.DataFrame:
        """
        Validate DataFrame against schema and clean duplicates.
        
        Args:
            df: DataFrame to validate
            source: Source identifier for logging
            
        Returns:
            Cleaned DataFrame
        """
        # Check for duplicates in input
        if df.columns.duplicated().any():
            duplicates = df.columns[df.columns.duplicated()].unique().tolist()
            msg = f"Input DataFrame{' from ' + source if source else ''} contains duplicate columns: {duplicates}"
            
            if self.strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                # Remove duplicates, keeping first
                df = df.loc[:, ~df.columns.duplicated()]
        
        # Check for unexpected columns
        actual_set = set(df.columns)
        unexpected = actual_set - self.expected_set
        if unexpected:
            msg = f"Unexpected columns{' in ' + source if source else ''}: {unexpected}"
            logger.warning(msg)
        
        # Check for missing expected columns
        missing = self.expected_set - actual_set
        if missing:
            msg = f"Missing expected columns{' in ' + source if source else ''}: {missing}"
            if self.strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
        
        return df
