"""
Data transformation utilities for snapshot processing.

Provides reusable transformation functions for tenure calculations,
age calculations, job level inference, and other data processing tasks.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

from .models import SnapshotConfig
from .constants import TENURE_BANDS, AGE_BANDS, LEVEL_BASED_DEFAULTS
from .exceptions import SnapshotBuildError
from .types import (
    CompensationAmount, TenureYears, AgeYears, ColumnName,
    TransformerConfig, CompensationExtractionResult
)
# Import unified schema
from cost_model.utils.frame_tools import _dedup_columns
from cost_model.schema import SnapshotColumns, migrate_legacy_columns

logger = logging.getLogger(__name__)


class SnapshotTransformer:
    """Handles data transformations for snapshot processing.
    
    This class provides reusable transformation functions for calculating tenure,
    age, job levels, compensation, and other derived data fields.
    """
    
    def __init__(self, config: SnapshotConfig) -> None:
        """Initialize the transformer.
        
        Args:
            config: Snapshot configuration containing transformation parameters.
        """
        self.config = config
        self._migration_done = False  # Track if migration has been performed
    
    def _ensure_schema_migration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure schema migration is performed once and deduplication is applied."""
        if not self._migration_done:
            df, migration_result = migrate_legacy_columns(df, schema_type="snapshot", strict_mode=False)
            if not migration_result.success:
                logger.warning(f"Column migration issues: {migration_result.warnings}")
            self._migration_done = True
        
        # Always apply deduplication to handle any other sources of duplicate columns
        df = _dedup_columns(df)
        return df
    
    def apply_tenure_calculations(self, df: pd.DataFrame, reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Calculate tenure and tenure bands for employees.
        
        Args:
            df: DataFrame with employee data
            reference_date: Date to calculate tenure from (defaults to start of config year)
            
        Returns:
            DataFrame with tenure calculations added
        """
        # Ensure schema migration and deduplication
        df = self._ensure_schema_migration(df)

        # Use unified schema column names
        hire_date_col = SnapshotColumns.EMP_HIRE_DATE
        if hire_date_col not in df.columns:
            raise SnapshotBuildError(f"Cannot calculate tenure: {hire_date_col} column missing. Available columns: {list(df.columns)}")
        
        # Use unified schema for output columns
        tenure_col = SnapshotColumns.EMP_TENURE
        tenure_band_col = SnapshotColumns.EMP_TENURE_BAND
        
        if reference_date is None:
            reference_date = datetime(self.config.start_year, 1, 1)
        
        logger.debug(f"Calculating tenure as of {reference_date} using column {hire_date_col}")
        
        # Calculate tenure in years
        hire_dates = pd.to_datetime(df[hire_date_col])
        tenure_days = (reference_date - hire_dates).dt.days
        df[tenure_col] = tenure_days / 365.25  # Account for leap years
        
        # Handle negative tenure (future hire dates)
        negative_tenure = df[tenure_col] < 0
        if negative_tenure.any():
            logger.warning(f"Found {negative_tenure.sum()} employees with future hire dates")
            df.loc[negative_tenure, tenure_col] = 0
        
        # Calculate tenure bands
        df[tenure_band_col] = df[tenure_col].apply(self._calculate_tenure_band)
        
        logger.debug(f"Calculated tenure for {len(df)} employees")
        return df
    
    def apply_age_calculations(self, df: pd.DataFrame, reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Calculate age and age bands for employees.
        
        Args:
            df: DataFrame with employee data
            reference_date: Date to calculate age from (defaults to start of config year)
            
        Returns:
            DataFrame with age calculations added
        """
        # Ensure schema migration and deduplication
        df = self._ensure_schema_migration(df)
        
        # Use unified schema column names
        birth_date_col = SnapshotColumns.EMP_BIRTH_DATE
        age_col = SnapshotColumns.EMP_AGE
        age_band_col = SnapshotColumns.EMP_AGE_BAND
        
        try:
            from cost_model.state.age import apply_age
        except ImportError:
            apply_age = None
        
        if reference_date is None:
            reference_date = datetime(self.config.start_year, 1, 1)
        
        logger.debug(f"Calculating age as of {reference_date}")
        
        # Use existing age calculation function if available
        if apply_age is not None:
            try:
                # Call apply_age with the required birth_col parameter
                df = apply_age(
                    df=df,
                    birth_col=birth_date_col,
                    as_of=reference_date,
                    out_age_col=age_col,
                    out_band_col=age_band_col
                )
                logger.debug(f"Applied age calculations using existing function")
            except Exception as e:
                logger.warning(f"Existing age function failed, using manual calculation: {e}")
                df = self._manual_age_calculation(df, reference_date)
        else:
            logger.debug("Existing age function not available, using manual calculation")
            df = self._manual_age_calculation(df, reference_date)
        
        return df
    
    def apply_contribution_calculations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate contribution amounts based on compensation and deferral rates.
        
        Args:
            df: DataFrame with employee data
            
        Returns:
            DataFrame with contribution calculations added
        """
        # Ensure schema migration and deduplication
        df = self._ensure_schema_migration(df)
        
        # Use unified schema column names
        comp_col = SnapshotColumns.EMP_GROSS_COMP
        deferral_col = SnapshotColumns.EMP_DEFERRAL_RATE
        
        if comp_col not in df.columns:
            raise SnapshotBuildError(f"Cannot calculate contributions: {comp_col} column missing. Available columns: {list(df.columns)}")
        
        # Use unified schema for output columns
        employee_contrib_col = SnapshotColumns.EMP_CONTRIBUTION
        employer_match_col = SnapshotColumns.EMPLOYER_MATCH_CONTRIBUTION
        
        logger.debug(f"Calculating contribution amounts using compensation column {comp_col}")
        
        if deferral_col not in df.columns:
            logger.warning(f"No deferral rate column {deferral_col} found, defaulting to 0.0")
            df[deferral_col] = 0.0
        
        # Calculate employee contributions
        df[employee_contrib_col] = df[comp_col] * df[deferral_col]
        
        # Calculate employer match (simple 50% of employee contribution up to 6% of compensation)
        max_match_base = df[comp_col] * 0.06  # 6% of compensation
        employee_contrib_eligible = df[employee_contrib_col].clip(upper=max_match_base)
        df[employer_match_col] = employee_contrib_eligible * 0.5  # 50% match
        
        logger.debug(f"Calculated contributions for {len(df)} employees")
        return df
    
    def infer_job_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Infer job levels for employees based on compensation and other factors.
        
        Args:
            df: DataFrame with employee data
            
        Returns:
            DataFrame with job level inference applied
        """
        # Ensure schema migration and deduplication
        df = self._ensure_schema_migration(df)
        
        # Use unified schema column names
        level_col = SnapshotColumns.EMP_LEVEL
        level_source_col = SnapshotColumns.EMP_LEVEL_SOURCE
        comp_col = SnapshotColumns.EMP_GROSS_COMP
        
        try:
            from cost_model.state.job_levels.loader import ingest_with_imputation
        except ImportError:
            ingest_with_imputation = None
        
        logger.debug("Inferring job levels")
        
        if ingest_with_imputation is not None:
            try:
                # Store original DataFrame structure
                original_df = df.copy()
                
                # Use existing job level inference if available
                level_result = ingest_with_imputation(df)
                
                # Merge the level inference results back into the original DataFrame
                # ingest_with_imputation returns only [EMP_ID, EMP_LEVEL, EMP_LEVEL_SOURCE]
                emp_id_col = SnapshotColumns.EMP_ID
                
                # Update the levels in the original DataFrame
                if emp_id_col in level_result.columns and emp_id_col in original_df.columns:
                    # Create a mapping of employee IDs to levels
                    level_mapping = level_result.set_index(emp_id_col)
                    
                    # Update levels in original DataFrame
                    for col in [level_col, level_source_col]:
                        if col in level_mapping.columns:
                            original_df[col] = original_df[emp_id_col].map(level_mapping[col]).fillna(original_df.get(col, pd.NA))
                
                df = original_df
                logger.debug("Applied job level inference using existing function")
            except Exception as e:
                logger.warning(f"Existing job level function failed, using fallback: {e}")
                df = self._fallback_job_level_inference(df)
        else:
            logger.warning("Job level inference function not available, using fallback")
            df = self._fallback_job_level_inference(df)
        
        return df
    
    def normalize_compensation_by_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize compensation values that are missing or unrealistic based on job level.
        
        Args:
            df: DataFrame with employee data
            
        Returns:
            DataFrame with normalized compensation
        """
        # Ensure schema migration and deduplication
        df = self._ensure_schema_migration(df)
        
        # Use unified schema column names
        level_col = SnapshotColumns.EMP_LEVEL
        comp_col = SnapshotColumns.EMP_GROSS_COMP
        
        logger.debug("Normalizing compensation by job level")
        
        if level_col not in df.columns:
            logger.warning(f"No job level column {level_col} available for compensation normalization")
            return df
            
        if comp_col not in df.columns:
            logger.warning(f"No compensation column {comp_col} available for normalization")
            return df
        
        # Fill missing compensation with level-based defaults
        missing_comp = df[comp_col].isna()
        # Safely handle any type of Series result
        try:
            missing_count = missing_comp.sum()
            # Convert to Python int safely
            if isinstance(missing_count, (int, float, np.number)):
                missing_count = int(missing_count)
            else:
                missing_count = 0  # Default if sum() returns something unexpected
        except Exception:
            missing_count = 0  # Handle any errors

        if missing_count > 0:
            logger.info(f"Filling {missing_count} missing compensation values")
            
            for level, default_comp in LEVEL_BASED_DEFAULTS.items():
                level_mask = (df[level_col] == level) & missing_comp
                df.loc[level_mask, comp_col] = default_comp
            
            # Fill any remaining missing values with global default
            still_missing = df[comp_col].isna()
            # Safely handle any type of Series result
            try:
                still_missing_count = still_missing.sum()
                # Convert to Python int safely
                if isinstance(still_missing_count, (int, float, np.number)):
                    still_missing_count = int(still_missing_count)
                else:
                    still_missing_count = 0  # Default if sum() returns something unexpected
            except Exception:
                still_missing_count = 0  # Handle any errors

            if still_missing_count > 0:
                from .constants import DEFAULT_COMPENSATION
                df.loc[still_missing, comp_col] = DEFAULT_COMPENSATION
                logger.warning(f"Used global default compensation for {still_missing_count} employees")
        
        return df
    
    def _calculate_tenure_band(self, tenure_years: float) -> str:
        """Calculate tenure band based on years of tenure."""
        if pd.isna(tenure_years):
            return 'UNKNOWN'
        
        for band_name, (min_years, max_years) in TENURE_BANDS.items():
            if min_years <= tenure_years < max_years:
                return band_name
        
        return 'VETERAN'  # Default for very long tenure
    
    def _calculate_age_band(self, age_years: float) -> str:
        """Calculate age band based on age in years."""
        if pd.isna(age_years):
            return 'UNKNOWN'
        
        for band_name, (min_age, max_age) in AGE_BANDS.items():
            if min_age <= age_years < max_age:
                return band_name
        
        return 'POST_RETIREMENT'  # Default for very old ages
    
    def _manual_age_calculation(self, df: pd.DataFrame, reference_date: datetime) -> pd.DataFrame:
        """Manual age calculation when existing function is not available."""
        # Use unified schema column names
        birth_date_col = SnapshotColumns.EMP_BIRTH_DATE
        age_col = SnapshotColumns.EMP_AGE
        age_band_col = SnapshotColumns.EMP_AGE_BAND
        
        if birth_date_col not in df.columns:
            logger.warning(f"No birth date column {birth_date_col} available for age calculation")
            df[age_col] = np.nan
            df[age_band_col] = 'UNKNOWN'
            return df
        
        logger.debug(f"Calculating age using column {birth_date_col}")
        
        # Calculate age in years
        birth_dates = pd.to_datetime(df[birth_date_col])
        age_days = (reference_date - birth_dates).dt.days
        df[age_col] = age_days / 365.25
        
        # Handle invalid ages
        invalid_age = (df[age_col] < 0) | (df[age_col] > 100)
        if invalid_age.any():
            logger.warning(f"Found {invalid_age.sum()} employees with invalid ages")
            df.loc[invalid_age, age_col] = np.nan
        
        # Calculate age bands
        df[age_band_col] = df[age_col].apply(self._calculate_age_band)
        
        return df
    
    def _fallback_job_level_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback job level inference based on compensation."""
        # Use unified schema column names
        level_col = SnapshotColumns.EMP_LEVEL
        level_source_col = SnapshotColumns.EMP_LEVEL_SOURCE
        comp_col = SnapshotColumns.EMP_GROSS_COMP
        
        if comp_col not in df.columns:
            logger.error(f"No compensation column {comp_col} found. Available columns: {list(df.columns)}")
            # Set default levels
            df[level_col] = 'BAND_1'
            df[level_source_col] = 'NO_COMPENSATION_DATA'
            return df
        
        # Handle empty DataFrame
        if df.empty:
            logger.warning("Empty DataFrame provided to fallback job level inference")
            return df
        
        # Simple compensation-based level assignment
        try:
            compensation = df[comp_col].fillna(0).astype(float)
            
            # Use numpy arrays to ensure proper boolean conditions for np.select
            conditions = [
                (compensation >= 100000).values,
                (compensation >= 80000).values,
                (compensation >= 60000).values,
                (compensation >= 40000).values,
                (compensation >= 20000).values
            ]
            
            choices = ['BAND_5', 'BAND_4', 'BAND_3', 'BAND_2', 'BAND_1']
            
            df[level_col] = np.select(conditions, choices, default='BAND_1')
            df[level_source_col] = 'INFERRED_FROM_COMPENSATION'
            
            logger.warning("Used fallback job level inference based on compensation")
            
        except Exception as e:
            logger.error(f"Error in fallback job level inference: {e}")
            # Set default values as final fallback
            df[level_col] = 'BAND_1'
            df[level_source_col] = 'FALLBACK_DEFAULT'
        
        return df