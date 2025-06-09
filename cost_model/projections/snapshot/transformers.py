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
    
    def apply_tenure_calculations(self, df: pd.DataFrame, reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Calculate tenure and tenure bands for employees.
        
        Args:
            df: DataFrame with employee data
            reference_date: Date to calculate tenure from (defaults to start of config year)
            
        Returns:
            DataFrame with tenure calculations added
        """
        # Check which hire date column is available
        hire_date_columns = ['EMP_HIRE_DATE', 'employee_hire_date']
        hire_date_col = None
        for col in hire_date_columns:
            if col in df.columns:
                hire_date_col = col
                break
        
        if hire_date_col is None:
            raise SnapshotBuildError(f"Cannot calculate tenure: No hire date column found. Available columns: {list(df.columns)}")
        
        # Use standardized column names for outputs
        EMP_TENURE = 'EMP_TENURE'
        EMP_TENURE_BAND = 'EMP_TENURE_BAND'
        
        if reference_date is None:
            reference_date = datetime(self.config.start_year, 1, 1)
        
        logger.debug(f"Calculating tenure as of {reference_date} using column {hire_date_col}")
        
        # Calculate tenure in years
        hire_dates = pd.to_datetime(df[hire_date_col])
        tenure_days = (reference_date - hire_dates).dt.days
        df[EMP_TENURE] = tenure_days / 365.25  # Account for leap years
        
        # Handle negative tenure (future hire dates)
        negative_tenure = df[EMP_TENURE] < 0
        if negative_tenure.any():
            logger.warning(f"Found {negative_tenure.sum()} employees with future hire dates")
            df.loc[negative_tenure, EMP_TENURE] = 0
        
        # Calculate tenure bands
        df[EMP_TENURE_BAND] = df[EMP_TENURE].apply(self._calculate_tenure_band)
        
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
        # Use standardized column names
        EMP_BIRTH_DATE = 'EMP_BIRTH_DATE'
        EMP_AGE = 'EMP_AGE'
        EMP_AGE_BAND = 'EMP_AGE_BAND'
        
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
                    birth_col=EMP_BIRTH_DATE,
                    as_of=reference_date,
                    out_age_col=EMP_AGE,
                    out_band_col=EMP_AGE_BAND
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
        # Use standardized column names
        EMP_GROSS_COMP = 'EMP_GROSS_COMP'
        EMP_DEFERRAL_RATE = 'EMP_DEFERRAL_RATE'
        EMP_EMPLOYEE_CONTRIB = 'EMP_EMPLOYEE_CONTRIB'
        EMP_EMPLOYER_MATCH = 'EMP_EMPLOYER_MATCH'
        
        logger.debug("Calculating contribution amounts")
        
        # Ensure required columns exist
        if EMP_GROSS_COMP not in df.columns:
            raise SnapshotBuildError("Cannot calculate contributions: EMP_GROSS_COMP column missing")
        
        if EMP_DEFERRAL_RATE not in df.columns:
            logger.warning("EMP_DEFERRAL_RATE missing, defaulting to 0.0")
            df[EMP_DEFERRAL_RATE] = 0.0
        
        # Calculate employee contributions
        df[EMP_EMPLOYEE_CONTRIB] = df[EMP_GROSS_COMP] * df[EMP_DEFERRAL_RATE]
        
        # Calculate employer match (simple 50% of employee contribution up to 6% of compensation)
        max_match_base = df[EMP_GROSS_COMP] * 0.06  # 6% of compensation
        employee_contrib_eligible = df[EMP_EMPLOYEE_CONTRIB].clip(upper=max_match_base)
        df[EMP_EMPLOYER_MATCH] = employee_contrib_eligible * 0.5  # 50% match
        
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
        # Use standardized column names
        EMP_LEVEL = 'EMP_LEVEL'
        EMP_LEVEL_SOURCE = 'EMP_LEVEL_SOURCE'
        EMP_GROSS_COMP = 'EMP_GROSS_COMP'
        
        try:
            from cost_model.state.job_levels.loader import ingest_with_imputation
        except ImportError:
            ingest_with_imputation = None
        
        logger.debug("Inferring job levels")
        
        if ingest_with_imputation is not None:
            try:
                # Use existing job level inference if available
                df = ingest_with_imputation(df)
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
        # Use standardized column names
        EMP_LEVEL = 'EMP_LEVEL'
        EMP_GROSS_COMP = 'EMP_GROSS_COMP'
        
        logger.debug("Normalizing compensation by job level")
        
        if EMP_LEVEL not in df.columns:
            logger.warning("No job level column available for compensation normalization")
            return df
        
        # Fill missing compensation with level-based defaults
        missing_comp = df[EMP_GROSS_COMP].isna()
        if missing_comp.any():
            logger.info(f"Filling {missing_comp.sum()} missing compensation values")
            
            for level, default_comp in LEVEL_BASED_DEFAULTS.items():
                level_mask = (df[EMP_LEVEL] == level) & missing_comp
                df.loc[level_mask, EMP_GROSS_COMP] = default_comp
            
            # Fill any remaining missing values with global default
            still_missing = df[EMP_GROSS_COMP].isna()
            if still_missing.any():
                from .constants import DEFAULT_COMPENSATION
                df.loc[still_missing, EMP_GROSS_COMP] = DEFAULT_COMPENSATION
                logger.warning(f"Used global default compensation for {still_missing.sum()} employees")
        
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
        # Use standardized column names
        EMP_BIRTH_DATE = 'EMP_BIRTH_DATE'
        EMP_AGE = 'EMP_AGE'
        EMP_AGE_BAND = 'EMP_AGE_BAND'
        
        if EMP_BIRTH_DATE not in df.columns:
            logger.warning("No birth date column available for age calculation")
            df[EMP_AGE] = np.nan
            df[EMP_AGE_BAND] = 'UNKNOWN'
            return df
        
        # Calculate age in years
        birth_dates = pd.to_datetime(df[EMP_BIRTH_DATE])
        age_days = (reference_date - birth_dates).dt.days
        df[EMP_AGE] = age_days / 365.25
        
        # Handle invalid ages
        invalid_age = (df[EMP_AGE] < 0) | (df[EMP_AGE] > 100)
        if invalid_age.any():
            logger.warning(f"Found {invalid_age.sum()} employees with invalid ages")
            df.loc[invalid_age, EMP_AGE] = np.nan
        
        # Calculate age bands
        df[EMP_AGE_BAND] = df[EMP_AGE].apply(self._calculate_age_band)
        
        return df
    
    def _fallback_job_level_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback job level inference based on compensation."""
        # Use standardized column names
        EMP_LEVEL = 'EMP_LEVEL'
        EMP_LEVEL_SOURCE = 'EMP_LEVEL_SOURCE'
        
        # Check which compensation column is available
        comp_columns = ['EMP_GROSS_COMP', 'employee_gross_compensation']
        comp_col = None
        for col in comp_columns:
            if col in df.columns:
                comp_col = col
                break
        
        if comp_col is None:
            logger.error(f"No compensation column found. Available columns: {list(df.columns)}")
            # Set default levels
            df[EMP_LEVEL] = 'BAND_1'
            df[EMP_LEVEL_SOURCE] = 'NO_COMPENSATION_DATA'
            return df
        
        # Simple compensation-based level assignment
        compensation = df[comp_col].fillna(0)
        
        conditions = [
            compensation >= 100000,
            compensation >= 80000,
            compensation >= 60000,
            compensation >= 40000,
            compensation >= 20000
        ]
        
        choices = ['BAND_5', 'BAND_4', 'BAND_3', 'BAND_2', 'BAND_1']
        
        df[EMP_LEVEL] = np.select(conditions, choices, default='BAND_1')
        df[EMP_LEVEL_SOURCE] = 'INFERRED_FROM_COMPENSATION'
        
        logger.warning("Used fallback job level inference based on compensation")
        return df