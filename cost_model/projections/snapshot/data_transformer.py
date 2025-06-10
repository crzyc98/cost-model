# cost_model/projections/snapshot/data_transformer.py
"""
Data transformation module for snapshot creation.
Handles tenure calculations, age calculations, job level inference, and other data transformations.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

from cost_model.state.snapshot.constants import SNAPSHOT_COLS as SNAPSHOT_COL_NAMES, SNAPSHOT_DTYPES
from cost_model.state.job_levels.loader import ingest_with_imputation
from cost_model.state.schema import (
    EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP, EMP_DEFERRAL_RATE,
    EMP_TENURE_BAND, EMP_TENURE, EMP_TERM_DATE, EMP_ACTIVE, EMP_LEVEL,
    EMP_LEVEL_SOURCE, EMP_EXITED, SIMULATION_YEAR, EMP_AGE, EMP_AGE_BAND
)

logger = logging.getLogger(__name__)


class DataTransformer:
    """Handles data transformation for snapshot creation."""
    
    def __init__(self):
        self.tenure_bands = {
            '<1': (0, 1),
            '1-3': (1, 3),
            '3-5': (3, 5),
            '5+': (5, float('inf'))
        }
    
    def calculate_tenure(self, census_df: pd.DataFrame, start_year: int) -> pd.DataFrame:
        """
        Calculate employee tenure based on hire date and start year.
        
        Args:
            census_df: Census DataFrame
            start_year: Simulation start year
            
        Returns:
            DataFrame with tenure calculated
        """
        logger.info("Calculating employee tenure")
        
        # Calculate tenure in years from hire date to start of simulation year
        start_date = pd.Timestamp(f'{start_year}-01-01')
        hire_dates = pd.to_datetime(census_df[EMP_HIRE_DATE])
        
        # Calculate tenure as years between hire date and start of simulation
        tenure_days = (start_date - hire_dates).dt.days
        tenure_years = tenure_days / 365.25  # Account for leap years
        
        # Ensure tenure is non-negative
        tenure_years = tenure_years.clip(lower=0)
        
        census_df[EMP_TENURE] = tenure_years.astype('float64')
        
        logger.info(f"Calculated tenure for {len(census_df)} employees. "
                   f"Range: {tenure_years.min():.2f} to {tenure_years.max():.2f} years")
        
        return census_df
    
    def assign_tenure_bands(self, census_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign tenure bands based on calculated tenure.
        
        Args:
            census_df: DataFrame with tenure calculated
            
        Returns:
            DataFrame with tenure bands assigned
        """
        def get_tenure_band(tenure: float) -> str:
            """Determine tenure band from tenure value."""
            if pd.isna(tenure):
                return pd.NA
            for band, (min_val, max_val) in self.tenure_bands.items():
                if min_val <= tenure < max_val:
                    return band
            return '5+'  # Default for values >= 5
        
        logger.info("Assigning tenure bands")
        
        if EMP_TENURE in census_df.columns:
            census_df[EMP_TENURE_BAND] = census_df[EMP_TENURE].apply(get_tenure_band)
            
            # Log distribution
            band_counts = census_df[EMP_TENURE_BAND].value_counts()
            logger.info(f"Tenure band distribution: {band_counts.to_dict()}")
        else:
            logger.warning("No tenure column found, setting tenure bands to NA")
            census_df[EMP_TENURE_BAND] = pd.NA
        
        return census_df
    
    def calculate_age_information(self, census_df: pd.DataFrame, start_year: int) -> pd.DataFrame:
        """
        Calculate age and age bands for employees.
        
        Args:
            census_df: Census DataFrame
            start_year: Simulation start year
            
        Returns:
            DataFrame with age information
        """
        from cost_model.state.age import apply_age
        
        logger.info("Calculating age information")
        
        # Apply age calculation using the existing age module
        census_df = apply_age(census_df, start_year)
        
        return census_df
    
    def infer_job_levels(self, census_df: pd.DataFrame) -> pd.DataFrame:
        """
        Infer job levels for employees using compensation data.
        
        Args:
            census_df: Census DataFrame with compensation
            
        Returns:
            DataFrame with job levels inferred
        """
        logger.info("Inferring job levels from compensation data")
        
        try:
            # Use existing job level inference logic
            census_df = ingest_with_imputation(
                census_df,
                compensation_col=EMP_GROSS_COMP,
                job_level_col=EMP_LEVEL,
                job_level_source_col=EMP_LEVEL_SOURCE
            )
            
            # Log results
            level_counts = census_df[EMP_LEVEL].value_counts().sort_index()
            source_counts = census_df[EMP_LEVEL_SOURCE].value_counts()
            
            logger.info(f"Job level distribution: {level_counts.to_dict()}")
            logger.info(f"Job level source distribution: {source_counts.to_dict()}")
            
        except Exception as e:
            logger.warning(f"Job level inference failed: {e}. Setting default values.")
            census_df[EMP_LEVEL] = pd.Series([pd.NA] * len(census_df), dtype='Int64')
            census_df[EMP_LEVEL_SOURCE] = pd.Series([pd.NA] * len(census_df), dtype='string')
        
        return census_df
    
    def set_employment_status(self, census_df: pd.DataFrame) -> pd.DataFrame:
        """
        Set employment status fields based on termination data.
        
        Args:
            census_df: Census DataFrame
            
        Returns:
            DataFrame with employment status set
        """
        logger.info("Setting employment status")
        
        # Determine active status
        has_term_dates = EMP_TERM_DATE in census_df.columns
        
        if has_term_dates:
            # Employee is active if they have no termination date
            active_status = census_df[EMP_TERM_DATE].isna()
            logger.info(f"Setting active status based on termination dates: "
                       f"{active_status.sum()} active out of {len(census_df)}")
        else:
            # If no termination dates, assume all are active
            active_status = pd.Series(True, index=census_df.index)
            logger.info("No termination dates found in census, assuming all employees are active")
        
        census_df[EMP_ACTIVE] = active_status
        census_df[EMP_EXITED] = ~active_status  # Exited is opposite of active
        
        return census_df
    
    def create_snapshot_structure(self, census_df: pd.DataFrame, start_year: int) -> pd.DataFrame:
        """
        Create the final snapshot structure with all required columns.
        
        Args:
            census_df: Processed census DataFrame
            start_year: Simulation start year
            
        Returns:
            Snapshot DataFrame with proper structure and types
        """
        logger.info("Creating snapshot structure")
        
        # Initialize snapshot data with all required columns
        snapshot_data = {
            EMP_ID: census_df[EMP_ID].astype('string'),
            EMP_HIRE_DATE: pd.to_datetime(census_df[EMP_HIRE_DATE]),
            EMP_BIRTH_DATE: pd.to_datetime(census_df[EMP_BIRTH_DATE]),
            EMP_GROSS_COMP: census_df[EMP_GROSS_COMP].astype('float64'),
            EMP_TERM_DATE: census_df.get(EMP_TERM_DATE, pd.NaT),
            EMP_ACTIVE: census_df[EMP_ACTIVE],
            EMP_DEFERRAL_RATE: census_df.get(EMP_DEFERRAL_RATE, 0.0).astype('float64'),
            EMP_TENURE: census_df.get(EMP_TENURE, 0.0).astype('float64'),
            EMP_TENURE_BAND: census_df.get(EMP_TENURE_BAND, pd.NA),
            EMP_LEVEL: census_df.get(EMP_LEVEL, pd.Series([pd.NA] * len(census_df), dtype='Int64')),
            EMP_LEVEL_SOURCE: census_df.get(EMP_LEVEL_SOURCE, pd.Series([pd.NA] * len(census_df), dtype='string')),
            EMP_EXITED: census_df.get(EMP_EXITED, False),
            SIMULATION_YEAR: start_year
        }
        
        # Add age columns if they exist
        if EMP_AGE in census_df.columns:
            snapshot_data[EMP_AGE] = census_df[EMP_AGE]
        if EMP_AGE_BAND in census_df.columns:
            snapshot_data[EMP_AGE_BAND] = census_df[EMP_AGE_BAND]
        
        # Create DataFrame
        snapshot_df = pd.DataFrame(snapshot_data)
        
        # Apply proper data types from SNAPSHOT_DTYPES
        for col, dtype in SNAPSHOT_DTYPES.items():
            if col in snapshot_df.columns:
                try:
                    snapshot_df[col] = snapshot_df[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Could not apply dtype {dtype} to column {col}: {e}")
        
        # Ensure all expected columns are present
        for col in SNAPSHOT_COL_NAMES:
            if col not in snapshot_df.columns:
                logger.warning(f"Adding missing column {col} with default values")
                # Add with appropriate default based on dtype
                if col in SNAPSHOT_DTYPES:
                    dtype = SNAPSHOT_DTYPES[col]
                    if 'int' in str(dtype).lower():
                        snapshot_df[col] = 0
                    elif 'float' in str(dtype).lower():
                        snapshot_df[col] = 0.0
                    elif 'bool' in str(dtype).lower():
                        snapshot_df[col] = False
                    else:
                        snapshot_df[col] = pd.NA
                else:
                    snapshot_df[col] = pd.NA
        
        logger.info(f"Created snapshot with {len(snapshot_df)} employees and {len(snapshot_df.columns)} columns")
        
        return snapshot_df
    
    def transform_census_to_snapshot(self, census_df: pd.DataFrame, start_year: int) -> pd.DataFrame:
        """
        Complete transformation pipeline from census to snapshot.
        
        Args:
            census_df: Processed census DataFrame
            start_year: Simulation start year
            
        Returns:
            Complete snapshot DataFrame
        """
        logger.info("Transforming census data to snapshot format")
        
        # Apply all transformations
        census_df = self.calculate_tenure(census_df, start_year)
        census_df = self.assign_tenure_bands(census_df)
        census_df = self.calculate_age_information(census_df, start_year)
        census_df = self.infer_job_levels(census_df)
        census_df = self.set_employment_status(census_df)
        
        # Create final snapshot structure
        snapshot_df = self.create_snapshot_structure(census_df, start_year)
        
        return snapshot_df