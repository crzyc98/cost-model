# cost_model/projections/snapshot/snapshot_builder.py
"""
Main snapshot builder orchestrator.
Coordinates census processing, data transformation, and validation to create snapshots.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Union, Tuple, Optional

from cost_model.state.snapshot.constants import SNAPSHOT_COLS as SNAPSHOT_COL_NAMES, SNAPSHOT_DTYPES
from .census_processor import CensusProcessor
from .data_transformer import DataTransformer
from .validation_engine import ValidationEngine

logger = logging.getLogger(__name__)


class SnapshotBuilder:
    """
    Main orchestrator for building employee snapshots from census data.
    
    Coordinates the census processing, data transformation, and validation
    to produce clean, validated snapshot DataFrames.
    """
    
    def __init__(self):
        self.census_processor = CensusProcessor()
        self.data_transformer = DataTransformer()
        self.validation_engine = ValidationEngine()
    
    def create_initial_snapshot(self, start_year: int, census_path: Union[str, Path]) -> pd.DataFrame:
        """
        Create the initial employee snapshot from census data.
        
        This is the main entry point that replaces the original create_initial_snapshot function.
        
        Args:
            start_year: The starting year for the simulation
            census_path: Path to the census data file (Parquet or CSV format)
        
        Returns:
            DataFrame containing the initial employee snapshot
        
        Raises:
            FileNotFoundError: If the census file doesn't exist
            ValueError: If the census data is invalid or missing required columns
        """
        logger.info("Creating initial snapshot for start year: %d from %s", start_year, str(census_path))
        
        try:
            # Step 1: Process census data
            census_df = self.census_processor.process_census_data(census_path, start_year)
            
            if census_df.empty:
                logger.warning("Census data is empty. Creating empty snapshot.")
                return pd.DataFrame(columns=SNAPSHOT_COL_NAMES).astype(SNAPSHOT_DTYPES)
            
            # Step 2: Transform to snapshot format
            snapshot_df = self.data_transformer.transform_census_to_snapshot(census_df, start_year)
            
            # Step 3: Validate the result
            is_valid, errors, warnings = self.validation_engine.validate_snapshot(
                snapshot_df, start_year
            )
            
            if not is_valid:
                error_msg = f"Snapshot validation failed: {'; '.join(errors)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info("Successfully created initial snapshot with %d employees", len(snapshot_df))
            return snapshot_df
            
        except Exception as e:
            logger.error("Failed to create initial snapshot: %s", str(e), exc_info=True)
            raise
    
    def build_enhanced_yearly_snapshot(
        self,
        start_of_year_snapshot: pd.DataFrame,
        end_of_year_snapshot: pd.DataFrame,
        year_events: pd.DataFrame,
        simulation_year: int
    ) -> pd.DataFrame:
        """
        Build an enhanced yearly snapshot including all employees active during the year.
        
        This method will be implemented to replace the large build_enhanced_yearly_snapshot function.
        For now, it delegates to the original implementation.
        
        Args:
            start_of_year_snapshot: Snapshot at the beginning of the year
            end_of_year_snapshot: Snapshot at the end of the year (after all events)
            year_events: Events that occurred during this simulation year
            simulation_year: The simulation year being processed
        
        Returns:
            Enhanced yearly snapshot DataFrame with all employees active during the year
        """
        # Import the original function for now
        from cost_model.projections.snapshot import build_enhanced_yearly_snapshot as original_build
        
        logger.info("Building enhanced yearly snapshot for year %d using original implementation", 
                   simulation_year)
        
        # Delegate to original implementation
        # TODO: Refactor this into modular components
        return original_build(
            start_of_year_snapshot,
            end_of_year_snapshot, 
            year_events,
            simulation_year
        )
    
    def validate_snapshot_integrity(self, snapshot_df: pd.DataFrame, simulation_year: int) -> Tuple[bool, list, list]:
        """
        Validate snapshot integrity and business rules.
        
        Args:
            snapshot_df: Snapshot DataFrame to validate
            simulation_year: Current simulation year
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        return self.validation_engine.validate_snapshot(snapshot_df, simulation_year)
    
    def create_empty_snapshot(self) -> pd.DataFrame:
        """
        Create an empty snapshot with proper schema.
        
        Returns:
            Empty DataFrame with correct columns and types
        """
        logger.info("Creating empty snapshot")
        return pd.DataFrame(columns=SNAPSHOT_COL_NAMES).astype(SNAPSHOT_DTYPES)