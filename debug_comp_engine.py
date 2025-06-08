#!/usr/bin/env python3
"""
Debug script to check what columns the compensation engine actually receives.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from logging_config import get_logger

# Import the necessary modules
from cost_model.config.loaders import load_config_to_namespace
from cost_model.projections.snapshot import create_initial_snapshot
from cost_model.projections.hazard import load_and_expand_hazard_table
from cost_model.engines.run_one_year.validation import validate_and_extract_hazard_slice

logger = get_logger(__name__)

def debug_compensation_engine():
    """Debug what columns the compensation engine receives."""
    
    logger.info("=" * 60)
    logger.info("DEBUGGING COMPENSATION ENGINE COLUMN ACCESS")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config_path = "config/dev_tiny.yaml"
        config = load_config_to_namespace(Path(config_path))
        global_params = config.global_parameters
        
        # Load initial snapshot
        census_path = "data/census_preprocessed.parquet"
        initial_snapshot = create_initial_snapshot(global_params.start_year, census_path)
        
        # Load hazard table
        logger.info("Loading hazard table...")
        hazard_table = load_and_expand_hazard_table('data/hazard_table.parquet')
        logger.info(f"Full hazard table columns: {hazard_table.columns.tolist()}")
        logger.info(f"Full hazard table shape: {hazard_table.shape}")
        
        # Extract hazard slice for year 2025
        year = 2025
        logger.info(f"\nExtracting hazard slice for year {year}...")
        hazard_slice = validate_and_extract_hazard_slice(hazard_table, year)
        logger.info(f"Hazard slice columns: {hazard_slice.columns.tolist()}")
        logger.info(f"Hazard slice shape: {hazard_slice.shape}")
        
        # Check for specific columns
        required_cols = ['merit_raise_pct', 'cola_pct']
        for col in required_cols:
            if col in hazard_slice.columns:
                sample_val = hazard_slice[col].iloc[0] if len(hazard_slice) > 0 else "N/A"
                logger.info(f"✅ Column '{col}' found. Sample value: {sample_val}")
            else:
                logger.error(f"❌ Column '{col}' MISSING!")
        
        # Check what the compensation engine would see
        logger.info(f"\nTesting what compensation engine would see...")
        
        # Simulate what happens in the comp engine
        if 'merit_raise_pct' in hazard_slice.columns:
            merit_col = 'merit_raise_pct'
            logger.info(f"✅ Compensation engine would use: {merit_col}")
        elif 'comp_raise_pct' in hazard_slice.columns:
            merit_col = 'comp_raise_pct'
            logger.info(f"⚠️  Compensation engine would fall back to: {merit_col}")
        else:
            logger.error("❌ Compensation engine would find NO merit column!")
            return False
        
        if 'cola_pct' in hazard_slice.columns:
            logger.info(f"✅ Compensation engine would use COLA column: cola_pct")
        else:
            logger.error("❌ Compensation engine would find NO COLA column!")
            return False
        
        # Show sample data
        if len(hazard_slice) > 0:
            logger.info(f"\nSample hazard slice data:")
            sample_row = hazard_slice.iloc[0]
            for col in ['simulation_year', 'employee_level', 'employee_tenure_band', 'merit_raise_pct', 'cola_pct']:
                if col in sample_row.index:
                    logger.info(f"  {col}: {sample_row[col]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in debugging: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = debug_compensation_engine()
    
    if success:
        print("\n🎉 Debugging completed successfully!")
        print("   The compensation engine should receive correct columns.")
    else:
        print("\n❌ Debugging revealed issues!")
        print("   The compensation engine may not receive correct columns.")