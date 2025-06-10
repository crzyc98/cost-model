# cost_model/engines/run_one_year/processors/promotion_processor.py
"""
Processor for handling employee promotions during yearly simulation.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from cost_model.engines.markov_promotion import apply_markov_promotions
from cost_model.state.schema import EMP_ACTIVE, EMP_ID, EMP_LEVEL, EMP_TENURE_BAND, SIMULATION_YEAR

from .base import BaseProcessor, ProcessorResult


class PromotionProcessor(BaseProcessor):
    """Handles Markov-chain based promotions for experienced employees."""

    def process(
        self,
        snapshot: pd.DataFrame,
        year: int,
        global_params: Any,
        hazard_table: pd.DataFrame,
        rng: Any,
        rng_seed_offset: int = 0,
        **kwargs,
    ) -> ProcessorResult:
        """
        Process promotions for experienced employees.

        Args:
            snapshot: Current employee snapshot
            year: Simulation year
            global_params: Global simulation parameters
            hazard_table: Hazard rate table
            rng: Random number generator
            rng_seed_offset: Seed offset for RNG

        Returns:
            ProcessorResult with promotion events and updated data
        """
        self.log_step_start(
            "Markov promotions/exits (experienced only)", snapshot_size=len(snapshot), year=year
        )

        result = ProcessorResult()

        try:
            # Validate inputs
            if not self.validate_inputs(
                snapshot=snapshot, year=year, global_params=global_params, hazard_table=hazard_table
            ):
                result.add_error("Invalid inputs for promotion processing")
                return result

            # Filter to experienced employees only (exclude new hires from this year)
            experienced_employees = (
                snapshot[snapshot[SIMULATION_YEAR] < year].copy()
                if SIMULATION_YEAR in snapshot.columns
                else snapshot.copy()
            )

            if experienced_employees.empty:
                self.logger.info("No experienced employees found for promotions")
                result.data = snapshot.copy()
                return result

            self.logger.info(
                f"Processing promotions for {len(experienced_employees)} experienced employees"
            )

            # Apply Markov promotions
            promotion_events = apply_markov_promotions(
                snapshot=experienced_employees,
                simulation_year=year,
                global_params=global_params,
                hazard_table=hazard_table,
                rng=rng,
                rng_seed_offset=rng_seed_offset,
            )

            # Log promotion results
            if not promotion_events.empty:
                self.logger.info(f"Generated {len(promotion_events)} promotion events")

                # Count promotion types if event type column exists
                if "event_type" in promotion_events.columns:
                    event_counts = promotion_events["event_type"].value_counts()
                    for event_type, count in event_counts.items():
                        self.logger.info(f"  {event_type}: {count} events")
            else:
                self.logger.info("No promotion events generated")

            result.events = promotion_events
            result.data = snapshot.copy()  # Snapshot will be updated later with events
            result.add_metadata("experienced_employees_count", len(experienced_employees))
            result.add_metadata("promotion_events_count", len(promotion_events))

        except Exception as e:
            self.logger.error(f"Error during promotion processing: {e}", exc_info=True)
            result.add_error(f"Promotion processing failed: {str(e)}")
            result.data = snapshot.copy()

        self.log_step_end(
            "Markov promotions/exits (experienced only)",
            events_generated=len(result.events),
            success=result.success,
        )

        return result

    def validate_inputs(
        self, snapshot: pd.DataFrame, year: int, global_params: Any, hazard_table: pd.DataFrame
    ) -> bool:
        """Validate inputs for promotion processing."""
        if snapshot.empty:
            self.logger.warning("Empty snapshot provided for promotion processing")
            return True  # Empty is valid, just means no promotions

        required_columns = [EMP_ID]
        missing_columns = [col for col in required_columns if col not in snapshot.columns]

        if missing_columns:
            self.logger.error(f"Missing required columns in snapshot: {missing_columns}")
            return False

        if hazard_table.empty:
            self.logger.warning("Empty hazard table provided")
            # This might be okay depending on configuration

        if year < 2000 or year > 2100:
            self.logger.warning(f"Unusual simulation year: {year}")

        return True
