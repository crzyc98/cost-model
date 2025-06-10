# cost_model/engines/run_one_year/processors/contribution_processor.py
"""
Processor for handling contribution calculations and plan rules during yearly simulation.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from cost_model.state.schema import EMP_ACTIVE, EMP_ID, SIMULATION_YEAR

from .base import BaseProcessor, ProcessorResult


class ContributionProcessor(BaseProcessor):
    """Handles contribution calculations and eligibility rules."""

    def process(
        self,
        snapshot: pd.DataFrame,
        year: int,
        global_params: Any,
        plan_rules: Dict[str, Any],
        **kwargs,
    ) -> ProcessorResult:
        """
        Process contribution calculations and plan eligibility.

        Args:
            snapshot: Current employee snapshot
            year: Simulation year
            global_params: Global simulation parameters
            plan_rules: Plan rules configuration

        Returns:
            ProcessorResult with contribution events and updated snapshot
        """
        self.log_step_start(
            "Apply contribution calculations and eligibility to final snapshot",
            snapshot_size=len(snapshot),
            year=year,
        )

        result = ProcessorResult()

        try:
            # Validate inputs
            if not self.validate_inputs(snapshot, year, global_params, plan_rules):
                result.add_error("Invalid inputs for contribution processing")
                return result

            # Start with the input snapshot
            updated_snapshot = snapshot.copy()

            # Apply contribution calculations
            updated_snapshot, contribution_events = self._apply_contribution_calculations(
                updated_snapshot, year, global_params, plan_rules
            )

            # Apply eligibility rules
            updated_snapshot, eligibility_events = self._apply_eligibility_rules(
                updated_snapshot, year, global_params, plan_rules
            )

            # Combine all events
            all_events = pd.DataFrame()
            if not contribution_events.empty:
                all_events = pd.concat([all_events, contribution_events], ignore_index=True)
            if not eligibility_events.empty:
                all_events = pd.concat([all_events, eligibility_events], ignore_index=True)

            # Log results
            self.logger.info(f"Contribution processing completed:")
            self.logger.info(f"  Contribution events: {len(contribution_events)}")
            self.logger.info(f"  Eligibility events: {len(eligibility_events)}")
            self.logger.info(f"  Total events: {len(all_events)}")

            result.data = updated_snapshot
            result.events = all_events
            result.add_metadata("contribution_events_count", len(contribution_events))
            result.add_metadata("eligibility_events_count", len(eligibility_events))

        except Exception as e:
            self.logger.error(f"Error during contribution processing: {e}", exc_info=True)
            result.add_error(f"Contribution processing failed: {str(e)}")
            result.data = snapshot.copy()

        self.log_step_end(
            "Apply contribution calculations and eligibility to final snapshot",
            events_generated=len(result.events),
            success=result.success,
        )

        return result

    def _apply_contribution_calculations(
        self, snapshot: pd.DataFrame, year: int, global_params: Any, plan_rules: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply contribution calculations to the snapshot.

        Args:
            snapshot: Employee snapshot
            year: Simulation year
            global_params: Global parameters
            plan_rules: Plan rules configuration

        Returns:
            Tuple of (updated_snapshot, contribution_events)
        """
        self.logger.info("Applying contribution calculations")

        # TODO: Implement actual contribution calculation logic
        # This is a placeholder that maintains the original structure

        try:
            # Import the actual contribution calculation logic
            from cost_model.plan_rules.contributions import apply_contributions

            updated_snapshot, events = apply_contributions(
                snapshot=snapshot,
                simulation_year=year,
                global_params=global_params,
                plan_rules=plan_rules,
            )

            return updated_snapshot, events

        except ImportError:
            self.logger.warning("Contribution calculation module not available, skipping")
            return snapshot.copy(), pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error in contribution calculations: {e}")
            return snapshot.copy(), pd.DataFrame()

    def _apply_eligibility_rules(
        self, snapshot: pd.DataFrame, year: int, global_params: Any, plan_rules: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply eligibility rules to the snapshot.

        Args:
            snapshot: Employee snapshot
            year: Simulation year
            global_params: Global parameters
            plan_rules: Plan rules configuration

        Returns:
            Tuple of (updated_snapshot, eligibility_events)
        """
        self.logger.info("Applying eligibility rules")

        # TODO: Implement actual eligibility rule logic
        # This is a placeholder that maintains the original structure

        try:
            # Import the actual eligibility rule logic
            from cost_model.plan_rules.eligibility import apply_eligibility

            updated_snapshot, events = apply_eligibility(
                snapshot=snapshot,
                simulation_year=year,
                global_params=global_params,
                plan_rules=plan_rules,
            )

            return updated_snapshot, events

        except ImportError:
            self.logger.warning("Eligibility rule module not available, skipping")
            return snapshot.copy(), pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error in eligibility rules: {e}")
            return snapshot.copy(), pd.DataFrame()

    def validate_inputs(
        self, snapshot: pd.DataFrame, year: int, global_params: Any, plan_rules: Dict[str, Any]
    ) -> bool:
        """Validate inputs for contribution processing."""
        if snapshot.empty:
            self.logger.warning("Empty snapshot provided for contribution processing")
            return True  # Empty is valid

        required_columns = [EMP_ID]
        missing_columns = [col for col in required_columns if col not in snapshot.columns]

        if missing_columns:
            self.logger.error(f"Missing required columns in snapshot: {missing_columns}")
            return False

        if plan_rules is None:
            self.logger.warning("No plan rules provided")
            # This might be okay depending on configuration

        return True
