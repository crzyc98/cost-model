# cost_model/engines/run_one_year/year_orchestrator.py
"""
Refactored orchestrator for yearly simulation using modular processors.

This replaces the monolithic orchestrator_original.py with a clean, modular design
that uses separate processor classes for each simulation step.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from cost_model.state.schema import EMP_ACTIVE, EMP_ID, SIMULATION_YEAR
from cost_model.state.snapshot import update as snapshot_update

from .processors import (
    ContributionProcessor,
    DiagnosticLogger,
    EventConsolidator,
    HiringProcessor,
    PromotionProcessor,
    TerminationProcessor,
)
from .validation import ensure_snapshot_cols, validate_eoy_snapshot

logger = logging.getLogger(__name__)


class YearOrchestrator:
    """
    Orchestrates all simulation steps for a single year using modular processors.

    This class replaces the monolithic run_one_year function with a clean,
    maintainable design that separates concerns and improves testability.
    """

    def __init__(self, enable_diagnostics: bool = True):
        """
        Initialize the orchestrator with processor instances.

        Args:
            enable_diagnostics: Whether to enable detailed diagnostic logging
        """
        self.promotion_processor = PromotionProcessor()
        self.termination_processor = TerminationProcessor()
        self.hiring_processor = HiringProcessor()
        self.contribution_processor = ContributionProcessor()
        self.event_consolidator = EventConsolidator()

        self.diagnostic_logger = DiagnosticLogger() if enable_diagnostics else None
        self.enable_diagnostics = enable_diagnostics

        # Track processing state
        self.current_snapshot = None
        self.accumulated_events = []
        self.processing_metadata = {}

    def run_one_year(
        self,
        event_log: pd.DataFrame,
        prev_snapshot: pd.DataFrame,
        year: int,
        global_params: Any,
        plan_rules: Dict[str, Any],
        hazard_table: pd.DataFrame,
        rng: Any,
        census_template_path: Optional[str] = None,
        rng_seed_offset: int = 0,
        deterministic_term: bool = False,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Orchestrate simulation for a single year using modular processors.

        Args:
            event_log: Previous event log
            prev_snapshot: Previous year's snapshot
            year: Simulation year
            global_params: Global simulation parameters
            plan_rules: Plan rules configuration
            hazard_table: Hazard rate table
            rng: Random number generator
            census_template_path: Path to census template
            rng_seed_offset: RNG seed offset
            deterministic_term: Whether to use deterministic termination

        Returns:
            Tuple of (updated_event_log, final_snapshot)
        """
        logger.info(f"Starting orchestrated simulation for year {year}")

        try:
            # Initialize processing state
            self.current_snapshot = prev_snapshot.copy()
            self.accumulated_events = []
            self.processing_metadata = {
                "year": year,
                "deterministic_term": deterministic_term,
                "rng_seed_offset": rng_seed_offset,
            }

            # Ensure snapshot has required columns
            self.current_snapshot = ensure_snapshot_cols(self.current_snapshot)

            # Step 1: Process promotions for experienced employees
            promotion_result = self._process_promotions(
                year, global_params, hazard_table, rng, rng_seed_offset
            )

            # Step 2: Process terminations for experienced employees
            termination_result = self._process_experienced_terminations(
                year, global_params, hazard_table, rng, rng_seed_offset, deterministic_term
            )

            # Step 3: Update snapshot with promotion/termination events
            self._update_snapshot_with_events(
                [promotion_result.events, termination_result.events], year
            )

            # Step 4: Process hiring
            hiring_result = self._process_hiring(
                year, global_params, rng, census_template_path, rng_seed_offset
            )

            # Step 5: Update snapshot with hire events
            self._update_snapshot_with_events([hiring_result.events], year)

            # Step 6: Process new hire terminations
            nh_termination_result = self._process_new_hire_terminations(
                year, global_params, rng, rng_seed_offset
            )

            # Step 7: Update snapshot with new hire termination events
            self._update_snapshot_with_events([nh_termination_result.events], year)

            # Step 8: Process contributions and eligibility
            contribution_result = self._process_contributions(year, global_params, plan_rules)

            # Step 9: Update snapshot with contribution events
            if not contribution_result.events.empty:
                self.current_snapshot = contribution_result.data

            # Step 10: Consolidate all events
            event_consolidation_result = self._consolidate_events(
                promotion_result.events,
                termination_result.events,
                hiring_result.events,
                contribution_result.events,
                nh_termination_result.events,
                year,
            )

            # Step 11: Final validation
            final_snapshot = self._finalize_snapshot(year)

            # Step 12: Build final event log
            updated_event_log = self._build_event_log(event_log, event_consolidation_result.data)

            # Log final summary
            if self.diagnostic_logger:
                self.diagnostic_logger.log_year_summary(
                    year, final_snapshot, event_consolidation_result.data
                )

            logger.info(f"Completed orchestrated simulation for year {year}")
            logger.info(f"  Final headcount: {len(final_snapshot)}")
            logger.info(f"  Total events: {len(event_consolidation_result.data)}")

            return updated_event_log, final_snapshot

        except Exception as e:
            logger.error(f"Error in year orchestration for year {year}: {e}", exc_info=True)
            raise

    def _process_promotions(
        self,
        year: int,
        global_params: Any,
        hazard_table: pd.DataFrame,
        rng: Any,
        rng_seed_offset: int,
    ):
        """Process promotions using the promotion processor."""
        snapshot_before = self.current_snapshot.copy()

        result = self.promotion_processor.process(
            snapshot=self.current_snapshot,
            year=year,
            global_params=global_params,
            hazard_table=hazard_table,
            rng=rng,
            rng_seed_offset=rng_seed_offset,
        )

        if self.diagnostic_logger:
            self.diagnostic_logger.log_step_diagnostics(
                "promotions", snapshot_before, self.current_snapshot, result.events, year
            )

        return result

    def _process_experienced_terminations(
        self,
        year: int,
        global_params: Any,
        hazard_table: pd.DataFrame,
        rng: Any,
        rng_seed_offset: int,
        deterministic_term: bool,
    ):
        """Process experienced employee terminations."""
        snapshot_before = self.current_snapshot.copy()

        result = self.termination_processor.process_experienced_terminations(
            snapshot=self.current_snapshot,
            year=year,
            global_params=global_params,
            hazard_table=hazard_table,
            rng=rng,
            rng_seed_offset=rng_seed_offset,
            deterministic_term=deterministic_term,
        )

        if self.diagnostic_logger:
            self.diagnostic_logger.log_step_diagnostics(
                "experienced_terminations",
                snapshot_before,
                self.current_snapshot,
                result.events,
                year,
            )

        return result

    def _process_hiring(
        self,
        year: int,
        global_params: Any,
        rng: Any,
        census_template_path: Optional[str],
        rng_seed_offset: int,
    ):
        """Process employee hiring."""
        snapshot_before = self.current_snapshot.copy()

        result = self.hiring_processor.process(
            snapshot=self.current_snapshot,
            year=year,
            global_params=global_params,
            rng=rng,
            census_template_path=census_template_path,
            rng_seed_offset=rng_seed_offset,
        )

        if self.diagnostic_logger:
            self.diagnostic_logger.log_step_diagnostics(
                "hiring", snapshot_before, self.current_snapshot, result.events, year
            )

        return result

    def _process_new_hire_terminations(
        self, year: int, global_params: Any, rng: Any, rng_seed_offset: int
    ):
        """Process new hire terminations."""
        snapshot_before = self.current_snapshot.copy()

        result = self.termination_processor.process_new_hire_terminations(
            snapshot=self.current_snapshot,
            year=year,
            global_params=global_params,
            rng=rng,
            rng_seed_offset=rng_seed_offset,
        )

        if self.diagnostic_logger:
            self.diagnostic_logger.log_step_diagnostics(
                "new_hire_terminations", snapshot_before, self.current_snapshot, result.events, year
            )

        return result

    def _process_contributions(self, year: int, global_params: Any, plan_rules: Dict[str, Any]):
        """Process contributions and eligibility."""
        snapshot_before = self.current_snapshot.copy()

        result = self.contribution_processor.process(
            snapshot=self.current_snapshot,
            year=year,
            global_params=global_params,
            plan_rules=plan_rules,
        )

        if self.diagnostic_logger:
            snapshot_after = result.data if result.success else self.current_snapshot
            self.diagnostic_logger.log_step_diagnostics(
                "contributions", snapshot_before, snapshot_after, result.events, year
            )

        return result

    def _update_snapshot_with_events(self, event_lists: List[pd.DataFrame], year: int):
        """Update the current snapshot with events from processors."""
        # Combine all events
        all_events = pd.DataFrame()
        for events in event_lists:
            if not events.empty:
                all_events = pd.concat([all_events, events], ignore_index=True)

        if not all_events.empty:
            # Apply events to snapshot
            self.current_snapshot = snapshot_update(self.current_snapshot, all_events, year)
            logger.debug(f"Updated snapshot with {len(all_events)} events")

    def _consolidate_events(
        self,
        promotion_events: pd.DataFrame,
        termination_events: pd.DataFrame,
        hiring_events: pd.DataFrame,
        contribution_events: pd.DataFrame,
        nh_termination_events: pd.DataFrame,
        year: int,
    ):
        """Consolidate all events using the event consolidator."""
        return self.event_consolidator.consolidate_events(
            promotion_events=promotion_events,
            termination_events=termination_events,
            hiring_events=hiring_events,
            contribution_events=contribution_events,
            new_hire_termination_events=nh_termination_events,
            year=year,
        )

    def _finalize_snapshot(self, year: int) -> pd.DataFrame:
        """Finalize and validate the snapshot."""
        logger.info("[STEP] Final snapshot validation")

        # Ensure all required columns are present
        final_snapshot = ensure_snapshot_cols(self.current_snapshot)

        # Validate the final snapshot
        validation_result = validate_eoy_snapshot(final_snapshot, year)
        if not validation_result["valid"]:
            for warning in validation_result["warnings"]:
                logger.warning(f"Snapshot validation warning: {warning}")
            for error in validation_result["errors"]:
                logger.error(f"Snapshot validation error: {error}")

        logger.info(f"Finalized snapshot with {len(final_snapshot)} employees")

        return final_snapshot

    def _build_event_log(
        self, prev_event_log: pd.DataFrame, year_events: pd.DataFrame
    ) -> pd.DataFrame:
        """Build the updated event log."""
        logger.info("[STEP] Build final event log")

        if year_events.empty:
            logger.info("No new events to add to event log")
            return prev_event_log.copy()

        # Concatenate previous events with new events
        if prev_event_log.empty:
            updated_event_log = year_events.copy()
        else:
            updated_event_log = pd.concat([prev_event_log, year_events], ignore_index=True)

        logger.info(
            f"Updated event log: {len(prev_event_log)} + {len(year_events)} = {len(updated_event_log)} events"
        )

        return updated_event_log

    def get_diagnostic_data(self) -> Dict[str, Any]:
        """Get diagnostic data from the orchestrator."""
        if self.diagnostic_logger:
            return self.diagnostic_logger.get_diagnostic_data()
        return {}


# Backward compatibility function
def run_one_year(
    event_log: pd.DataFrame,
    prev_snapshot: pd.DataFrame,
    year: int,
    global_params: Any,
    plan_rules: Dict[str, Any],
    hazard_table: pd.DataFrame,
    rng: Any,
    census_template_path: Optional[str] = None,
    rng_seed_offset: int = 0,
    deterministic_term: bool = False,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backward compatibility wrapper for the refactored orchestrator.

    This function maintains the same interface as the original run_one_year
    function but uses the new modular YearOrchestrator internally.
    """
    orchestrator = YearOrchestrator(enable_diagnostics=kwargs.get("enable_diagnostics", True))

    return orchestrator.run_one_year(
        event_log=event_log,
        prev_snapshot=prev_snapshot,
        year=year,
        global_params=global_params,
        plan_rules=plan_rules,
        hazard_table=hazard_table,
        rng=rng,
        census_template_path=census_template_path,
        rng_seed_offset=rng_seed_offset,
        deterministic_term=deterministic_term,
        **kwargs,
    )
