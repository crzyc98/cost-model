# cost_model/engines/run_one_year/orchestrator/__init__.py
"""
Refactored orchestrator package for run_one_year simulation engine.

This package provides a decomposed, modular implementation of the workforce simulation
orchestration logic, with extracted diagnostic utilities and event consolidation.

The main entry point is the run_one_year() function, which maintains the exact same
signature and behavior as the original implementation.
"""
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from logging_config import get_diagnostic_logger, get_logger

from cost_model.state.event_log import EVENT_COLS
from cost_model.state.schema import (
    ACTIVE_STATUS,
    EMP_ACTIVE,
    EMP_BIRTH_DATE,
    EMP_CONTR,
    EMP_HIRE_DATE,
    EMP_ID,
    EMP_STATUS_EOY,
    EMPLOYER_CORE,
    EMPLOYER_MATCH,
    IS_ELIGIBLE,
)

# Import validation and utility functions
from ..validation import ensure_snapshot_cols, validate_and_extract_hazard_slice

# Import orchestrator components
from .base import YearContext, ensure_simulation_year_column, filter_valid_employee_ids

# Import extracted diagnostic and event consolidation utilities
from .diagnostic_utils import DiagnosticTracker, log_headcount_stage, n_active
from .event_consolidation import EventConsolidationManager, consolidate_events
from .hiring import HiringOrchestrator
from .promotion import PromotionOrchestrator
from .termination import TerminationOrchestrator
from .validator import SnapshotValidator


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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates simulation for a single year using refactored modular components.

    This function maintains the exact same signature and behavior as the original
    monolithic implementation, but uses extracted diagnostic utilities and
    event consolidation logic for better maintainability.

    Args:
        event_log: Cumulative event log from previous years
        prev_snapshot: Workforce snapshot from end of previous year
        year: Current simulation year
        global_params: Global simulation parameters
        plan_rules: Plan configuration rules
        hazard_table: Hazard rates table for all years
        rng: Random number generator
        census_template_path: Optional path to census template
        rng_seed_offset: Offset for year-specific RNG seeding
        deterministic_term: Whether to use deterministic terminations

    Returns:
        Tuple of (new_events, final_snapshot) where:
        - new_events: DataFrame of all events generated this year
        - final_snapshot: Final workforce snapshot at end of year
    """
    logger = get_logger(__name__)
    diag_logger = get_diagnostic_logger(__name__)

    logger.info(f"[RUN_ONE_YEAR] Simulating year {year} with refactored orchestrator")

    # Initialize diagnostic tracking
    diagnostic_tracker = DiagnosticTracker(diag_logger, year)
    event_manager = EventConsolidationManager(logger, year)

    # Initialize orchestrator components
    hiring_orchestrator = HiringOrchestrator(logger)
    termination_orchestrator = TerminationOrchestrator(logger)
    promotion_orchestrator = PromotionOrchestrator(logger)
    validator = SnapshotValidator(logger)

    try:
        # Prepare inputs and validate
        snapshot, hazard_slice = _prepare_inputs(
            prev_snapshot, year, hazard_table, logger, diagnostic_tracker
        )

        # Create year context
        year_context = YearContext.create(
            year=year,
            global_params=global_params,
            plan_rules=plan_rules,
            hazard_slice=hazard_slice,
            rng=rng,
            rng_seed_offset=rng_seed_offset,
            deterministic_term=deterministic_term,
            census_template_path=census_template_path,
        )

        # Step 1: Process promotions for experienced employees
        promotion_events, snapshot = promotion_orchestrator.get_events(snapshot, year_context)
        event_manager.add_events(promotion_events, "promotion")
        diagnostic_tracker.track_stage(snapshot, "Post-Promotions")

        # Capture start count before terminations for hiring calculations
        start_count_before_terminations = len(snapshot[snapshot[EMP_ACTIVE]] if EMP_ACTIVE in snapshot.columns else snapshot)

        # Step 2: Process terminations for experienced employees
        termination_events, snapshot = termination_orchestrator.get_experienced_termination_events(
            snapshot, year_context
        )
        event_manager.add_events(termination_events, "termination")
        diagnostic_tracker.track_stage(snapshot, "Post-Experienced-Terminations")

        # Step 3: Process hiring
        hiring_events, snapshot = hiring_orchestrator.get_events(
            snapshot, year_context, start_count=start_count_before_terminations
        )
        event_manager.add_events(hiring_events, "hiring")
        diagnostic_tracker.track_stage(snapshot, "Post-Hiring")

        # Step 4: Process new hire terminations
        nh_termination_events, snapshot = termination_orchestrator.get_new_hire_termination_events(
            snapshot, year_context
        )
        event_manager.add_events(nh_termination_events, "nh_termination")
        diagnostic_tracker.track_stage(snapshot, "Post-New-Hire-Terminations")

        # Step 5: Apply contribution calculations
        contribution_events, snapshot = _apply_contribution_calculations(
            snapshot, year, global_params, plan_rules, logger
        )
        event_manager.add_events(contribution_events, "contribution")
        diagnostic_tracker.track_stage(snapshot, "Post-Contributions")

        # Step 6: Apply compensation events (merit raises and COLA)
        compensation_events = _generate_compensation_events(snapshot, year_context, logger)
        event_manager.add_events(compensation_events, "compensation")

        # Step 7: Final validation
        validator.validate_eoy(snapshot)
        final_snapshot = snapshot
        diagnostic_tracker.track_stage(final_snapshot, "Final-Snapshot")

        # Step 8: Consolidate all events
        consolidated_events = event_manager.consolidate_all()

        # Log summary
        logger.info(f"[RUN_ONE_YEAR] Completed year {year}")
        logger.info(f"  Final headcount: {len(final_snapshot)}")
        logger.info(f"  Active employees: {n_active(final_snapshot)}")
        logger.info(f"  Total events: {len(consolidated_events)}")

        return consolidated_events, final_snapshot

    except Exception as e:
        logger.error(f"Error in orchestrated simulation for year {year}: {e}", exc_info=True)
        raise


def _prepare_inputs(
    prev_snapshot: pd.DataFrame,
    year: int,
    hazard_table: pd.DataFrame,
    logger: logging.Logger,
    diagnostic_tracker: DiagnosticTracker,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare and validate inputs for orchestration.

    Args:
        prev_snapshot: Previous year's snapshot
        year: Current simulation year
        hazard_table: Hazard rates table
        logger: Logger instance
        diagnostic_tracker: Diagnostic tracking instance

    Returns:
        Tuple of (prepared_snapshot, hazard_slice)
    """
    logger.info(f"[PREPARE_INPUTS] Preparing inputs for year {year}")

    # Ensure snapshot has required columns
    snapshot = ensure_snapshot_cols(prev_snapshot)

    # Track initial state
    diagnostic_tracker.track_stage(snapshot, "Initial-Snapshot")

    # Validate and extract hazard slice
    hazard_slice = validate_and_extract_hazard_slice(hazard_table, year)

    # Filter to valid employee IDs
    snapshot = filter_valid_employee_ids(snapshot, logger)

    # Ensure simulation year column
    snapshot = ensure_simulation_year_column(snapshot, year)

    logger.info(f"Prepared snapshot with {len(snapshot)} employees for year {year}")

    return snapshot, hazard_slice


def _apply_contribution_calculations(
    snapshot: pd.DataFrame,
    year: int,
    global_params: Any,
    plan_rules: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply contribution calculations and plan rules to snapshot.

    Args:
        snapshot: Current employee snapshot
        year: Simulation year
        global_params: Global parameters
        plan_rules: Plan rules configuration
        logger: Logger instance

    Returns:
        Tuple of (contribution_events, updated_snapshot)
    """
    logger.info("[APPLY_CONTRIBUTIONS] Processing contribution calculations")

    # TODO: Replace with actual contribution calculation logic
    # This is a placeholder that maintains the original structure
    try:
        # Import and apply contribution calculation logic
        from cost_model.plan_rules.contributions import apply_contributions

        updated_snapshot, events = apply_contributions(
            snapshot=snapshot,
            simulation_year=year,
            global_params=global_params,
            plan_rules=plan_rules,
        )

        logger.info(f"Applied contribution calculations: {len(events)} events generated")
        return events, updated_snapshot

    except ImportError:
        logger.warning("Contribution calculation module not available, skipping")
        return pd.DataFrame(), snapshot.copy()
    except Exception as e:
        logger.error(f"Error in contribution calculations: {e}")
        return pd.DataFrame(), snapshot.copy()


def _generate_compensation_events(
    snapshot: pd.DataFrame, year_context: YearContext, logger: logging.Logger
) -> List[pd.DataFrame]:
    """
    Generate compensation-related events (merit raises and COLA).

    Args:
        snapshot: Current employee snapshot
        year_context: Year-specific context containing hazard_slice and other params
        logger: Logger instance

    Returns:
        List of compensation event DataFrames [merit_events, cola_events]
    """
    logger.info("[GENERATE_COMPENSATION] Processing merit raises and COLA events")

    try:
        # Import the bump function from comp engine
        from cost_model.engines.comp import bump

        # Generate both merit and COLA events using the bump function
        # bump returns [merit_events_df, cola_events_df]
        compensation_events = bump(
            snapshot=snapshot,
            hazard_slice=year_context.hazard_slice,
            as_of=pd.Timestamp(f"{year_context.year}-12-31"),
            rng=year_context.rng,
        )

        total_events = sum(len(df) for df in compensation_events if not df.empty)
        logger.info(f"Generated {total_events} total compensation events (merit + COLA)")
        
        # Log breakdown
        if len(compensation_events) >= 2:
            merit_count = len(compensation_events[0]) if not compensation_events[0].empty else 0
            cola_count = len(compensation_events[1]) if not compensation_events[1].empty else 0
            logger.info(f"  Merit events: {merit_count}")
            logger.info(f"  COLA events: {cola_count}")

        return compensation_events

    except ImportError as e:
        logger.warning(f"Compensation module not available: {e}")
        return [pd.DataFrame(), pd.DataFrame()]
    except Exception as e:
        logger.error(f"Error generating compensation events: {e}")
        return [pd.DataFrame(), pd.DataFrame()]


# Backward compatibility exports
__all__ = [
    "run_one_year",
    "n_active",
    "log_headcount_stage",
    "DiagnosticTracker",
    "EventConsolidationManager",
]
