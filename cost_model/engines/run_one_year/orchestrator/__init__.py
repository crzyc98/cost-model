# cost_model/engines/run_one_year/orchestrator/__init__.py
"""
Orchestrator package for run_one_year simulation engine.

This package provides a decomposed, modular implementation of the workforce simulation
orchestration logic, replacing the monolithic orchestrator.py file.

The main entry point is the run_one_year() function, which maintains the exact same
signature and behavior as the original implementation.
"""
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from logging_config import get_logger, get_diagnostic_logger

from cost_model.state.event_log import EVENT_COLS
from cost_model.state.schema import (
    EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_STATUS_EOY,
    ACTIVE_STATUS, EMP_CONTR, EMPLOYER_CORE, EMPLOYER_MATCH, IS_ELIGIBLE
)

# Import validation and utility functions
from ..validation import ensure_snapshot_cols, validate_and_extract_hazard_slice

# Import orchestrator components
from .base import YearContext, filter_valid_employee_ids, ensure_simulation_year_column
from .hiring import HiringOrchestrator
from .termination import TerminationOrchestrator
from .promotion import PromotionOrchestrator
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
    deterministic_term: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates simulation for a single year, following the new hiring/termination flow:
      1. Markov promotions/exits (experienced only)
      2. Hazard-based terminations (experienced only)
      3. Update snapshot to survivors
      4. Compute headcount targets (gross/net)
      5. Generate/apply hires
      6. Deterministic new-hire terminations
      7. Final snapshot + validation

    This function maintains the exact same signature and behavior as the original
    monolithic implementation, but uses the decomposed orchestrator components.

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
        **kwargs: Additional parameters (for compatibility)

    Returns:
        Tuple of (new_events, final_snapshot) where:
        - new_events: DataFrame of all events generated this year
        - final_snapshot: Final workforce snapshot at end of year
    """
    logger = get_logger(__name__)
    diag_logger = get_diagnostic_logger(__name__)

    logger.info(f"[RUN_ONE_YEAR] Simulating year {year}")

    # Initialize orchestrator components
    hiring_orchestrator = HiringOrchestrator(logger)
    termination_orchestrator = TerminationOrchestrator(logger)
    promotion_orchestrator = PromotionOrchestrator(logger)
    validator = SnapshotValidator(logger)

    # Validate and prepare inputs
    snapshot = _prepare_inputs(prev_snapshot, year, logger, diag_logger)
    hazard_slice = validate_and_extract_hazard_slice(hazard_table, year)

    # Create year context
    year_context = YearContext.create(
        year=year,
        hazard_slice=hazard_slice,
        global_params=global_params,
        plan_rules=plan_rules,
        rng=rng,
        rng_seed_offset=rng_seed_offset,
        census_template_path=census_template_path,
        deterministic_term=deterministic_term
    )

    # Track all events generated this year and the start count
    all_events = []
    start_count = len(snapshot)  # Count before any changes

    # Step 1: Compensation events (annual raises and COLA) - Apply FIRST so promotions use updated compensation
    compensation_events = _generate_compensation_events(snapshot, year_context, logger)
    all_events.extend(compensation_events)

    # Step 1b: Apply compensation events to snapshot
    snapshot = _apply_compensation_events_to_snapshot(snapshot, compensation_events, year_context, logger)

    # Step 2: Markov promotions/exits (experienced only) - Apply AFTER compensation so raises use updated values
    promotion_events, snapshot = promotion_orchestrator.get_events(snapshot, year_context)
    all_events.extend(promotion_events)
    validator.validate(snapshot, "markov_promotions", year_context)

    # Step 3: Hazard-based terminations (experienced only)
    termination_events, snapshot = termination_orchestrator.get_experienced_termination_events(
        snapshot, year_context
    )
    all_events.extend(termination_events)
    validator.validate(snapshot, "experienced_terminations", year_context)

    # Step 4: Hiring
    # Extract the first termination events DataFrame if available
    term_events_for_hiring = None
    if termination_events and len(termination_events) > 0 and not termination_events[0].empty:
        term_events_for_hiring = termination_events[0]

    hiring_events, snapshot = hiring_orchestrator.get_events(
        snapshot, year_context,
        terminated_events=term_events_for_hiring,
        start_count=start_count
    )
    all_events.extend(hiring_events)
    validator.validate(snapshot, "hiring", year_context)

    # Step 5: New hire terminations
    nh_termination_events, snapshot = termination_orchestrator.get_new_hire_termination_events(
        snapshot, year_context
    )
    all_events.extend(nh_termination_events)
    validator.validate(snapshot, "new_hire_terminations", year_context)

    # Step 6: Apply contribution calculations and eligibility
    snapshot = _apply_contribution_calculations(snapshot, year_context, logger)

    # Step 7: Final validation
    validator.validate_eoy(snapshot)

    # Step 8: Consolidate and return events
    new_events = _consolidate_events(all_events, event_log, year, logger)

    logger.info(
        f"[RUN_ONE_YEAR] Year {year} complete. Final headcount: {len(snapshot)}, "
        f"New events: {len(new_events)}"
    )

    return new_events, snapshot


def _prepare_inputs(
    prev_snapshot: pd.DataFrame,
    year: int,
    logger: logging.Logger,
    diag_logger: logging.Logger
) -> pd.DataFrame:
    """
    Prepare and validate input snapshot.

    Args:
        prev_snapshot: Input snapshot from previous year
        year: Current simulation year
        logger: Main logger
        diag_logger: Diagnostic logger

    Returns:
        Cleaned and validated snapshot
    """
    # Validate EMP_ID in prev_snapshot
    if EMP_ID in prev_snapshot.columns and prev_snapshot[EMP_ID].isna().any():
        na_count = prev_snapshot[EMP_ID].isna().sum()
        diag_logger.warning(
            f"Year {year}: Input snapshot (prev_snapshot) contains {na_count} "
            f"records with NA {EMP_ID}. These records will be dropped."
        )
        prev_snapshot = prev_snapshot.dropna(subset=[EMP_ID])

    # Ensure required columns
    snapshot = ensure_snapshot_cols(prev_snapshot)

    # Filter valid employee IDs
    snapshot = filter_valid_employee_ids(snapshot, logger)

    # Ensure simulation year
    snapshot = ensure_simulation_year_column(snapshot, year)

    logger.info(f"[PREP] Prepared snapshot with {len(snapshot)} employees for year {year}")

    return snapshot


def _apply_contribution_calculations(
    snapshot: pd.DataFrame,
    year_context: YearContext,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Apply contribution calculations and eligibility to the final snapshot.

    Args:
        snapshot: Current workforce snapshot
        year_context: Year-specific context
        logger: Logger instance

    Returns:
        Snapshot with contribution calculations applied
    """
    logger.info("[STEP] Apply contribution calculations and eligibility to final snapshot")

    # Ensure required columns exist
    snapshot_copy = snapshot.copy()

    # Set employee status for EOY
    if EMP_STATUS_EOY not in snapshot_copy.columns:
        snapshot_copy[EMP_STATUS_EOY] = ACTIVE_STATUS

    # Apply contribution calculations
    try:
        from cost_model.rules.contributions import apply as apply_contributions
        from cost_model.rules.validators import ContributionsRule, MatchRule, NonElectiveRule

        # Get contribution rules from plan_rules
        contrib_config = year_context.plan_rules.get('contributions', {})
        match_config = year_context.plan_rules.get('match', {})
        nec_config = year_context.plan_rules.get('non_elective', {})

        # Convert SimpleNamespace to dict if needed
        if hasattr(contrib_config, '__dict__'):
            contrib_config = contrib_config.__dict__
        if hasattr(match_config, '__dict__'):
            match_config = match_config.__dict__
        if hasattr(nec_config, '__dict__'):
            nec_config = nec_config.__dict__

        # Create rule objects
        contrib_rules = ContributionsRule(**contrib_config) if contrib_config else ContributionsRule(enabled=True)

        # Handle MatchRule with proper default tiers
        if match_config:
            match_rules = MatchRule(**match_config)
        else:
            # Create a default match rule with at least one tier to satisfy validation
            match_rules = MatchRule(
                tiers=[{'match_rate': 0.0, 'cap_deferral_pct': 0.0}],
                dollar_cap=None
            )

        nec_rules = NonElectiveRule(**nec_config) if nec_config else NonElectiveRule(rate=0.0)

        # Get IRS limits from plan_rules and convert to Pydantic models
        irs_limits_raw = year_context.plan_rules.get('irs_limits', {})
        irs_limits = {}

        logger.debug(f"Plan rules keys: {list(year_context.plan_rules.keys())}")
        logger.debug(f"Raw IRS limits type: {type(irs_limits_raw)}")

        if irs_limits_raw:
            from cost_model.config.models import IRSYearLimits
            from types import SimpleNamespace

            # Handle both dict and SimpleNamespace objects
            if isinstance(irs_limits_raw, dict):
                items = irs_limits_raw.items()
            elif isinstance(irs_limits_raw, SimpleNamespace):
                items = vars(irs_limits_raw).items()
            else:
                logger.warning(f"Unexpected IRS limits type: {type(irs_limits_raw)}. Expected dict or SimpleNamespace.")
                items = []

            for year, limits_obj in items:
                try:
                    # Ensure year is an integer (contributions.apply expects integer keys)
                    year_int = int(year) if not isinstance(year, int) else year

                    # Convert to dictionary first if it's a SimpleNamespace
                    if isinstance(limits_obj, SimpleNamespace):
                        limits_dict = vars(limits_obj)
                    elif isinstance(limits_obj, dict):
                        limits_dict = limits_obj
                    elif hasattr(limits_obj, 'compensation_limit'):
                        # Already an IRSYearLimits object
                        irs_limits[year_int] = limits_obj
                        logger.debug(f"Using existing IRSYearLimits object for year {year_int}")
                        continue
                    else:
                        logger.warning(f"Invalid IRS limits format for year {year}: {type(limits_obj)}. Skipping this year.")
                        continue

                    # Convert dictionary to IRSYearLimits model
                    irs_limits[year_int] = IRSYearLimits(**limits_dict)
                    logger.debug(f"Successfully converted IRS limits for year {year_int}: {limits_dict}")

                except Exception as e:
                    logger.warning(f"Failed to convert IRS limits for year {year}: {e}. Skipping this year.")

        if not irs_limits:
            logger.warning(f"IRS limits missing or could not be converted for {year_context.year}. Contribution calculations might fail.")

        # Apply contributions
        snapshot_copy = apply_contributions(
            df=snapshot_copy,
            contrib_rules=contrib_rules,
            match_rules=match_rules,
            nec_rules=nec_rules,
            irs_limits=irs_limits,
            simulation_year=year_context.year,
            year_start=year_context.as_of,
            year_end=year_context.end_of_year
        )

        logger.info(f"Applied contribution calculations to {len(snapshot_copy)} employees")

    except Exception as e:
        logger.warning(f"Error applying contribution calculations: {e}")
        # Ensure contribution columns exist with defaults
        for col in [EMP_CONTR, EMPLOYER_CORE, EMPLOYER_MATCH]:
            if col not in snapshot_copy.columns:
                snapshot_copy[col] = 0.0

    # Apply eligibility calculations
    try:
        from cost_model.utils.date_utils import calculate_age

        # Get eligibility parameters
        min_age = getattr(year_context.global_params, 'min_eligibility_age', 21)
        min_service_months = getattr(year_context.global_params, 'min_service_months', 12)

        # Initialize eligibility column
        if IS_ELIGIBLE not in snapshot_copy.columns:
            snapshot_copy[IS_ELIGIBLE] = False

        # Calculate eligibility for each employee
        for idx, row in snapshot_copy.iterrows():
            birth_date = row.get(EMP_BIRTH_DATE)
            hire_date = row.get(EMP_HIRE_DATE)

            if pd.notna(birth_date) and pd.notna(hire_date):
                # Convert to datetime if needed
                if not isinstance(birth_date, pd.Timestamp):
                    birth_date = pd.to_datetime(birth_date)
                if not isinstance(hire_date, pd.Timestamp):
                    hire_date = pd.to_datetime(hire_date)

                # Calculate age at end of year
                age = calculate_age(birth_date, year_context.end_of_year)

                # Calculate service in months
                service_months = (
                    (year_context.end_of_year.year - hire_date.year) * 12 +
                    (year_context.end_of_year.month - hire_date.month)
                )

                # Check eligibility criteria
                if age >= min_age and service_months >= min_service_months:
                    snapshot_copy.loc[idx, IS_ELIGIBLE] = True

        eligible_count = snapshot_copy[IS_ELIGIBLE].sum()
        logger.info(f"Determined eligibility for {len(snapshot_copy)} employees: {eligible_count} eligible")

    except Exception as e:
        logger.warning(f"Error applying eligibility calculations: {e}")
        if IS_ELIGIBLE not in snapshot_copy.columns:
            snapshot_copy[IS_ELIGIBLE] = False

    return snapshot_copy


def _apply_compensation_events_to_snapshot(
    snapshot: pd.DataFrame,
    compensation_events: List[pd.DataFrame],
    year_context: YearContext,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Apply compensation events to update employee compensation in the snapshot.

    Args:
        snapshot: Current workforce snapshot
        compensation_events: List of compensation event DataFrames
        year_context: Year-specific context and parameters
        logger: Logger instance

    Returns:
        Updated snapshot with compensation changes applied
    """
    logger.info("[STEP] Applying compensation events to snapshot")

    # Collect all compensation events into a single DataFrame
    events_to_apply = []
    for event_df in compensation_events:
        if isinstance(event_df, pd.DataFrame) and not event_df.empty:
            events_to_apply.append(event_df)

    if not events_to_apply:
        logger.info("[COMP] No compensation events to apply")
        return snapshot

    # Concatenate all events
    all_comp_events = pd.concat(events_to_apply, ignore_index=True)

    # Apply events using the snapshot update mechanism
    from cost_model.state.snapshot_update import update
    updated_snapshot = update(
        prev_snapshot=snapshot,
        new_events=all_comp_events,
        snapshot_year=year_context.year
    )

    # Count how many employees had compensation updated
    comp_events = all_comp_events[all_comp_events['event_type'] == 'EVT_COMP']
    cola_events = all_comp_events[all_comp_events['event_type'] == 'EVT_COLA']

    comp_count = len(comp_events) if not comp_events.empty else 0
    cola_count = len(cola_events) if not cola_events.empty else 0

    logger.info(f"[COMP] Applied {comp_count} compensation updates and {cola_count} COLA updates to snapshot")

    return updated_snapshot


def _apply_compensation_events_to_snapshot(
    snapshot: pd.DataFrame,
    compensation_events: List[pd.DataFrame],
    year_context: YearContext,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Apply compensation events to update employee compensation in the snapshot.

    Args:
        snapshot: Current workforce snapshot
        compensation_events: List of compensation event DataFrames
        year_context: Year-specific context and parameters
        logger: Logger instance

    Returns:
        Updated snapshot with compensation changes applied
    """
    logger.info("[STEP] Applying compensation events to snapshot")

    # Collect all compensation events into a single DataFrame
    events_to_apply = []
    for event_df in compensation_events:
        if isinstance(event_df, pd.DataFrame) and not event_df.empty:
            events_to_apply.append(event_df)

    if not events_to_apply:
        logger.info("[COMP] No compensation events to apply")
        return snapshot

    # Concatenate all events
    all_comp_events = pd.concat(events_to_apply, ignore_index=True)

    # Apply events using the snapshot update mechanism
    from cost_model.state.snapshot_update import update
    updated_snapshot = update(
        prev_snapshot=snapshot,
        new_events=all_comp_events,
        snapshot_year=year_context.year
    )

    # Count how many employees had compensation updated
    comp_events = all_comp_events[all_comp_events['event_type'] == 'EVT_COMP']
    cola_events = all_comp_events[all_comp_events['event_type'] == 'EVT_COLA']

    comp_count = len(comp_events) if not comp_events.empty else 0
    cola_count = len(cola_events) if not cola_events.empty else 0

    logger.info(f"[COMP] Applied {comp_count} compensation updates and {cola_count} COLA updates to snapshot")

    return updated_snapshot


def _generate_compensation_events(
    snapshot: pd.DataFrame,
    year_context: YearContext,
    logger: logging.Logger
) -> List[pd.DataFrame]:
    """
    Generate comprehensive compensation events (annual raises and COLA) for active employees.

    This function generates both EVT_COMP (standard raises) and EVT_COLA (cost of living adjustments) events.

    Args:
        snapshot: Current workforce snapshot
        year_context: Year-specific context and parameters
        logger: Logger instance

    Returns:
        List containing the event DataFrames for both raise and COLA events
    """
    logger.info("[STEP] Generating compensation events (annual raises and COLA)")
    import traceback
    all_events = []

    try:
        # Import the compensation function
        from cost_model.engines.comp import bump

        # Generate both standard compensation events (raises) and COLA events
        # The bump() function now handles both EVT_COMP and EVT_COLA events internally
        try:
            comp_events = bump(
                snapshot=snapshot,
                hazard_slice=year_context.hazard_slice,
                as_of=year_context.as_of,
                rng=year_context.year_rng
            )
            all_events.extend(comp_events)

            # Count events by type for logging
            total_comp_events = 0
            total_cola_events = 0
            for df in comp_events:
                if not df.empty and 'event_type' in df.columns:
                    comp_count = (df['event_type'] == 'EVT_COMP').sum()
                    cola_count = (df['event_type'] == 'EVT_COLA').sum()
                    total_comp_events += comp_count
                    total_cola_events += cola_count

            logger.info(f"[COMP] Generated {total_comp_events} standard compensation events and {total_cola_events} COLA events")

        except KeyError as e:
            if 'cola_pct' in str(e):
                logger.warning("[COMP] No COLA events generated: 'cola_pct' not found in hazard slice")
            else:
                logger.error(f"[COMP] Error in compensation generation: {e}")
                logger.error(f"[COMP] Hazard columns: {year_context.hazard_slice.columns.tolist()}")
                logger.error(f"[COMP] Traceback: {traceback.format_exc()}")
        except Exception as e:
            logger.error(f"[COMP] Error in compensation generation: {e}")
            logger.error(f"[COMP] Traceback: {traceback.format_exc()}")

    except Exception as e:
        logger.error(f"[COMP] Unexpected error in compensation event generation: {e}")
        logger.error(f"[COMP] Traceback: {traceback.format_exc()}")

    # Log summary of all generated events
    total_events = sum(len(df) for df in all_events if not df.empty)
    event_types = []
    for df in all_events:
        if not df.empty and 'event_type' in df.columns:
            event_types.extend(df['event_type'].unique())

    logger.info(f"[COMP] Total compensation events for year {year_context.year}: {total_events}")
    logger.info(f"[COMP] Event types generated: {sorted(list(set(event_types)))}")

    return all_events


def _consolidate_events(
    all_events: List[pd.DataFrame],
    event_log: pd.DataFrame,
    year: int,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Consolidate all events generated during the year into a single DataFrame.

    Args:
        all_events: List of event DataFrames from each orchestrator
        event_log: Existing cumulative event log
        year: Current simulation year
        logger: Logger instance

    Returns:
        Consolidated DataFrame of all new events for this year
    """
    logger.info("[STEP] Consolidating events")

    # Collect all non-empty DataFrames
    events_to_concat = []
    for event_df in all_events:
        # Ensure we only process DataFrame objects
        if isinstance(event_df, pd.DataFrame) and not event_df.empty:
            events_to_concat.append(event_df)

    # Filter out empty DataFrames before concatenation to avoid FutureWarning
    non_empty_events = [df for df in events_to_concat if not df.empty]

    if not non_empty_events:
        logger.info("No events generated this year")
        return pd.DataFrame(columns=EVENT_COLS)

    # Validate events before concatenation
    validated_events = []
    required_cols = ['event_time', 'event_type', EMP_ID]

    for i, df in enumerate(non_empty_events):
        # Check for required columns
        is_valid = True
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"DataFrame {i} is missing required column '{col}'. Skipping.")
                is_valid = False
                break

        if not is_valid:
            continue

        # Check for NAs in required columns
        for col in required_cols:
            if df[col].isna().any():
                na_count = df[col].isna().sum()
                logger.warning(f"DataFrame {i} has {na_count} NA values in required column '{col}'. Fixing.")

                # For event_time, fill with year timestamp
                if col == 'event_time' and na_count > 0:
                    df = df.copy()
                    df.loc[:, col] = df[col].fillna(pd.Timestamp(f"{year}-01-01"))

                # For event_type and employee_id, drop rows with NA values
                if (col == 'event_type' or col == EMP_ID) and na_count > 0:
                    df = df.dropna(subset=[col])

        # Only add valid, non-empty DataFrames
        if not df.empty:
            validated_events.append(df)

    # Concatenate validated events
    if not validated_events:
        logger.info("No valid events after validation")
        return pd.DataFrame(columns=EVENT_COLS)

    try:
        new_events = pd.concat(validated_events, ignore_index=True)
    except Exception as e:
        logger.error(f"Error concatenating events: {e}")
        return pd.DataFrame(columns=EVENT_COLS)

    # Ensure event_id is unique and present
    if 'event_id' in new_events.columns:
        if new_events['event_id'].isna().any():
            logger.warning("Generating missing event_ids for new events")
            mask = new_events['event_id'].isna()
            new_events.loc[mask, 'event_id'] = [str(uuid.uuid4()) for _ in range(mask.sum())]
    else:
        logger.warning("event_id column missing in new events, generating new event_ids")
        new_events.loc[:, 'event_id'] = [str(uuid.uuid4()) for _ in range(len(new_events))]

    # Ensure simulation_year is set
    new_events = ensure_simulation_year_column(new_events, year)

    # Sort events by timestamp
    if 'event_time' in new_events.columns and not new_events.empty:
        new_events.loc[:, 'event_time'] = pd.to_datetime(new_events['event_time'], errors='coerce')
        new_events = new_events.sort_values('event_time', ignore_index=True)

    # Combine with existing event log - filter empty DataFrames to avoid FutureWarning
    dfs_to_concat = [df for df in [event_log, new_events] if not df.empty]
    if dfs_to_concat:
        cumulative_events = pd.concat(dfs_to_concat, ignore_index=True)
    else:
        # Both DataFrames are empty, return empty DataFrame with correct columns
        cumulative_events = pd.DataFrame(columns=EVENT_COLS)

    logger.info(f"Consolidated {len(new_events)} new events for year {year}")

    return cumulative_events


# Re-export the main function for backward compatibility
__all__ = ['run_one_year']
