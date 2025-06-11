"""
Yearly snapshot processing functionality.

Handles the complex logic for building enhanced yearly snapshots that include
all employees active during a specific year, including terminated employees.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .event_processor import EventProcessor
from .exceptions import SnapshotBuildError
from .logging_utils import get_snapshot_logger, progress_context, timing_decorator
from .models import SnapshotConfig
from .transformers import SnapshotTransformer
from .types import (
    CompensationAmount,
    CompensationExtractionResult,
    EmployeeId,
    EmployeeSet,
    Logger,
    SimulationYear,
)
from .validators import SnapshotValidator
from cost_model.state.event_log import EVT_HIRE, EVT_TERM, EVT_NEW_HIRE_TERM

logger = get_snapshot_logger(__name__)


class YearlySnapshotProcessor:
    """Handles building enhanced yearly snapshots.

    This class manages the complex process of building yearly snapshots that include
    all employees who were active at any point during a simulation year, including
    those who were terminated.
    """

    def __init__(self, config: Optional[SnapshotConfig] = None) -> None:
        """Initialize the yearly snapshot processor.

        Args:
            config: Optional snapshot configuration. If None, uses default.
        """
        self.config = config or SnapshotConfig(start_year=2025)
        self.event_processor = EventProcessor()
        self.transformer = SnapshotTransformer(self.config)
        self.validator = SnapshotValidator(self.config)

    @timing_decorator(logger)
    def identify_employees_active_during_year(
        self,
        start_of_year_snapshot: pd.DataFrame,
        year_events: pd.DataFrame,
        simulation_year: SimulationYear,
    ) -> EmployeeSet:
        """
        Identify all employees who were active at any point during the year.

        Args:
            start_of_year_snapshot: Snapshot at beginning of year
            year_events: Events during the year
            simulation_year: Year being processed

        Returns:
            Set of employee IDs active during the year
        """
        logger.debug(f"Identifying employees active during {simulation_year}")

        # Use original schema column names for compatibility
        EMP_ID = "employee_id"
        EMP_ACTIVE = "active"

        # Step 1: Get employees active at start of year
        soy_active_employees = (
            start_of_year_snapshot[start_of_year_snapshot[EMP_ACTIVE] == True].copy()
            if EMP_ACTIVE in start_of_year_snapshot.columns
            else start_of_year_snapshot.copy()
        )

        soy_active_ids = (
            set(soy_active_employees[EMP_ID].unique())
            if EMP_ID in soy_active_employees.columns
            else set()
        )
        logger.debug(f"Employees active at start of year: {len(soy_active_ids)}")

        # Step 2: Get employees hired during the year
        hired_this_year = set()
        if (
            year_events is not None
            and not year_events.empty
            and "event_type" in year_events.columns
        ):
            hire_events = year_events[year_events["event_type"] == EVT_HIRE]
            if not hire_events.empty and EMP_ID in hire_events.columns:
                hired_this_year = set(hire_events[EMP_ID].unique())

        logger.debug(f"Employees hired during year: {len(hired_this_year)}")

        # Step 3: Get employees terminated during the year
        terminated_this_year = set()
        if year_events is not None and not year_events.empty:
            term_events = year_events[year_events["event_type"].isin([EVT_TERM, EVT_NEW_HIRE_TERM])]
            if not term_events.empty and EMP_ID in term_events.columns:
                terminated_this_year = set(term_events[EMP_ID].unique())

        logger.debug(f"Employees terminated during year: {len(terminated_this_year)}")

        # Step 4: Union all sets to get employees active at any point
        active_during_year_ids = soy_active_ids.union(hired_this_year).union(terminated_this_year)

        logger.info(
            f"Total employees active during {simulation_year}: {len(active_during_year_ids)}"
        )

        return active_during_year_ids

    def build_base_yearly_snapshot(
        self, end_of_year_snapshot: pd.DataFrame, active_during_year_ids: EmployeeSet
    ) -> pd.DataFrame:
        """
        Build the base yearly snapshot from end-of-year data.

        Args:
            end_of_year_snapshot: Snapshot at end of year
            active_during_year_ids: IDs of employees active during year

        Returns:
            Base yearly snapshot DataFrame
        """
        logger.debug("Building base yearly snapshot from EOY data")

        EMP_ID = "employee_id"

        # Filter EOY snapshot to include only employees active during the year
        if EMP_ID in end_of_year_snapshot.columns:
            eoy_employees = end_of_year_snapshot[
                end_of_year_snapshot[EMP_ID].isin(active_during_year_ids)
            ].copy()
        else:
            logger.warning("No employee_id column in EOY snapshot")
            eoy_employees = end_of_year_snapshot.copy()

        logger.debug(f"Base yearly snapshot contains {len(eoy_employees)} employees from EOY")

        return eoy_employees

    def identify_missing_employees(
        self, base_snapshot: pd.DataFrame, active_during_year_ids: EmployeeSet
    ) -> EmployeeSet:
        """
        Identify employees missing from the base snapshot.

        Args:
            base_snapshot: Base yearly snapshot
            active_during_year_ids: All employees who should be included

        Returns:
            Set of employee IDs missing from base snapshot
        """
        EMP_ID = "employee_id"

        if EMP_ID not in base_snapshot.columns:
            logger.warning("Cannot identify missing employees - no employee_id column")
            return set()

        base_snapshot_ids = set(base_snapshot[EMP_ID].unique())
        missing_ids = active_during_year_ids - base_snapshot_ids

        logger.debug(f"Missing employees from base snapshot: {len(missing_ids)}")

        return missing_ids

    def reconstruct_missing_employees(
        self,
        missing_ids: EmployeeSet,
        start_of_year_snapshot: pd.DataFrame,
        year_events: pd.DataFrame,
        simulation_year: SimulationYear,
    ) -> pd.DataFrame:
        """
        Reconstruct data for employees missing from EOY snapshot.

        Args:
            missing_ids: Employee IDs to reconstruct
            start_of_year_snapshot: SOY snapshot for terminated employees
            year_events: Events during the year
            simulation_year: Year being processed

        Returns:
            DataFrame with reconstructed employee data
        """
        if not missing_ids:
            logger.debug("No missing employees to reconstruct")
            return pd.DataFrame()

        logger.debug(f"Reconstructing {len(missing_ids)} missing employees")

        EMP_ID = "employee_id"
        missing_employees = []

        # Step 1: Handle terminated employees from SOY snapshot
        if EMP_ID in start_of_year_snapshot.columns:
            soy_terminated = start_of_year_snapshot[
                start_of_year_snapshot[EMP_ID].isin(missing_ids)
            ].copy()

            if not soy_terminated.empty:
                # Enrich SOY rows with termination metadata from events
                try:
                    from cost_model.state.event_log import EVT_TERM
                    # Get termination events for these employees in the current year
                    term_events = year_events[
                        (year_events["event_type"] == EVT_TERM)
                        & (year_events[EMP_ID].isin(soy_terminated[EMP_ID]))
                    ][[EMP_ID, "event_time"]].copy()
                    term_events = term_events.drop_duplicates(subset=[EMP_ID])
                    term_events.rename(columns={"event_time": "employee_termination_date"}, inplace=True)
                    # Drop stale termination-date column to avoid _x/_y suffix mess
                    if "employee_termination_date" in soy_terminated.columns:
                        soy_terminated = soy_terminated.drop(columns=["employee_termination_date"])
                    # Merge termination date into snapshot rows
                    soy_terminated = soy_terminated.merge(term_events, on=EMP_ID, how="left")
                    # After merge, ensure proper column naming if suffixes occurred
                    # (happens if previous column slipped through)
                    for col_candidate in [
                        "employee_termination_date_y",
                        "employee_termination_date_x",
                    ]:
                        if col_candidate in soy_terminated.columns:
                            soy_terminated["employee_termination_date"] = soy_terminated[
                                col_candidate
                            ].where(
                                soy_terminated["employee_termination_date"].isna(),
                                soy_terminated["employee_termination_date"],
                            )
                            soy_terminated = soy_terminated.drop(columns=[col_candidate])
                    # Normalize to date (remove time) for cleaner snapshot column
                    soy_terminated["employee_termination_date"] = pd.to_datetime(
                        soy_terminated["employee_termination_date"], errors="coerce"
                    ).dt.date
                except Exception as e:
                    logger.warning(
                        f"Failed to merge termination dates for experienced terms: {e}")

                # Mark status flags for terminated employees reconstructed from SOY
                soy_terminated["active"] = False
                soy_terminated["exited"] = True
                soy_terminated["employee_status_eoy"] = "TERMINATED"

                missing_employees.append(soy_terminated)
                reconstructed_from_soy = set(soy_terminated[EMP_ID].unique())
                logger.debug(
                    f"Reconstructed {len(reconstructed_from_soy)} employees from SOY snapshot"
                )
                missing_ids = missing_ids - reconstructed_from_soy

        # Step 2: Reconstruct terminated new hires from events
        if missing_ids and year_events is not None and not year_events.empty:
            new_hire_terminated = self._reconstruct_terminated_new_hires(
                missing_ids, year_events, simulation_year
            )
            if not new_hire_terminated.empty:
                missing_employees.append(new_hire_terminated)
                logger.debug(
                    f"Reconstructed {len(new_hire_terminated)} new hire terminations from events"
                )

        # Combine all reconstructed employees
        if missing_employees:
            result = pd.concat(missing_employees, ignore_index=True)
            logger.info(f"Successfully reconstructed {len(result)} missing employees")
            return result
        else:
            logger.warning(f"Could not reconstruct {len(missing_ids)} missing employees")
            return pd.DataFrame()

    def _reconstruct_terminated_new_hires(
        self, missing_ids: EmployeeSet, year_events: pd.DataFrame, simulation_year: SimulationYear
    ) -> pd.DataFrame:
        """
        Reconstruct data for new hires who were terminated during the year.

        This is the most complex part of the yearly snapshot processing,
        as these employees don't appear in either SOY or EOY snapshots.
        """
        logger.debug("Reconstructing terminated new hires from events")

        EMP_ID = "employee_id"

        # Filter for new hire termination events
        nht_events = year_events[
            (year_events["event_type"] == EVT_NEW_HIRE_TERM)
            & (year_events[EMP_ID].isin(missing_ids))
        ]

        if nht_events.empty:
            logger.debug("No new hire termination events found for missing employees")
            return pd.DataFrame()

        logger.debug(f"Processing {len(nht_events)} new hire termination events")

        reconstructed_employees = []

        for _, nht_event in nht_events.iterrows():
            try:
                employee_data = self._reconstruct_single_terminated_new_hire(
                    nht_event, year_events, simulation_year
                )
                if employee_data is not None:
                    reconstructed_employees.append(employee_data)
            except Exception as e:
                logger.warning(f"Failed to reconstruct employee {nht_event.get(EMP_ID)}: {e}")

        if reconstructed_employees:
            result = pd.DataFrame(reconstructed_employees)
            logger.debug(f"Successfully reconstructed {len(result)} terminated new hires")
            return result
        else:
            return pd.DataFrame()

    def _reconstruct_single_terminated_new_hire(
        self,
        termination_event: pd.Series,
        year_events: pd.DataFrame,
        simulation_year: SimulationYear,
    ) -> Optional[Dict[str, Any]]:
        """
        Reconstruct a single terminated new hire from events.

        Args:
            termination_event: The termination event for this employee
            year_events: All events during the year
            simulation_year: Year being processed

        Returns:
            Dictionary with employee data or None if reconstruction fails
        """
        EMP_ID = "employee_id"
        emp_id = termination_event.get(EMP_ID)

        if pd.isna(emp_id):
            logger.warning("Termination event missing employee ID")
            return None

        logger.debug(f"Reconstructing terminated new hire: {emp_id}")

        # Extract compensation using existing logic
        compensation_result = self.event_processor.extract_compensation_for_employee(
            emp_id, year_events
        )

        # Get hire date from events
        hire_date = self._extract_hire_date_from_events(emp_id, year_events)
        term_date = termination_event.get("event_date")

        # Build employee record
        employee_data = {
            EMP_ID: emp_id,
            "employee_hire_date": hire_date,
            "employee_termination_date": term_date,
            "employee_gross_compensation": compensation_result.compensation,
            "active": False,
            "exited": True,
            "employee_status_eoy": "TERMINATED",
            "simulation_year": simulation_year,
            "employee_deferral_rate": 0.0,  # Default for new hires
            "employee_contribution": 0.0,
            "employer_core_contribution": 0.0,
            "employer_match_contribution": 0.0,
            "is_eligible": False,  # New hires typically not eligible immediately
        }

        # Calculate tenure for terminated new hire
        if hire_date and term_date:
            try:
                hire_dt = pd.to_datetime(hire_date)
                term_dt = pd.to_datetime(term_date)
                tenure_days = (term_dt - hire_dt).days
                employee_data["employee_tenure"] = max(0, tenure_days / 365.25)
            except Exception as e:
                logger.warning(f"Could not calculate tenure for {emp_id}: {e}")
                employee_data["employee_tenure"] = 0.0
        else:
            employee_data["employee_tenure"] = 0.0

        # Set tenure band
        tenure_years = employee_data["employee_tenure"]
        if tenure_years < 1:
            employee_data["employee_tenure_band"] = "NEW_HIRE"
        elif tenure_years < 5:
            employee_data["employee_tenure_band"] = "EARLY_CAREER"
        else:
            employee_data["employee_tenure_band"] = "MID_CAREER"

        logger.debug(
            f"Reconstructed new hire termination for {emp_id}: "
            f"tenure={tenure_years:.2f}y, comp=${compensation_result.compensation:,.0f}"
        )

        return employee_data

    def _extract_hire_date_from_events(
        self, emp_id: EmployeeId, year_events: pd.DataFrame
    ) -> Optional[str]:
        """Extract hire date for an employee from events."""
        EMP_ID = "employee_id"

        hire_events = year_events[
            (year_events["event_type"] == EVT_HIRE) & (year_events[EMP_ID] == emp_id)
        ]

        if not hire_events.empty:
            # Get the most recent hire event
            hire_event = hire_events.iloc[-1]
            return hire_event.get("event_date")

        return None
