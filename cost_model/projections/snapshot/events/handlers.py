"""
Concrete event handlers for different event types.

This module provides specific implementations for handling different
types of workforce events (hiring, termination, promotion, compensation).
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from cost_model.schema import EventColumns, EventTypes, SnapshotColumns

from .base import BaseEventProcessor, EventContext, EventProcessingError

logger = logging.getLogger(__name__)


class HireEventHandler(BaseEventProcessor):
    """Handler for hiring events."""

    def __init__(self):
        super().__init__([EventTypes.HIRE])

    def process_events(
        self, snapshot: pd.DataFrame, events: pd.DataFrame, context: EventContext
    ) -> pd.DataFrame:
        """Process hiring events."""
        if events.empty:
            return snapshot

        start_time = pd.Timestamp.now()
        self.logger.debug(f"Processing {len(events)} hire events")

        try:
            # Validate events first
            validation_errors = self.validate_events(events, context)
            if validation_errors:
                raise EventProcessingError(
                    f"Hire event validation failed: {validation_errors}", event_type=EventTypes.HIRE
                )

            # Create new employee records for hires
            new_employees = self._create_new_employee_records(events, context)

            # Add new employees to snapshot
            updated_snapshot = pd.concat([snapshot, new_employees], ignore_index=True)

            # Update statistics
            processing_time = (pd.Timestamp.now() - start_time).total_seconds()
            self.stats.processing_time_seconds += processing_time
            self.stats.add_processed_event(success=True, rows_affected=len(new_employees))

            self._log_processing_summary(EventTypes.HIRE, len(events), processing_time)
            return updated_snapshot

        except Exception as e:
            self.logger.error(f"Error processing hire events: {str(e)}")
            self.stats.add_processed_event(success=False)
            raise EventProcessingError(f"Failed to process hire events: {str(e)}", EventTypes.HIRE)

    def validate_events(self, events: pd.DataFrame, context: EventContext) -> List[str]:
        """Validate hire events."""
        errors = self._validate_common_fields(events, context)

        # Hire-specific validation
        required_fields = [EventColumns.GROSS_COMPENSATION]
        for field in required_fields:
            if field not in events.columns:
                errors.append(f"Hire events missing required field: {field}")
            elif events[field].isna().any():
                null_count = events[field].isna().sum()
                errors.append(f"Found {null_count} hire events with null {field}")

        # Validate compensation is positive
        if EventColumns.GROSS_COMPENSATION in events.columns:
            zero_comp = events[EventColumns.GROSS_COMPENSATION] <= 0
            if zero_comp.any():
                errors.append(
                    f"Found {zero_comp.sum()} hire events with zero or negative compensation"
                )

        return errors

    def _create_new_employee_records(
        self, hire_events: pd.DataFrame, context: EventContext
    ) -> pd.DataFrame:
        """Create new employee records from hire events."""
        new_employees = pd.DataFrame()

        # Map event fields to snapshot fields
        field_mapping = {
            EventColumns.EMP_ID: SnapshotColumns.EMP_ID,
            EventColumns.EVENT_DATE: SnapshotColumns.EMP_HIRE_DATE,
            EventColumns.GROSS_COMPENSATION: SnapshotColumns.EMP_GROSS_COMP,
            EventColumns.JOB_LEVEL: SnapshotColumns.EMP_LEVEL,
            EventColumns.DEFERRAL_RATE: SnapshotColumns.EMP_DEFERRAL_RATE,
        }

        # Copy mapped fields
        for event_field, snapshot_field in field_mapping.items():
            if event_field in hire_events.columns:
                new_employees[snapshot_field] = hire_events[event_field]

        # Set default values for new hires
        new_employees[SnapshotColumns.EMP_ACTIVE] = True
        new_employees[SnapshotColumns.SIMULATION_YEAR] = context.simulation_year
        new_employees[SnapshotColumns.EMP_TENURE] = 0.0
        new_employees[SnapshotColumns.EMP_TENURE_BAND] = "NEW_HIRE"
        new_employees[SnapshotColumns.EMP_LEVEL_SOURCE] = "HIRE_EVENT"

        # Set default deferral rate if not provided
        if SnapshotColumns.EMP_DEFERRAL_RATE not in new_employees.columns:
            new_employees[SnapshotColumns.EMP_DEFERRAL_RATE] = 0.0

        # Set default job level if not provided
        if SnapshotColumns.EMP_LEVEL not in new_employees.columns:
            new_employees[SnapshotColumns.EMP_LEVEL] = "BAND_1"

        # Calculate initial contribution amounts
        new_employees = self._calculate_initial_contributions(new_employees)

        self.logger.debug(f"Created {len(new_employees)} new employee records")
        return new_employees

    def _calculate_initial_contributions(self, new_employees: pd.DataFrame) -> pd.DataFrame:
        """Calculate initial contribution amounts for new hires."""
        # Employee contributions
        new_employees[SnapshotColumns.EMP_CONTRIBUTION] = (
            new_employees[SnapshotColumns.EMP_GROSS_COMP]
            * new_employees[SnapshotColumns.EMP_DEFERRAL_RATE]
        )

        # Employer match (simple 50% up to 6% of compensation)
        max_match_base = new_employees[SnapshotColumns.EMP_GROSS_COMP] * 0.06
        employee_contrib_eligible = new_employees[SnapshotColumns.EMP_CONTRIBUTION].clip(
            upper=max_match_base
        )
        new_employees[SnapshotColumns.EMPLOYER_MATCH_CONTRIBUTION] = employee_contrib_eligible * 0.5

        # Default employer core contribution
        new_employees[SnapshotColumns.EMPLOYER_CORE_CONTRIBUTION] = 0.0

        return new_employees


class TerminationEventHandler(BaseEventProcessor):
    """Handler for termination events."""

    def __init__(self):
        super().__init__([EventTypes.TERMINATION, EventTypes.NEW_HIRE_TERMINATION])

    def process_events(
        self, snapshot: pd.DataFrame, events: pd.DataFrame, context: EventContext
    ) -> pd.DataFrame:
        """Process termination events."""
        if events.empty:
            return snapshot

        start_time = pd.Timestamp.now()
        self.logger.debug(f"Processing {len(events)} termination events")

        try:
            # Validate events first
            validation_errors = self.validate_events(events, context)
            if validation_errors:
                raise EventProcessingError(
                    f"Termination event validation failed: {validation_errors}",
                    event_type="TERMINATION",
                )

            # Get employee mask for terminations
            employee_ids = events[EventColumns.EMP_ID].tolist()
            employee_mask = self._get_employee_mask(snapshot, employee_ids)

            # Apply termination updates
            updates = {
                SnapshotColumns.EMP_ACTIVE: False,
                SnapshotColumns.EMP_TERM_DATE: events[EventColumns.EVENT_DATE].tolist(),
                SnapshotColumns.EMP_EXITED: True,
                SnapshotColumns.EMP_STATUS_EOY: "TERMINATED",
            }

            # Add termination reason if available
            if EventColumns.TERMINATION_REASON in events.columns:
                updates["termination_reason"] = events[EventColumns.TERMINATION_REASON].tolist()

            updated_snapshot = self._update_snapshot_safely(
                snapshot, updates, employee_mask, "termination"
            )

            # Update statistics
            processing_time = (pd.Timestamp.now() - start_time).total_seconds()
            self.stats.processing_time_seconds += processing_time

            self._log_processing_summary("TERMINATION", len(events), processing_time)
            return updated_snapshot

        except Exception as e:
            self.logger.error(f"Error processing termination events: {str(e)}")
            self.stats.add_processed_event(success=False)
            raise EventProcessingError(
                f"Failed to process termination events: {str(e)}", "TERMINATION"
            )

    def validate_events(self, events: pd.DataFrame, context: EventContext) -> List[str]:
        """Validate termination events."""
        errors = self._validate_common_fields(events, context)

        # Termination-specific validation
        # Check that termination dates are not too far in the future
        if EventColumns.EVENT_DATE in events.columns:
            term_dates = pd.to_datetime(events[EventColumns.EVENT_DATE])
            future_terms = term_dates > context.reference_date
            if future_terms.any():
                self.stats.add_warning()
                self.logger.warning(
                    f"Found {future_terms.sum()} termination events with future dates"
                )

        return errors


class PromotionEventHandler(BaseEventProcessor):
    """Handler for promotion events."""

    def __init__(self):
        super().__init__([EventTypes.PROMOTION])

    def process_events(
        self, snapshot: pd.DataFrame, events: pd.DataFrame, context: EventContext
    ) -> pd.DataFrame:
        """Process promotion events."""
        if events.empty:
            return snapshot

        start_time = pd.Timestamp.now()
        self.logger.debug(f"Processing {len(events)} promotion events")

        try:
            # Validate events first
            validation_errors = self.validate_events(events, context)
            if validation_errors:
                raise EventProcessingError(
                    f"Promotion event validation failed: {validation_errors}",
                    event_type=EventTypes.PROMOTION,
                )

            # Get employee mask for promotions
            employee_ids = events[EventColumns.EMP_ID].tolist()
            employee_mask = self._get_employee_mask(snapshot, employee_ids)

            # Prepare updates
            updates = {}

            # Update job level if provided
            if EventColumns.JOB_LEVEL in events.columns:
                updates[SnapshotColumns.EMP_LEVEL] = events[EventColumns.JOB_LEVEL].tolist()
                updates[SnapshotColumns.EMP_LEVEL_SOURCE] = "PROMOTION_EVENT"

            # Update compensation if provided
            if EventColumns.GROSS_COMPENSATION in events.columns:
                updates[SnapshotColumns.EMP_GROSS_COMP] = events[
                    EventColumns.GROSS_COMPENSATION
                ].tolist()

            # Update deferral rate if provided
            if EventColumns.DEFERRAL_RATE in events.columns:
                updates[SnapshotColumns.EMP_DEFERRAL_RATE] = events[
                    EventColumns.DEFERRAL_RATE
                ].tolist()

            updated_snapshot = self._update_snapshot_safely(
                snapshot, updates, employee_mask, "promotion"
            )

            # Recalculate contributions if compensation or deferral rate changed
            if any(
                col in updates
                for col in [SnapshotColumns.EMP_GROSS_COMP, SnapshotColumns.EMP_DEFERRAL_RATE]
            ):
                updated_snapshot = self._recalculate_contributions(updated_snapshot, employee_mask)

            # Update statistics
            processing_time = (pd.Timestamp.now() - start_time).total_seconds()
            self.stats.processing_time_seconds += processing_time

            self._log_processing_summary(EventTypes.PROMOTION, len(events), processing_time)
            return updated_snapshot

        except Exception as e:
            self.logger.error(f"Error processing promotion events: {str(e)}")
            self.stats.add_processed_event(success=False)
            raise EventProcessingError(
                f"Failed to process promotion events: {str(e)}", EventTypes.PROMOTION
            )

    def validate_events(self, events: pd.DataFrame, context: EventContext) -> List[str]:
        """Validate promotion events."""
        errors = self._validate_common_fields(events, context)

        # Promotion-specific validation
        has_job_level = EventColumns.JOB_LEVEL in events.columns
        has_compensation = EventColumns.GROSS_COMPENSATION in events.columns

        if not has_job_level and not has_compensation:
            errors.append("Promotion events should have either job level or compensation change")

        return errors

    def _recalculate_contributions(
        self, snapshot: pd.DataFrame, employee_mask: pd.Series
    ) -> pd.DataFrame:
        """Recalculate contributions for promoted employees."""
        if not employee_mask.any():
            return snapshot

        # Get promoted employees
        promoted_employees = snapshot[employee_mask].copy()

        # Recalculate employee contributions
        snapshot.loc[employee_mask, SnapshotColumns.EMP_CONTRIBUTION] = (
            promoted_employees[SnapshotColumns.EMP_GROSS_COMP]
            * promoted_employees[SnapshotColumns.EMP_DEFERRAL_RATE]
        )

        # Recalculate employer match
        max_match_base = promoted_employees[SnapshotColumns.EMP_GROSS_COMP] * 0.06
        employee_contrib_eligible = promoted_employees[SnapshotColumns.EMP_CONTRIBUTION].clip(
            upper=max_match_base
        )
        snapshot.loc[employee_mask, SnapshotColumns.EMPLOYER_MATCH_CONTRIBUTION] = (
            employee_contrib_eligible * 0.5
        )

        return snapshot


class CompensationEventHandler(BaseEventProcessor):
    """Handler for compensation change events."""

    def __init__(self):
        super().__init__([EventTypes.COMPENSATION])

    def process_events(
        self, snapshot: pd.DataFrame, events: pd.DataFrame, context: EventContext
    ) -> pd.DataFrame:
        """Process compensation change events."""
        if events.empty:
            return snapshot

        start_time = pd.Timestamp.now()
        self.logger.debug(f"Processing {len(events)} compensation events")

        try:
            # Validate events first
            validation_errors = self.validate_events(events, context)
            if validation_errors:
                raise EventProcessingError(
                    f"Compensation event validation failed: {validation_errors}",
                    event_type=EventTypes.COMPENSATION,
                )

            # Get employee mask for compensation changes
            employee_ids = events[EventColumns.EMP_ID].tolist()
            employee_mask = self._get_employee_mask(snapshot, employee_ids)

            # Prepare updates
            updates = {
                SnapshotColumns.EMP_GROSS_COMP: events[EventColumns.GROSS_COMPENSATION].tolist()
            }

            # Update deferral rate if provided
            if EventColumns.DEFERRAL_RATE in events.columns:
                updates[SnapshotColumns.EMP_DEFERRAL_RATE] = events[
                    EventColumns.DEFERRAL_RATE
                ].tolist()

            updated_snapshot = self._update_snapshot_safely(
                snapshot, updates, employee_mask, "compensation change"
            )

            # Recalculate contributions based on new compensation
            updated_snapshot = self._recalculate_contributions(updated_snapshot, employee_mask)

            # Update statistics
            processing_time = (pd.Timestamp.now() - start_time).total_seconds()
            self.stats.processing_time_seconds += processing_time

            self._log_processing_summary(EventTypes.COMPENSATION, len(events), processing_time)
            return updated_snapshot

        except Exception as e:
            self.logger.error(f"Error processing compensation events: {str(e)}")
            self.stats.add_processed_event(success=False)
            raise EventProcessingError(
                f"Failed to process compensation events: {str(e)}", EventTypes.COMPENSATION
            )

    def validate_events(self, events: pd.DataFrame, context: EventContext) -> List[str]:
        """Validate compensation events."""
        errors = self._validate_common_fields(events, context)

        # Compensation-specific validation
        if EventColumns.GROSS_COMPENSATION not in events.columns:
            errors.append("Compensation events missing compensation amount")
        else:
            missing_comp = events[EventColumns.GROSS_COMPENSATION].isna()
            if missing_comp.any():
                errors.append(f"Found {missing_comp.sum()} compensation events without amount")

            # Validate compensation is positive
            zero_comp = events[EventColumns.GROSS_COMPENSATION] <= 0
            if zero_comp.any():
                errors.append(
                    f"Found {zero_comp.sum()} compensation events with zero or negative amount"
                )

        return errors

    def _recalculate_contributions(
        self, snapshot: pd.DataFrame, employee_mask: pd.Series
    ) -> pd.DataFrame:
        """Recalculate contributions for employees with compensation changes."""
        if not employee_mask.any():
            return snapshot

        # Get affected employees
        affected_employees = snapshot[employee_mask].copy()

        # Recalculate employee contributions
        snapshot.loc[employee_mask, SnapshotColumns.EMP_CONTRIBUTION] = (
            affected_employees[SnapshotColumns.EMP_GROSS_COMP]
            * affected_employees[SnapshotColumns.EMP_DEFERRAL_RATE]
        )

        # Recalculate employer match
        max_match_base = affected_employees[SnapshotColumns.EMP_GROSS_COMP] * 0.06
        employee_contrib_eligible = affected_employees[SnapshotColumns.EMP_CONTRIBUTION].clip(
            upper=max_match_base
        )
        snapshot.loc[employee_mask, SnapshotColumns.EMPLOYER_MATCH_CONTRIBUTION] = (
            employee_contrib_eligible * 0.5
        )

        return snapshot
