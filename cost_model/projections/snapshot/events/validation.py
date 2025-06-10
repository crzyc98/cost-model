"""
Event validation utilities.

This module provides comprehensive validation for events before processing,
ensuring data quality and business rule compliance.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from cost_model.schema import EventColumns, EventTypes, SnapshotColumns
from cost_model.schema.validation import ValidationResult

from .base import EventContext

logger = logging.getLogger(__name__)


class EventValidator:
    """
    Comprehensive validator for event data.

    This class provides validation for event structure, data quality,
    and business rules before events are processed.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the event validator.

        Args:
            strict_mode: If True, treat warnings as errors
        """
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def validate_events(self, events: pd.DataFrame, context: EventContext) -> List[str]:
        """
        Validate events and return list of error messages.

        Args:
            events: Events DataFrame to validate
            context: Processing context

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        try:
            # Basic structure validation
            errors.extend(self._validate_structure(events))

            # Required fields validation
            errors.extend(self._validate_required_fields(events))

            # Data type validation
            errors.extend(self._validate_data_types(events))

            # Data quality validation
            errors.extend(self._validate_data_quality(events, context))

            # Business rules validation
            errors.extend(self._validate_business_rules(events, context))

            # Event-specific validation
            errors.extend(self._validate_event_specific_rules(events, context))

        except Exception as e:
            errors.append(f"Critical validation error: {str(e)}")
            self.logger.error(f"Error during event validation: {str(e)}", exc_info=True)

        return errors

    def validate_event_consistency(
        self, events: pd.DataFrame, snapshot: pd.DataFrame, context: EventContext
    ) -> List[str]:
        """
        Validate consistency between events and current snapshot.

        Args:
            events: Events to validate
            snapshot: Current snapshot
            context: Processing context

        Returns:
            List of consistency error messages
        """
        errors = []

        try:
            # Check that employees in events exist in snapshot (except for hires)
            hire_events = events[events[EventColumns.EVENT_TYPE] == EventTypes.HIRE]
            non_hire_events = events[events[EventColumns.EVENT_TYPE] != EventTypes.HIRE]

            if not non_hire_events.empty and SnapshotColumns.EMP_ID in snapshot.columns:
                snapshot_emp_ids = set(snapshot[SnapshotColumns.EMP_ID])
                event_emp_ids = set(non_hire_events[EventColumns.EMP_ID])
                missing_employees = event_emp_ids - snapshot_emp_ids

                if missing_employees:
                    errors.append(
                        f"Events reference {len(missing_employees)} employees not in snapshot: "
                        f"{sorted(list(missing_employees))[:10]}{'...' if len(missing_employees) > 10 else ''}"
                    )

            # Check for duplicate hire events
            if not hire_events.empty:
                duplicate_hires = hire_events[EventColumns.EMP_ID].duplicated()
                if duplicate_hires.any():
                    duplicate_count = duplicate_hires.sum()
                    errors.append(f"Found {duplicate_count} duplicate hire events")

            # Check termination events for already terminated employees
            term_events = events[
                events[EventColumns.EVENT_TYPE].isin(
                    [EventTypes.TERMINATION, EventTypes.NEW_HIRE_TERMINATION]
                )
            ]

            if not term_events.empty and all(
                col in snapshot.columns
                for col in [SnapshotColumns.EMP_ID, SnapshotColumns.EMP_ACTIVE]
            ):
                term_emp_ids = set(term_events[EventColumns.EMP_ID])
                inactive_emp_ids = set(
                    snapshot[snapshot[SnapshotColumns.EMP_ACTIVE] == False][SnapshotColumns.EMP_ID]
                )
                already_terminated = term_emp_ids & inactive_emp_ids

                if already_terminated:
                    errors.append(
                        f"Termination events for {len(already_terminated)} already inactive employees"
                    )

        except Exception as e:
            errors.append(f"Error validating event consistency: {str(e)}")
            self.logger.error(f"Error validating event consistency: {str(e)}", exc_info=True)

        return errors

    def _validate_structure(self, events: pd.DataFrame) -> List[str]:
        """Validate basic DataFrame structure."""
        errors = []

        if events is None:
            errors.append("Events DataFrame is None")
            return errors

        if events.empty:
            # Empty events is valid, just log it
            self.logger.info("Events DataFrame is empty")
            return errors

        # Check for duplicate column names
        if len(events.columns) != len(set(events.columns)):
            duplicates = [col for col in events.columns if list(events.columns).count(col) > 1]
            errors.append(f"Duplicate column names in events: {set(duplicates)}")

        # Check for unnamed columns
        unnamed_cols = [col for col in events.columns if str(col).startswith("Unnamed:")]
        if unnamed_cols:
            errors.append(f"Unnamed columns in events: {unnamed_cols}")

        return errors

    def _validate_required_fields(self, events: pd.DataFrame) -> List[str]:
        """Validate required fields are present."""
        errors = []

        if events.empty:
            return errors

        required_columns = [EventColumns.EVENT_TYPE, EventColumns.EVENT_DATE, EventColumns.EMP_ID]

        missing_columns = [col for col in required_columns if col not in events.columns]
        if missing_columns:
            errors.append(f"Missing required event columns: {missing_columns}")

        return errors

    def _validate_data_types(self, events: pd.DataFrame) -> List[str]:
        """Validate data types of event columns."""
        errors = []

        if events.empty:
            return errors

        # Validate event types
        if EventColumns.EVENT_TYPE in events.columns:
            valid_event_types = {et.value for et in EventTypes}
            invalid_types = ~events[EventColumns.EVENT_TYPE].isin(valid_event_types)
            if invalid_types.any():
                invalid_values = events.loc[invalid_types, EventColumns.EVENT_TYPE].unique()
                errors.append(f"Invalid event types: {list(invalid_values)}")

        # Validate dates
        if EventColumns.EVENT_DATE in events.columns:
            try:
                event_dates = pd.to_datetime(events[EventColumns.EVENT_DATE], errors="coerce")
                invalid_dates = event_dates.isna() & events[EventColumns.EVENT_DATE].notna()
                if invalid_dates.any():
                    errors.append(f"Found {invalid_dates.sum()} invalid event dates")
            except Exception as e:
                errors.append(f"Error parsing event dates: {str(e)}")

        # Validate numeric fields (exclude job_level as it can be string)
        numeric_fields = [EventColumns.GROSS_COMPENSATION, EventColumns.DEFERRAL_RATE]

        for field in numeric_fields:
            if field in events.columns:
                try:
                    numeric_values = pd.to_numeric(events[field], errors="coerce")
                    invalid_numeric = numeric_values.isna() & events[field].notna()
                    if invalid_numeric.any():
                        errors.append(
                            f"Found {invalid_numeric.sum()} invalid numeric values in {field}"
                        )
                except Exception as e:
                    errors.append(f"Error validating numeric field {field}: {str(e)}")

        return errors

    def _validate_data_quality(self, events: pd.DataFrame, context: EventContext) -> List[str]:
        """Validate data quality issues."""
        errors = []

        if events.empty:
            return errors

        # Check for null values in critical fields
        critical_fields = [EventColumns.EVENT_TYPE, EventColumns.EMP_ID, EventColumns.EVENT_DATE]
        for field in critical_fields:
            if field in events.columns:
                null_count = events[field].isna().sum()
                if null_count > 0:
                    errors.append(f"Found {null_count} null values in critical field {field}")

        # Check employee ID format
        if EventColumns.EMP_ID in events.columns:
            emp_ids = events[EventColumns.EMP_ID]

            # Check for empty employee IDs
            empty_ids = (emp_ids == "") | emp_ids.isna()
            if empty_ids.any():
                errors.append(f"Found {empty_ids.sum()} empty employee IDs")

            # Check for reasonable employee ID length (basic sanity check)
            if not emp_ids.empty:
                id_lengths = emp_ids.astype(str).str.len()
                very_long_ids = id_lengths > 50
                if very_long_ids.any():
                    errors.append(f"Found {very_long_ids.sum()} suspiciously long employee IDs")

        # Validate compensation values
        if EventColumns.GROSS_COMPENSATION in events.columns:
            compensation = events[EventColumns.GROSS_COMPENSATION]

            # Check for negative compensation
            negative_comp = compensation < 0
            if negative_comp.any():
                errors.append(f"Found {negative_comp.sum()} events with negative compensation")

            # Check for unrealistic compensation values
            very_high_comp = compensation > 2000000  # $2M threshold
            if very_high_comp.any():
                message = f"Found {very_high_comp.sum()} events with very high compensation (>$2M)"
                if self.strict_mode:
                    errors.append(message)
                else:
                    self.logger.warning(message)

        # Validate deferral rates
        if EventColumns.DEFERRAL_RATE in events.columns:
            deferral_rates = events[EventColumns.DEFERRAL_RATE]

            # Check for rates outside valid range
            invalid_rates = (deferral_rates < 0) | (deferral_rates > 1)
            if invalid_rates.any():
                errors.append(
                    f"Found {invalid_rates.sum()} events with invalid deferral rates (outside 0-1)"
                )

        return errors

    def _validate_business_rules(self, events: pd.DataFrame, context: EventContext) -> List[str]:
        """Validate business rules."""
        errors = []

        if events.empty:
            return errors

        # Validate event dates are not too far in the future
        if EventColumns.EVENT_DATE in events.columns:
            event_dates = pd.to_datetime(events[EventColumns.EVENT_DATE], errors="coerce")
            max_future_date = context.reference_date + timedelta(days=365)  # 1 year in future

            future_events = event_dates > max_future_date
            if future_events.any():
                message = f"Found {future_events.sum()} events with dates too far in future"
                if self.strict_mode:
                    errors.append(message)
                else:
                    self.logger.warning(message)

        # Validate hire events have required fields
        hire_events = events[events[EventColumns.EVENT_TYPE] == EventTypes.HIRE]
        if not hire_events.empty:
            required_hire_fields = [EventColumns.GROSS_COMPENSATION]
            for field in required_hire_fields:
                if field not in hire_events.columns:
                    errors.append(f"Hire events missing required field: {field}")
                elif hire_events[field].isna().any():
                    null_count = hire_events[field].isna().sum()
                    errors.append(f"Found {null_count} hire events with null {field}")

        # Validate termination events
        term_events = events[
            events[EventColumns.EVENT_TYPE].isin(
                [EventTypes.TERMINATION, EventTypes.NEW_HIRE_TERMINATION]
            )
        ]
        if not term_events.empty:
            # Termination events should have termination reason if available
            if EventColumns.TERMINATION_REASON in term_events.columns:
                missing_reason = term_events[EventColumns.TERMINATION_REASON].isna()
                if missing_reason.any():
                    message = f"Found {missing_reason.sum()} termination events without reason"
                    if self.strict_mode:
                        errors.append(message)
                    else:
                        self.logger.warning(message)

        return errors

    def _validate_event_specific_rules(
        self, events: pd.DataFrame, context: EventContext
    ) -> List[str]:
        """Validate rules specific to each event type."""
        errors = []

        if events.empty:
            return errors

        # Group events by type and validate each group
        for event_type in events[EventColumns.EVENT_TYPE].unique():
            type_events = events[events[EventColumns.EVENT_TYPE] == event_type]

            if event_type == EventTypes.HIRE:
                errors.extend(self._validate_hire_events(type_events, context))
            elif event_type in [EventTypes.TERMINATION, EventTypes.NEW_HIRE_TERMINATION]:
                errors.extend(self._validate_termination_events(type_events, context))
            elif event_type == EventTypes.PROMOTION:
                errors.extend(self._validate_promotion_events(type_events, context))
            elif event_type == EventTypes.COMPENSATION:
                errors.extend(self._validate_compensation_events(type_events, context))

        return errors

    def _validate_hire_events(self, hire_events: pd.DataFrame, context: EventContext) -> List[str]:
        """Validate hire-specific business rules."""
        errors = []

        # Hire events should have positive compensation
        if EventColumns.GROSS_COMPENSATION in hire_events.columns:
            zero_comp = hire_events[EventColumns.GROSS_COMPENSATION] <= 0
            if zero_comp.any():
                errors.append(
                    f"Found {zero_comp.sum()} hire events with zero or negative compensation"
                )

        # Check for reasonable hire dates (not too far in past or future)
        if EventColumns.EVENT_DATE in hire_events.columns:
            hire_dates = pd.to_datetime(hire_events[EventColumns.EVENT_DATE])
            very_old_hires = hire_dates < (
                context.reference_date - timedelta(days=365 * 50)
            )  # 50 years ago
            very_future_hires = hire_dates > (
                context.reference_date + timedelta(days=365 * 2)
            )  # 2 years future

            if very_old_hires.any():
                message = f"Found {very_old_hires.sum()} hire events with very old dates"
                if self.strict_mode:
                    errors.append(message)
                else:
                    self.logger.warning(message)

            if very_future_hires.any():
                errors.append(
                    f"Found {very_future_hires.sum()} hire events with future dates beyond reasonable range"
                )

        return errors

    def _validate_termination_events(
        self, term_events: pd.DataFrame, context: EventContext
    ) -> List[str]:
        """Validate termination-specific business rules."""
        errors = []

        # Termination events should not be in the future (beyond current simulation year)
        if EventColumns.EVENT_DATE in term_events.columns:
            term_dates = pd.to_datetime(term_events[EventColumns.EVENT_DATE])
            future_terms = term_dates > context.reference_date

            if future_terms.any():
                message = f"Found {future_terms.sum()} termination events with future dates"
                if self.strict_mode:
                    errors.append(message)
                else:
                    self.logger.warning(message)

        return errors

    def _validate_promotion_events(
        self, promo_events: pd.DataFrame, context: EventContext
    ) -> List[str]:
        """Validate promotion-specific business rules."""
        errors = []

        # Promotion events should have job level information
        if EventColumns.JOB_LEVEL in promo_events.columns:
            missing_level = promo_events[EventColumns.JOB_LEVEL].isna()
            if missing_level.any():
                errors.append(f"Found {missing_level.sum()} promotion events without job level")

        # Promotions should typically include compensation changes
        if EventColumns.GROSS_COMPENSATION in promo_events.columns:
            missing_comp = promo_events[EventColumns.GROSS_COMPENSATION].isna()
            if missing_comp.any():
                message = f"Found {missing_comp.sum()} promotion events without compensation change"
                if self.strict_mode:
                    errors.append(message)
                else:
                    self.logger.warning(message)

        return errors

    def _validate_compensation_events(
        self, comp_events: pd.DataFrame, context: EventContext
    ) -> List[str]:
        """Validate compensation-specific business rules."""
        errors = []

        # Compensation events must have compensation amount
        if EventColumns.GROSS_COMPENSATION not in comp_events.columns:
            errors.append("Compensation events missing compensation amount")
        else:
            missing_comp = comp_events[EventColumns.GROSS_COMPENSATION].isna()
            if missing_comp.any():
                errors.append(f"Found {missing_comp.sum()} compensation events without amount")

        return errors
