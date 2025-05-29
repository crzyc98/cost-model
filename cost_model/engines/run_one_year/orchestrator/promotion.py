# cost_model/engines/run_one_year/orchestrator/promotion.py
"""
Promotion and compensation orchestration module.

Handles Markov chain-based promotions, merit increases, and cost-of-living adjustments.
"""
import logging
from typing import List, Optional, Tuple
import pandas as pd

from cost_model.engines.markov_promotion import apply_markov_promotions
from cost_model.state.schema import EMP_ID, EMP_LEVEL
from cost_model.state.event_log import EVENT_PANDAS_DTYPES
from .base import YearContext, filter_valid_employee_ids


class PromotionOrchestrator:
    """
    Orchestrates all promotion and compensation-related activities for a simulation year.

    This includes:
    - Markov chain-based promotions with associated raises
    - Merit increases and cost-of-living adjustments
    - Generating promotion, raise, and compensation events
    - Updating employee levels and compensation in snapshots
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the promotion orchestrator.

        Args:
            logger: Optional logger instance. If None, creates a new one.
        """
        self.logger = logger or logging.getLogger(__name__)

    def get_events(
        self,
        snapshot: pd.DataFrame,
        year_context: YearContext
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """
        Generate promotion and compensation events and update the snapshot.

        Args:
            snapshot: Current workforce snapshot
            year_context: Year-specific context and parameters

        Returns:
            Tuple of (event_list, updated_snapshot) where:
            - event_list: List of [promotion_events, raise_events, exit_events]
            - updated_snapshot: Snapshot with promotions applied and exits removed
        """
        self.logger.info("[STEP] Markov promotions/exits (experienced only)")

        # Apply Markov promotions
        promotion_events, raise_events, exit_events = self._apply_markov_promotions(
            snapshot, year_context
        )

        # Update snapshot with promotions and remove exits
        updated_snapshot = self._update_snapshot_with_promotions(
            snapshot, promotion_events, raise_events, exit_events
        )

        self.logger.info(
            f"[MARKOV] Promotions: {len(promotion_events)}, "
            f"Raises: {len(raise_events)}, Exits: {len(exit_events)}"
        )

        return [promotion_events, raise_events, exit_events], updated_snapshot

    def _apply_markov_promotions(
        self,
        snapshot: pd.DataFrame,
        year_context: YearContext
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply Markov chain-based promotions to the workforce.

        Args:
            snapshot: Current workforce snapshot
            year_context: Year-specific context and parameters

        Returns:
            Tuple of (promotion_events, raise_events, exit_events)
        """
        # Get promotion configuration
        promotion_raise_config = getattr(year_context.global_params, 'promotion_raise_config', {})

        # Apply Markov promotions
        promotions_df, raises_df, exited_df = apply_markov_promotions(
            snapshot=snapshot,
            promo_time=year_context.as_of,  # Promotions at start of year
            rng=year_context.year_rng,
            promotion_raise_config=promotion_raise_config,
            simulation_year=year_context.year,
            global_params=year_context.global_params
        )

        # Validate and clean events
        promotions_df = self._validate_events(promotions_df, "promotion")
        raises_df = self._validate_events(raises_df, "raise")
        exited_df = self._validate_events(exited_df, "exit")

        return promotions_df, raises_df, exited_df

    def _validate_events(self, events: pd.DataFrame, event_type: str) -> pd.DataFrame:
        """
        Validate and clean event DataFrames.

        Args:
            events: Events DataFrame to validate
            event_type: Type of events for logging

        Returns:
            Cleaned events DataFrame
        """
        if events.empty:
            return events

        # Filter out invalid employee IDs
        events = filter_valid_employee_ids(events, self.logger)

        # Ensure proper dtypes
        for col, dtype in EVENT_PANDAS_DTYPES.items():
            if col in events.columns:
                try:
                    events[col] = events[col].astype(dtype)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Could not convert {event_type} events column {col} to {dtype}: {e}")

        return events

    def _update_snapshot_with_promotions(
        self,
        snapshot: pd.DataFrame,
        promotion_events: pd.DataFrame,
        raise_events: pd.DataFrame,
        exit_events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Update the snapshot with promotion results and remove exited employees.

        Args:
            snapshot: Current workforce snapshot
            promotion_events: Promotion events DataFrame
            raise_events: Raise events DataFrame
            exit_events: Exit events DataFrame
            year_context: Year-specific context

        Returns:
            Updated snapshot with promotions applied and exits removed
        """
        updated_snapshot = snapshot.copy()

        # Apply promotions (level changes)
        if not promotion_events.empty:
            updated_snapshot = self._apply_promotions_to_snapshot(
                updated_snapshot, promotion_events
            )

        # Apply raises (compensation changes)
        if not raise_events.empty:
            updated_snapshot = self._apply_raises_to_snapshot(
                updated_snapshot, raise_events
            )

        # Remove exited employees
        if not exit_events.empty:
            exited_emp_ids = set(exit_events[EMP_ID].unique())
            # Handle both index-based and column-based employee ID filtering
            if EMP_ID in updated_snapshot.columns:
                updated_snapshot = updated_snapshot[~updated_snapshot[EMP_ID].isin(exited_emp_ids)].copy()
            else:
                # If EMP_ID is in the index
                updated_snapshot = updated_snapshot.loc[~updated_snapshot.index.isin(exited_emp_ids)].copy()
            self.logger.info(f"[MARKOV] Removed {len(exited_emp_ids)} exited employees")

        return updated_snapshot

    def _apply_promotions_to_snapshot(
        self,
        snapshot: pd.DataFrame,
        promotion_events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply promotion level changes to the snapshot.

        Args:
            snapshot: Current workforce snapshot
            promotion_events: Promotion events DataFrame

        Returns:
            Updated snapshot with new levels
        """
        snapshot_copy = snapshot.copy()

        for _, event in promotion_events.iterrows():
            emp_id = event[EMP_ID]

            # Extract new level from event
            new_level = self._extract_new_level_from_promotion(event)

            if new_level is not None:
                # Update employee level in snapshot
                mask = snapshot_copy[EMP_ID] == emp_id
                if mask.any():
                    snapshot_copy.loc[mask, EMP_LEVEL] = new_level
                    self.logger.debug(f"Updated level for {emp_id} to {new_level}")

        return snapshot_copy

    def _apply_raises_to_snapshot(
        self,
        snapshot: pd.DataFrame,
        raise_events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply compensation raises to the snapshot.

        Args:
            snapshot: Current workforce snapshot
            raise_events: Raise events DataFrame

        Returns:
            Updated snapshot with new compensation
        """
        snapshot_copy = snapshot.copy()

        for _, event in raise_events.iterrows():
            emp_id = event[EMP_ID]

            # Extract new compensation from event
            new_comp = self._extract_new_compensation_from_raise(event)

            if new_comp is not None:
                # Update employee compensation in snapshot
                mask = snapshot_copy[EMP_ID] == emp_id
                if mask.any():
                    from cost_model.state.schema import EMP_GROSS_COMP
                    snapshot_copy.loc[mask, EMP_GROSS_COMP] = new_comp
                    self.logger.debug(f"Updated compensation for {emp_id} to {new_comp}")

        return snapshot_copy

    def _extract_new_level_from_promotion(self, promotion_event: pd.Series) -> Optional[int]:
        """
        Extract the new job level from a promotion event.

        Args:
            promotion_event: Single promotion event row

        Returns:
            New job level or None if not found
        """
        import json

        # Try value_json first
        if 'value_json' in promotion_event and pd.notna(promotion_event['value_json']):
            try:
                value_data = json.loads(promotion_event['value_json'])
                if isinstance(value_data, dict):
                    to_level = value_data.get('to_level')
                    if to_level is not None:
                        return int(to_level)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # Try meta
        if 'meta' in promotion_event and pd.notna(promotion_event['meta']):
            from .base import safe_get_meta
            level = safe_get_meta(promotion_event['meta'], 'to_level')
            if level is not None:
                try:
                    return int(level)
                except (ValueError, TypeError):
                    pass

        return None

    def _extract_new_compensation_from_raise(self, raise_event: pd.Series) -> Optional[float]:
        """
        Extract the new compensation from a raise event.

        Args:
            raise_event: Single raise event row

        Returns:
            New compensation or None if not found
        """
        # Try value_num first
        if 'value_num' in raise_event and pd.notna(raise_event['value_num']):
            return float(raise_event['value_num'])

        # Try value_json
        if 'value_json' in raise_event and pd.notna(raise_event['value_json']):
            import json
            try:
                value_data = json.loads(raise_event['value_json'])
                if isinstance(value_data, dict):
                    new_comp = value_data.get('new_compensation') or value_data.get('new_comp')
                    if new_comp is not None:
                        return float(new_comp)
                elif isinstance(value_data, (int, float)):
                    return float(value_data)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        return None
