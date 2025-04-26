import logging
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import numpy as np

logger = logging.getLogger(__name__)

# Centralized default for annual compensation increase rate
DEFAULT_COMP_INCREASE_RATE = 0.03
DEFAULT_COMP_INCREASE_STATUSES = ["Active"]

class CompensationMixin:
    f"""Mixin providing compensation update logic.

    Each eligible agent’s gross_compensation is increased by the scenario’s
    `annual_compensation_increase_rate` (default: {DEFAULT_COMP_INCREASE_RATE}).
    Eligibility is determined by `comp_increase_statuses` in scenario_config (default: {DEFAULT_COMP_INCREASE_STATUSES}).
    After this runs, gross_compensation will always be a Decimal rounded to two places (bankers’ rounding).

    Attributes required on self:
      - self.status (e.g. 'Active', 'Terminated', etc.)
      - self.gross_compensation (numeric or Decimal)
      - self.model.scenario_config (dict) containing 'annual_compensation_increase_rate' and optionally 'comp_increase_statuses', 'comp_min_cap', 'comp_max_cap'

    Note: Decimal is used for financial precision. If you later need to vectorize/batch updates, consider a float-based fast path for performance.
    """

    def _update_compensation(self):
        """Apply annual compensation increase to eligible agents."""
        # Get allowed statuses from config, default to ["Active"]
        valid_statuses = self.model.scenario_config.get("comp_increase_statuses", DEFAULT_COMP_INCREASE_STATUSES)
        if getattr(self, "status", None) not in valid_statuses:
            return

        try:
            # 1) Fetch and validate the increase rate
            rate = self.model.scenario_config.get("annual_compensation_increase_rate", DEFAULT_COMP_INCREASE_RATE)
            decimal_rate = Decimal(str(rate))
            if decimal_rate < 0:
                logger.warning("%r: Negative increase rate: %s. Skipping update.", self, decimal_rate)
                return

            # 2) Cast current comp to Decimal
            old_comp = Decimal(str(self.gross_compensation))

            # 3) Compute the new compensation
            updated = old_comp * (Decimal("1") + decimal_rate)
            new_comp = updated.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            # Apply min/max cap logic
            min_cap = self.model.scenario_config.get("comp_min_cap")
            max_cap = self.model.scenario_config.get("comp_max_cap")
            if min_cap is not None:
                new_comp = max(new_comp, Decimal(str(min_cap)))
            if max_cap is not None:
                new_comp = min(new_comp, Decimal(str(max_cap)))

            # 4) Assign and log
            self.gross_compensation = new_comp
            logger.debug(
                "%r: Compensation increased from %s to %s (rate=%s)",
                self, old_comp, new_comp, decimal_rate
            )

        except (InvalidOperation, ValueError) as ve:
            logger.warning(
                "%r: Invalid compensation or rate - cannot update: %s", self, ve
            )
        except Exception as e:
            logger.exception(
                "%r: Unexpected error in _update_compensation(): %s", self, e
            )

    @classmethod
    def batch_update(cls, agents, rate, assign_as_decimal=True):
        """
        Efficiently apply a compensation increase to a sequence of agents using NumPy vectorization.
        Only agents with eligible status (per their model config) are updated.

        Args:
            agents: Iterable of agent objects with .status and .gross_compensation
            rate: Annual increase rate (float or Decimal)
            assign_as_decimal: If True (default), assign results as Decimal; else, assign as float
        Returns:
            Number of agents updated
        """
        rate_f = float(rate)
        valid_statuses = agents[0].model.scenario_config.get("comp_increase_statuses", DEFAULT_COMP_INCREASE_STATUSES)
        eligible = [agent for agent in agents if getattr(agent, "status", None) in valid_statuses]
        if not eligible:
            logger.info("batch_update: No eligible agents found.")
            return 0
        comp_arr = np.array([float(agent.gross_compensation) for agent in eligible], dtype=np.float64)
        new_comps = comp_arr * (1.0 + rate_f)

        # Apply min/max cap logic
        min_cap = agents[0].model.scenario_config.get("comp_min_cap")
        max_cap = agents[0].model.scenario_config.get("comp_max_cap")
        if min_cap is not None:
            new_comps = np.maximum(new_comps, float(min_cap))
        if max_cap is not None:
            new_comps = np.minimum(new_comps, float(max_cap))

        count = 0
        for agent, new_val in zip(eligible, new_comps):
            if assign_as_decimal:
                agent.gross_compensation = Decimal(str(new_val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            else:
                agent.gross_compensation = float(round(new_val, 2))
            count += 1
        logger.info("batch_update: Updated compensation for %d agents at rate %s.", count, rate_f)
        return count
