from decimal import Decimal, ROUND_HALF_UP


class CompensationMixin:
    """Mixin providing compensation update logic."""

    def _update_compensation(self):
        """Updates the agent's gross compensation based on annual increase rules."""
        if self.status == 'Active':
            # Use the correct config key
            increase_rate = self.model.scenario_config.get('annual_compensation_increase_rate', 0.03)
            decimal_increase_rate = Decimal(str(increase_rate))
            # Convert to Decimal, apply increase, then round to 2 decimals
            updated = Decimal(str(self.gross_compensation)) * (Decimal('1.0') + decimal_increase_rate)
            self.gross_compensation = updated.quantize(
                Decimal('0.01'), rounding=ROUND_HALF_UP
            )
