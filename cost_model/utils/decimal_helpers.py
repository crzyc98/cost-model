# utils/decimal_helpers.py

from decimal import Decimal, ROUND_HALF_UP, getcontext

# Shared zero constant for financial calculations
ZERO_DECIMAL = Decimal('0.00')
# Set global rounding mode to half-up
getcontext().rounding = ROUND_HALF_UP
# Standard quantization unit for money
TWO_PLACES = Decimal('0.01')

def to_money(d: Decimal) -> Decimal:
    """Quantize Decimal to two places with ROUND_HALF_UP rounding."""
    return d.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
