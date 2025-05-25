# Financial Utilities

## Core Financial Functions

### Decimal Arithmetic
- **Location**: `cost_model.utils.decimal_helpers`
- **Description**: Precise decimal calculations
- **Key Features**:
  - `ZERO_DECIMAL`: Decimal('0.00')
  - `TWO_PLACES`: Decimal('0.01')
  - `to_money()`: Round to 2 decimal places
  - `decimal_quantize()`: Custom quantization
- **Search Tags**: `module:decimal_helpers`, `utils:decimal`

### Currency Handling
- **Location**: `cost_model.utils.currency`
- **Description**: Currency formatting and conversion
- **Features**:
  - Format currency amounts
  - Convert between currencies
  - Handle currency symbols
- **Search Tags**: `module:currency`, `utils:money`

## Financial Calculations

### Interest & Growth
- **Location**: `cost_model.utils.finance`
- **Description**: Financial math functions
- **Key Functions**:
  - `compound_interest()`: Calculate compound growth
  - `present_value()`: Calculate PV of future amount
  - `future_value()`: Calculate FV of present amount
  - `pmt()`: Calculate loan payment
- **Search Tags**: `module:finance`, `utils:calculations`

### Tax Calculations
- **Location**: `cost_model.utils.tax`
- **Description**: Tax-related calculations
- **Features**:
  - Tax bracket lookups
  - Withholding calculations
  - Tax credit applications
- **Search Tags**: `module:tax`, `utils:taxes`

## Usage Examples

### Basic Financial Calculations
```python
from decimal import Decimal
from cost_model.utils.decimal_helpers import to_money, TWO_PLACES
from cost_model.utils.finance import compound_interest

# Precise decimal arithmetic
amount = to_money(Decimal('100.499'))  # 100.50

# Compound interest calculation
principal = Decimal('1000')
rate = Decimal('0.05')  # 5%
years = 10
future_value = compound_interest(principal, rate, years)  # 1628.89
```

### Currency Formatting
```python
from cost_model.utils.currency import format_currency

# Format currency amounts
amount = Decimal('1234.56')
formatted = format_currency(amount, 'USD')  # '$1,234.56'
```

### Tax Calculations
```python
from cost_model.utils.tax import calculate_withholding

# Calculate tax withholding
gross_income = Decimal('75000')
filing_status = 'single'
withholding = calculate_withholding(
    gross_income, 
    filing_status,
    allowances=2
)
```

## Performance Considerations

### Decimal vs Float
- Use `Decimal` for financial calculations
- Avoid floating-point for money
- Set precision globally for consistency

### Caching
- Tax brackets are cached
- Exchange rates are cached
- Common calculations are memoized

## Best Practices
1. Always use `Decimal` for money
2. Set explicit precision
3. Handle rounding consistently
4. Document currency assumptions
5. Validate financial inputs

## Related Documentation
- [Data Types](../04_data/data_types.md)
- [Configuration](../03_config/financial_settings.md)
