# Date & Time Utilities

## Core Date Functions

### Date Calculations
- **Location**: `cost_model.utils.date_utils`
- **Description**: Date arithmetic and manipulation
- **Key Functions**:
  - `calculate_age()`: Calculate age from birth date
  - `calculate_tenure()`: Calculate employment tenure
  - `is_anniversary()`: Check if date is an anniversary
  - `add_years()`: Add years to a date
  - `last_day_of_month()`: Get last day of month
- **Search Tags**: `module:date_utils`, `utils:dates`

### Date Ranges
- **Location**: `cost_model.utils.date_ranges`
- **Description**: Working with date ranges and periods
- **Features**:
  - Generate date ranges
  - Check date overlaps
  - Split ranges by year/quarter/month
- **Search Tags**: `module:date_ranges`, `utils:ranges`

## Timezone Handling

### Timezone Utilities
- **Location**: `cost_model.utils.timezone_utils`
- **Description**: Timezone conversion and handling
- **Features**:
  - Convert between timezones
  - Handle daylight saving time
  - Localize naive datetimes
- **Search Tags**: `module:timezone_utils`, `utils:timezones`

## Business Calendar

### BusinessDate
- **Location**: `cost_model.utils.business_date`
- **Description**: Business day calculations
- **Features**:
  - Business day arithmetic
  - Holiday calendar integration
  - Business hours calculations
- **Search Tags**: `class:BusinessDate`, `utils:business_dates`

## Usage Examples

### Basic Date Calculations
```python
from cost_model.utils.date_utils import calculate_age, calculate_tenure
from datetime import date

# Calculate age
birth_date = date(1985, 6, 15)
age = calculate_age(birth_date, date(2023, 1, 1))  # 37

# Calculate tenure
hire_date = date(2020, 1, 15)
tenure = calculate_tenure(hire_date, date(2023, 1, 1))  # 3.0
```

### Working with Business Dates
```python
from cost_model.utils.business_date import BusinessDate
from datetime import date, timedelta

# Create business date with holiday calendar
biz_date = BusinessDate(date(2023, 12, 24))  # Christmas Eve
next_biz_day = biz_date + timedelta(days=1)  # Skips Christmas

# Check if business day
is_biz_day = BusinessDate.is_business_day(date(2023, 1, 1))  # False
```

### Timezone Conversions
```python
from datetime import datetime
from cost_model.utils.timezone_utils import (
    convert_tz,
    LOCAL_TZ,
    UTC
)

# Convert UTC to local time
utc_dt = datetime(2023, 1, 1, 12, 0, tzinfo=UTC)
local_dt = convert_tz(utc_dt, LOCAL_TZ)
```

## Performance Considerations

### Caching
- Date parsing and timezone conversions are cached
- Business day calculations use memoization
- Holiday calendars are loaded once and cached

### Best Practices
1. Always use timezone-aware datetimes
2. Use business date for financial calculations
3. Cache frequently used date calculations
4. Avoid string parsing in tight loops

## Related Documentation
- [Data Types](../04_data/data_types.md)
- [Configuration](../03_config/time_settings.md)
