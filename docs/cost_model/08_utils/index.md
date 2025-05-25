# Utilities

This document details the utility modules and helper functions used throughout the Workforce Simulation & Cost Model system.

## Date Utilities

### Date Utilities Module

- **Location**: `cost_model.utils.date_utils`
- **Description**: Date calculation and manipulation functions
- **Key Functions**:
  - `calculate_age(birth_date, current_date)`: Calculates age in years from birth date
  - `calculate_tenure(hire_date, current_date)`: Calculates tenure in fractional years
  - `get_random_dates_in_year(year, count, rng, day_of_month=15)`: Generates random dates within a year

- **Example**:
  ```python
  import pandas as pd
  from cost_model.utils.date_utils import calculate_age, calculate_tenure, get_random_dates_in_year
  import numpy as np

  # Calculate age for a single employee
  birth_date = pd.Timestamp('1985-06-15')
  current_date = pd.Timestamp('2023-01-01')
  age = calculate_age(birth_date, current_date)  # Returns 37

  # Calculate tenure for multiple employees
  hire_dates = pd.Series([
      '2020-01-15', '2018-06-01', '2022-11-15'
  ]).apply(pd.Timestamp)
  tenures = calculate_tenure(hire_dates, current_date)
  
  # Generate random hire dates for simulation
  rng = np.random.default_rng(42)
  hire_dates = get_random_dates_in_year(2023, 10, rng)
  ```

## Decimal Helpers

### Decimal Helpers Module

- **Location**: `cost_model.utils.decimal_helpers`
- **Description**: Utilities for precise decimal arithmetic in financial calculations
- **Key Components**:
  - `ZERO_DECIMAL`: Decimal('0.00') constant for financial zero
  - `TWO_PLACES`: Decimal('0.01') constant for money quantization
  - `to_money(d)`: Quantizes decimal to two places with ROUND_HALF_UP

- **Example**:
  ```python
  from decimal import Decimal
  from cost_model.utils.decimal_helpers import ZERO_DECIMAL, to_money
  
  # Using decimal for financial calculations
  price = Decimal('19.99')
  quantity = 3
  subtotal = price * quantity  # 59.97
  tax = (subtotal * Decimal('0.08')).quantize(Decimal('0.01'))
  total = to_money(subtotal + tax)  # 64.77
  
  # Using the zero constant
  balance = ZERO_DECIMAL
  balance += Decimal('100.50')
  ```

## Status Enums

### Status Enums Module

- **Location**: `cost_model.utils.status_enums`
- **Description**: Enumerations for employment status and enrollment methods
- **Key Enums**:
  - `EnrollmentMethod`: Methods of plan enrollment
    - `AE`: Auto-enrollment
    - `MANUAL`: Manual enrollment
    - `NONE`: Not enrolled
  - `EmploymentStatus`: Employee status values
    - `PREV_TERMINATED`: Previously terminated
    - `TERMINATED`: Recently terminated
    - `NEW_HIRE`: New hire
    - `ACTIVE_INITIAL`: Active in initial period
    - `ACTIVE_CONTINUOUS`: Active continuously
    - `NOT_HIRED`: Not yet hired
    - `INACTIVE`: Inactive status

- **Example**:
  ```python
  from cost_model.utils.status_enums import EmploymentStatus, EnrollmentMethod
  
  # Check employment status
  status = EmploymentStatus.NEW_HIRE
  if status == EmploymentStatus.NEW_HIRE:
      print("New hire detected")
  
  # Set enrollment method
  enrollment = EnrollmentMethod.AE
  if enrollment == EnrollmentMethod.AE:
      print("Auto-enrollment enabled")

## Example Usage

### Processing Employee Data with Utilities

```python
import pandas as pd
from datetime import datetime
from decimal import Decimal
from cost_model.utils.date_utils import calculate_age, calculate_tenure
from cost_model.utils.decimal_helpers import ZERO_DECIMAL, to_money
from cost_model.utils.status_enums import EmploymentStatus, EnrollmentMethod

def process_employee_hire(employee_data, current_date):
    """Process new hire data with utility functions."""
    # Calculate age and tenure
    birth_date = pd.to_datetime(employee_data['birth_date'])
    hire_date = pd.to_datetime(employee_data['hire_date'])
    
    age = calculate_age(birth_date, current_date)
    tenure = calculate_tenure(hire_date, current_date)
    
    # Set initial compensation with decimal precision
    base_salary = to_money(Decimal(str(employee_data['base_salary'])))
    
    # Set employment status
    status = EmploymentStatus.NEW_HIRE
    
    # Set enrollment method based on eligibility
    if employee_data.get('auto_enroll_eligible', False):
        enrollment = EnrollmentMethod.AE
    else:
        enrollment = EnrollmentMethod.MANUAL
    
    return {
        'employee_id': employee_data['id'],
        'age': age,
        'tenure': tenure,
        'base_salary': float(base_salary),
        'status': status.value,
        'enrollment_method': enrollment.value,
        'hire_date': hire_date.strftime('%Y-%m-%d'),
        'last_updated': current_date.strftime('%Y-%m-%d')
    }

def calculate_benefits_contribution(employee_records, benefit_rate=Decimal('0.15')):
    """Calculate employer benefit contributions with decimal precision."""
    results = []
    
    for emp in employee_records:
        try:
            # Convert salary to Decimal for precise calculation
            salary = Decimal(str(emp['base_salary']))
            
            # Calculate employer contribution (15% of salary by default)
            contribution = to_money(salary * benefit_rate)
            
            results.append({
                'employee_id': emp['employee_id'],
                'salary': float(salary),
                'employer_contribution': float(contribution),
                'contribution_rate': float(benefit_rate)
            })
            
        except (ValueError, KeyError) as e:
            print(f"Error processing employee {emp.get('employee_id', 'unknown')}: {str(e)}")
    
    return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    # Current date for calculations
    current_date = pd.Timestamp('2023-06-15')
    
    # Sample employee data
    employees = [
        {
            'id': 'E1001',
            'birth_date': '1985-05-20',
            'hire_date': '2023-01-10',
            'base_salary': 75000.00,
            'auto_enroll_eligible': True
        },
        {
            'id': 'E1002',
            'birth_date': '1990-11-15',
            'hire_date': '2022-06-01',
            'base_salary': 85000.50,
            'auto_enroll_eligible': False
        }
    ]
    
    # Process each employee
    processed_employees = [process_employee_hire(emp, current_date) for emp in employees]
    
    # Calculate benefits
    benefits_df = calculate_benefits_contribution(processed_employees)
    
    # Display results
    print("\nProcessed Employees:")
    for emp in processed_employees:
        print(f"ID: {emp['employee_id']}, Status: {emp['status']}, "
              f"Age: {emp['age']}, Tenure: {emp['tenure']:.2f} years, "
              f"Salary: ${emp['base_salary']:,.2f}")
    
    print("\nBenefit Contributions:")
    print(benefits_df.to_string(index=False))
```

## Related Documentation

- [Class Inventory](02_class_inventory.md) - Complete list of all classes
- [Data Classes](04_data_classes.md) - Data handling utilities
- [Configuration Classes](03_config_classes.md) - Configuration management
- [Projection Engine](05_projection_engine.md) - How to use the projection utilities
- [Plan Rules](06_plan_rules.md) - Documentation on plan rule utilities
