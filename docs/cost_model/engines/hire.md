# Hire Engine Documentation

## Overview
The `hire` module is responsible for generating new hire events and their initial compensation data in the workforce simulation. It handles the creation of new employee records, assignment of roles, and calculation of starting compensation based on configured parameters.

## Key Components

### `run()` Function

The main function that generates new hire events and their compensation data.

#### Parameters
- `snapshot`: Current workforce snapshot DataFrame
- `hires_to_make`: Number of new hires to generate
- `hazard_slice`: Hazard rate data for the current simulation year
- `rng`: Random number generator for reproducibility
- `census_template_path`: Path to census data template (currently not used)
- `global_params`: Global simulation parameters
- `terminated_events`: DataFrame of termination events (for replacement hires)

#### Returns
- List containing a single DataFrame of hire events with compensation data

## Recent Changes

### Fix: Missing Compensation for New Hires

**Date**: 2025-05-27  
**Files Modified**: `cost_model/engines/hire.py`

#### Issue
New hire events were being created without their corresponding compensation values, resulting in warnings about missing compensation data and potentially incorrect financial calculations in the simulation.

#### Root Cause
The `run()` function was correctly sampling compensation values for new hires (stored in `starting_comps`) but not including them in the generated hire events. The `value_num` field in the event creation was explicitly set to `None` instead of using the sampled compensation values.

#### Changes Made
1. Modified the hire event creation loop to include the index in the enumeration
2. Updated the `create_event` call to include the sampled compensation value in the `value_num` parameter

#### Before:
```python
for eid, role, dt, bd_raw, co in zip(new_ids, role_choices, hire_dates, birth_dates, clone_of):
    # ...
    hire_events.append(create_event(
        event_time=dt,
        employee_id=eid,
        event_type=EVT_HIRE,
        value_num=None,  # Compensation not included
        value_json=json.dumps(payload),
        meta=f"Hire event for {eid} in {simulation_year}"
    ))
```

#### After:
```python
for i, (eid, role, dt, bd_raw, co) in enumerate(zip(new_ids, role_choices, hire_dates, birth_dates, clone_of)):
    # ...
    hire_events.append(create_event(
        event_time=dt,
        employee_id=eid,
        event_type=EVT_HIRE,
        value_num=starting_comps[i],  # Now includes sampled compensation
        value_json=json.dumps(payload),
        meta=f"Hire event for {eid} in {simulation_year}"
    ))
```

#### Impact
- Ensures new hire events include their corresponding compensation values
- Eliminates warnings about missing compensation data
- Improves the accuracy of financial calculations in the simulation
- Maintains data consistency between hire events and employee records

## Related Components
- `cost_model.engines.run_one_year.orchestrator`: Processes hire events and creates new employee records
- `cost_model.state.snapshot_update`: Updates the workforce snapshot with new hire data
- `cost_model.state.schema`: Defines the schema for hire events and employee records
