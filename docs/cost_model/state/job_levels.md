
This module defines the job level taxonomy and provides utilities for managing
and validating job levels within the cost model system.

## QuickStart

To use job levels in your code:

```python
from cost_model.state.job_levels import (
    EMP_LEVEL,
    JobLevel,
    LEVEL_TAXONOMY,
    get_level_by_id,
    validate_level
)

# Access a specific level
manager_level = LEVEL_TAXONOMY[2]
print(f"Manager level description: {manager_level.description}")

# Validate a level
is_valid = validate_level(3)  # Returns True

# Get all levels
all_levels = get_all_levels()
for level in all_levels:
    print(f"Level {level.level_id}: {level.name}")
```