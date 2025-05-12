# cost_model/projections/config.py
"""
Configuration loading and conversion utilities.

## QuickStart

To load and manipulate configurations programmatically:

```python
from pathlib import Path
from cost_model.projections.config import load_config_to_namespace, dict_to_simplenamespace

# Load a configuration file into a SimpleNamespace
config_path = 'config/my_scenario.yaml'
config = load_config_to_namespace(config_path)

# Access configuration values using dot notation
start_year = config.global_parameters.start_year
projection_years = config.global_parameters.projection_years
print(f"Projecting from {start_year} for {projection_years} years")

# Check if a configuration section exists
has_auto_enroll = hasattr(config.plan_rules, 'auto_enrollment')
if has_auto_enroll:
    auto_enroll_rate = config.plan_rules.auto_enrollment.default_rate
    print(f"Auto-enrollment default rate: {auto_enroll_rate:.1%}")

# Modify configuration programmatically
config.global_parameters.random_seed = 12345
config.plan_rules.employer_match.tiers[0].employer_match_rate = 0.75

# Create a configuration from a dictionary
modified_config = {
    'global_parameters': {
        'start_year': 2025,
        'projection_years': 10,
        'random_seed': 42
    },
    'plan_rules': {
        'auto_enrollment': {
            'enabled': True,
            'default_rate': 0.05
        }
    }
}
config_ns = dict_to_simplenamespace(modified_config)
```

This allows you to load, inspect, and modify configuration settings programmatically.
"""

# YAML loading + namespace conversion
import yaml
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Union


def dict_to_simplenamespace(d: Any) -> Any:
    """Recursively convert a dictionary to a SimpleNamespace."""
    if isinstance(d, dict):
        kwargs = {str(k): dict_to_simplenamespace(v) for k, v in d.items()}
        return SimpleNamespace(**kwargs)
    elif isinstance(d, list):
        return [dict_to_simplenamespace(item) for item in d]
    return d


def load_config_to_namespace(config_path: Union[Path, str]) -> SimpleNamespace:
    """Load a YAML configuration file and convert it to a SimpleNamespace."""
    if isinstance(config_path, str):
        config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return dict_to_simplenamespace(config_dict)
