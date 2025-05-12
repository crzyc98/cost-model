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