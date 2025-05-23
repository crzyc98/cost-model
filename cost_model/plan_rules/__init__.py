"""
Plan rules package: Exports loader for plan rule configuration for projections and engines.
"""

def load_plan_rules(config_ns):
    """
    Loads plan rules from the config namespace or returns an empty dict if not present.
    Handles SimpleNamespace and dict types.
    """
    if hasattr(config_ns, 'plan_rules'):
        plan_rules = config_ns.plan_rules
        if isinstance(plan_rules, dict):
            return plan_rules
        # Handle SimpleNamespace
        return vars(plan_rules)
    return {}

# Re-export public API if needed
