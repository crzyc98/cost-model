# Make loaders a package

from .scenario_loader import load as load_scenarios
from .mc_loader import load_mc_package

__all__ = ['load_scenarios', 'load_mc_package']
