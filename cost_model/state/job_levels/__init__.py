__all__ = ['JobLevel', 'ConfigError', 'init_job_levels', 'refresh_job_levels', 'LEVEL_TAXONOMY', '_COMP_INTERVALS',
           '_WARNING_COUNTS', 'MAX_WARNINGS', 'assign_levels_to_dataframe', 'get_level_by_compensation', 
           'get_level_distribution', 'sample_new_hire_compensation', 'sample_new_hires_vectorized', 
           'sample_mixed_new_hires', 'load_from_yaml', 'load_job_levels_from_config', 'get_warning_counts']

from .models import JobLevel, ConfigError
from .utils import assign_levels_to_dataframe
from .assign import get_level_by_compensation, get_level_distribution
from .sampling import sample_new_hire_compensation, sample_new_hires_vectorized, sample_mixed_new_hires
from .loader import load_from_yaml, load_job_levels_from_config
from .init import init_job_levels, refresh_job_levels, get_warning_counts

# Initialize module-level state
init_job_levels(reset_warnings=True)

# Expose state variables
import cost_model.state.job_levels.state as _state
LEVEL_TAXONOMY = _state.LEVEL_TAXONOMY
_COMP_INTERVALS = _state._COMP_INTERVALS
_WARNING_COUNTS = _state._WARNING_COUNTS
MAX_WARNINGS = _state.MAX_WARNINGS
