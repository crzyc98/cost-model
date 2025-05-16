from typing import Dict
from .models import JobLevel

# Default job level definitions
DEFAULT_LEVELS: Dict[int, JobLevel] = {
    0: JobLevel(
        level_id=0,
        name="Hourly",
        description="Non-exempt employees paid hourly wages",
        min_compensation=0,
        max_compensation=55000,
        comp_base_salary=40000,
        comp_age_factor=0.005,
        comp_stochastic_std_dev=0.1,
        avg_annual_merit_increase=0.03,
        promotion_probability=0.08,
        target_bonus_percent=0.0
    ),
    1: JobLevel(
        level_id=1,
        name="Staff",
        description="Entry-level exempt employees, individual contributors",
        min_compensation=56000,
        max_compensation=80000,
        comp_base_salary=65000,
        comp_age_factor=0.006,
        comp_stochastic_std_dev=0.1,
        avg_annual_merit_increase=0.035,
        promotion_probability=0.12,
        target_bonus_percent=0.05
    ),
    2: JobLevel(
        level_id=2,
        name="Manager",
        description="First-level management with direct reports",
        min_compensation=81000,
        max_compensation=120000,
        comp_base_salary=95000,
        comp_age_factor=0.007,
        comp_stochastic_std_dev=0.1,
        avg_annual_merit_increase=0.04,
        promotion_probability=0.08,
        target_bonus_percent=0.15
    ),
    3: JobLevel(
        level_id=3,
        name="SrMgr",
        description="Mid-level management, often managing other managers",
        min_compensation=121000,
        max_compensation=160000,
        comp_base_salary=135000,
        comp_age_factor=0.008,
        comp_stochastic_std_dev=0.1,
        avg_annual_merit_increase=0.045,
        promotion_probability=0.05,
        target_bonus_percent=0.25
    ),
    4: JobLevel(
        level_id=4,
        name="Exec",
        description="Senior leadership roles (VP, C-suite, etc.)",
        min_compensation=161000,
        max_compensation=10000000,  # Set to $10M to accommodate highly compensated executives
        comp_base_salary=200000,
        comp_age_factor=0.01,
        comp_stochastic_std_dev=0.15,
        avg_annual_merit_increase=0.05,
        promotion_probability=0.02,
        target_bonus_percent=0.40
    )
}
