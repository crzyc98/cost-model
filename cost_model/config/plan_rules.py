from pydantic import BaseModel
from typing import Optional

class EligibilityConfig(BaseModel):
    min_age: int = 21
    min_service_months: int = 12

class AutoEnrollmentConfig(BaseModel):
    enabled: bool = False
    default_rate: float = 0.03
    window_days: int = 90

class EnrollmentConfig(BaseModel):
    window_days: int = 30
    allow_opt_out: bool = True
    default_rate: float = 0.05
    auto_enrollment: Optional[AutoEnrollmentConfig] = None
    voluntary_enrollment_rate: float = 0.0

class ContributionConfig(BaseModel):
    default_deferral_rate: float = 0.06
    max_deferral_rate: float = 0.15
    employer_match_pct: float = 0.5

class AutoIncreaseConfig(BaseModel):
    increase_pct: float = 0.01
    cap: float = 0.10
    frequency_years: int = 1
