from pydantic import BaseModel, Field
from typing import List, Dict, Optional


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


class EligibilityEventsConfig(BaseModel):
    # e.g. milestone_months: List[int], event_type: str, ...
    milestone_months: Optional[list[int]] = None
    event_type: Optional[str] = None

class ContributionIncreaseConfig(BaseModel):
    # e.g. increase_pct: float, eligible_roles: List[str], …
    increase_pct: Optional[float] = None
    eligible_roles: Optional[list[str]] = None

class ProactiveDecreaseConfig(BaseModel):
    # e.g. decrease_pct: float, irs_limit_pct: float, …
    decrease_pct: Optional[float] = None
    irs_limit_pct: Optional[float] = None

"""
Definition of Pydantic config models for Plan Rules engines.
Each config model specifies parameters needed by its corresponding engine stub.
"""

class EligibilityEventsConfig(BaseModel):
    """
    Configuration for milestone-based eligibility events.

    Attributes:
        milestone_months: List of service-month milestones (in months) at which to emit events.
        milestone_years: List of service-year milestones (in years) at which to emit events (converted to months).
        event_type_map: Mapping from each milestone month to the event_type string to emit.
    """
    milestone_months: List[int] = Field(
        ..., description="Service month milestones for eligibility events (e.g., [12, 24, 36])."
    )
    milestone_years: List[int] = Field(
        default_factory=list, description="Service year milestones for eligibility events (e.g., [1, 5, 10])."
    )
    event_type_map: Dict[int, str] = Field(
        ..., description="Mapping from milestone month to event_type (e.g., {12: 'EVT_1YR_ANNIV'})."
    )


class ContributionIncreaseConfig(BaseModel):
    """
    Configuration for contribution increase event emission.

    Attributes:
        min_increase_pct: Minimum increase in deferral rate (as a decimal, e.g., 0.01 for 1%) required to trigger the event.
        event_type: Event type string to emit (default: 'EVT_CONTRIB_INCREASE').
    """
    min_increase_pct: float = Field(
        ..., description="Minimum pct increase to trigger event (e.g., 0.01 for 1%)."
    )
    event_type: str = Field(
        "EVT_CONTRIB_INCREASE", description="Event type to emit."
    )

class ProactiveDecreaseConfig(BaseModel):
    """
    Configuration for proactive rate decreases to respect contribution limits.

    Attributes:
        lookback_months: Number of months to look back for contributions.
        threshold_pct: Minimum drop from historical high to current rate to trigger event.
        event_type: Event type string to emit.
    """
    lookback_months: int = Field(
        ..., description="Window (in months) for averaging contributions to decide decrease."
    )
    threshold_pct: float = Field(
        ..., description="Minimum drop from high-water mark to current rate to trigger decrease (e.g., 0.05 for 5%)."
    )
    event_type: str = Field(
        "EVT_PROACTIVE_DECREASE", description="Event type to emit."
    )

