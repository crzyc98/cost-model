import mesa
import pandas as pd
from dateutil.relativedelta import relativedelta
from decimal import Decimal, ROUND_HALF_UP # For precision
import logging # Ensure logging is imported
from .state import StateMixin
from utils.decimal_helpers import ZERO_DECIMAL  # Shared decimal helper
from .behavior import BehaviorMixin
from .compensation import CompensationMixin
from .eligibility import EligibilityMixin
from .deferral import DeferralMixin
from .contributions import ContributionsMixin

class EmployeeAgent(BehaviorMixin, StateMixin, CompensationMixin, EligibilityMixin, DeferralMixin, ContributionsMixin, mesa.Agent):
    """An agent representing an employee in the retirement plan simulation."""

    def __init__(self, unique_id: int, model: "RetirementPlanModel", initial_state: dict) -> None:
        """
        Create a new employee agent.
        Args:
            unique_id: Unique identifier for the agent.
            model: The model instance the agent belongs to.
            initial_state: Dictionary containing initial attributes from the census.
        """
        super().__init__(unique_id, model)
        self.logger = logging.getLogger(f"EmployeeAgent.{unique_id}")

        # Core Demographics & Status
        self.birth_date = initial_state.get('birth_date')
        self.hire_date = initial_state.get('hire_date')
        self.termination_date = initial_state.get('termination_date', None)
        self.participation_date = initial_state.get('participation_date', None)
        self.status = initial_state.get('status', 'Active')
        self.role = initial_state.get('role')
        # Ensure gross_compensation is Decimal for arithmetic
        self.gross_compensation = Decimal(str(initial_state.get('gross_compensation', 0))).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP
        )
        self.employment_status = None
        self.hire_year = pd.to_datetime(self.hire_date).year if self.hire_date else None
        self.is_new_hire = False  # Flag to track new hires separately
        # Initialize active flag and employment_status
        self._initialize_employment_status()

        # Plan-Specific State
        self.is_eligible: bool = False
        initial_deferral_percentage = initial_state.get('pre_tax_deferral_percentage', 0.0)
        self.deferral_rate: Decimal = Decimal(str(initial_deferral_percentage)) / Decimal('100.0')
        self.is_participating: bool = self.deferral_rate > ZERO_DECIMAL
        # Estimate participation_date for initial participants if not provided
        if self.is_participating and (self.participation_date is None or pd.isna(self.participation_date)):
            # Sample first contribution date between hire_date and end-of-year before simulation starts
            baseline = pd.Timestamp(year=self.model.start_year-1, month=12, day=31)
            tenure_days = (baseline - self.hire_date).days if self.hire_date and not pd.isna(self.hire_date) else 0
            offset = self.model.random.randint(0, tenure_days) if tenure_days > 0 else 0
            self.participation_date = self.hire_date + pd.Timedelta(days=offset)
        self.enrollment_method: str = initial_state.get('enrollment_method', 'None')
        self.ae_opted_out: bool = str(initial_state.get('ae_opted_out', 'False')).lower() in ('true', '1', 't', 'y', 'yes')
        self.ai_opted_out: bool = str(initial_state.get('ai_opted_out', 'False')).lower() in ('true', '1', 't', 'y', 'yes')
        self.behavioral_profile: str = initial_state.get('behavioral_profile', 'Default')
        self.contributions_current_year: dict = {}
        self.prorated_compensation_for_reporting: Decimal = ZERO_DECIMAL

    # All state logic moved to StateMixin and BehaviorMixin
