class BehaviorMixin:
    """Mixin orchestrating the annual agent step."""

    def step(self):
        """Execute one simulation year for the agent by invoking mixin logic in sequence."""
        # Skip if not active
        if not getattr(self, 'is_active', False):
            return

        # 1. Compensation update
        self._update_compensation()
        # 2. Eligibility update
        self._update_eligibility()
        # 3. Deferral decision
        self._make_deferral_decision()
        # 4. Contribution calculation
        self._calculate_contributions()
        # 5. Finalize status for reporting
        self._determine_status_for_year()
