from decimal import Decimal
from utils.decimal_helpers import ZERO_DECIMAL  # Shared decimal helper
import pandas as pd  # For handling dates when capturing participation_date

class DeferralMixin:
    """Mixin providing deferral decision logic."""

    def _make_deferral_decision(self):
        """Handles the agent's decision regarding participation and deferral rate.
           Incorporates AE/AI features and voluntary behavioral changes."""
        was_participating = self.is_participating
        is_new_hire_flag = getattr(self, 'is_new_hire', False)

        if is_new_hire_flag:
            print(f"DEBUG NH {self.unique_id} Year {self.model.year}: Entering _make_deferral_decision. Eligible: {self.is_eligible}, Participating: {self.is_participating}, AE Opt-Out: {self.ae_opted_out}, Initial Rate: {self.deferral_rate}")

        # --- Eligibility Check ---
        if not self.is_eligible:
            if was_participating:
                self.is_participating = True
            else:
                self.is_participating = False
                self.deferral_rate = ZERO_DECIMAL
                self.enrollment_method = self.ENROLL_METHOD_NONE
            if is_new_hire_flag:
                print(f"DEBUG NH {self.unique_id} Year {self.model.year}: Ineligible branch. Final Rate: {self.deferral_rate}")
            return

        # --- Agent is Eligible ---
        if is_new_hire_flag:
            print(f"DEBUG NH {self.unique_id} Year {self.model.year}: Eligible branch.")

        ae_enabled = self.model.ae_enabled
        ai_enabled = self.model.ai_enabled
        current_rate = self.deferral_rate

        # --- Decision Logic for Eligible Agents ---
        if not was_participating:
            if is_new_hire_flag:
                print(f"DEBUG NH {self.unique_id} Year {self.model.year}: Entering 'not was_participating' block.")
                print(f"DEBUG NH {self.unique_id} Year {self.model.year}: Was not participating branch.")

            ae_enabled = self.model.ae_enabled
            if is_new_hire_flag:
                print(f"DEBUG NH PRE-AE CHECK {self.unique_id} Year {self.model.year}: ae_enabled={ae_enabled}, self.ae_opted_out={self.ae_opted_out}")
            if ae_enabled and not self.ae_opted_out:
                if is_new_hire_flag:
                    print(f"DEBUG NH {self.unique_id} Year {self.model.year}: AE enabled & not opted out branch.")

                prob_opt_out = self.model.ae_prob_opt_out
                prob_stay_default = self.model.ae_prob_stay_default
                prob_opt_down = self.model.ae_prob_opt_down
                prob_increase_match = self.model.ae_prob_increase_match
                prob_increase_high = self.model.ae_prob_increase_high

                default_rate = self.model.ae_default_rate
                opt_down_target = self.model.ae_opt_down_target
                rate_for_match = self.model.rate_for_max_match
                increase_high_target = self.model.ae_increase_high_target

                if is_new_hire_flag:
                    print(f"DEBUG NH RATES {self.unique_id} Year {self.model.year}: Default Rate = {default_rate}")
                    print(f"DEBUG NH RATES {self.unique_id} Year {self.model.year}: Opt Down Target = {opt_down_target}")
                    print(f"DEBUG NH RATES {self.unique_id} Year {self.model.year}: Rate for Match = {rate_for_match}")
                    print(f"DEBUG NH RATES {self.unique_id} Year {self.model.year}: Increase High Target = {increase_high_target}")

                rand_choice = Decimal(str(self.random.random()))
                new_rate = ZERO_DECIMAL
                self.is_participating = False
                self.enrollment_method = self.ENROLL_METHOD_NONE
                self.ae_opted_out = True
                outcome_desc = "Opt Out"

                cumulative_prob = ZERO_DECIMAL
                if rand_choice <= (cumulative_prob + prob_opt_out):
                    pass
                else:
                    cumulative_prob += prob_opt_out
                    if rand_choice <= (cumulative_prob + prob_stay_default):
                        new_rate = default_rate
                        self.is_participating = True
                        self.enrollment_method = self.ENROLL_METHOD_AE
                        self.ae_opted_out = False
                        outcome_desc = "Stay Default"
                    else:
                        cumulative_prob += prob_stay_default
                        if rand_choice <= (cumulative_prob + prob_opt_down):
                            new_rate = opt_down_target
                            self.is_participating = True
                            self.enrollment_method = self.ENROLL_METHOD_AE
                            self.ae_opted_out = False
                            outcome_desc = "Opt Down"
                        else:
                            cumulative_prob += prob_opt_down
                            if rand_choice <= (cumulative_prob + prob_increase_match):
                                new_rate = rate_for_match
                                self.is_participating = True
                                self.enrollment_method = self.ENROLL_METHOD_AE
                                self.ae_opted_out = False
                                outcome_desc = "Increase to Match"
                            else:
                                new_rate = increase_high_target
                                self.is_participating = True
                                self.enrollment_method = self.ENROLL_METHOD_AE
                                self.ae_opted_out = False
                                outcome_desc = "Increase High"

                self.deferral_rate = new_rate
                if is_new_hire_flag:
                    print(f"DEBUG NH {self.unique_id} Year {self.model.year}: AE Outcome '{outcome_desc}' (Rand: {rand_choice:.4f}). Rate set to {self.deferral_rate}")
            else:
                self.is_participating = False
                self.deferral_rate = ZERO_DECIMAL
                self.enrollment_method = self.ENROLL_METHOD_NONE
                if is_new_hire_flag:
                    print(f"DEBUG NH NO AE {self.unique_id} Year {self.model.year}: AE skipped. Enabled={ae_enabled}, OptedOut={self.ae_opted_out}")
                    print(f"DEBUG NH {self.unique_id} Year {self.model.year}: AE disabled or opted out branch. Final Rate: {self.deferral_rate}")
        elif was_participating:
            pass

        if is_new_hire_flag:
            print(f"DEBUG NH {self.unique_id} Year {self.model.year}: Exiting _make_deferral_decision. Final Rate: {self.deferral_rate}")

        self.is_participating = self.deferral_rate > ZERO_DECIMAL
        # Set participation_date when agent first participates
        if not was_participating and self.is_participating and (self.participation_date is None or pd.isna(self.participation_date)):
            # Auto-enrollment: hire_date + window_days
            if self.enrollment_method == self.ENROLL_METHOD_AE:
                days = self.model.ae_config.get('window_days', 0)
                if self.hire_date and not pd.isna(self.hire_date):
                    self.participation_date = self.hire_date + pd.Timedelta(days=days)
            # Manual enrollment: random within voluntary window
            elif self.enrollment_method == self.ENROLL_METHOD_MANUAL:
                bparams = self.model.scenario_config.get('behavioral_params', {})
                max_days = bparams.get('voluntary_window_days', 180)
                if self.hire_date and not pd.isna(self.hire_date):
                    offset = self.random.randint(0, max_days)
                    self.participation_date = self.hire_date + pd.Timedelta(days=offset)
            else:
                # Fallback: use hire_date
                self.participation_date = self.hire_date
