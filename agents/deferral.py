import logging
from decimal import Decimal, InvalidOperation
import pandas as pd  # type: ignore[import]
from utils.decimal_helpers import ZERO_DECIMAL  # Shared decimal helper
from typing import Protocol, Any, Optional

logger = logging.getLogger(__name__)


class ModelProtocol(Protocol):
    year: int
    scenario_config: dict
    random: Any


class DeferralMixin:
    """Mixin providing deferral decision logic (AE, AI, voluntary)

    Required on self:
      - is_eligible: bool
      - is_participating: bool
      - deferral_rate: Decimal or float
      - hire_date: pd.Timestamp
      - participation_date: Optional[pd.Timestamp]
      - unique_id: Any
      - random: Random-like
      - model: ModelProtocol
    """
    # Mypy attribute declarations
    model: ModelProtocol
    deferral_rate: Decimal
    is_participating: bool
    is_eligible: bool
    hire_date: pd.Timestamp
    participation_date: Optional[pd.Timestamp]
    unique_id: Any
    random: Any
    enrollment_method: str

    ENROLL_METHOD_NONE = 'none'
    ENROLL_METHOD_AE = 'ae'
    ENROLL_METHOD_MANUAL = 'manual'

    def _make_deferral_decision(
        self,
        model: Optional[ModelProtocol] = None
    ) -> None:
        """Orchestrate AE, AI, and voluntary changes using scenario config."""
        if model is None:
            model = self.model

        self.deferral_rate = self._safe_decimal(
            self.deferral_rate,
            'deferral_rate'
        )

        if not self.is_eligible:
            self._handle_ineligible()
            return

        rules = model.scenario_config.get('plan_rules', {})
        ae_cfg = rules.get('auto_enrollment', {})
        ai_cfg = rules.get('auto_increase', {})

        # Auto-enrollment
        if not self.is_participating:
            if ae_cfg.get('enabled', True):
                self._handle_auto_enroll(ae_cfg, model)
            else:
                logger.debug("%r AE disabled or opted out", self)
                self._reset_participation()

        # Auto-increase
        if self.is_participating and ai_cfg.get('enabled', False):
            self._handle_auto_increase(ai_cfg)

        # Voluntary
        if self.is_participating:
            self._handle_voluntary_change(model)

        self.is_participating = self.deferral_rate > ZERO_DECIMAL

        if self.is_participating and not self.participation_date:
            self._assign_participation_date(ae_cfg, model)

    def _handle_ineligible(self) -> None:
        logger.debug(
            "%r is not eligible for deferral; resetting participation",
            self
        )
        self._reset_participation()

    def _handle_auto_enroll(
        self,
        ae_cfg: dict,
        model: ModelProtocol
    ) -> None:
        """Perform AE decision based on centralized config."""
        dist = ae_cfg.get('outcome_distribution', {})
        try:
            keys = [
                'prob_opt_out', 'prob_stay_default',
                'prob_opt_down', 'prob_increase_to_match',
                'prob_increase_high'
            ]
            probs = [Decimal(str(dist.get(k, 0))) for k in keys]
        except Exception as e:
            logger.warning(
                "%r: Error parsing AE outcome_distribution: %s",
                self,
                e
            )
            probs = [Decimal('0')] * 5

        total = sum(probs)
        if total > 1:
            logger.warning(
                '%r AE probabilities sum to >1: %s',
                self,
                total
            )

        rand = Decimal(str(self.random.random()))
        cum = ZERO_DECIMAL
        choices = [
            ('opt_out', ZERO_DECIMAL),
            (
                'stay_default',
                Decimal(str(ae_cfg.get('default_rate', 0)))
            ),
            (
                'opt_down',
                Decimal(str(ae_cfg.get('opt_down_target_rate', 0)))
            ),
            (
                'inc_match',
                Decimal(str(ae_cfg.get('match_target', 0)))
            ),
            (
                'inc_high',
                Decimal(
                    str(ae_cfg.get('increase_high_target_rate', 0))
                )
            )
        ]
        methods = {
            'opt_out':   (self.ENROLL_METHOD_NONE, False),
            'stay_default': (self.ENROLL_METHOD_AE, True),
            'opt_down':  (self.ENROLL_METHOD_AE, True),
            'inc_match': (self.ENROLL_METHOD_AE, True),
            'inc_high':  (self.ENROLL_METHOD_AE, True)
        }

        for prob, (name, rate) in zip(probs, choices):
            if rand <= cum + prob:
                method, partake = methods[name]
                self.enrollment_method = method
                self.is_participating = partake
                self.deferral_rate = rate
                logger.debug(
                    "%r AE choice=%s rate=%s rand=%s",
                    self, name, rate, rand
                )
                return

            cum += prob

        logger.debug("%r AE fallback to opt_out", self)
        self._reset_participation()

    def _handle_auto_increase(
        self,
        ai_cfg: dict
    ) -> None:
        """Increase deferral_rate by step_rate, capped at cap_rate."""
        step = Decimal(str(ai_cfg.get('increase_rate', 0)))
        cap = Decimal(str(ai_cfg.get('cap_rate', 1)))
        new_rate = min(self.deferral_rate + step, cap)
        logger.debug(
            "%r AI: bumping deferral_rate from %s to %s (step=%s, cap=%s)",
            self,
            self.deferral_rate,
            new_rate,
            step,
            cap,
        )
        self.deferral_rate = new_rate

    def _handle_voluntary_change(
        self,
        model: ModelProtocol
    ) -> None:
        """Handle voluntary change stub."""
        bp = (
            model.scenario_config.get('plan_rules', {})
            .get('behavioral_params', {})
        )
        logger.debug(
            "%r: No voluntary logic implemented yet "
            "(behavioral_params=%s)",
            self,
            bp,
        )

    def _reset_participation(self) -> None:
        self.is_participating = False
        self.deferral_rate = ZERO_DECIMAL
        self.enrollment_method = self.ENROLL_METHOD_NONE

    def _assign_participation_date(
        self,
        ae_cfg: dict,
        model: ModelProtocol
    ) -> None:
        hd = getattr(self, 'hire_date', None)
        window_days = int(ae_cfg.get('window_days', 0))
        if (
            self.enrollment_method == self.ENROLL_METHOD_AE
            and hd is not None
        ):
            self.participation_date = hd + pd.Timedelta(
                days=window_days
            )
        elif (
            self.enrollment_method == self.ENROLL_METHOD_MANUAL
            and hd is not None
        ):
            offset = self.random.randint(0, 180)
            self.participation_date = hd + pd.Timedelta(
                days=offset
            )
        else:
            self.participation_date = hd

    def _safe_decimal(
        self,
        value: Any,
        name: str
    ) -> Decimal:
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError) as e:
            logger.warning(
                "%r: Invalid decimal for %s: %s",
                self,
                name,
                e
            )
            return ZERO_DECIMAL
