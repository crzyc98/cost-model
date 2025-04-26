import logging

logger = logging.getLogger(__name__)

class BehaviorMixin:
    """Mixin orchestrating the annual agent step, with tracing, error handling, and hooks."""

    def step(self):
        """Execute one simulation year for the agent by invoking mixin logic in sequence."""
        if not getattr(self, 'is_active', False):
            return

        for method_name in (
            '_update_compensation',
            '_update_eligibility',
            '_make_deferral_decision',
            '_calculate_contributions',
            '_determine_status_for_year',
        ):
            self._run_step(method_name)

    def _run_step(self, method_name: str):
        """Internal wrapper to pre/post hook, logging, and error isolation."""
        self._pre_step(method_name)
        logger.debug(f"{self!r}: BEGIN {method_name}")
        try:
            getattr(self, method_name)()
        except Exception as e:
            logger.exception(f"{self!r}: Error in {method_name}: {e}")
            # decide: swallow or re-raise?  for now, swallow and continue
        finally:
            logger.debug(f"{self!r}: END   {method_name}")
            self._post_step(method_name)

    # Hooks you can override in subclasses or at runtime:
    def _pre_step(self, step_name: str):
        """Called right before each step; override for diagnostics or interventions."""
        pass

    def _post_step(self, step_name: str):
        """Called right after each step; override for diagnostics or interventions."""
        pass
