"""
Event handler registry for managing event processors.

This module provides a registry system for managing different event handlers
and dispatching events to the appropriate processors.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set

from .base import BaseEventProcessor, EventProcessorProtocol

logger = logging.getLogger(__name__)


class EventHandlerRegistry:
    """
    Registry for managing event handlers.

    This class maintains a mapping of event types to their corresponding
    handlers and provides dispatch functionality.
    """

    def __init__(self):
        """Initialize the registry."""
        self._handlers: Dict[str, BaseEventProcessor] = {}
        self._type_mappings: Dict[str, Set[str]] = defaultdict(set)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def register_handler(self, event_type: str, handler: BaseEventProcessor):
        """
        Register an event handler for a specific event type.

        Args:
            event_type: The event type to handle
            handler: The handler instance

        Raises:
            ValueError: If handler doesn't support the event type
            TypeError: If handler doesn't implement the required interface
        """
        # Validate handler interface
        if not isinstance(handler, EventProcessorProtocol):
            raise TypeError(f"Handler must implement EventProcessorProtocol, got {type(handler)}")

        # Validate that handler can handle this event type
        if not handler.can_handle(event_type):
            raise ValueError(
                f"Handler {handler.__class__.__name__} cannot handle event type '{event_type}'"
            )

        # Register the handler
        self._handlers[event_type] = handler

        # Update type mappings
        handler_class = handler.__class__.__name__
        self._type_mappings[handler_class].add(event_type)

        self.logger.info(f"Registered handler {handler_class} for event type '{event_type}'")

    def unregister_handler(self, event_type: str):
        """
        Unregister a handler for a specific event type.

        Args:
            event_type: The event type to unregister
        """
        if event_type in self._handlers:
            handler = self._handlers[event_type]
            handler_class = handler.__class__.__name__

            del self._handlers[event_type]
            self._type_mappings[handler_class].discard(event_type)

            # Clean up empty handler class entries
            if not self._type_mappings[handler_class]:
                del self._type_mappings[handler_class]

            self.logger.info(f"Unregistered handler for event type '{event_type}'")
        else:
            self.logger.warning(f"No handler registered for event type '{event_type}'")

    def get_handler(self, event_type: str) -> Optional[BaseEventProcessor]:
        """
        Get the handler for a specific event type.

        Args:
            event_type: The event type to get handler for

        Returns:
            The handler instance or None if not found
        """
        return self._handlers.get(event_type)

    def has_handler(self, event_type: str) -> bool:
        """
        Check if a handler is registered for an event type.

        Args:
            event_type: The event type to check

        Returns:
            True if a handler is registered
        """
        return event_type in self._handlers

    def get_supported_types(self) -> Set[str]:
        """
        Get all supported event types.

        Returns:
            Set of supported event types
        """
        return set(self._handlers.keys())

    def get_handlers_by_class(self, handler_class_name: str) -> Dict[str, BaseEventProcessor]:
        """
        Get all handlers of a specific class.

        Args:
            handler_class_name: Name of the handler class

        Returns:
            Dictionary mapping event types to handlers
        """
        result = {}
        for event_type in self._type_mappings.get(handler_class_name, set()):
            if event_type in self._handlers:
                result[event_type] = self._handlers[event_type]
        return result

    def get_registry_summary(self) -> Dict[str, any]:
        """
        Get a summary of the current registry state.

        Returns:
            Dictionary with registry summary information
        """
        handler_classes = defaultdict(list)
        for event_type, handler in self._handlers.items():
            handler_classes[handler.__class__.__name__].append(event_type)

        return {
            "total_handlers": len(self._handlers),
            "total_event_types": len(self.get_supported_types()),
            "handler_classes": dict(handler_classes),
            "supported_types": sorted(list(self.get_supported_types())),
        }

    def validate_registry(self) -> List[str]:
        """
        Validate the current registry state.

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Check for duplicate handlers
        handler_instances = defaultdict(list)
        for event_type, handler in self._handlers.items():
            handler_id = id(handler)
            handler_instances[handler_id].append(event_type)

        for handler_id, event_types in handler_instances.items():
            if len(event_types) > 1:
                issues.append(
                    f"Same handler instance registered for multiple event types: {event_types}"
                )

        # Validate handler capabilities
        for event_type, handler in self._handlers.items():
            if not handler.can_handle(event_type):
                issues.append(
                    f"Handler {handler.__class__.__name__} registered for '{event_type}' "
                    f"but claims it cannot handle this type"
                )

        return issues

    def clear_registry(self):
        """Clear all registered handlers."""
        cleared_count = len(self._handlers)
        self._handlers.clear()
        self._type_mappings.clear()
        self.logger.info(f"Cleared registry, removed {cleared_count} handlers")

    def copy_from_registry(self, other_registry: "EventHandlerRegistry"):
        """
        Copy handlers from another registry.

        Args:
            other_registry: Registry to copy from
        """
        for event_type, handler in other_registry._handlers.items():
            self.register_handler(event_type, handler)

        self.logger.info(f"Copied {len(other_registry._handlers)} handlers from other registry")

    def __len__(self) -> int:
        """Get number of registered handlers."""
        return len(self._handlers)

    def __contains__(self, event_type: str) -> bool:
        """Check if event type is supported."""
        return event_type in self._handlers

    def __repr__(self) -> str:
        """String representation of registry."""
        return (
            f"EventHandlerRegistry("
            f"handlers={len(self._handlers)}, "
            f"types={sorted(list(self.get_supported_types()))}"
            f")"
        )
