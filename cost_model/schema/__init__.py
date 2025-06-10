"""
Unified schema system for the cost model.

This module provides centralized definitions for all column names, event types,
and data schemas used throughout the cost model. It eliminates inconsistencies
and provides type-safe access to schema elements.

Example Usage:
    >>> from cost_model.schema import SnapshotColumns, EventTypes
    >>> emp_id_col = SnapshotColumns.EMP_ID
    >>> hire_event = EventTypes.HIRE
"""

from .columns import SnapshotColumns, EventColumns, PlanRuleColumns
from .events import EventTypes, EventStatus
from .dtypes import SnapshotDTypes, EventDTypes
from .validation import SchemaValidator, validate_snapshot_schema, validate_event_schema
from .migration import (
    LegacyColumnMapper, 
    migrate_legacy_columns, 
    detect_schema_version, 
    get_migration_recommendations
)

__all__ = [
    # Column definitions
    'SnapshotColumns',
    'EventColumns', 
    'PlanRuleColumns',
    
    # Event definitions
    'EventTypes',
    'EventStatus',
    
    # Data type definitions
    'SnapshotDTypes',
    'EventDTypes',
    
    # Validation utilities
    'SchemaValidator',
    'validate_snapshot_schema',
    'validate_event_schema',
    
    # Migration utilities
    'LegacyColumnMapper',
    'migrate_legacy_columns',
    'detect_schema_version',
    'get_migration_recommendations',
]

# Version information
SCHEMA_VERSION = "2.0.0"
COMPATIBILITY_VERSION = "1.0.0"