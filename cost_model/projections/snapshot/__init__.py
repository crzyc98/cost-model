"""
Refactored snapshot processing module.

This module provides a clean, modular interface for snapshot creation and processing.
It maintains backward compatibility while offering improved maintainability,
testability, and performance monitoring.

The refactoring breaks down the original monolithic functions into focused,
reusable components following the Extract Method and Extract Class patterns.

## Architecture Overview

The refactored system follows these design principles:

1. **Single Responsibility Principle**: Each class has a focused responsibility
2. **Dependency Injection**: Components accept configuration for flexibility  
3. **Type Safety**: Comprehensive type hints prevent runtime errors
4. **Performance Monitoring**: Built-in timing and memory tracking
5. **Structured Logging**: Detailed, contextual logging throughout

## Core Components

- **CensusLoader**: Handles loading and preprocessing census data
- **SnapshotTransformer**: Provides data transformation utilities
- **SnapshotValidator**: Comprehensive data validation capabilities
- **YearlySnapshotProcessor**: Complex yearly snapshot logic
- **ContributionsProcessor**: Plan rule calculations
- **StatusProcessor**: Employee status determination

## Key Features

- **Modular Design**: Easy to test, maintain, and extend
- **Performance Monitoring**: Built-in timing decorators and memory tracking
- **Type Safety**: Comprehensive TypedDict definitions and type hints
- **Error Handling**: Custom exception hierarchy with detailed context
- **Structured Logging**: Contextual logging with performance metrics
- **Backward Compatibility**: Drop-in replacement for existing functions

## Usage Examples

```python
# Basic usage - create initial snapshot
from cost_model.projections.snapshot import create_initial_snapshot

snapshot = create_initial_snapshot(
    start_year=2025,
    census_path="data/census_preprocessed.parquet"
)

# Enhanced yearly snapshot
from cost_model.projections.snapshot import build_enhanced_yearly_snapshot

yearly_snapshot = build_enhanced_yearly_snapshot(
    start_of_year_snapshot=soy_snapshot,
    end_of_year_snapshot=eoy_snapshot, 
    year_events=events_df,
    simulation_year=2025
)

# Custom configuration
from cost_model.projections.snapshot.models import SnapshotConfig
from cost_model.projections.snapshot.census_loader import CensusLoader

config = SnapshotConfig(
    start_year=2025,
    enable_validation=True,
    enable_timing=True,
    log_level="DEBUG"
)

loader = CensusLoader(config)
```

## Performance Features

- **Timing Decorators**: Automatic function timing with detailed metrics
- **Memory Tracking**: Optional memory usage monitoring (requires psutil)
- **Progress Indicators**: For long-running operations
- **Performance Checkpoints**: Track progress through complex operations
- **Structured Metrics**: JSON-compatible performance data

## Type Safety

The module provides comprehensive type definitions:

- **Core Types**: EmployeeId, SimulationYear, CompensationAmount, etc.
- **Data Structures**: TypedDict definitions for all major data structures
- **Protocols**: Interface definitions for component contracts
- **Union Types**: Flexible parameter types where appropriate

## Error Handling

Custom exception hierarchy provides detailed error context:

- **SnapshotError**: Base exception class
- **CensusDataError**: Census data loading/validation issues
- **ValidationError**: Data validation failures  
- **SnapshotBuildError**: Snapshot construction errors
- **ConfigurationError**: Configuration-related problems

Each exception includes rich context for debugging:
- Operation name and processing step
- Employee ID and simulation year (when relevant)
- Data shape and additional diagnostic information
"""

from typing import Optional
import pandas as pd
from datetime import datetime

from .builder import create_initial_snapshot, build_enhanced_yearly_snapshot
from .event_processor import EventProcessor
from .consolidator import consolidate_snapshots_to_parquet
from .types import SimulationYear, FilePath

# Create a default event processor instance for backward compatibility
_default_event_processor = EventProcessor()

def update_snapshot_with_events(
    prev_snapshot: pd.DataFrame, 
    events: pd.DataFrame, 
    target_date: datetime
) -> pd.DataFrame:
    """Backward compatibility wrapper for update_snapshot_with_events.
    
    Args:
        prev_snapshot: Previous snapshot DataFrame
        events: Events DataFrame to apply
        target_date: Target date for snapshot update
        
    Returns:
        Updated snapshot DataFrame
    """
    return _default_event_processor.update_snapshot_with_events(prev_snapshot, events, target_date)

__all__ = [
    'create_initial_snapshot',
    'build_enhanced_yearly_snapshot', 
    'update_snapshot_with_events',
    'consolidate_snapshots_to_parquet'
]