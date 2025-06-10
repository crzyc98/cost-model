# Snapshot Refactoring Implementation Roadmap

## Immediate Action Plan (Next 2 Weeks)

### Priority 1: Critical Issues Resolution (Days 1-3)

#### Day 1: Schema Unification
**Goal**: Eliminate column naming inconsistencies

```python
# Create cost_model/schema/__init__.py
from enum import Enum
from typing import Dict, Any

class SnapshotColumns(Enum):
    """Centralized column definitions"""
    EMP_ID = "employee_id"
    EMP_HIRE_DATE = "employee_hire_date"  
    EMP_BIRTH_DATE = "employee_birth_date"
    EMP_GROSS_COMP = "employee_gross_compensation"
    EMP_TERM_DATE = "employee_termination_date"
    EMP_ACTIVE = "active"
    EMP_DEFERRAL_RATE = "employee_deferral_rate"
    # ... rest of columns

class EventTypes(Enum):
    """Event type definitions"""
    HIRE = "hire"
    TERMINATION = "termination"
    COMPENSATION = "compensation"
    PROMOTION = "promotion"
    # ... rest of event types

# Backward compatibility mapping
LEGACY_COLUMN_MAP = {
    "EMP_ID": SnapshotColumns.EMP_ID.value,
    "EMP_HIRE_DATE": SnapshotColumns.EMP_HIRE_DATE.value,
    # ... complete mapping
}
```

**Tasks**:
- [ ] Create unified schema module
- [ ] Update all existing modules to use schema constants
- [ ] Create migration utility for legacy column names
- [ ] Add schema validation utilities

#### Day 2: Event Processing Consolidation
**Goal**: Create unified event handling

```python
# Create cost_model/events/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd

class EventProcessor(ABC):
    """Base class for all event processors"""
    
    @abstractmethod
    def supported_events(self) -> List[str]:
        """Return list of supported event types"""
        pass
    
    @abstractmethod
    def process_events(
        self, 
        snapshot: pd.DataFrame, 
        events: pd.DataFrame
    ) -> pd.DataFrame:
        """Process events and return updated snapshot"""
        pass
    
    @abstractmethod
    def validate_events(self, events: pd.DataFrame) -> List[str]:
        """Validate events and return error messages"""
        pass

class EventProcessingEngine:
    """Orchestrates event processing"""
    
    def __init__(self):
        self.processors: Dict[str, EventProcessor] = {}
    
    def register_processor(self, event_type: str, processor: EventProcessor):
        self.processors[event_type] = processor
    
    def process_all_events(
        self, 
        snapshot: pd.DataFrame, 
        events: pd.DataFrame
    ) -> pd.DataFrame:
        # Group events by type and process in order
        pass
```

**Tasks**:
- [ ] Extract event processing logic from existing modules
- [ ] Create base event processor interface
- [ ] Implement specific processors for each event type
- [ ] Create event processing pipeline

#### Day 3: Snapshot Update Simplification
**Goal**: Break down 708-line update function

```python
# Create cost_model/snapshots/updaters/pipeline.py
from typing import List, Protocol
from dataclasses import dataclass

@dataclass
class UpdateContext:
    """Context passed through update pipeline"""
    snapshot: pd.DataFrame
    events: pd.DataFrame
    year: int
    metadata: Dict[str, Any]

class UpdateStep(Protocol):
    """Protocol for update pipeline steps"""
    def execute(self, context: UpdateContext) -> UpdateContext: ...

class SnapshotUpdatePipeline:
    """Manages snapshot update through pipeline of steps"""
    
    def __init__(self):
        self.steps: List[UpdateStep] = []
    
    def add_step(self, step: UpdateStep):
        self.steps.append(step)
    
    def execute_update(self, context: UpdateContext) -> pd.DataFrame:
        for step in self.steps:
            context = step.execute(context)
        return context.snapshot

# Example update steps
class EmployeeReconciliationStep:
    def execute(self, context: UpdateContext) -> UpdateContext:
        # Handle new hires and terminations
        pass

class CompensationUpdateStep:
    def execute(self, context: UpdateContext) -> UpdateContext:
        # Update compensation based on events
        pass

class TenureCalculationStep:
    def execute(self, context: UpdateContext) -> UpdateContext:
        # Calculate tenure for all employees
        pass
```

**Tasks**:
- [ ] Identify distinct update operations in current function
- [ ] Create pipeline-based update system
- [ ] Implement individual update steps
- [ ] Add comprehensive error handling and rollback

### Priority 2: Performance Optimization (Days 4-7)

#### Day 4-5: Vectorization Implementation
**Goal**: Replace iterative operations with vectorized ones

```python
# Create cost_model/optimizations/vectorized.py
import pandas as pd
import numpy as np
from typing import Optional

class VectorizedCalculations:
    """High-performance vectorized calculations"""
    
    @staticmethod
    def calculate_tenure_vectorized(
        hire_dates: pd.Series, 
        reference_date: pd.Timestamp,
        term_dates: Optional[pd.Series] = None
    ) -> pd.Series:
        """Vectorized tenure calculation"""
        hire_dates = pd.to_datetime(hire_dates)
        
        if term_dates is not None:
            # For terminated employees, use termination date
            term_dates = pd.to_datetime(term_dates)
            end_dates = term_dates.where(term_dates.notna(), reference_date)
        else:
            end_dates = reference_date
        
        tenure_days = (end_dates - hire_dates).dt.days
        return (tenure_days / 365.25).clip(lower=0)
    
    @staticmethod
    def calculate_age_vectorized(
        birth_dates: pd.Series,
        reference_date: pd.Timestamp
    ) -> pd.Series:
        """Vectorized age calculation"""
        birth_dates = pd.to_datetime(birth_dates)
        age_days = (reference_date - birth_dates).dt.days
        return (age_days / 365.25).clip(lower=0, upper=100)
    
    @staticmethod
    def apply_tenure_bands_vectorized(tenure_years: pd.Series) -> pd.Series:
        """Vectorized tenure band assignment"""
        conditions = [
            tenure_years < 1,
            tenure_years < 5,
            tenure_years < 15,
            tenure_years < 25
        ]
        choices = ['NEW_HIRE', 'EARLY_CAREER', 'MID_CAREER', 'SENIOR']
        return pd.Series(np.select(conditions, choices, default='VETERAN'), 
                        index=tenure_years.index)
```

**Tasks**:
- [ ] Replace all iterative calculations with vectorized versions
- [ ] Implement efficient data type usage (categorical for enums)
- [ ] Add memory usage monitoring
- [ ] Benchmark performance improvements

#### Day 6-7: Caching and Memory Optimization
**Goal**: Implement intelligent caching and memory management

```python
# Create cost_model/optimizations/caching.py
from functools import wraps, lru_cache
from typing import Callable, Any, Dict
import hashlib
import pickle

class SnapshotCache:
    """Intelligent caching for expensive snapshot operations"""
    
    def __init__(self, max_size: int = 128):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from function arguments"""
        key_data = pickle.dumps((args, sorted(kwargs.items())))
        return hashlib.md5(key_data).hexdigest()
    
    def cached_calculation(self, func: Callable) -> Callable:
        """Decorator for caching expensive calculations"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = self._get_cache_key(*args, **kwargs)
            
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            result = func(*args, **kwargs)
            
            if len(self.cache) >= self.max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = result
            return result
        
        return wrapper

# Memory optimization utilities
class MemoryOptimizer:
    """Memory optimization utilities for large DataFrames"""
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency"""
        optimized = df.copy()
        
        for col in optimized.columns:
            col_type = optimized[col].dtype
            
            if col_type == 'object':
                # Convert to categorical if few unique values
                unique_ratio = optimized[col].nunique() / len(optimized)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    optimized[col] = optimized[col].astype('category')
            
            elif col_type == 'int64':
                # Downcast integers if possible
                optimized[col] = pd.to_numeric(optimized[col], downcast='integer')
            
            elif col_type == 'float64':
                # Downcast floats if possible
                optimized[col] = pd.to_numeric(optimized[col], downcast='float')
        
        return optimized
```

### Priority 3: Architecture Improvements (Days 8-14)

#### Day 8-10: Modular Event Handlers
**Goal**: Implement specific event handlers

```python
# Create cost_model/events/handlers/hire.py
from ..base import EventProcessor
from ...schema import EventTypes, SnapshotColumns
import pandas as pd

class HireEventProcessor(EventProcessor):
    """Processes hire events"""
    
    def supported_events(self) -> List[str]:
        return [EventTypes.HIRE.value]
    
    def process_events(
        self, 
        snapshot: pd.DataFrame, 
        events: pd.DataFrame
    ) -> pd.DataFrame:
        """Add new hires to snapshot"""
        hire_events = events[
            events['event_type'] == EventTypes.HIRE.value
        ]
        
        if hire_events.empty:
            return snapshot
        
        # Get new employee IDs not in current snapshot
        existing_ids = set(snapshot[SnapshotColumns.EMP_ID.value])
        new_hires = hire_events[
            ~hire_events[SnapshotColumns.EMP_ID.value].isin(existing_ids)
        ]
        
        if new_hires.empty:
            return snapshot
        
        # Create new employee records
        new_records = self._create_new_hire_records(new_hires)
        
        # Append to snapshot
        return pd.concat([snapshot, new_records], ignore_index=True)
    
    def _create_new_hire_records(self, hire_events: pd.DataFrame) -> pd.DataFrame:
        """Create new employee records from hire events"""
        # Implementation here
        pass
    
    def validate_events(self, events: pd.DataFrame) -> List[str]:
        """Validate hire events"""
        errors = []
        
        # Check required fields
        required_fields = [
            SnapshotColumns.EMP_ID.value,
            SnapshotColumns.EMP_HIRE_DATE.value,
            SnapshotColumns.EMP_GROSS_COMP.value
        ]
        
        for field in required_fields:
            if field not in events.columns:
                errors.append(f"Missing required field: {field}")
            elif events[field].isna().any():
                errors.append(f"Null values found in required field: {field}")
        
        return errors
```

**Tasks**:
- [ ] Implement hire event processor
- [ ] Implement termination event processor  
- [ ] Implement compensation event processor
- [ ] Implement promotion event processor
- [ ] Add comprehensive validation for each event type

#### Day 11-12: Update Pipeline Implementation
**Goal**: Create production-ready update pipeline

```python
# Create cost_model/snapshots/engine.py
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

@dataclass
class UpdateConfiguration:
    """Configuration for snapshot updates"""
    validation_level: str = "standard"  # "none", "basic", "standard", "strict"
    performance_mode: str = "balanced"  # "fast", "balanced", "memory_optimized"
    cache_enabled: bool = True
    batch_size: int = 10000
    max_memory_usage: Optional[int] = None  # MB
    
@dataclass  
class UpdateResult:
    """Result of snapshot update operation"""
    success: bool
    updated_snapshot: Optional[pd.DataFrame]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

class SnapshotEngine:
    """Main engine for snapshot operations"""
    
    def __init__(self, config: UpdateConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.event_engine = EventProcessingEngine()
        self.update_pipeline = SnapshotUpdatePipeline()
        self._setup_default_pipeline()
    
    def update_snapshot(
        self,
        current_snapshot: pd.DataFrame,
        events: pd.DataFrame,
        target_year: int
    ) -> UpdateResult:
        """Main entry point for snapshot updates"""
        start_time = time.time()
        
        try:
            # Validate inputs
            validation_errors = self._validate_inputs(
                current_snapshot, events, target_year
            )
            if validation_errors:
                return UpdateResult(
                    success=False,
                    updated_snapshot=None,
                    errors=validation_errors
                )
            
            # Create update context
            context = UpdateContext(
                snapshot=current_snapshot.copy(),
                events=events,
                year=target_year,
                metadata={"config": self.config}
            )
            
            # Execute update pipeline
            updated_snapshot = self.update_pipeline.execute_update(context)
            
            # Final validation
            final_validation = self._validate_result(updated_snapshot)
            
            execution_time = time.time() - start_time
            
            return UpdateResult(
                success=True,
                updated_snapshot=updated_snapshot,
                warnings=final_validation,
                metrics=self._collect_metrics(current_snapshot, updated_snapshot),
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.exception("Snapshot update failed")
            return UpdateResult(
                success=False,
                updated_snapshot=None,
                errors=[f"Update failed: {str(e)}"],
                execution_time=time.time() - start_time
            )
```

#### Day 13-14: Integration and Testing
**Goal**: Integrate all components and create comprehensive tests

```python
# Create tests/test_snapshot_integration.py
import pytest
import pandas as pd
from cost_model.snapshots import SnapshotEngine, UpdateConfiguration
from cost_model.schema import SnapshotColumns, EventTypes

class TestSnapshotIntegration:
    """Integration tests for snapshot system"""
    
    @pytest.fixture
    def sample_snapshot(self):
        """Create sample snapshot for testing"""
        return pd.DataFrame({
            SnapshotColumns.EMP_ID.value: ['EMP001', 'EMP002', 'EMP003'],
            SnapshotColumns.EMP_HIRE_DATE.value: [
                '2020-01-01', '2021-06-15', '2019-03-10'
            ],
            SnapshotColumns.EMP_GROSS_COMP.value: [75000, 65000, 85000],
            SnapshotColumns.EMP_ACTIVE.value: [True, True, True]
        })
    
    @pytest.fixture
    def sample_events(self):
        """Create sample events for testing"""
        return pd.DataFrame({
            'event_type': [EventTypes.HIRE.value, EventTypes.COMPENSATION.value],
            SnapshotColumns.EMP_ID.value: ['EMP004', 'EMP002'],
            'event_date': ['2024-01-15', '2024-03-01'],
            SnapshotColumns.EMP_GROSS_COMP.value: [70000, 70000]
        })
    
    def test_full_update_pipeline(self, sample_snapshot, sample_events):
        """Test complete update pipeline"""
        config = UpdateConfiguration(validation_level="standard")
        engine = SnapshotEngine(config)
        
        result = engine.update_snapshot(
            current_snapshot=sample_snapshot,
            events=sample_events,
            target_year=2024
        )
        
        assert result.success
        assert result.updated_snapshot is not None
        assert len(result.updated_snapshot) == 4  # Original 3 + 1 new hire
        
    def test_performance_benchmarks(self, large_snapshot, large_events):
        """Test performance with large datasets"""
        config = UpdateConfiguration(performance_mode="fast")
        engine = SnapshotEngine(config)
        
        result = engine.update_snapshot(
            current_snapshot=large_snapshot,
            events=large_events,
            target_year=2024
        )
        
        assert result.success
        assert result.execution_time < 30.0  # Should complete within 30 seconds
```

**Tasks**:
- [ ] Create comprehensive integration tests
- [ ] Implement performance benchmarks
- [ ] Add error handling edge cases
- [ ] Create migration utilities from old system
- [ ] Document API and usage examples

## Success Criteria

### Week 1 Deliverables
- [ ] Unified schema system with backward compatibility
- [ ] Event processing framework with basic handlers
- [ ] Simplified snapshot update pipeline (< 100 lines main function)
- [ ] 50% reduction in code duplication

### Week 2 Deliverables  
- [ ] Vectorized calculations for all mathematical operations
- [ ] Caching system for expensive operations
- [ ] Complete event handler implementations
- [ ] Production-ready snapshot engine with comprehensive error handling
- [ ] 90% test coverage for new components

### Performance Targets
- [ ] 50% reduction in snapshot update time
- [ ] 30% reduction in memory usage
- [ ] Zero data integrity issues
- [ ] Complete backward compatibility

This roadmap provides a practical, phased approach to transforming the snapshot system from its current complex, duplicated state to a clean, maintainable, and performant architecture.