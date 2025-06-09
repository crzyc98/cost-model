Looking at this `snapshot.py` file, I can see it's a complex module that handles workforce snapshots during projections. Let me create a comprehensive refactoring plan that will make the code more maintainable, testable, and easier to understand.

## Refactoring Plan for `snapshot.py`

### Overview
The file currently contains 900+ lines with multiple responsibilities mixed together. The main issues are:
- Functions are too long and do too many things
- Business logic is mixed with data processing
- Hard-coded values scattered throughout
- Limited error handling and validation
- Complex nested logic that's hard to follow

### Phase 1: Initial Setup and Structure

1. **Create new module structure**
   ```
   cost_model/projections/snapshot/
   ├── __init__.py
   ├── builder.py          # Main snapshot building logic
   ├── census_loader.py    # Census data loading and validation
   ├── event_processor.py  # Event processing logic
   ├── validators.py       # Data validation functions
   ├── transformers.py     # Data transformation utilities
   ├── constants.py        # All constants and defaults
   └── models.py          # Data models/schemas
   ```

2. **Extract all constants and defaults**
   - Move `DEFAULT_COMPENSATION`, `LEVEL_BASED_DEFAULTS` to `constants.py`
   - Create configuration dataclasses for better type safety
   - Group related constants together

### Phase 2: Break Down Core Functions

3. **Refactor `create_initial_snapshot`** (currently 400+ lines)
   
   Break into smaller functions:
   - `load_census_file()` - Handle file loading with proper error handling
   - `standardize_column_names()` - Column mapping logic
   - `validate_required_columns()` - Column validation
   - `filter_terminated_employees()` - Termination filtering
   - `initialize_employee_data()` - Create initial data structure
   - `infer_missing_data()` - Handle job levels, tenure, age calculations
   - `create_snapshot_dataframe()` - Final DataFrame creation

4. **Refactor `build_enhanced_yearly_snapshot`** (currently 600+ lines)
   
   Break into:
   - `identify_active_employees()` - Determine who was active during year
   - `reconstruct_terminated_employees()` - Handle terminated employee data
   - `merge_employee_sources()` - Combine different data sources
   - `apply_year_end_calculations()` - Apply contributions, tenure, age
   - `validate_snapshot_data()` - Final validation and logging

5. **Extract `_extract_compensation_for_employee`** logic
   - Create a `CompensationExtractor` class with clear priority rules
   - Make the priority logic configurable
   - Add proper logging and error handling

### Phase 3: Create Reusable Components

6. **Create data validators**
   ```python
   class SnapshotValidator:
       def validate_census_data(self, df: pd.DataFrame) -> ValidationResult
       def validate_snapshot_completeness(self, df: pd.DataFrame) -> ValidationResult
       def validate_compensation_data(self, df: pd.DataFrame) -> ValidationResult
   ```

7. **Create data transformers**
   ```python
   class SnapshotTransformer:
       def apply_tenure_calculations(self, df: pd.DataFrame) -> pd.DataFrame
       def apply_age_calculations(self, df: pd.DataFrame) -> pd.DataFrame
       def apply_contribution_calculations(self, df: pd.DataFrame) -> pd.DataFrame
   ```

8. **Create event processors**
   ```python
   class EventProcessor:
       def process_hire_events(self, events: pd.DataFrame) -> pd.DataFrame
       def process_termination_events(self, events: pd.DataFrame) -> pd.DataFrame
       def extract_employee_from_events(self, emp_id: str, events: pd.DataFrame) -> dict
   ```

### Phase 4: Improve Error Handling and Logging

9. **Add custom exceptions**
   ```python
   class SnapshotError(Exception): pass
   class CensusDataError(SnapshotError): pass
   class ValidationError(SnapshotError): pass
   class EventProcessingError(SnapshotError): pass
   ```

10. **Enhance logging**
    - Add structured logging with context
    - Create log decorators for timing and debugging
    - Add progress indicators for long operations

### Phase 5: Add Type Safety and Documentation

11. **Add type hints everywhere**
    - Create TypedDict or dataclasses for complex data structures
    - Use Protocol for defining interfaces
    - Add mypy configuration

12. **Create comprehensive documentation**
    - Add docstrings with examples
    - Create usage examples
    - Document the data flow and transformations

### Phase 6: Testing Infrastructure

13. **Create test fixtures**
    ```python
    # tests/fixtures/snapshot_fixtures.py
    def sample_census_data() -> pd.DataFrame
    def sample_events_data() -> pd.DataFrame
    def expected_snapshot_output() -> pd.DataFrame
    ```

14. **Add unit tests for each component**
    - Test each extracted function independently
    - Add edge case testing
    - Create integration tests for the full workflow

### Phase 7: Performance Optimizations

15. **Optimize DataFrame operations**
    - Use vectorized operations instead of iterrows()
    - Batch similar operations together
    - Add caching for expensive calculations

16. **Add performance monitoring**
    - Add timing decorators
    - Log memory usage for large operations
    - Identify and optimize bottlenecks

### Implementation Order

1. **Week 1**: Create new structure and extract constants
2. **Week 2**: Break down `create_initial_snapshot`
3. **Week 3**: Break down `build_enhanced_yearly_snapshot`
4. **Week 4**: Create reusable components and validators
5. **Week 5**: Add error handling and improve logging
6. **Week 6**: Add type safety and documentation
7. **Week 7**: Create comprehensive tests
8. **Week 8**: Performance optimizations and final cleanup

### Key Refactoring Patterns to Apply

1. **Extract Method**: Break large functions into smaller, focused ones
2. **Replace Magic Numbers**: Use named constants
3. **Introduce Parameter Object**: Group related parameters
4. **Replace Conditional with Polymorphism**: For event type handling
5. **Extract Class**: For complex logic like compensation extraction
6. **Introduce Null Object**: For handling missing data gracefully

### Success Metrics

- No function longer than 50 lines
- Test coverage > 90%
- All functions have type hints
- Performance improvement of at least 20%
- Clear separation of concerns
- Easier to add new event types or validation rules

This plan provides a structured approach to refactoring the code while maintaining functionality and improving maintainability.