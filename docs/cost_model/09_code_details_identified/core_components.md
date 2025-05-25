# Core Components Documentation

## Logging System

### Logging Configuration
- **Location**: `logging_config.py`
- **Purpose**: Centralized logging configuration
- **Key Features**:
  - Multiple log files by concern
  - Proper log rotation (10MB files, 5 backups)
  - Thread-safe initialization
  - Debug mode support
- **Search Tags**: `logging`, `configuration`, `debugging`

### Log Files
1. `projection_events.log` (INFO+)
   - Workflow events
   - State transitions
   - Major milestones

2. `performance_metrics.log` (INFO+)
   - Timing information
   - Resource usage
   - Performance counters

3. `warnings_errors.log` (WARNING+)
   - All warnings and errors
   - Stack traces
   - Error context

4. `debug_detail.log` (DEBUG, debug mode only)
   - Detailed debug information
   - Variable dumps
   - Execution traces

## Configuration System

### Scenario Definition
- **Location**: `config/scenarios.py`
- **Purpose**: Defines simulation scenarios
- **Key Components**:
  - Employee demographics
  - Compensation structure
  - Benefit plans
  - Economic assumptions
- **Search Tags**: `configuration`, `scenarios`

### Runtime Configuration
- **Location**: `config/runtime.py`
- **Purpose**: Runtime settings and overrides
- **Features**:
  - Environment variables
  - Command-line arguments
  - Configuration files
  - Default values
- **Search Tags**: `configuration`, `runtime`

## Data Models

### Employee Model
- **Location**: `models/employee.py`
- **Fields**:
  - `employee_id`: Unique identifier
  - `hire_date`: Employment start date
  - `termination_date`: Employment end date (if any)
  - `compensation`: Annual compensation
  - `department`: Department code
  - `job_level`: Job level/grade
  - `status`: Employment status
- **Search Tags**: `models`, `employee`, `data`

### Event Model
- **Location**: `models/events.py`
- **Purpose**: Tracks state changes
- **Event Types**:
  - `HIRE`: New employee
  - `TERM`: Termination
  - `PROMO`: Promotion
  - `COMP_CHG`: Compensation change
  - `DEPT_TRANSFER`: Department transfer
- **Search Tags**: `models`, `events`, `state_changes`

## Core Engine

### Simulation Engine
- **Location**: `engine/simulator.py`
- **Purpose**: Main simulation loop
- **Features**:
  - Annual cycle processing
  - Event scheduling
  - State management
  - Result aggregation
- **Search Tags**: `engine`, `simulation`, `core`

### Rules Engine
- **Location**: `engine/rules.py`
- **Purpose**: Applies business rules
- **Rule Types**:
  - Eligibility rules
  - Vesting schedules
  - Benefit calculations
  - Compliance checks
- **Search Tags**: `engine`, `rules`, `business_logic`

## Data Access

### Repository Layer
- **Location**: `data/repository.py`
- **Purpose**: Data persistence
- **Features**:
  - CRUD operations
  - Query building
  - Connection management
  - Transaction support
- **Search Tags**: `data`, `database`, `persistence`

### Data Loaders
- **Location**: `data/loaders/`
- **Purpose**: Data import/export
- **Supported Formats**:
  - CSV
  - Excel
  - Parquet
  - JSON
- **Search Tags**: `data`, `import`, `export`

## Utilities

### Date Utilities
- **Location**: `utils/dates.py`
- **Features**:
  - Business day calculations
  - Holiday calendars
  - Date ranges
  - Age/tenure calculations
- **Search Tags**: `utils`, `dates`, `time`

### Financial Utilities
- **Location**: `utils/finance.py`
- **Features**:
  - Interest calculations
  - Amortization
  - Present/future value
  - Currency formatting
- **Search Tags**: `utils`, `finance`, `money`

## Testing

### Test Framework
- **Location**: `tests/`
- **Components**:
  - Unit tests
  - Integration tests
  - Performance tests
  - Test fixtures
- **Search Tags**: `testing`, `quality`, `coverage`

### Test Data
- **Location**: `tests/data/`
- **Contents**:
  - Sample datasets
  - Mock objects
  - Test scenarios
  - Expected results
- **Search Tags**: `testing`, `data`, `fixtures`

## Documentation

### API Reference
- **Location**: `docs/api/`
- **Contents**:
  - Module documentation
  - Class references
  - Function signatures
  - Type information
- **Search Tags**: `documentation`, `api`, `reference`

### User Guide
- **Location**: `docs/guide/`
- **Contents**:
  - Getting started
  - Tutorials
  - How-to guides
  - Best practices
- **Search Tags**: `documentation`, `guide`, `tutorial`

## Deployment

### Containerization
- **Location**: `Dockerfile`, `docker-compose.yml`
- **Purpose**: Container setup
- **Features**:
  - Multi-stage builds
  - Environment configuration
  - Health checks
  - Resource limits
- **Search Tags**: `deployment`, `docker`, `containers`

### CI/CD
- **Location**: `.github/workflows/`
- **Purpose**: Automation
- **Features**:
  - Test automation
  - Code quality checks
  - Release management
  - Deployment pipelines
- **Search Tags**: `ci`, `cd`, `automation`
