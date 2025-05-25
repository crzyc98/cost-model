# Event Logging

## Event Log
- **Location**: `state.event_log`
- **Description**: Tracks all state changes for audit and analysis
- **Key Attributes**:
  - `event_id`: Unique event identifier (UUID)
  - `event_type`: Type of event (see below)
  - `employee_id`: Affected employee
  - `effective_date`: When the event occurred
  - `details`: Event-specific details (JSON)
  - `simulation_year`: Simulation year when event occurred
- **Search Tags**: `class:EventLog`, `state:events`

## Event Types

### Employee Events
- `EVT_HIRE`: Employee hire event
- `EVT_TERM`: Employee termination
- `EVT_REHIRE`: Employee rehire after termination
- `EVT_LEAVE`: Employee leave of absence
- `EVT_RETURN`: Employee return from leave

### Compensation Events
- `EVT_COMP`: Base compensation change
- `EVT_COLA`: Cost of living adjustment
- `EVT_BONUS`: One-time bonus payment
- `EVT_STOCK`: Equity grant or vesting

### Job Events
- `EVT_PROMOTION`: Job level increase
- `EVT_DEMOTION`: Job level decrease
- `EVT_TRANSFER`: Department/role transfer
- `EVT_TITLE_CHANGE`: Title change without level change

### Retirement Plan Events
- `EVT_ENROLL`: Plan enrollment
- `EVT_CONTRIB`: Contribution change
- `EVT_LOAN`: Loan initiation/repayment
- `EVT_WITHDRAWAL`: Hardship withdrawal
- `EVT_ROLLOVER`: Rollover in/out

## Event Log Operations

### log_event()
- **Location**: `state.event_log.log_event`
- **Description**: Records a new event in the log
- **Parameters**:
  - `event_type`: Type of event
  - `employee_id`: Affected employee
  - `effective_date`: When the event occurred
  - `details`: Event-specific details
  - `simulation_year`: Current simulation year
- **Returns**: Created event record
- **Search Tags**: `function:log_event`, `events:create`

### get_employee_events()
- **Location**: `state.event_log.get_employee_events`
- **Description**: Retrieves all events for a specific employee
- **Search Tags**: `function:get_employee_events`, `events:query`

## Event Schema

```python
class EventLog(BaseModel):
    event_id: UUID
    event_type: str
    employee_id: str
    effective_date: date
    details: Dict[str, Any]
    simulation_year: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
```
