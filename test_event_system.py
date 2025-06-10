"""
Test script to verify the unified event handling system is working correctly.

This script tests the event processing framework and event handlers.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from cost_model.schema import SnapshotColumns, EventColumns, EventTypes
from cost_model.projections.snapshot.events import EventProcessor, EventContext
from cost_model.projections.snapshot.models import SnapshotConfig

def test_event_processor_initialization():
    """Test event processor initialization and registry."""
    print("=== Testing Event Processor Initialization ===")
    
    config = SnapshotConfig(start_year=2024)
    processor = EventProcessor(config)
    
    # Check supported event types
    supported_types = processor.get_supported_event_types()
    print(f"Supported event types: {sorted(list(supported_types))}")
    
    expected_types = {EventTypes.HIRE, EventTypes.TERMINATION, EventTypes.NEW_HIRE_TERMINATION, 
                     EventTypes.PROMOTION, EventTypes.COMPENSATION}
    assert expected_types.issubset(supported_types), f"Missing expected event types"
    
    print("✅ Event processor initialization tests passed!\n")


def test_hire_events():
    """Test hire event processing."""
    print("=== Testing Hire Events ===")
    
    # Create initial snapshot
    snapshot = pd.DataFrame({
        SnapshotColumns.EMP_ID: ['EMP001', 'EMP002'],
        SnapshotColumns.EMP_HIRE_DATE: ['2020-01-01', '2021-01-01'],
        SnapshotColumns.EMP_GROSS_COMP: [75000, 85000],
        SnapshotColumns.EMP_ACTIVE: [True, True],
        SnapshotColumns.SIMULATION_YEAR: [2024, 2024]
    })
    
    # Create hire events
    hire_events = pd.DataFrame({
        EventColumns.EVENT_TYPE: [EventTypes.HIRE, EventTypes.HIRE],
        EventColumns.EVENT_DATE: ['2024-03-01', '2024-04-15'],
        EventColumns.EMP_ID: ['EMP003', 'EMP004'],
        EventColumns.GROSS_COMPENSATION: [90000, 70000],
        EventColumns.JOB_LEVEL: ['BAND_3', 'BAND_2'],
        EventColumns.DEFERRAL_RATE: [0.06, 0.04]
    })
    
    print(f"Initial snapshot: {len(snapshot)} employees")
    print(f"Hire events: {len(hire_events)} events")
    
    # Process events
    config = SnapshotConfig(start_year=2024)
    processor = EventProcessor(config)
    
    result = processor.process_events(
        snapshot=snapshot,
        events=hire_events,
        simulation_year=2024,
        reference_date=datetime(2024, 1, 1)
    )
    
    print(f"Processing success: {result.success}")
    print(f"Events processed: {result.events_processed}")
    print(f"Final snapshot: {len(result.updated_snapshot)} employees")
    
    # Verify results
    assert result.success, f"Event processing failed: {result.errors}"
    assert len(result.updated_snapshot) == 4, f"Expected 4 employees, got {len(result.updated_snapshot)}"
    
    # Check new employees are active
    new_employees = result.updated_snapshot[
        result.updated_snapshot[SnapshotColumns.EMP_ID].isin(['EMP003', 'EMP004'])
    ]
    assert len(new_employees) == 2, "Should have 2 new employees"
    assert all(new_employees[SnapshotColumns.EMP_ACTIVE]), "New employees should be active"
    
    print("✅ Hire event tests passed!\n")


def test_termination_events():
    """Test termination event processing."""
    print("=== Testing Termination Events ===")
    
    # Create snapshot with active employees
    snapshot = pd.DataFrame({
        SnapshotColumns.EMP_ID: ['EMP001', 'EMP002', 'EMP003'],
        SnapshotColumns.EMP_HIRE_DATE: ['2020-01-01', '2021-01-01', '2022-01-01'],
        SnapshotColumns.EMP_GROSS_COMP: [75000, 85000, 90000],
        SnapshotColumns.EMP_ACTIVE: [True, True, True],
        SnapshotColumns.SIMULATION_YEAR: [2024, 2024, 2024]
    })
    
    # Create termination events
    term_events = pd.DataFrame({
        EventColumns.EVENT_TYPE: [EventTypes.TERMINATION],
        EventColumns.EVENT_DATE: ['2024-06-15'],
        EventColumns.EMP_ID: ['EMP002'],
        EventColumns.TERMINATION_REASON: ['VOLUNTARY']
    })
    
    print(f"Initial active employees: {snapshot[SnapshotColumns.EMP_ACTIVE].sum()}")
    print(f"Termination events: {len(term_events)} events")
    
    # Process events
    config = SnapshotConfig(start_year=2024)
    processor = EventProcessor(config)
    
    result = processor.process_events(
        snapshot=snapshot,
        events=term_events,
        simulation_year=2024,
        reference_date=datetime(2024, 1, 1)
    )
    
    print(f"Processing success: {result.success}")
    print(f"Final active employees: {result.updated_snapshot[SnapshotColumns.EMP_ACTIVE].sum()}")
    
    # Verify results
    assert result.success, f"Event processing failed: {result.errors}"
    assert result.updated_snapshot[SnapshotColumns.EMP_ACTIVE].sum() == 2, "Should have 2 active employees"
    
    # Check terminated employee
    terminated_emp = result.updated_snapshot[
        result.updated_snapshot[SnapshotColumns.EMP_ID] == 'EMP002'
    ]
    assert len(terminated_emp) == 1, "Should find terminated employee"
    assert not terminated_emp[SnapshotColumns.EMP_ACTIVE].iloc[0], "Employee should be inactive"
    
    print("✅ Termination event tests passed!\n")


def test_promotion_events():
    """Test promotion event processing."""
    print("=== Testing Promotion Events ===")
    
    # Create snapshot
    snapshot = pd.DataFrame({
        SnapshotColumns.EMP_ID: ['EMP001', 'EMP002'],
        SnapshotColumns.EMP_HIRE_DATE: ['2020-01-01', '2021-01-01'],
        SnapshotColumns.EMP_GROSS_COMP: [75000, 85000],
        SnapshotColumns.EMP_LEVEL: ['BAND_2', 'BAND_3'],
        SnapshotColumns.EMP_ACTIVE: [True, True],
        SnapshotColumns.EMP_DEFERRAL_RATE: [0.05, 0.06],
        SnapshotColumns.EMP_CONTRIBUTION: [3750, 5100],  # 75000*0.05, 85000*0.06
        SnapshotColumns.EMPLOYER_MATCH_CONTRIBUTION: [1875, 2550],  # 50% of employee contrib
        SnapshotColumns.SIMULATION_YEAR: [2024, 2024]
    })
    
    # Create promotion events
    promo_events = pd.DataFrame({
        EventColumns.EVENT_TYPE: [EventTypes.PROMOTION],
        EventColumns.EVENT_DATE: ['2024-07-01'],
        EventColumns.EMP_ID: ['EMP001'],
        EventColumns.JOB_LEVEL: ['BAND_3'],
        EventColumns.GROSS_COMPENSATION: [95000]
    })
    
    print(f"Initial compensation for EMP001: ${snapshot[snapshot[SnapshotColumns.EMP_ID] == 'EMP001'][SnapshotColumns.EMP_GROSS_COMP].iloc[0]:,}")
    
    # Process events
    config = SnapshotConfig(start_year=2024)
    processor = EventProcessor(config)
    
    result = processor.process_events(
        snapshot=snapshot,
        events=promo_events,
        simulation_year=2024,
        reference_date=datetime(2024, 1, 1)
    )
    
    print(f"Processing success: {result.success}")
    
    # Verify results
    assert result.success, f"Event processing failed: {result.errors}"
    
    # Check promoted employee
    promoted_emp = result.updated_snapshot[
        result.updated_snapshot[SnapshotColumns.EMP_ID] == 'EMP001'
    ]
    assert len(promoted_emp) == 1, "Should find promoted employee"
    assert promoted_emp[SnapshotColumns.EMP_GROSS_COMP].iloc[0] == 95000, "Compensation should be updated"
    assert promoted_emp[SnapshotColumns.EMP_LEVEL].iloc[0] == 'BAND_3', "Job level should be updated"
    
    print(f"Final compensation for EMP001: ${promoted_emp[SnapshotColumns.EMP_GROSS_COMP].iloc[0]:,}")
    print("✅ Promotion event tests passed!\n")


def test_event_validation():
    """Test event validation functionality."""
    print("=== Testing Event Validation ===")
    
    config = SnapshotConfig(start_year=2024)
    processor = EventProcessor(config)
    
    # Test valid events
    valid_events = pd.DataFrame({
        EventColumns.EVENT_TYPE: [EventTypes.HIRE],
        EventColumns.EVENT_DATE: ['2024-01-15'],
        EventColumns.EMP_ID: ['EMP001'],
        EventColumns.GROSS_COMPENSATION: [85000]
    })
    
    errors = processor.validate_events_only(
        events=valid_events,
        simulation_year=2024,
        reference_date=datetime(2024, 1, 1)
    )
    
    print(f"Valid events validation errors: {len(errors)}")
    assert len(errors) == 0, f"Valid events should not have errors: {errors}"
    
    # Test invalid events
    invalid_events = pd.DataFrame({
        EventColumns.EVENT_TYPE: [EventTypes.HIRE, 'INVALID_TYPE'],
        EventColumns.EVENT_DATE: ['2024-01-15', '2024-02-01'],
        EventColumns.EMP_ID: ['EMP001', 'EMP002'],
        EventColumns.GROSS_COMPENSATION: [85000, -1000]  # Negative compensation
    })
    
    errors = processor.validate_events_only(
        events=invalid_events,
        simulation_year=2024,
        reference_date=datetime(2024, 1, 1)
    )
    
    print(f"Invalid events validation errors: {len(errors)}")
    assert len(errors) > 0, "Invalid events should have errors"
    
    print("✅ Event validation tests passed!\n")


def test_event_processing_order():
    """Test that events are processed in correct order."""
    print("=== Testing Event Processing Order ===")
    
    # Create snapshot
    snapshot = pd.DataFrame({
        SnapshotColumns.EMP_ID: ['EMP001'],
        SnapshotColumns.EMP_HIRE_DATE: ['2020-01-01'],
        SnapshotColumns.EMP_GROSS_COMP: [75000],
        SnapshotColumns.EMP_ACTIVE: [True],
        SnapshotColumns.SIMULATION_YEAR: [2024]
    })
    
    # Create mixed events that should be processed in specific order
    mixed_events = pd.DataFrame({
        EventColumns.EVENT_TYPE: [
            EventTypes.COMPENSATION,    # Should be processed last
            EventTypes.HIRE,           # Should be processed second  
            EventTypes.TERMINATION     # Should be processed first
        ],
        EventColumns.EVENT_DATE: [
            '2024-01-03',
            '2024-01-02', 
            '2024-01-01'
        ],
        EventColumns.EMP_ID: [
            'EMP001',  # Compensation change
            'EMP002',  # New hire
            'EMP001'   # Termination
        ],
        EventColumns.GROSS_COMPENSATION: [95000, 80000, None]
    })
    
    print(f"Processing {len(mixed_events)} mixed events")
    
    # Process events
    config = SnapshotConfig(start_year=2024)
    processor = EventProcessor(config)
    
    result = processor.process_events(
        snapshot=snapshot,
        events=mixed_events,
        simulation_year=2024,
        reference_date=datetime(2024, 1, 1)
    )
    
    print(f"Processing success: {result.success}")
    print(f"Handler stats: {list(result.handler_stats.keys())}")
    
    # Verify results - EMP001 should be terminated, EMP002 should be hired
    assert result.success, f"Event processing failed: {result.errors}"
    
    emp001 = result.updated_snapshot[result.updated_snapshot[SnapshotColumns.EMP_ID] == 'EMP001']
    emp002 = result.updated_snapshot[result.updated_snapshot[SnapshotColumns.EMP_ID] == 'EMP002']
    
    assert len(emp001) == 1 and not emp001[SnapshotColumns.EMP_ACTIVE].iloc[0], "EMP001 should be terminated"
    assert len(emp002) == 1 and emp002[SnapshotColumns.EMP_ACTIVE].iloc[0], "EMP002 should be active"
    
    print("✅ Event processing order tests passed!\n")


def main():
    """Run all tests."""
    print("🚀 Testing Unified Event Handling System\n")
    
    try:
        test_event_processor_initialization()
        test_hire_events()
        test_termination_events()
        test_promotion_events()
        test_event_validation()
        test_event_processing_order()
        
        print("🎉 All event system tests passed! Event handling system is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()