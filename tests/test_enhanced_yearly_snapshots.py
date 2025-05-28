"""
Tests for enhanced yearly snapshot generation functionality.

This module tests the implementation of the approved solution for yearly snapshots
that include all employees who were active at any point during each year.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from unittest.mock import Mock, patch

from cost_model.projections.snapshot import build_enhanced_yearly_snapshot
from cost_model.state.schema import (
    EMP_ID, EMP_ACTIVE, EMP_TERM_DATE, EMP_STATUS_EOY,
    SIMULATION_YEAR, EVT_HIRE, EVT_TERM, EVT_NEW_HIRE_TERM, EMP_HIRE_DATE
)


class TestEnhancedYearlySnapshots:
    """Test suite for enhanced yearly snapshot generation."""

    def setup_method(self):
        """Set up test data for each test method."""
        # Create sample start-of-year snapshot
        self.soy_snapshot = pd.DataFrame({
            EMP_ID: ['emp_001', 'emp_002', 'emp_003'],
            EMP_ACTIVE: [True, True, True],
            EMP_TERM_DATE: [pd.NaT, pd.NaT, pd.NaT],
            EMP_HIRE_DATE: [
                datetime(2020, 1, 1),
                datetime(2021, 6, 15),
                datetime(2022, 3, 10)
            ],
            'gross_comp': [75000.0, 85000.0, 65000.0]
        })

        # Create sample end-of-year snapshot
        self.eoy_snapshot = pd.DataFrame({
            EMP_ID: ['emp_001', 'emp_002', 'emp_004', 'emp_005'],
            EMP_ACTIVE: [True, False, True, True],
            EMP_TERM_DATE: [
                pd.NaT,
                datetime(2024, 8, 15),  # Terminated during year
                pd.NaT,
                pd.NaT
            ],
            EMP_HIRE_DATE: [
                datetime(2020, 1, 1),
                datetime(2021, 6, 15),
                datetime(2024, 4, 1),   # Hired during year
                datetime(2024, 9, 1)    # Hired during year
            ],
            'gross_comp': [78000.0, 85000.0, 70000.0, 60000.0]
        })

        # Create sample year events
        self.year_events = pd.DataFrame({
            'event_type': [EVT_HIRE, EVT_HIRE, EVT_TERM],
            EMP_ID: ['emp_004', 'emp_005', 'emp_002'],
            'simulation_year': [2024, 2024, 2024],
            'event_time': [
                datetime(2024, 4, 1),
                datetime(2024, 9, 1),
                datetime(2024, 8, 15)
            ]
        })

        self.simulation_year = 2024

    def test_enhanced_snapshot_includes_all_active_during_year(self):
        """Test that enhanced snapshot includes all employees active during the year."""
        result = build_enhanced_yearly_snapshot(
            start_of_year_snapshot=self.soy_snapshot,
            end_of_year_snapshot=self.eoy_snapshot,
            year_events=self.year_events,
            simulation_year=self.simulation_year
        )

        # Should include:
        # - emp_001: active at SOY and EOY
        # - emp_002: active at SOY, terminated during year
        # - emp_004: hired during year, active at EOY
        # - emp_005: hired during year, active at EOY
        # Should NOT include:
        # - emp_003: was active at SOY but not in EOY snapshot (left before EOY)

        expected_employee_ids = {'emp_001', 'emp_002', 'emp_004', 'emp_005'}
        actual_employee_ids = set(result[EMP_ID].unique())

        assert actual_employee_ids == expected_employee_ids, (
            f"Expected employees {expected_employee_ids}, got {actual_employee_ids}"
        )

    def test_employee_status_eoy_determination(self):
        """Test that employee_status_eoy is correctly determined."""
        result = build_enhanced_yearly_snapshot(
            start_of_year_snapshot=self.soy_snapshot,
            end_of_year_snapshot=self.eoy_snapshot,
            year_events=self.year_events,
            simulation_year=self.simulation_year
        )

        # Check status for each employee
        status_by_id = result.set_index(EMP_ID)[EMP_STATUS_EOY].to_dict()

        assert status_by_id['emp_001'] == 'Active', "emp_001 should be Active"
        assert status_by_id['emp_002'] == 'Terminated', "emp_002 should be Terminated"
        assert status_by_id['emp_004'] == 'Active', "emp_004 should be Active"
        assert status_by_id['emp_005'] == 'Active', "emp_005 should be Active"

    def test_simulation_year_column_set(self):
        """Test that simulation_year column is correctly set."""
        result = build_enhanced_yearly_snapshot(
            start_of_year_snapshot=self.soy_snapshot,
            end_of_year_snapshot=self.eoy_snapshot,
            year_events=self.year_events,
            simulation_year=self.simulation_year
        )

        assert SIMULATION_YEAR in result.columns, "simulation_year column should be present"
        assert all(result[SIMULATION_YEAR] == self.simulation_year), (
            "All rows should have correct simulation_year"
        )

    def test_empty_year_events_handling(self):
        """Test handling of empty year events."""
        empty_events = pd.DataFrame(columns=['event_type', EMP_ID, 'simulation_year'])

        result = build_enhanced_yearly_snapshot(
            start_of_year_snapshot=self.soy_snapshot,
            end_of_year_snapshot=self.eoy_snapshot,
            year_events=empty_events,
            simulation_year=self.simulation_year
        )

        # Should still include employees from SOY that are in EOY
        assert len(result) > 0, "Should include employees even with no events"
        assert EMP_STATUS_EOY in result.columns, "Should have status column"

    def test_none_year_events_handling(self):
        """Test handling of None year events."""
        result = build_enhanced_yearly_snapshot(
            start_of_year_snapshot=self.soy_snapshot,
            end_of_year_snapshot=self.eoy_snapshot,
            year_events=None,
            simulation_year=self.simulation_year
        )

        # Should still work with None events
        assert len(result) > 0, "Should include employees even with None events"
        assert EMP_STATUS_EOY in result.columns, "Should have status column"

    def test_employee_data_from_eoy_snapshot(self):
        """Test that employee data comes from end-of-year snapshot."""
        result = build_enhanced_yearly_snapshot(
            start_of_year_snapshot=self.soy_snapshot,
            end_of_year_snapshot=self.eoy_snapshot,
            year_events=self.year_events,
            simulation_year=self.simulation_year
        )

        # Check that compensation data comes from EOY snapshot
        emp_001_result = result[result[EMP_ID] == 'emp_001'].iloc[0]
        emp_001_eoy = self.eoy_snapshot[self.eoy_snapshot[EMP_ID] == 'emp_001'].iloc[0]

        assert emp_001_result['gross_comp'] == emp_001_eoy['gross_comp'], (
            "Employee data should come from EOY snapshot"
        )

    @patch('cost_model.projections.snapshot.logging.getLogger')
    def test_logging_output(self, mock_logger):
        """Test that appropriate logging messages are generated."""
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        build_enhanced_yearly_snapshot(
            start_of_year_snapshot=self.soy_snapshot,
            end_of_year_snapshot=self.eoy_snapshot,
            year_events=self.year_events,
            simulation_year=self.simulation_year
        )

        # Verify logging calls were made
        assert mock_logger_instance.info.call_count >= 3, (
            "Should log building, employee counts, and completion messages"
        )

    def test_schema_column_constants_used(self):
        """Test that schema column constants are properly used."""
        result = build_enhanced_yearly_snapshot(
            start_of_year_snapshot=self.soy_snapshot,
            end_of_year_snapshot=self.eoy_snapshot,
            year_events=self.year_events,
            simulation_year=self.simulation_year
        )

        # Verify required schema columns are present
        required_columns = [EMP_ID, EMP_STATUS_EOY, SIMULATION_YEAR]
        for col in required_columns:
            assert col in result.columns, f"Required schema column {col} should be present"

    def test_edge_case_no_soy_active_employees(self):
        """Test edge case where no employees are active at start of year."""
        empty_soy = pd.DataFrame({
            EMP_ID: [],
            EMP_ACTIVE: [],
            EMP_TERM_DATE: [],
            EMP_HIRE_DATE: [],
            'gross_comp': []
        })

        result = build_enhanced_yearly_snapshot(
            start_of_year_snapshot=empty_soy,
            end_of_year_snapshot=self.eoy_snapshot,
            year_events=self.year_events,
            simulation_year=self.simulation_year
        )

        # Should still include hired employees from EOY snapshot
        hired_employees = set(self.year_events[
            self.year_events['event_type'] == EVT_HIRE
        ][EMP_ID].unique())

        result_employees = set(result[EMP_ID].unique())
        assert hired_employees.issubset(result_employees), (
            "Should include employees hired during year even with empty SOY"
        )

    def test_new_hire_terminations_included(self):
        """Test that new hire terminations are properly included in yearly snapshots."""
        # Create test data with new hire terminations
        soy_snapshot = pd.DataFrame({
            EMP_ID: ['emp_001', 'emp_002'],
            EMP_ACTIVE: [True, True],
            EMP_TERM_DATE: [pd.NaT, pd.NaT],
            EMP_HIRE_DATE: [
                datetime(2020, 1, 1),
                datetime(2021, 6, 15)
            ],
            'gross_comp': [75000.0, 85000.0]
        })

        # EOY snapshot excludes terminated new hires (they're not active at EOY)
        eoy_snapshot = pd.DataFrame({
            EMP_ID: ['emp_001', 'emp_002', 'emp_004'],  # emp_003 and emp_005 terminated
            EMP_ACTIVE: [True, True, True],
            EMP_TERM_DATE: [pd.NaT, pd.NaT, pd.NaT],
            EMP_HIRE_DATE: [
                datetime(2020, 1, 1),
                datetime(2021, 6, 15),
                datetime(2024, 4, 1)   # New hire who survived
            ],
            'gross_comp': [78000.0, 85000.0, 70000.0]
        })

        # Events include new hires and new hire terminations
        year_events = pd.DataFrame({
            'event_type': [EVT_HIRE, EVT_HIRE, EVT_HIRE, EVT_NEW_HIRE_TERM, EVT_NEW_HIRE_TERM],
            EMP_ID: ['emp_003', 'emp_004', 'emp_005', 'emp_003', 'emp_005'],
            'simulation_year': [2024, 2024, 2024, 2024, 2024],
            'event_time': [
                datetime(2024, 2, 1),   # emp_003 hired
                datetime(2024, 4, 1),   # emp_004 hired
                datetime(2024, 6, 1),   # emp_005 hired
                datetime(2024, 8, 15),  # emp_003 terminated (new hire term)
                datetime(2024, 10, 1)   # emp_005 terminated (new hire term)
            ]
        })

        result = build_enhanced_yearly_snapshot(
            start_of_year_snapshot=soy_snapshot,
            end_of_year_snapshot=eoy_snapshot,
            year_events=year_events,
            simulation_year=2024
        )

        # Should include all employees who were active during the year:
        # - emp_001, emp_002: active at SOY and EOY
        # - emp_003: hired and terminated during year (new hire termination)
        # - emp_004: hired during year, active at EOY
        # - emp_005: hired and terminated during year (new hire termination)
        expected_employee_ids = {'emp_001', 'emp_002', 'emp_003', 'emp_004', 'emp_005'}
        actual_employee_ids = set(result[EMP_ID].unique())

        assert actual_employee_ids == expected_employee_ids, (
            f"Expected employees {expected_employee_ids}, got {actual_employee_ids}. "
            f"Missing new hire terminations: {expected_employee_ids - actual_employee_ids}"
        )

        # Verify status for terminated new hires
        status_by_id = result.set_index(EMP_ID)[EMP_STATUS_EOY].to_dict()
        assert status_by_id['emp_003'] == 'Terminated', "emp_003 (new hire termination) should be Terminated"
        assert status_by_id['emp_005'] == 'Terminated', "emp_005 (new hire termination) should be Terminated"
        assert status_by_id['emp_004'] == 'Active', "emp_004 (surviving new hire) should be Active"

    def test_mixed_termination_types_included(self):
        """Test that both regular and new hire terminations are included."""
        soy_snapshot = pd.DataFrame({
            EMP_ID: ['emp_001', 'emp_002'],
            EMP_ACTIVE: [True, True],
            EMP_TERM_DATE: [pd.NaT, pd.NaT],
            EMP_HIRE_DATE: [
                datetime(2020, 1, 1),
                datetime(2021, 6, 15)
            ],
            'gross_comp': [75000.0, 85000.0]
        })

        eoy_snapshot = pd.DataFrame({
            EMP_ID: ['emp_001', 'emp_004'],  # emp_002 and emp_003 terminated
            EMP_ACTIVE: [True, True],
            EMP_TERM_DATE: [pd.NaT, pd.NaT],
            EMP_HIRE_DATE: [
                datetime(2020, 1, 1),
                datetime(2024, 4, 1)
            ],
            'gross_comp': [78000.0, 70000.0]
        })

        # Mix of regular termination and new hire termination
        year_events = pd.DataFrame({
            'event_type': [EVT_HIRE, EVT_HIRE, EVT_TERM, EVT_NEW_HIRE_TERM],
            EMP_ID: ['emp_003', 'emp_004', 'emp_002', 'emp_003'],
            'simulation_year': [2024, 2024, 2024, 2024],
            'event_time': [
                datetime(2024, 2, 1),   # emp_003 hired
                datetime(2024, 4, 1),   # emp_004 hired
                datetime(2024, 8, 15),  # emp_002 terminated (experienced)
                datetime(2024, 10, 1)   # emp_003 terminated (new hire)
            ]
        })

        result = build_enhanced_yearly_snapshot(
            start_of_year_snapshot=soy_snapshot,
            end_of_year_snapshot=eoy_snapshot,
            year_events=year_events,
            simulation_year=2024
        )

        # Should include all employees active during year
        expected_employee_ids = {'emp_001', 'emp_002', 'emp_003', 'emp_004'}
        actual_employee_ids = set(result[EMP_ID].unique())

        assert actual_employee_ids == expected_employee_ids, (
            f"Expected employees {expected_employee_ids}, got {actual_employee_ids}"
        )

        # Verify both termination types are marked as terminated
        status_by_id = result.set_index(EMP_ID)[EMP_STATUS_EOY].to_dict()
        assert status_by_id['emp_002'] == 'Terminated', "emp_002 (experienced termination) should be Terminated"
        assert status_by_id['emp_003'] == 'Terminated', "emp_003 (new hire termination) should be Terminated"

    def test_new_hire_termination_dates_populated(self):
        """Test that termination dates are properly populated for new hire terminations."""
        soy_snapshot = pd.DataFrame({
            EMP_ID: ['emp_001'],
            EMP_ACTIVE: [True],
            EMP_TERM_DATE: [pd.NaT],
            EMP_HIRE_DATE: [datetime(2020, 1, 1)],
            'gross_comp': [75000.0]
        })

        eoy_snapshot = pd.DataFrame({
            EMP_ID: ['emp_001'],  # emp_002 terminated
            EMP_ACTIVE: [True],
            EMP_TERM_DATE: [pd.NaT],
            EMP_HIRE_DATE: [datetime(2020, 1, 1)],
            'gross_comp': [78000.0]
        })

        term_date = datetime(2024, 8, 15)
        year_events = pd.DataFrame({
            'event_type': [EVT_HIRE, EVT_NEW_HIRE_TERM],
            EMP_ID: ['emp_002', 'emp_002'],
            'simulation_year': [2024, 2024],
            'event_time': [
                datetime(2024, 2, 1),   # emp_002 hired
                term_date                # emp_002 terminated
            ]
        })

        result = build_enhanced_yearly_snapshot(
            start_of_year_snapshot=soy_snapshot,
            end_of_year_snapshot=eoy_snapshot,
            year_events=year_events,
            simulation_year=2024
        )

        # Should include the terminated new hire
        assert 'emp_002' in result[EMP_ID].values, "Should include terminated new hire emp_002"

        # Check termination date is populated
        emp_002_data = result[result[EMP_ID] == 'emp_002'].iloc[0]
        assert pd.notna(emp_002_data[EMP_TERM_DATE]), "Termination date should be populated for new hire termination"
        assert emp_002_data[EMP_TERM_DATE] == term_date, "Termination date should match event date"

    def test_tenure_calculation_for_terminated_new_hires(self):
        """Test that tenure is correctly calculated for new hires who terminate within the same year."""
        # Create snapshots with no employees (all new hires)
        soy_snapshot = pd.DataFrame({
            EMP_ID: [],
            EMP_ACTIVE: [],
            EMP_TERM_DATE: [],
            EMP_HIRE_DATE: [],
            'gross_comp': []
        })

        eoy_snapshot = pd.DataFrame({
            EMP_ID: ['emp_004'],  # Only surviving new hire
            EMP_ACTIVE: [True],
            EMP_TERM_DATE: [pd.NaT],
            EMP_HIRE_DATE: [datetime(2024, 6, 1)],
            'gross_comp': [75000.0]
        })

        # Create events with new hires and terminations at different intervals
        year_events = pd.DataFrame({
            'event_type': [
                'EVT_HIRE', 'EVT_HIRE', 'EVT_HIRE', 'EVT_HIRE',
                'EVT_NEW_HIRE_TERM', 'EVT_NEW_HIRE_TERM', 'EVT_NEW_HIRE_TERM'
            ],
            EMP_ID: [
                'emp_001', 'emp_002', 'emp_003', 'emp_004',
                'emp_001', 'emp_002', 'emp_003'
            ],
            'simulation_year': [2024] * 7,
            'event_time': [
                datetime(2024, 1, 1),   # emp_001 hired Jan 1
                datetime(2024, 3, 15),  # emp_002 hired Mar 15
                datetime(2024, 6, 1),   # emp_003 hired Jun 1
                datetime(2024, 6, 1),   # emp_004 hired Jun 1 (survives)
                datetime(2024, 1, 2),   # emp_001 terminated Jan 2 (1 day tenure)
                datetime(2024, 9, 15),  # emp_002 terminated Sep 15 (~6 months tenure)
                datetime(2024, 12, 1)   # emp_003 terminated Dec 1 (~6 months tenure)
            ],
            'value_json': [
                '{"compensation": 50000, "level": 1, "role": "Regular"}',
                '{"compensation": 60000, "level": 2, "role": "Regular"}',
                '{"compensation": 70000, "level": 2, "role": "Regular"}',
                '{"compensation": 75000, "level": 3, "role": "Regular"}',
                None, None, None  # Termination events don't have value_json
            ]
        })

        result = build_enhanced_yearly_snapshot(
            start_of_year_snapshot=soy_snapshot,
            end_of_year_snapshot=eoy_snapshot,
            year_events=year_events,
            simulation_year=2024
        )

        # Verify all employees are included
        assert len(result) == 4, "Should include all 4 employees (3 terminated + 1 active)"

        # Check emp_001 (1 day tenure)
        emp_001_data = result[result[EMP_ID] == 'emp_001'].iloc[0]
        assert emp_001_data['employee_tenure'] > 0, "emp_001 should have positive tenure"
        assert emp_001_data['employee_tenure'] < 0.01, "emp_001 should have very small tenure (~1 day)"
        assert emp_001_data['employee_tenure_band'] == '0-1', "emp_001 should be in 0-1 tenure band"

        # Check emp_002 (~6 months tenure)
        emp_002_data = result[result[EMP_ID] == 'emp_002'].iloc[0]
        expected_tenure_002 = (datetime(2024, 9, 15) - datetime(2024, 3, 15)).days / 365.25
        assert abs(emp_002_data['employee_tenure'] - expected_tenure_002) < 0.01, f"emp_002 tenure should be ~{expected_tenure_002:.3f} years"
        assert emp_002_data['employee_tenure_band'] == '0-1', "emp_002 should be in 0-1 tenure band"

        # Check emp_003 (~6 months tenure)
        emp_003_data = result[result[EMP_ID] == 'emp_003'].iloc[0]
        expected_tenure_003 = (datetime(2024, 12, 1) - datetime(2024, 6, 1)).days / 365.25
        assert abs(emp_003_data['employee_tenure'] - expected_tenure_003) < 0.01, f"emp_003 tenure should be ~{expected_tenure_003:.3f} years"
        assert emp_003_data['employee_tenure_band'] == '0-1', "emp_003 should be in 0-1 tenure band"

        # Check emp_004 (surviving employee - should have tenure calculated to end of year)
        emp_004_data = result[result[EMP_ID] == 'emp_004'].iloc[0]
        # For surviving employees, tenure should be calculated to end of year, not termination date
        # This will be handled by the existing tenure calculation logic for active employees
        assert emp_004_data['employee_tenure'] > 0, "emp_004 should have positive tenure"

        # Verify all terminated employees have proper status
        terminated_employees = result[result['employee_status_eoy'] == 'Terminated']
        assert len(terminated_employees) == 3, "Should have 3 terminated employees"

        # Verify none of the terminated employees have zero tenure
        for _, emp in terminated_employees.iterrows():
            assert emp['employee_tenure'] > 0, f"Employee {emp[EMP_ID]} should have positive tenure, got {emp['employee_tenure']}"
            assert pd.notna(emp['employee_tenure_band']), f"Employee {emp[EMP_ID]} should have valid tenure band"


if __name__ == "__main__":
    pytest.main([__file__])
