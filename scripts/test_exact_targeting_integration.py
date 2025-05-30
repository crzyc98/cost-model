#!/usr/bin/env python3
"""
Test script to verify exact headcount targeting integration.

This script tests the integrated exact targeting logic to ensure it properly
enforces the YAML-configured target growth and new hire termination rates.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from types import SimpleNamespace

from cost_model.engines.run_one_year.orchestrator.hiring import HiringOrchestrator
from cost_model.engines.run_one_year.orchestrator.base import YearContext
from cost_model.state.schema import EMP_ID, EMP_ACTIVE, EMP_HIRE_DATE, EMP_GROSS_COMP


def create_test_snapshot(num_employees: int = 100) -> pd.DataFrame:
    """Create a test workforce snapshot."""
    employee_ids = [f"EMP_{i:04d}" for i in range(num_employees)]
    
    snapshot_data = {
        EMP_ID: employee_ids,
        EMP_ACTIVE: [True] * num_employees,
        EMP_HIRE_DATE: pd.to_datetime('2024-01-01'),  # All hired before 2025
        EMP_GROSS_COMP: [65000.0] * num_employees,
        'employee_termination_date': pd.NaT,
        'employee_birth_date': pd.to_datetime('1990-01-01'),
        'employee_deferral_rate': 0.05,
        'employee_tenure_band': '1-3',
        'employee_tenure': 1.5,
        'employee_level': 1,
        'employee_level_source': 'existing',
        'employee_exited': False,
        'simulation_year': 2024
    }
    
    snapshot = pd.DataFrame(snapshot_data)
    snapshot = snapshot.set_index(EMP_ID, drop=False)
    
    return snapshot


def create_test_global_params() -> SimpleNamespace:
    """Create test global parameters with exact targeting values."""
    attrition = SimpleNamespace()
    attrition.new_hire_termination_rate = 0.25  # 25% as configured in YAML
    
    global_params = SimpleNamespace()
    global_params.target_growth = 0.03  # 3% as configured in YAML
    global_params.attrition = attrition
    
    return global_params


def create_test_hazard_slice() -> pd.DataFrame:
    """Create a minimal hazard slice for testing."""
    return pd.DataFrame({
        'simulation_year': [2025],
        'employee_level': [1],
        'tenure_band': ['1-3'],
        'term_rate': [0.15],
        'new_hire_termination_rate': [0.25],
        'cola_pct': [0.02],
        'merit_raise_pct': [0.03]
    })


def create_test_year_context() -> YearContext:
    """Create a test year context."""
    global_params = create_test_global_params()
    hazard_slice = create_test_hazard_slice()
    
    # Create a simple RNG
    rng = np.random.default_rng(42)
    
    year_context = SimpleNamespace()
    year_context.year = 2025
    year_context.global_params = global_params
    year_context.hazard_slice = hazard_slice
    year_context.year_rng = rng
    year_context.as_of = pd.Timestamp('2025-01-01')
    year_context.end_of_year = pd.Timestamp('2025-12-31')
    year_context.census_template_path = None
    year_context.deterministic_term = True
    
    return year_context


def test_exact_targeting_integration():
    """Test the exact targeting integration."""
    print("=== Testing Exact Headcount Targeting Integration ===\n")
    
    # Create test data
    snapshot = create_test_snapshot(100)  # Start with 100 employees
    year_context = create_test_year_context()
    
    # Simulate experienced terminations (18 employees as in your example)
    experienced_terms = 18
    survivors_snapshot = snapshot.iloc[experienced_terms:].copy()  # Remove first 18
    
    print(f"Test scenario:")
    print(f"- Start of year: {len(snapshot)} employees")
    print(f"- Experienced terminations: {experienced_terms}")
    print(f"- Survivors after experienced terms: {len(survivors_snapshot)}")
    print(f"- Target growth rate: {year_context.global_params.target_growth:.1%}")
    print(f"- New hire termination rate: {year_context.global_params.attrition.new_hire_termination_rate:.1%}")
    print()
    
    # Initialize hiring orchestrator
    hiring_orchestrator = HiringOrchestrator()
    
    # Test the exact targeting calculation
    try:
        start_count, survivor_count, target_eoy, gross_hires, forced_terminations = hiring_orchestrator._compute_targets(
            snapshot=survivors_snapshot,
            year_context=year_context,
            start_count=len(snapshot)  # Pass the original start count
        )
        
        print("Exact targeting results:")
        print(f"- Start count: {start_count}")
        print(f"- Survivor count: {survivor_count}")
        print(f"- Target EOY: {target_eoy}")
        print(f"- Gross hires needed: {gross_hires}")
        print(f"- Forced terminations needed: {forced_terminations}")
        print()
        
        # Verify the math
        expected_target = round(start_count * (1 + year_context.global_params.target_growth))
        expected_net_needed = expected_target - survivor_count
        expected_gross = round(expected_net_needed / (1 - year_context.global_params.attrition.new_hire_termination_rate))
        
        print("Verification:")
        print(f"- Expected target: {expected_target} (actual: {target_eoy}) ✓" if target_eoy == expected_target else f"- Expected target: {expected_target} (actual: {target_eoy}) ✗")
        print(f"- Expected gross hires: {expected_gross} (actual: {gross_hires}) ✓" if gross_hires == expected_gross else f"- Expected gross hires: {expected_gross} (actual: {gross_hires}) ✗")
        print(f"- Expected forced terms: 0 (actual: {forced_terminations}) ✓" if forced_terminations == 0 else f"- Expected forced terms: 0 (actual: {forced_terminations}) ✗")
        print()
        
        # Test new hire termination calculation
        expected_nh_terms = round(gross_hires * year_context.global_params.attrition.new_hire_termination_rate)
        expected_final_headcount = survivor_count + gross_hires - expected_nh_terms - forced_terminations
        
        print("Expected simulation outcome:")
        print(f"- Gross hires: {gross_hires}")
        print(f"- Expected new hire terminations: {expected_nh_terms}")
        print(f"- Expected final headcount: {expected_final_headcount}")
        print(f"- Target headcount: {target_eoy}")
        print(f"- Difference: {expected_final_headcount - target_eoy}")
        
        if expected_final_headcount == target_eoy:
            print("✅ Exact targeting is working correctly!")
        else:
            print("❌ Exact targeting has a discrepancy")
            
    except Exception as e:
        print(f"❌ Error testing exact targeting: {e}")
        import traceback
        traceback.print_exc()


def test_forced_termination_scenario():
    """Test a scenario that requires forced terminations."""
    print("\n" + "="*60)
    print("=== Testing Forced Termination Scenario ===\n")
    
    # Create test data
    snapshot = create_test_snapshot(100)
    year_context = create_test_year_context()
    
    # Simulate very low experienced terminations (only 2 employees)
    # This should trigger forced terminations
    experienced_terms = 2
    survivors_snapshot = snapshot.iloc[experienced_terms:].copy()  # Remove first 2
    
    # Use negative growth to force the scenario
    year_context.global_params.target_growth = -0.05  # -5% shrinkage
    
    print(f"Forced termination test scenario:")
    print(f"- Start of year: {len(snapshot)} employees")
    print(f"- Experienced terminations: {experienced_terms}")
    print(f"- Survivors after experienced terms: {len(survivors_snapshot)}")
    print(f"- Target growth rate: {year_context.global_params.target_growth:.1%}")
    print()
    
    # Initialize hiring orchestrator
    hiring_orchestrator = HiringOrchestrator()
    
    try:
        start_count, survivor_count, target_eoy, gross_hires, forced_terminations = hiring_orchestrator._compute_targets(
            snapshot=survivors_snapshot,
            year_context=year_context,
            start_count=len(snapshot)
        )
        
        print("Forced termination results:")
        print(f"- Start count: {start_count}")
        print(f"- Survivor count: {survivor_count}")
        print(f"- Target EOY: {target_eoy}")
        print(f"- Gross hires needed: {gross_hires}")
        print(f"- Forced terminations needed: {forced_terminations}")
        print()
        
        if forced_terminations > 0:
            print(f"✅ Forced termination logic is working! Need to terminate {forced_terminations} employees.")
        else:
            print("❌ Expected forced terminations but got 0")
            
    except Exception as e:
        print(f"❌ Error testing forced terminations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_exact_targeting_integration()
    test_forced_termination_scenario()
