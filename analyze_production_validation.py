#!/usr/bin/env python3
"""
Analyze the production validation results for Campaign 6 config_101.
"""
import pandas as pd
import numpy as np

def main():
    print("=== PRODUCTION VALIDATION RESULTS - CAMPAIGN 6 CONFIG_101 ===")
    print()
    
    try:
        # Load the summary statistics
        summary_df = pd.read_parquet('validation_run_production_fixed/config_101_production_validation_fixed_summary_statistics.parquet')
        print("Summary Statistics:")
        print(summary_df.to_string(index=False))
        print()
        
        # Load consolidated snapshots for detailed analysis
        consolidated_df = pd.read_parquet('validation_run_production_fixed/consolidated_snapshots.parquet')
        print("=== DETAILED YEAR-BY-YEAR ANALYSIS ===")
        
        # Calculate key metrics by year
        for year in sorted(consolidated_df['simulation_year'].unique()):
            year_data = consolidated_df[consolidated_df['simulation_year'] == year]
            active_employees = year_data[year_data['active'] == True]
            
            print(f'\nYear {year}:')
            print(f'  Total Employees: {len(year_data)}')
            print(f'  Active Employees: {len(active_employees)}')
            
            if len(active_employees) > 0:
                avg_comp = active_employees['employee_gross_compensation'].mean()
                total_payroll = active_employees['employee_gross_compensation'].sum()
                print(f'  Average Compensation: ${avg_comp:,.2f}')
                print(f'  Total Payroll: ${total_payroll:,.2f}')
                
                # Tenure distribution
                if 'employee_tenure_band' in active_employees.columns:
                    tenure_dist = active_employees['employee_tenure_band'].value_counts().sort_index()
                    print(f'  Tenure Distribution: {dict(tenure_dist)}')
        
        # Calculate growth metrics
        initial_year = consolidated_df['simulation_year'].min()
        final_year = consolidated_df['simulation_year'].max()
        
        initial_active = len(consolidated_df[(consolidated_df['simulation_year'] == initial_year) & (consolidated_df['active'] == True)])
        final_active = len(consolidated_df[(consolidated_df['simulation_year'] == final_year) & (consolidated_df['active'] == True)])
        
        initial_payroll = consolidated_df[(consolidated_df['simulation_year'] == initial_year) & (consolidated_df['active'] == True)]['employee_gross_compensation'].sum()
        final_payroll = consolidated_df[(consolidated_df['simulation_year'] == final_year) & (consolidated_df['active'] == True)]['employee_gross_compensation'].sum()
        
        headcount_growth = (final_active - initial_active) / initial_active * 100
        payroll_growth = (final_payroll - initial_payroll) / initial_payroll * 100
        
        print(f'\n=== OVERALL PERFORMANCE METRICS ===')
        print(f'Headcount Growth: {headcount_growth:.2f}% (Target: 3.0%)')
        print(f'Payroll Growth: {payroll_growth:.2f}%')
        print(f'Initial Active Employees: {initial_active}')
        print(f'Final Active Employees: {final_active}')
        print(f'Net Change: {final_active - initial_active} employees')
        
        # Load employment status summary
        try:
            employment_df = pd.read_parquet('validation_run_production_fixed/config_101_production_validation_fixed_employment_status_summary.parquet')
            print(f'\n=== EMPLOYMENT STATUS SUMMARY ===')
            print(employment_df.to_string(index=False))
        except Exception as e:
            print(f"Could not load employment status summary: {e}")
            
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
