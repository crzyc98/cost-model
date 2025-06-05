#!/usr/bin/env python3
"""
Fix for headcount plotting issue - diagnose and provide corrected plotting code.
"""

import pandas as pd
import plotly.express as px
from pathlib import Path

def diagnose_emp_status_df(emp_status_df):
    """Diagnose the structure of emp_status_df and suggest fixes"""
    
    print("🔍 DIAGNOSING emp_status_df STRUCTURE")
    print("=" * 50)
    
    if emp_status_df.empty:
        print("❌ DataFrame is empty!")
        return None
    
    print(f"📊 DataFrame shape: {emp_status_df.shape}")
    print(f"📋 Available columns: {list(emp_status_df.columns)}")
    print(f"📈 Data types:\n{emp_status_df.dtypes}")
    
    print(f"\n📄 First few rows:")
    print(emp_status_df.head())
    
    # Check for common column name variations
    possible_year_cols = [col for col in emp_status_df.columns if 'year' in col.lower()]
    possible_active_cols = [col for col in emp_status_df.columns if 'active' in col.lower()]
    possible_term_cols = [col for col in emp_status_df.columns if 'term' in col.lower()]
    
    print(f"\n🔍 COLUMN ANALYSIS:")
    print(f"   Year-like columns: {possible_year_cols}")
    print(f"   Active-like columns: {possible_active_cols}")
    print(f"   Termination-like columns: {possible_term_cols}")
    
    return {
        'year_cols': possible_year_cols,
        'active_cols': possible_active_cols,
        'term_cols': possible_term_cols,
        'all_cols': list(emp_status_df.columns)
    }

def create_adaptive_plot(emp_status_df):
    """Create a plot that adapts to the actual column structure"""
    
    if emp_status_df.empty:
        print("❌ Cannot create plot: DataFrame is empty")
        return None
    
    # Diagnose the structure
    structure = diagnose_emp_status_df(emp_status_df)
    
    print(f"\n🎯 CREATING ADAPTIVE PLOT")
    print("=" * 30)
    
    # Try to identify the year column
    year_col = None
    for col in structure['year_cols']:
        if emp_status_df[col].dtype in ['int64', 'float64'] or 'year' in col.lower():
            year_col = col
            break
    
    if not year_col and structure['year_cols']:
        year_col = structure['year_cols'][0]
    elif not year_col:
        # Look for any numeric column that could be a year
        numeric_cols = emp_status_df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if emp_status_df[col].min() > 2000 and emp_status_df[col].max() < 2100:
                year_col = col
                break
    
    if not year_col:
        print("❌ Could not identify year column")
        return None
    
    print(f"✅ Using year column: '{year_col}'")
    
    # Identify plottable columns (numeric, not the year column)
    numeric_cols = emp_status_df.select_dtypes(include=['int64', 'float64']).columns
    plot_cols = [col for col in numeric_cols if col != year_col and not col.endswith('Rate')]
    
    print(f"📊 Plottable columns: {plot_cols}")
    
    if not plot_cols:
        print("❌ No suitable columns found for plotting")
        return None
    
    # Create the plot
    try:
        # Prepare data for plotting
        plot_data = emp_status_df[[year_col] + plot_cols].copy()
        
        # Create the interactive plot
        fig = px.line(plot_data, x=year_col, y=plot_cols,
                     title='Headcount Trends by Employment Status',
                     markers=True,
                     labels={'value': 'Count', 'variable': 'Status'},
                     template='plotly_white')
        
        # Update layout
        fig.update_layout(
            xaxis=dict(tickmode='linear', dtick=1),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        print("✅ Plot created successfully!")
        return fig
        
    except Exception as e:
        print(f"❌ Error creating plot: {e}")
        return None

def generate_fixed_code(emp_status_df):
    """Generate the corrected code based on actual DataFrame structure"""
    
    structure = diagnose_emp_status_df(emp_status_df)
    
    # Identify year column
    year_col = None
    for col in structure['year_cols']:
        year_col = col
        break
    
    if not year_col:
        numeric_cols = emp_status_df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if emp_status_df[col].min() > 2000 and emp_status_df[col].max() < 2100:
                year_col = col
                break
    
    # Identify plottable columns
    numeric_cols = emp_status_df.select_dtypes(include=['int64', 'float64']).columns
    plot_cols = [col for col in numeric_cols if col != year_col and not col.endswith('Rate')]
    
    print(f"\n💻 CORRECTED CODE:")
    print("=" * 20)
    
    code = f"""
# Fixed plotting code based on your actual DataFrame structure
if not emp_status_df.empty:
    # Check available columns
    print(f"Available columns: {{list(emp_status_df.columns)}}")
    
    # Use the actual column names from your DataFrame
    year_col = '{year_col}'
    plot_cols = {plot_cols}
    
    # Prepare data for plotting
    plot_data = emp_status_df[[year_col] + plot_cols].copy()
    
    # Create the interactive plot
    fig = px.line(plot_data, x=year_col, y=plot_cols,
                 title='Headcount Trends by Employment Status',
                 markers=True,
                 labels={{'value': 'Count', 'variable': 'Status'}},
                 template='plotly_white')
    
    # Update layout
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    fig.show()
else:
    print("❌ emp_status_df is empty - cannot create plot")
"""
    
    print(code)
    return code

# Example usage - you would call this with your actual emp_status_df
if __name__ == "__main__":
    print("🔧 Headcount Plot Diagnostic Tool")
    print("=" * 40)
    print("To use this tool:")
    print("1. Import this module in your notebook")
    print("2. Call diagnose_emp_status_df(emp_status_df)")
    print("3. Call generate_fixed_code(emp_status_df)")
    print("4. Use the generated code in your notebook")
