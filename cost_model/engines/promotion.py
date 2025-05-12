# cost_model/engines/promotion.py
"""
Engine for simulating employee promotions and associated merit raises during workforce simulations.

## QuickStart

To simulate employee promotions programmatically:

```python
import pandas as pd
import numpy as np
from pathlib import Path
from cost_model.engines.promotion import promote
from cost_model.utils.columns import EMP_ID, EMP_ROLE, EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_TERM_DATE

# Create a sample workforce snapshot
snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'],
    'employee_role': ['Engineer', 'Manager', 'Engineer', 'Analyst', 'Senior Engineer'],
    'employee_gross_compensation': [75000.0, 85000.0, 65000.0, 55000.0, 95000.0],
    'active': [True, True, True, True, True],
    'employee_hire_date': [
        pd.Timestamp('2023-06-01'),
        pd.Timestamp('2022-03-15'),
        pd.Timestamp('2024-01-10'),
        pd.Timestamp('2024-11-05'),
        pd.Timestamp('2021-05-20')
    ],
    'employee_termination_date': [None, None, None, None, None],
    'tenure_band': ['1-3', '3-5', '1-3', '0-1', '3-5'],
    # Flag employees eligible for promotion
    'eligible_for_promotion': [True, False, False, False, True]
}).set_index('employee_id')

# Define promotion rules
promotion_rules = {
    # Role progression map
    'next_title': {
        'Engineer': 'Senior Engineer',
        'Senior Engineer': 'Principal Engineer',
        'Analyst': 'Senior Analyst',
        'Senior Analyst': 'Principal Analyst',
        'Manager': 'Senior Manager',
        'Senior Manager': 'Director'
    },
    # Merit raise percentage for promotions
    'merit_pct': 0.15  # 15% raise with promotion
}

# Set the promotion and raise dates
promotion_date = pd.Timestamp('2025-07-01')  # Promotions effective July 1
raise_date = pd.Timestamp('2025-07-15')      # Raises effective July 15 (two weeks later)

# Generate promotion and raise events
promotion_events = promote(
    snapshot=snapshot.copy(),
    rules=promotion_rules,
    promo_time=promotion_date,
    raise_time=raise_date
)

# Examine the promotion events
if promotion_events and len(promotion_events) >= 2:
    # First DataFrame contains promotion events
    promo_df = promotion_events[0]
    # Second DataFrame contains raise events
    raise_df = promotion_events[1]
    
    # Check if we have promotion events
    if not promo_df.empty:
        print(f"Generated {len(promo_df)} promotion events:")
        
        # Display promotion details
        print("\nPromotion details:")
        for _, event in promo_df.iterrows():
            emp_id = event['employee_id']
            event_time = event['event_time']
            value_json = event['value_json']
            
            # Parse the JSON data
            if value_json and pd.notna(value_json):
                import json
                data = json.loads(value_json)
                old_role = data.get('from', 'Unknown')
                new_role = data.get('to', 'Unknown')
                
                print(f"  {emp_id}: {old_role} → {new_role} (Effective: {event_time.strftime('%Y-%m-%d')})")
        
        # Update the snapshot with new roles
        for _, event in promo_df.iterrows():
            emp_id = event['employee_id']
            value_json = event['value_json']
            
            if value_json and pd.notna(value_json):
                data = json.loads(value_json)
                new_role = data.get('to', None)
                
                if new_role and emp_id in snapshot.index:
                    snapshot.at[emp_id, 'employee_role'] = new_role
    else:
        print("No promotion events generated")
    
    # Check if we have raise events
    if not raise_df.empty:
        print(f"\nGenerated {len(raise_df)} merit raise events:")
        
        # Display raise details
        print("\nMerit raise details:")
        for _, event in raise_df.iterrows():
            emp_id = event['employee_id']
            event_time = event['event_time']
            raise_amount = event['value_num']
            
            # Get the old compensation from the snapshot
            old_comp = snapshot.loc[emp_id, 'employee_gross_compensation']
            
            # Calculate the new compensation
            new_comp = old_comp + raise_amount
            pct_increase = (raise_amount / old_comp) * 100
            
            print(f"  {emp_id}: ${old_comp:,.2f} → ${new_comp:,.2f} (+${raise_amount:,.2f}, +{pct_increase:.1f}%) (Effective: {event_time.strftime('%Y-%m-%d')})")
        
        # Update the snapshot with new compensation values
        for _, event in raise_df.iterrows():
            emp_id = event['employee_id']
            raise_amount = event['value_num']
            
            if emp_id in snapshot.index:
                old_comp = snapshot.at[emp_id, 'employee_gross_compensation']
                snapshot.at[emp_id, 'employee_gross_compensation'] = old_comp + raise_amount
    else:
        print("\nNo merit raise events generated")
    
    # Analyze the updated workforce
    print("\nUpdated workforce after promotions and raises:")
    for emp_id, row in snapshot.loc[snapshot['eligible_for_promotion']].iterrows():
        print(f"  {emp_id}: {row['employee_role']} at ${row['employee_gross_compensation']:,.2f}")
    
    # Save the events and updated snapshot
    output_dir = Path('output/promotions')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not promo_df.empty:
        promo_df.to_parquet(output_dir / 'promotion_events.parquet')
    
    if not raise_df.empty:
        raise_df.to_parquet(output_dir / 'merit_raise_events.parquet')
    
    snapshot.to_parquet(output_dir / 'snapshot_after_promotions.parquet')
else:
    print("No promotion or raise events generated")
```

This demonstrates how to simulate employee promotions and associated merit raises based on eligibility criteria and a defined role progression map.
"""

import pandas as pd
import json
from typing import List
from cost_model.state.event_log import EVENT_COLS, EVT_PROMOTION, EVT_RAISE, create_event
from cost_model.utils.columns import EMP_ID, EMP_ROLE, EMP_GROSS_COMP

def promote(
    snapshot: pd.DataFrame,
    rules: dict,
    promo_time: pd.Timestamp,
    raise_time: pd.Timestamp = None
) -> List[pd.DataFrame]:
    """
    Vectorized promotion and merit-raise event emission.
    - Uses rules dict for role hierarchy (rules['next_title']) and merit pct (rules['merit_pct']).
    - Allows separate timestamps for promotions and raises (raise_time defaults to promo_time).
    - Returns [promotions_df, raises_df] (both EVENT_COLS-compliant).
    """
    if "eligible_for_promotion" not in snapshot.columns:
        # If the column is missing, no one is eligible
        return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)]
    promotees = snapshot[snapshot["eligible_for_promotion"] == True].copy()
    if promotees.empty:
        return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)]

    # Promotion event
    next_title_map = rules.get("next_title", {})
    merit_pct = rules.get("merit_pct", 0.10)
    raise_time = raise_time or promo_time

    # Build new titles
    new_titles = promotees[EMP_ROLE].map(next_title_map).fillna("Senior " + promotees[EMP_ROLE].astype(str))

    promotions = pd.DataFrame({
        "event_time": promo_time,
        EMP_ID: promotees[EMP_ID],
        "event_type": EVT_PROMOTION,
        "value_json": promotees.apply(
            lambda r: json.dumps({"from": r[EMP_ROLE], "to": new_titles.loc[r.name]}), axis=1
        ),
        "meta": "Promotion based on eligibility"
    })
    # Merit raises
    raises = pd.DataFrame({
        "event_time": raise_time,
        EMP_ID: promotees[EMP_ID],
        "event_type": EVT_RAISE,
        "value_num": promotees[EMP_GROSS_COMP] * merit_pct,
        "value_json": promotees.apply(
            lambda r: json.dumps({
                "pct": merit_pct,
                "old_comp": r[EMP_GROSS_COMP],
                "new_comp": r[EMP_GROSS_COMP] * (1 + merit_pct)
            }), axis=1
        ),
        "meta": "Merit raise concurrent with promotion"
    })
    # Assign event_id if needed, slice to schema
    for df in (promotions, raises):
        if "event_id" not in df:
            df["event_id"] = df.index.map(lambda i: f"evt_{i:06d}")
    promotions = promotions[EVENT_COLS]
    raises = raises[EVENT_COLS]
    return [promotions, raises]
