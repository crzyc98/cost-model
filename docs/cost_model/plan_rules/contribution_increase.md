
Run Example:
>>> import pandas as pd
>>> from cost_model.plan_rules.contribution_increase import run
>>> from cost_model.config.plan_rules import ContributionIncreaseConfig
>>> # Build snapshot: current deferral rates
>>> snap = pd.DataFrame({
...     EMP_ID:['E1','E2'],
...     EMP_DEFERRAL_RATE:[0.02, 0.05]
... }).set_index(EMP_ID)
>>> # Build prior events: enrollment at 2% for E1, 4% for E2
>>> events = pd.DataFrame([
...     {'event_id':'1','event_time':pd.Timestamp('2025-01-01'),
...      EMP_ID:'E1','event_type':'EVT_ENROLL','value_num':0.02,'value_json':None,'meta':None},
...     {'event_id':'2','event_time':pd.Timestamp('2025-01-01'),
...      EMP_ID:'E2','event_type':'EVT_ENROLL','value_num':0.04,'value_json':None,'meta':None},
... ])
>>> cfg = ContributionIncreaseConfig(min_increase_pct=0.01, event_type='EVT_CONTRIB_INCR')
>>> evs = run(snap, events, pd.Timestamp('2025-06-30'), cfg)
>>> df = evs[0]
>>> set(df[EMP_ID])
{'E2'}
>>> df.loc[df[EMP_ID]=='E2','value_json'].iloc[0]
'{"old_rate": 0.04, "new_rate": 0.05, "delta": 0.010000000000000009}'
