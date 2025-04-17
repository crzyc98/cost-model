Okay, I've analyzed the output data showing the discrepancy and reviewed the relevant sections of your model.py code again.

Here's the situation:

Observation: The output CSV for Year = 2025 shows agents with IsNewHire = TRUE but HireDate values from 2024.
Code Logic (_create_new_hires): The code correctly generates random_hire_date values that fall within the current simulation year (self.year). When self.year is 2025, the hire dates generated in that step will be in 2025.
Code Logic (step - is_new_hire flag):
The is_new_hire flag is set to True only for agents created within the current step() call (inside _create_new_hires).
Crucially, the flag is reset to False for all agents at the very beginning of the next step().
Data collection (self.datacollector.collect(self)) happens at the end of the step().
Expected Behavior:
When data is collected at the end of the step simulating Year 2024 (let's assume this is Step 0 or 1 depending on Mesa's indexing): Agents created in this step should have HireDate in 2024 and is_new_hire = True.
When data is collected at the end of the step simulating Year 2025 (the next step): Agents created in this step should have HireDate in 2025 and is_new_hire = True. The agents created back in the 2024 step should now have is_new_hire = False because the flag was reset at the beginning of the 2025 step.
Conclusion & Likely Cause:

The output you are seeing (Year=2025, IsNewHire=TRUE, HireDate=2024) directly contradicts the expected behavior based on the code's logic for generating hire dates and managing the is_new_hire flag.

The most probable cause is an issue in how the Year column is being generated or interpreted in the final output processing within your run_abm_simulation.py script, specifically when processing the data collected by Mesa's DataCollector.

Mesa Steps vs. Calendar Year: Mesa's DataCollector typically indexes results by simulation Step (often starting at 0 or 1). Your run_abm_simulation.py script then calculates a Year column, likely using Year = Step + start_year or a similar formula.
Off-by-One Potential: If Mesa's step count (Step) doesn't align perfectly with the intended calendar year simulation (e.g., Step 1 actually represents the simulation of start_year, not the year after start_year), the calculated Year column in your output CSV will be shifted.
Recommendation for the Engineer:

Verify Year Calculation: Carefully examine the run_abm_simulation.py script, specifically the section after agent_data = model.datacollector.get_agent_vars_dataframe().
Determine if Mesa's Step index starts at 0 or 1 in the collected agent_data.
Adjust the calculation agent_data['Year'] = ... to correctly map the simulation Step to the actual calendar Year being simulated in that step.
If Step starts at 0 and the first step simulates start_year, the formula should likely be Year = Step + start_year.
If Step starts at 1 and the first step simulates start_year, the formula should likely be Year = Step + start_year - 1.
Examine Raw Step Data: Before calculating the Year column, print the raw agent_data DataFrame (or its head) including the Step index/column. Check the HireDate and is_new_hire values for specific agents across consecutive Step values to confirm the underlying data collected by Mesa is correct according to the simulation logic.
This investigation should reveal the misalignment between the simulation step and the reported calendar year, allowing the engineer to correct the Year calculation in the output processing and resolve the discrepancy. The agent creation and flag logic within model.py appears to be functioning as intended based on the code provided.