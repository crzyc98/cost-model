# Employer Match and Non-Elective Contribution Formulas

## Implementation in the Cost Model

The cost model supports both employer match and non-elective contribution (NEC) formulas through a tiered structure defined in the YAML configuration. This approach allows for flexible modeling of various plan designs.

### Employer Match Configuration

Match formulas are defined in the `employer_match` section of the `plan_rules` configuration:

```yaml
plan_rules:
  employer_match:
    tiers:
      - match_rate: 1.0        # 100% match rate
        cap_deferral_pct: 0.03  # on first 3% of pay
      - match_rate: 0.5        # 50% match rate
        cap_deferral_pct: 0.02  # on next 2% of pay
    dollar_cap: 5000          # Optional annual dollar cap
```

### Non-Elective Contribution Configuration

NEC formulas are defined in the `employer_nec` section:

```yaml
plan_rules:
  employer_nec:
    rate: 0.03  # 3% of eligible compensation
```

## Common Types of 401(k) Matching Formulas

There are several standard 401(k) employer matching formulas used in the U.S., each with its own structure and rationale:

### Single-Tier (Simple) Match

- The employer matches a fixed percentage of the employee's contribution, up to a certain percentage of pay.
- **Example:** 50% match on the first 6% of pay.
  - If an employee earns $60,000 and contributes 6% ($3,600), the employer matches 50% of that ($1,800).
- **Example:** 100% match on the first 4% of pay.
  - Employee contributes 4% of $60,000 ($2,400), employer matches $2,400.

### Multi-Tier Match

- The employer matches different percentages at different contribution levels.
- **Example:** 100% match on the first 3% of pay, plus 50% match on the next 2% of pay.
  - For $60,000 salary:
    - 3% = $1,800 matched at 100% = $1,800
    - Next 2% = $1,200 matched at 50% = $600
    - Total match = $2,400
- **Example:** 100% match on the first 1% of pay, then 50% on the next 5%.
  - 1% = $600 matched at 100% = $600
  - Next 5% = $3,000 matched at 50% = $1,500
  - Total match = $2,100

### Dollar Cap Match

- The employer matches up to a fixed dollar amount, regardless of the employee's salary or contribution percentage.
- **Example:** 50% match on the first 6% of pay, up to a maximum employer contribution of $2,000 per year.

### Safe Harbor Match Formulas

- These are standardized formulas that help employers automatically satisfy certain IRS nondiscrimination tests.
- **Basic Safe Harbor Match:** 100% match on the first 3% of pay, plus 50% on the next 2%.
  - Total possible match: 4% of pay.
- **Enhanced Safe Harbor Match:** Must be at least as generous as the basic match, often 100% on the first 4% or 6% of pay.
- **QACA Safe Harbor Match (Qualified Automatic Contribution Arrangement):** 100% on the first 1% of pay, plus 50% on the next 5%.
  - Total possible match: 3.5% of pay.

### Stretch Match

- The employer matches a lower percentage but on a higher portion of pay, incentivizing higher employee contributions.
- **Example:** 25% match up to 12% of pay (maximum match is 3% of pay if the employee contributes 12%).

## Example Table for Engineering Implementation

| Formula Type        | Example Formula                                             | Max Employer Match (%) | Notes                                | YAML Configuration |
|---------------------|------------------------------------------------------------|-----------------------|--------------------------------------|--------------------|
| Single-Tier         | 50% on first 6% of pay                                     | 3%                    | Most common                          | `tiers: [{match_rate: 0.5, cap_deferral_pct: 0.06}]` |
| Single-Tier         | 100% on first 4% of pay                                    | 4%                    | Safe Harbor Enhanced                 | `tiers: [{match_rate: 1.0, cap_deferral_pct: 0.04}]` |
| Multi-Tier          | 100% on first 3%, 50% on next 2%                           | 4%                    | Safe Harbor Basic                    | `tiers: [{match_rate: 1.0, cap_deferral_pct: 0.03}, {match_rate: 0.5, cap_deferral_pct: 0.02}]` |
| Multi-Tier          | 100% on first 1%, 50% on next 5%                           | 3.5%                  | QACA Safe Harbor                     | `tiers: [{match_rate: 1.0, cap_deferral_pct: 0.01}, {match_rate: 0.5, cap_deferral_pct: 0.05}]` |
| Dollar Cap          | 50% on first 6%, up to $2,000/year                         | Variable              | Less common                          | `tiers: [{match_rate: 0.5, cap_deferral_pct: 0.06}], dollar_cap: 2000` |
| Stretch Match       | 25% on first 12% of pay                                    | 3%                    | Encourages higher employee deferral  | `tiers: [{match_rate: 0.25, cap_deferral_pct: 0.12}]` |

## Implementation Notes

- The cost model applies IRS limits to contributions as defined in the `irs_limits` section of the configuration.
- Catch-up contributions for employees over the catch-up eligibility age are handled automatically.
- Non-elective contributions are applied to all eligible employees regardless of their deferral rate.
- The model handles prorated contributions for mid-year hires and terminations.
- Vesting schedules and eligibility rules are handled separately from the match formula itself.
