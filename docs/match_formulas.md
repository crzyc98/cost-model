## Common Types of 401(k) Matching Formulas

There are several standard 401(k) employer matching formulas used in the U.S., each with its own structure and rationale. Here are the most prevalent types, with examples you can provide to your engineer for forecasting tool development:

**Single-Tier (Simple) Match**

- The employer matches a fixed percentage of the employee’s contribution, up to a certain percentage of pay.
- **Example:** 50% match on the first 6% of pay.
  - If an employee earns $60,000 and contributes 6% ($3,600), the employer matches 50% of that ($1,800)[1][3][4].
- **Example:** 100% match on the first 4% of pay.
  - Employee contributes 4% of $60,000 ($2,400), employer matches $2,400[2][6].

**Multi-Tier Match**

- The employer matches different percentages at different contribution levels.
- **Example:** 100% match on the first 3% of pay, plus 50% match on the next 2% of pay.
  - For $60,000 salary:
    - 3% = $1,800 matched at 100% = $1,800
    - Next 2% = $1,200 matched at 50% = $600
    - Total match = $2,400[2][4][5][6].
- **Example:** 100% match on the first 1% of pay, then 50% on the next 5%.
  - 1% = $600 matched at 100% = $600
  - Next 5% = $3,000 matched at 50% = $1,500
  - Total match = $2,100[7].

**Dollar Cap Match**

- The employer matches up to a fixed dollar amount, regardless of the employee’s salary or contribution percentage.
- **Example:** 50% match on the first 6% of pay, up to a maximum employer contribution of $2,000 per year[1][4].

**Safe Harbor Match Formulas**

- These are standardized formulas that help employers automatically satisfy certain IRS nondiscrimination tests.
- **Basic Safe Harbor Match:** 100% match on the first 3% of pay, plus 50% on the next 2%.
  - Total possible match: 4% of pay[6][7].
- **Enhanced Safe Harbor Match:** Must be at least as generous as the basic match, often 100% on the first 4% or 6% of pay[6][7].
- **QACA Safe Harbor Match (Qualified Automatic Contribution Arrangement):** 100% on the first 1% of pay, plus 50% on the next 5%.
  - Total possible match: 3.5% of pay[7].

**Stretch Match**

- The employer matches a lower percentage but on a higher portion of pay, incentivizing higher employee contributions.
- **Example:** 25% match up to 12% of pay (maximum match is 3% of pay if the employee contributes 12%)[7].

## Example Table for Engineering Implementation

| Formula Type        | Example Formula                                             | Max Employer Match (%) | Notes                                |
|---------------------|------------------------------------------------------------|-----------------------|--------------------------------------|
| Single-Tier         | 50% on first 6% of pay                                     | 3%                    | Most common                          |
| Single-Tier         | 100% on first 4% of pay                                    | 4%                    | Safe Harbor Enhanced                 |
| Multi-Tier          | 100% on first 3%, 50% on next 2%                           | 4%                    | Safe Harbor Basic                    |
| Multi-Tier          | 100% on first 1%, 50% on next 5%                           | 3.5%                  | QACA Safe Harbor                     |
| Dollar Cap          | 50% on first 6%, up to $2,000/year                         | Variable              | Less common                          |
| Stretch Match       | 25% on first 12% of pay                                    | 3%                    | Encourages higher employee deferral  |

## Notes for Implementation

- The formulas may be written in plan documents using different language (e.g., "50 cents on the dollar up to 6%," "100% match up to 4%," etc.)[3].
- Some plans combine matching and nonmatching (profit-sharing) contributions[1].
- Vesting schedules and eligibility rules may further affect employer contributions but are separate from the match formula itself[7].

These examples should allow your engineer to build a flexible forecasting tool that accommodates the most common and IRS-compliant 401(k) matching structures.

## Common Types of 401(k) Matching Formulas

There are several standard 401(k) employer matching formulas used in the U.S., each with its own structure and rationale. Here are the most prevalent types, with examples you can provide to your engineer for forecasting tool development:

**Single-Tier (Simple) Match**

- The employer matches a fixed percentage of the employee’s contribution, up to a certain percentage of pay.
- **Example:** 50% match on the first 6% of pay.
  - If an employee earns $60,000 and contributes 6% ($3,600), the employer matches 50% of that ($1,800)[1][3][4].
- **Example:** 100% match on the first 4% of pay.
  - Employee contributes 4% of $60,000 ($2,400), employer matches $2,400[2][6].

**Multi-Tier Match**

- The employer matches different percentages at different contribution levels.
- **Example:** 100% match on the first 3% of pay, plus 50% match on the next 2% of pay.
  - For $60,000 salary:
    - 3% = $1,800 matched at 100% = $1,800
    - Next 2% = $1,200 matched at 50% = $600
    - Total match = $2,400[2][4][5][6].
- **Example:** 100% match on the first 1% of pay, then 50% on the next 5%.
  - 1% = $600 matched at 100% = $600
  - Next 5% = $3,000 matched at 50% = $1,500
  - Total match = $2,100[7].

**Dollar Cap Match**

- The employer matches up to a fixed dollar amount, regardless of the employee’s salary or contribution percentage.
- **Example:** 50% match on the first 6% of pay, up to a maximum employer contribution of $2,000 per year[1][4].

**Safe Harbor Match Formulas**

- These are standardized formulas that help employers automatically satisfy certain IRS nondiscrimination tests.
- **Basic Safe Harbor Match:** 100% match on the first 3% of pay, plus 50% on the next 2%.
  - Total possible match: 4% of pay[6][7].
- **Enhanced Safe Harbor Match:** Must be at least as generous as the basic match, often 100% on the first 4% or 6% of pay[6][7].
- **QACA Safe Harbor Match (Qualified Automatic Contribution Arrangement):** 100% on the first 1% of pay, plus 50% on the next 5%.
  - Total possible match: 3.5% of pay[7].

**Stretch Match**

- The employer matches a lower percentage but on a higher portion of pay, incentivizing higher employee contributions.
- **Example:** 25% match up to 12% of pay (maximum match is 3% of pay if the employee contributes 12%)[7].

## Example Table for Engineering Implementation

| Formula Type        | Example Formula                                             | Max Employer Match (%) | Notes                                |
|---------------------|------------------------------------------------------------|-----------------------|--------------------------------------|
| Single-Tier         | 50% on first 6% of pay                                     | 3%                    | Most common                          |
| Single-Tier         | 100% on first 4% of pay                                    | 4%                    | Safe Harbor Enhanced                 |
| Multi-Tier          | 100% on first 3%, 50% on next 2%                           | 4%                    | Safe Harbor Basic                    |
| Multi-Tier          | 100% on first 1%, 50% on next 5%                           | 3.5%                  | QACA Safe Harbor                     |
| Dollar Cap          | 50% on first 6%, up to $2,000/year                         | Variable              | Less common                          |
| Stretch Match       | 25% on first 12% of pay                                    | 3%                    | Encourages higher employee deferral  |

## Notes for Implementation

- The formulas may be written in plan documents using different language (e.g., "50 cents on the dollar up to 6%," "100% match up to 4%," etc.)[3].
- Some plans combine matching and nonmatching (profit-sharing) contributions[1].
- Vesting schedules and eligibility rules may further affect employer contributions but are separate from the match formula itself[7].

These examples should allow your engineer to build a flexible forecasting tool that accommodates the most common and IRS-compliant 401(k) matching structures.
