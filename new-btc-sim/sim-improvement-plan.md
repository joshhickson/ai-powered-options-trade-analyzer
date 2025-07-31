Simulator Improvement & Analysis Plan
Objective: To enhance the leverage_war_days.py simulator to more accurately model and predict a range of probable outcomes for the leveraged Bitcoin accumulation strategy.

Current Status: The project is at an advanced stage. A functional simulator exists that implements the core strategy logic, runs against historical data, logs events, and visualizes the results of a single backtest.

Proposed Enhancements
This plan outlines four key improvements to increase the realism and analytical power of the simulation. They are ordered from the most direct and foundational enhancements to more advanced analytical methods.

1. Model Loan Interest Accrual
Description: The current simulation does not appear to model the cost of borrowing (11.5% APR). In reality, accrued interest increases the total loan balance daily, which directly impacts the Loan-to-Value (LTV) ratio and increases the risk of a margin call over time.

Implementation Steps:

Define a DAILY_INTEREST_RATE constant calculated as LOAN_APR / 365.

Within the main simulation loop (for each time step), update the LoanBalanceUSD state variable by adding the interest accrued for that period: LoanBalanceUSD += LoanBalanceUSD * DAILY_INTEREST_RATE.

Ensure all subsequent LTV calculations in that time step use this updated loan balance.

Expected Impact: This will provide a more realistic LTV calculation, leading to a more accurate assessment of the strategy's risk profile and historical performance. It will correctly penalize long periods without a price increase.

2. Incorporate Transaction & Loan Fees
Description: Real-world execution of this strategy involves costs beyond interest. These include fees for trading BTC and potentially fees for originating the loan itself. Omitting them overstates the strategy's profitability.

Implementation Steps:

Add new configuration constants:

TRADING_FEE_PERCENT (e.g., 0.001 for 0.1%).

LOAN_ORIGINATION_FEE_PERCENT (e.g., 0.01 for 1%) or a fixed USD amount.

When simulating a BTC purchase (both the initial buy and subsequent buys after closing a loan), reduce the final BTC amount by the TRADING_FEE_PERCENT.

When a new loan cycle starts, reduce the cash received from the loan by the origination fee. This means the initial cash available to buy more BTC will be slightly lower.

Expected Impact: A more conservative and realistic calculation of the final accumulated BTC, reflecting the true net gains of the strategy.

3. Implement Monte Carlo Analysis
Description: A single backtest shows what happened on one historical path. A Monte Carlo analysis simulates the strategy against thousands of statistically possible future price paths to understand the range of potential outcomes and their probabilities.

Implementation Steps:

Analyze Historical Data: Calculate the mean and standard deviation of the historical daily log returns of the BTC price data. This captures its average drift and volatility.

Build a Path Generator: Create a new function that generates a randomized price path for the simulation period. This is typically done using a Geometric Brownian Motion (GBM) model, which uses the historical drift, volatility, and a random factor to project future prices.

Wrap the Simulator: Encapsulate the existing simulation logic in a master loop that runs for a large number of iterations (e.g., 1,000 or 10,000).

Run and Collect Data: In each iteration, generate a new random price path and run the complete strategy simulation against it. Store the key outcomes of each run (e.g., Final BTC, Was Goal Reached?, Was Liquidation Triggered?, Time to Goal/Liquidation).

Analyze Results: Analyze the distribution of the collected outcomes to answer probabilistic questions (e.g., "What is the probability of liquidation within 2 years?").

Expected Impact: Transforms the tool from a historical review into a predictive model, providing a much richer understanding of the strategy's risk and reward profile.

4. Add Parameter Sensitivity Analysis
Description: The strategy's success is highly dependent on its core rules, such as the MARGIN_CALL_LTV (85%) and the PROFIT_TAKE_PRICE_INCREASE_USD ($30,000). A sensitivity analysis automatically tests how the final outcome changes when these key parameters are adjusted.

Implementation Steps:

Define Parameter Ranges: For key variables, define a range of values to test (e.g., MARGIN_CALL_LTV from 0.80 to 0.90; PROFIT_TAKE_PRICE_INCREASE_USD from $20,000 to $40,000).

Create Outer Loops: Add another layer of loops around the core simulation that iterates through each combination of the defined parameter ranges.

Store and Visualize: Run the simulation for each combination and store the results. A heatmap is an excellent way to visualize the output, showing which parameter combinations lead to the most favorable outcomes.

Expected Impact: Helps to identify the most critical levers in the strategy and allows for data-driven optimization of its rules for maximum robustness and profitability.