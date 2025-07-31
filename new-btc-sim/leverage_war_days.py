#!/usr/bin/env python3
"""
loan_cycle_simulator.py

This script executes a historical backtest of a compounding, leveraged BTC
loan strategy based on the user-defined 'mermaid diagram' logic. It uses
real historical price data to simulate the strategy's performance, including
handling margin calls with a dedicated backup collateral pool.

The goal is to provide a definitive, data-driven answer on the historical
viability and performance of this specific trading strategy.
"""

import datetime as dt
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

# --- Configuration based on Mermaid Diagram ---
INITIAL_USD_CAPITAL = 30000.0
BTC_GOAL = 1.0
LOAN_APR = 0.115
TRADING_FEE_PERCENT = 0.001  # 0.1% fee on all trades
LOAN_ORIGINATION_FEE_PERCENT = 0.01  # 1% of loan amount
# LTV = Loan-to-Value Ratio
MARGIN_CALL_LTV = 0.85
LIQUIDATION_LTV = 0.90
CURE_LTV_TARGET = 0.75 # When curing a margin call, add BTC to reach this LTV
# Exit a loan cycle when the price is $30,000 higher than the entry price
PROFIT_TAKE_PRICE_INCREASE_USD = 30000.0

def get_historical_data(interval: int = 1440, periods: int = 720) -> pd.Series:
    """
    Fetches historical price data. For the simulation, we need a continuous
    stream of high-resolution data.
    """
    timeframe_map = {1440: "days", 60: "hours", 15: "minutes"}
    timeframe_unit = timeframe_map.get(interval, "periods")
    duration = periods
    if timeframe_unit == "days": duration = periods
    elif timeframe_unit == "hours": duration = (interval * periods) / 60
    elif timeframe_unit == "minutes": duration = (interval * periods)

    print(f"📈 Fetching {periods} periods of {interval}-minute BTC/USD data for simulation ({duration:.0f} {timeframe_unit})...")

    since_timestamp = int((dt.datetime.now() - dt.timedelta(days=periods if interval == 1440 else (periods * interval / 1440))).timestamp())

    url = "https://api.kraken.com/0/public/OHLC"
    params = {'pair': 'XBTUSD', 'interval': interval, 'since': since_timestamp}

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if data.get('error'):
            print(f"   ❌ Kraken API Error: {', '.join(data['error'])}")
            return pd.Series(dtype=float)

        pair_name = list(data['result'].keys())[0]
        ohlc_data = data['result'][pair_name]

        df = pd.DataFrame(ohlc_data, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time').sort_index()
        btc_series = df['close'].astype(float)
        print(f"   ✅ Loaded {len(btc_series)} data points from {btc_series.index.min()} to {btc_series.index.max()}.")
        return btc_series
    except Exception as e:
        print(f"   ❌ Could not fetch historical data: {e}")
        return pd.Series(dtype=float)

class Loan:
    """Represents a single loan cycle."""
    def __init__(self, cash_received_from_loan, collateral_btc, entry_price, entry_date):
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.collateral_btc = collateral_btc

        # The actual loan principal is higher due to the origination fee.
        # The fee is added to the principal, so interest accrues on it.
        origination_fee = cash_received_from_loan * LOAN_ORIGINATION_FEE_PERCENT
        self.loan_amount_usd = cash_received_from_loan + origination_fee

        self.deferred_interest_usd = 0.0
        self.is_active = True

        # BTC is purchased with the cash received, minus trading fees.
        btc_bought_before_fees = cash_received_from_loan / entry_price
        self.btc_purchased = btc_bought_before_fees * (1 - TRADING_FEE_PERCENT)

        self.exit_date = None
        self.exit_price = None

    def get_balance(self):
        return self.loan_amount_usd + self.deferred_interest_usd

    def accrue_interest(self):
        daily_rate = LOAN_APR / 365
        interest = self.get_balance() * daily_rate
        self.deferred_interest_usd += interest

    def get_ltv(self, current_price):
        collateral_value = self.collateral_btc * current_price
        if collateral_value == 0: return float('inf')
        return self.get_balance() / collateral_value

    def close(self, exit_price, exit_date):
        self.is_active = False
        self.exit_date = exit_date
        self.exit_price = exit_price
        proceeds_usd = self.btc_purchased * exit_price
        net_usd = proceeds_usd - self.get_balance()
        net_btc = net_usd / exit_price
        return net_btc

class Simulator:
    """Manages the entire loan strategy simulation over a given price series."""
    def __init__(self):
        """Initializes the simulator. State is reset before each run."""
        self._reset_state()

    def _reset_state(self):
        """Resets all state variables to allow for a fresh simulation run."""
        self.log = []
        self.total_btc = 0
        self.backup_btc = 0
        self.current_loan = None
        self.cycle_count = 0
        self.is_liquidated = False

    def log_event(self, timestamp, event_type, details):
        self.log.append({
            'Timestamp': timestamp,
            'EventType': event_type,
            'TotalBTC': self.total_btc,
            'BackupBTC': self.backup_btc,
            'CollateralBTC': self.current_loan.collateral_btc if self.current_loan else 0,
            'LoanBalanceUSD': self.current_loan.get_balance() if self.current_loan else 0,
            'BTCPriceUSD': details.get('price', 0),
            'LTV': self.current_loan.get_ltv(details.get('price', 0)) if self.current_loan else 0,
            'Details': details
        })

    def run(self, price_series: pd.Series, margin_call_ltv: float, profit_take_usd: float):
        """
        Runs a complete simulation against a given price series with dynamic parameters.
        """
        self._reset_state()
        start_price = price_series.iloc[0]
        start_date = price_series.index[0]

        # Step A, B, C: Initial capital deployment
        initial_btc_purchase = INITIAL_USD_CAPITAL / start_price
        self.total_btc = initial_btc_purchase * (1 - TRADING_FEE_PERCENT)
        self.log_event(start_date, 'START', {'price': start_price, 'details': f'Initial buy of {self.total_btc:.4f} BTC'})

        # Main simulation loop through the price history
        for timestamp, price in price_series.items():
            if self.current_loan and self.current_loan.is_active:
                self.current_loan.accrue_interest()
                ltv = self.current_loan.get_ltv(price)

                # Step F -> G -> H: Monitor and handle margin calls
                if ltv >= margin_call_ltv:
                    required_collateral_value = self.current_loan.get_balance() / CURE_LTV_TARGET
                    required_collateral_btc = required_collateral_value / price
                    btc_to_add = required_collateral_btc - self.current_loan.collateral_btc

                    if self.backup_btc >= btc_to_add > 0:
                        self.backup_btc -= btc_to_add
                        self.current_loan.collateral_btc += btc_to_add
                        self.log_event(timestamp, 'MARGIN_CALL_CURED', {'price': price, 'cured_with_btc': btc_to_add})
                    else:
                        self.log_event(timestamp, 'LIQUIDATION', {'price': price, 'details': 'Insufficient backup BTC to cure margin call.'})
                        self.total_btc = 0
                        self.backup_btc = 0
                        self.is_liquidated = True
                        break

                # Step F -> I1: Check for profit-taking exit (fixed $30k increase)
                if price >= self.current_loan.entry_price + profit_take_usd:
                    self.close_cycle(price, timestamp)

            elif self.total_btc > 0:
                self.start_new_cycle(price, timestamp)

        return pd.DataFrame(self.log)

    def start_new_cycle(self, price, timestamp):
        self.cycle_count += 1

        # Logic for the VERY FIRST loan cycle
        if self.cycle_count == 1:
            loan_amount = 10000.0
            collateral_usd_target = 14000.0
            collateral_btc = collateral_usd_target / price
        # Logic for ALL SUBSEQUENT loan cycles
        else:
            # Use half of total BTC as collateral for the next, larger loan
            collateral_btc = self.total_btc / 2.0
            # Loan amount is now dynamic based on the larger collateral
            loan_amount = collateral_btc * price * 0.70 # Safe 70% LTV

        # Check if we have enough BTC for the required collateral
        if self.total_btc < collateral_btc:
            return # Wait until we have enough BTC

        # The loan contract has a minimum loan size
        if loan_amount < 10000:
            return # Can't get a loan, so we wait

        self.backup_btc = self.total_btc - collateral_btc
        self.current_loan = Loan(loan_amount, collateral_btc, price, timestamp)
        self.log_event(timestamp, f'CYCLE_{self.cycle_count}_START', {'price': price, 'loan_amount': loan_amount})

    def close_cycle(self, price, timestamp):
        net_btc_profit = self.current_loan.close(price, timestamp)

        # Recombine all BTC holdings for the next cycle
        self.total_btc = self.current_loan.collateral_btc + self.backup_btc + net_btc_profit
        self.backup_btc = 0

        self.log_event(timestamp, f'CYCLE_{self.cycle_count}_CLOSE', {'price': price, 'net_btc_profit': net_btc_profit})
        self.current_loan = None

        if self.total_btc >= BTC_GOAL:
            self.log_event(timestamp, 'GOAL_REACHED', {'price': price, 'details': f'Total BTC {self.total_btc:.4f} exceeds goal of {BTC_GOAL:.4f} BTC.'})


def generate_gbm_path(start_price, mu, sigma, days, dt=1):
    """Generates a random price path using Geometric Brownian Motion."""
    n_steps = int(days / dt)
    prices = np.zeros(n_steps + 1)
    prices[0] = start_price
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal()
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return prices

def run_monte_carlo_simulation(historical_prices, num_simulations=1000, margin_call_ltv=MARGIN_CALL_LTV, profit_take_usd=PROFIT_TAKE_PRICE_INCREASE_USD, show_progress=True):
    """
    Runs the Monte Carlo simulation for a given set of parameters.
    """
    if show_progress:
        print(f"🇲🇨 Starting Monte Carlo analysis with {num_simulations} simulations...")

    log_returns = np.log(historical_prices / historical_prices.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()
    start_price = historical_prices.iloc[-1] # Start simulation from the last known price
    sim_days = 720 # 2 years

    sim = Simulator()
    results = []

    for i in range(num_simulations):
        if show_progress and (i + 1) % 100 == 0:
            print(f"   ... running simulation {i+1}/{num_simulations}")

        # Generate a new price path
        dates = pd.to_datetime(pd.date_range(start=historical_prices.index[-1], periods=sim_days + 1, freq='D'))
        price_path_array = generate_gbm_path(start_price, mu, sigma, sim_days)
        price_series = pd.Series(price_path_array, index=dates)

        # Run the simulation
        log_df = sim.run(price_series, margin_call_ltv, profit_take_usd)

        # Store the outcome
        final_btc = sim.total_btc
        was_liquidated = sim.is_liquidated
        goal_reached = not log_df[log_df['EventType'] == 'GOAL_REACHED'].empty
        results.append({
            'FinalBTC': final_btc,
            'Liquidated': was_liquidated,
            'GoalReached': goal_reached
        })

    results_df = pd.DataFrame(results)
    print("✅ Monte Carlo analysis finished.")

    # --- Analyze and Print Results ---
    liquidation_rate = results_df['Liquidated'].mean() * 100
    goal_rate = results_df['GoalReached'].mean() * 100
    avg_final_btc = results_df['FinalBTC'].mean()
    median_final_btc = results_df['FinalBTC'].median()
    avg_btc_if_not_liquidated = results_df[~results_df['Liquidated']]['FinalBTC'].mean()


    print("\n" + "=" * 60)
    print("Monte Carlo Simulation Results")
    print("=" * 60)
    print(f"Total Simulations: {num_simulations}")
    print(f"Liquidation Rate: {liquidation_rate:.2f}%")
    print(f"BTC Goal ({BTC_GOAL}) Reached Rate: {goal_rate:.2f}%")
    print(f"Average Final BTC (all outcomes): {avg_final_btc:.4f} BTC")
    print(f"Median Final BTC (all outcomes): {median_final_btc:.4f} BTC")
    print(f"Average Final BTC (non-liquidated): {avg_btc_if_not_liquidated:.4f} BTC")
    print("-" * 60)

    # Optional: Plot distribution of outcomes
    export_dir = Path("exports") / f"monte_carlo_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.hist(results_df['FinalBTC'], bins=50, edgecolor='black')
    plt.title('Distribution of Final BTC Holdings')
    plt.xlabel('Final BTC')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(export_dir / 'monte_carlo_btc_distribution.png')
    print(f"💾 Saved distribution plot to {export_dir / 'monte_carlo_btc_distribution.png'}")
    results_df.to_csv(export_dir / 'monte_carlo_results.csv', index=False)
    print(f"💾 Saved raw results to {export_dir / 'monte_carlo_results.csv'}")

    return results_df


def run_historical_backtest(historical_prices, export_dir):
    """Runs the original historical backtest and saves logs and charts."""
    print("🚀 Starting historical loan cycle simulation...")
    sim = Simulator()
    results_df = sim.run(historical_prices, MARGIN_CALL_LTV, PROFIT_TAKE_PRICE_INCREASE_USD)
    print("✅ Historical simulation finished.")

    if results_df.empty:
        print("⚠️ No simulation events were logged.")
        return

    results_df.to_csv(export_dir / 'simulation_log.csv', index=False)
    print(f"💾 Saved detailed log to {export_dir / 'simulation_log.csv'}")

    print("🎨 Generating summary charts...")
    fig, ax1 = plt.subplots(figsize=(18, 9))

    ax1.plot(results_df['Timestamp'], results_df['TotalBTC'], color='blue', label='Total BTC Holdings', marker='.', markersize=4)
    ax1.set_ylabel('Total BTC', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(historical_prices.index, historical_prices, color='black', alpha=0.3, label='BTC Price (USD)')
    ax2.set_yscale('log')
    ax2.set_ylabel('BTC Price (USD, Log Scale)', color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')

    margin_calls = results_df[results_df['EventType'] == 'MARGIN_CALL_CURED']
    if not margin_calls.empty:
        ax1.scatter(margin_calls['Timestamp'], margin_calls['TotalBTC'], color='orange', s=100, zorder=5, label='Margin Call Cured')

    liquidations = results_df[results_df['EventType'] == 'LIQUIDATION']
    if not liquidations.empty:
        ax1.scatter(liquidations['Timestamp'], liquidations['TotalBTC'], color='red', marker='X', s=200, zorder=5, label='LIQUIDATION')

    ax1.axhline(y=BTC_GOAL, color='green', linestyle='--', label=f'BTC Goal ({BTC_GOAL})')

    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
    fig.suptitle('Historical Loan Strategy Simulation (2-Year Backtest)', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(export_dir / 'simulation_summary.png', dpi=200)
    print(f"   - Saved: simulation_summary.png")

    print("\n" + "=" * 60)
    print("✅ Backtest finished successfully!")
    print(f"   All results have been saved to the '{export_dir}' directory.")
    print("=" * 60)


import seaborn as sns

def run_sensitivity_analysis(historical_prices, mc_sims_per_case=100):
    """
    Runs a sensitivity analysis by varying key parameters and executing a
    Monte Carlo simulation for each parameter combination.
    """
    print("📈 Starting sensitivity analysis...")

    # --- Define Parameter Ranges ---
    ltv_range = np.arange(0.80, 0.91, 0.05)  # From 80% to 90%
    profit_take_range = np.arange(20000, 40001, 5000) # From $20k to $40k

    total_cases = len(ltv_range) * len(profit_take_range)
    print(f"   - Testing {len(ltv_range)} LTV values and {len(profit_take_range)} Profit Take values.")
    print(f"   - Total cases to simulate: {total_cases}")
    print(f"   - Monte Carlo simulations per case: {mc_sims_per_case}")

    all_results = []
    case_num = 0

    for ltv in ltv_range:
        for profit_take in profit_take_range:
            case_num += 1
            print(f"   - Running case {case_num}/{total_cases}: LTV={ltv:.2f}, ProfitTake=${profit_take:,.0f}")

            # Run the Monte Carlo simulation for this parameter combination
            results_df = run_monte_carlo_simulation(
                historical_prices,
                num_simulations=mc_sims_per_case,
                margin_call_ltv=ltv,
                profit_take_usd=profit_take,
                show_progress=False # Keep the output clean
            )

            # Get the summary statistics
            liquidation_rate = results_df['Liquidated'].mean()
            avg_btc_if_not_liquidated = results_df[~results_df['Liquidated']]['FinalBTC'].mean()

            all_results.append({
                'LTV': ltv,
                'ProfitTake': profit_take,
                'LiquidationRate': liquidation_rate,
                'AvgFinalBTC': avg_btc_if_not_liquidated
            })

    results_df = pd.DataFrame(all_results)
    print("✅ Sensitivity analysis finished.")

    # --- Visualize Results ---
    export_dir = Path("exports") / f"sensitivity_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(export_dir / 'sensitivity_raw_results.csv', index=False)
    print(f"💾 Saved raw sensitivity results to {export_dir / 'sensitivity_raw_results.csv'}")

    # Create Heatmap for Liquidation Rate
    liquidation_pivot = results_df.pivot(index='LTV', columns='ProfitTake', values='LiquidationRate')
    plt.figure(figsize=(12, 8))
    sns.heatmap(liquidation_pivot, annot=True, fmt=".2%", cmap="Reds")
    plt.title('Liquidation Rate Sensitivity')
    plt.xlabel('Profit Take Threshold (USD)')
    plt.ylabel('Margin Call LTV')
    plt.savefig(export_dir / 'sensitivity_liquidation_rate.png')
    print(f"💾 Saved Liquidation Rate heatmap to {export_dir / 'sensitivity_liquidation_rate.png'}")

    # Create Heatmap for Average Final BTC
    btc_pivot = results_df.pivot(index='LTV', columns='ProfitTake', values='AvgFinalBTC')
    plt.figure(figsize=(12, 8))
    sns.heatmap(btc_pivot, annot=True, fmt=".4f", cmap="Greens")
    plt.title('Average Final BTC (Non-Liquidated) Sensitivity')
    plt.xlabel('Profit Take Threshold (USD)')
    plt.ylabel('Margin Call LTV')
    plt.savefig(export_dir / 'sensitivity_avg_final_btc.png')
    print(f"💾 Saved Average Final BTC heatmap to {export_dir / 'sensitivity_avg_final_btc.png'}")


def main():
    """Main function to run the simulation and generate outputs."""
    import argparse
    parser = argparse.ArgumentParser(description="Run a BTC leverage simulation.")
    parser.add_argument(
        '--mode',
        type=str,
        default='historical',
        choices=['historical', 'montecarlo', 'sensitivity'],
        help="The simulation mode to run ('historical', 'montecarlo', or 'sensitivity')."
    )
    parser.add_argument(
        '-n', '--num-simulations',
        type=int,
        default=100,
        help="Number of simulations per case for Monte Carlo or Sensitivity modes."
    )
    args = parser.parse_args()

    historical_prices = get_historical_data(interval=1440, periods=720)
    if historical_prices.empty:
        print("❌ Aborting simulation due to data fetch failure.")
        return

    if args.mode == 'historical':
        export_dir = Path("exports") / f"simulation_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        export_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created export directory: {export_dir}")
        run_historical_backtest(historical_prices, export_dir)
    elif args.mode == 'montecarlo':
        run_monte_carlo_simulation(historical_prices, num_simulations=args.num_simulations)
    elif args.mode == 'sensitivity':
        run_sensitivity_analysis(historical_prices, mc_sims_per_case=args.num_simulations)


if __name__ == "__main__":
    main()
