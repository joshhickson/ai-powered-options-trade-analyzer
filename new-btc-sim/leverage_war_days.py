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
    def __init__(self, loan_amount_usd, collateral_btc, entry_price, entry_date):
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.collateral_btc = collateral_btc
        self.loan_amount_usd = loan_amount_usd
        self.deferred_interest_usd = 0.0
        self.is_active = True
        self.btc_purchased = loan_amount_usd / entry_price
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
    """Manages the entire loan strategy simulation over a historical period."""
    def __init__(self, historical_prices):
        self.prices = historical_prices
        self.log = []
        self.total_btc = 0
        self.backup_btc = 0
        self.current_loan = None
        self.cycle_count = 0

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

    def run(self):
        print("🚀 Starting historical loan cycle simulation...")
        start_price = self.prices.iloc[0]
        start_date = self.prices.index[0]

        # Step A, B, C: Initial capital deployment
        self.total_btc = INITIAL_USD_CAPITAL / start_price
        self.log_event(start_date, 'START', {'price': start_price, 'details': f'Initial buy of {self.total_btc:.4f} BTC'})

        # Main simulation loop through the price history
        for timestamp, price in self.prices.items():
            if self.current_loan and self.current_loan.is_active:
                self.current_loan.accrue_interest()
                ltv = self.current_loan.get_ltv(price)

                # Step F -> G -> H: Monitor and handle margin calls
                if ltv >= MARGIN_CALL_LTV:
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
                        break

                # Step F -> I1: Check for profit-taking exit (fixed $30k increase)
                if price >= self.current_loan.entry_price + PROFIT_TAKE_PRICE_INCREASE_USD:
                    self.close_cycle(price, timestamp)

            elif self.total_btc > 0:
                self.start_new_cycle(price, timestamp)

        print("✅ Simulation finished.")
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


def main():
    """Main function to run the simulation and generate outputs."""
    export_dir = Path("exports") / f"simulation_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created export directory: {export_dir}")

    historical_prices = get_historical_data(interval=1440, periods=720)

    if historical_prices.empty:
        print("❌ Aborting simulation due to data fetch failure.")
        return

    sim = Simulator(historical_prices)
    results_df = sim.run()

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
    print("✅ Simulation finished successfully!")
    print(f"   All results have been saved to the '{export_dir}' directory.")
    print("=" * 60)

if __name__ == "__main__":
    main()
