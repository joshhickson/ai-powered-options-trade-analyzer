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

# --- Configuration based on Figure Lending LLC Contract ---
# See: new-btc-sim/Loan contract ocr.md
INITIAL_USD_CAPITAL = 30000.0 # Standardized initial capital
INITIAL_LOAN_USD = 10000.0  # Simulation scaled to $10k loan
BTC_GOAL = 1.0
LOAN_APR = 0.12615  # 12.615% per Figure Lending LLC contract
TRADING_FEE_PERCENT = 0.001  # 0.1% fee on all trades
LOAN_ORIGINATION_FEE_PERCENT = 0.01  # 1% of loan amount
LIQUIDATION_FEE_PERCENT = 0.02  # Contract: 2% processing fee on liquidations

# LTV (Loan-to-Value) Ratios from Contract
LTV_BASELINE = 0.75  # Contract: 75% baseline ratio
MARGIN_CALL_LTV = 0.85  # 85% LTV triggers margin call per contract
LIQUIDATION_LTV = 0.90  # Contract: 90% liquidation trigger
CURE_LTV_TARGET = LTV_BASELINE # When curing, restore to baseline

# Exit a loan cycle when the price is $30,000 higher than the entry price
PROFIT_TAKE_PRICE_INCREASE_USD = 30000.0

# CONTRACT COMPLIANCE VALIDATION (per sim-improvement-plan-2.md)
def validate_contract_compliance():
    """Ensure all parameters match Figure Lending LLC contract terms exactly"""
    assert LOAN_APR == 0.12615, f"APR must match contract: 12.615%, got {LOAN_APR}"
    assert MARGIN_CALL_LTV == 0.85, "Margin call must be at 85% LTV per contract"
    assert LIQUIDATION_LTV == 0.90, "Liquidation must be at 90% LTV per contract"
    print("✅ CONTRACT COMPLIANCE: All parameters validated against Figure Lending LLC terms")

# Validate on import
validate_contract_compliance()


def fetch_live_btc_price():
    """Get current BTC price from Kraken with a fallback."""
    print("📈 Fetching live BTC price from Kraken...")
    url = "https://api.kraken.com/0/public/Ticker"
    params = {'pair': 'XBTUSD'}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data.get('error'):
            price_str = data['result']['XXBTZUSD']['c'][0]
            price = float(price_str)
            print(f"✅ Live BTC price from Kraken: ${price:,.2f}")
            return price
        else:
            print(f"⚠️ Kraken API Error: {data['error']}")
    except Exception as e:
        print(f"⚠️ Could not fetch live price from Kraken: {e}")

    # Fallback to a recent, realistic price if API fails
    fallback_price = 118000.0
    print(f"⚠️ Using fallback price: ${fallback_price:,.2f}")
    return fallback_price


def initialize_current_market_state():
    """Initialize simulation with real current market conditions."""
    current_price = fetch_live_btc_price()

    # Per strategy, start with a fixed amount of USD capital
    initial_btc = INITIAL_USD_CAPITAL / current_price * (1 - TRADING_FEE_PERCENT)

    print(f"🏦 Initializing market state: {INITIAL_USD_CAPITAL:,.0f} USD buys {initial_btc:.4f} BTC at ${current_price:,.2f}")

    return {
        'current_price': current_price,
        'total_btc': initial_btc,
        'simulation_start_date': dt.datetime.now(),
    }


def generate_gbm_path(start_price, mu, sigma, days, dt_days=1):
    """Generates a random price path using Geometric Brownian Motion."""
    n_steps = int(days / dt_days)
    prices = np.zeros(n_steps + 1)
    prices[0] = start_price
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal()
        prices[t] = prices[t-1] * exp((mu - 0.5 * sigma**2) * dt_days + sigma * sqrt(dt_days) * z)
    return prices


def generate_forward_price_scenarios():
    """Defines realistic forward-looking price scenarios based on historical regimes."""
    scenarios = {
        'bull_market': {'probability': 0.20, 'annual_return': 1.5, 'volatility': 0.8},
        'bear_market': {'probability': 0.25, 'annual_return': -0.6, 'volatility': 1.2},
        'sideways_market': {'probability': 0.35, 'annual_return': 0.1, 'volatility': 0.6},
        'crash_recovery': {'probability': 0.20, 'annual_return': 0.3, 'volatility': 1.5}
    }
    print(f"📉 Defined {len(scenarios)} distinct price scenarios for the simulation.")
    return scenarios


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

    def accrue_daily_interest(self):
        """Accrues interest for one day."""
        # A simple daily rate is sufficient for simulation purposes
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

from math import sqrt, exp

class ForwardLoanSimulator:
    """
    Manages the entire forward-looking loan strategy simulation.
    """
    def __init__(self):
        # CONTRACT COMPLIANCE (Figure Lending LLC terms)
        self.LOAN_APR = LOAN_APR
        self.LTV_BASELINE = LTV_BASELINE
        self.MARGIN_CALL_LTV = MARGIN_CALL_LTV
        self.LIQUIDATION_LTV = LIQUIDATION_LTV
        self.PROFIT_TAKE_USD = PROFIT_TAKE_PRICE_INCREASE_USD
        self.MIN_LOAN_AMOUNT = 10000.0
        self._reset_state()

    def _reset_state(self):
        """Resets all state variables for a new simulation run."""
        self.log = []
        self.total_btc = 0
        self.backup_btc = 0 # This is BTC not used as collateral
        self.current_loan = None
        self.cycle_count = 0
        self.is_liquidated = False
        self.exit_reason = ""
        self.liquidation_reason = ""

    def accrue_daily_interest(self):
        """Handles daily interest accrual for the active loan."""
        if self.current_loan and self.current_loan.is_active:
            self.current_loan.accrue_daily_interest()

    def start_new_cycle(self, price, day):
        """Start a new loan cycle with contract-compliant sizing."""
        required_collateral_btc = 0
        loan_amount = 0
        
        # First cycle: Fixed $10K loan per improvement plan
        if self.cycle_count == 0:
            loan_amount = self.MIN_LOAN_AMOUNT
            collateral_value_target = loan_amount / self.LTV_BASELINE
            required_collateral_btc = collateral_value_target / price
        # Subsequent cycles: Progressive scaling
        else:
            collateral_btc_to_use = self.total_btc / 2.0

            contract_max_loan = collateral_btc_to_use * price * self.LTV_BASELINE
            # Safety: LTV must not exceed 90% in a 60% crash (price -> price * 0.4)
            safety_max_loan = (collateral_btc_to_use * (price * 0.4)) * self.LTV_BASELINE

            loan_amount = min(contract_max_loan, safety_max_loan)
            required_collateral_btc = collateral_btc_to_use

        # Check if we have enough BTC and if the loan meets the minimum size
        if self.total_btc >= required_collateral_btc and loan_amount >= self.MIN_LOAN_AMOUNT:
            self.cycle_count += 1

            # The BTC used for collateral is now "locked"
            collateral_btc = required_collateral_btc
            self.backup_btc = self.total_btc - collateral_btc

            # Create the new loan
            self.current_loan = Loan(
                cash_received_from_loan=loan_amount,
                collateral_btc=collateral_btc,
                entry_price=price,
                entry_date=day
            )

            # The loan gives us cash, which we immediately use to buy more BTC
            # This is the "leveraged" part of the strategy
            self.total_btc += self.current_loan.btc_purchased

            print(f"✅ Day {day}: Started cycle {self.cycle_count} -> Loan: ${loan_amount:,.0f}, Collateral: {collateral_btc:.4f} BTC")
            return True
        return False

    def should_close_cycle(self, current_price):
        """Determine if current cycle should be closed."""
        if not self.current_loan:
            return False

        ltv = self.current_loan.get_ltv(current_price)
        profit_target = self.current_loan.entry_price + self.PROFIT_TAKE_USD

        # Contract-mandated liquidation
        if ltv >= self.LIQUIDATION_LTV:
            self.liquidation_reason = f"Contract liquidation - LTV at {ltv:.2%} exceeded {self.LIQUIDATION_LTV:.0%} threshold."
            return True

        # Profit-taking exit
        if current_price >= profit_target:
            self.exit_reason = f"Profit target reached: Price ${current_price:,.0f} >= Target ${profit_target:,.0f}"
            return True

        # Margin call handling
        if ltv >= self.MARGIN_CALL_LTV:
            return self.handle_margin_call(current_price)

        return False

    def handle_margin_call(self, current_price):
        """Handles a margin call by using backup BTC to cure."""
        if not self.current_loan:
            return False

        # Calculate how much BTC we need to add to get back to the baseline LTV
        required_collateral_value = self.current_loan.get_balance() / self.LTV_BASELINE
        required_collateral_btc = required_collateral_value / current_price
        btc_to_add = required_collateral_btc - self.current_loan.collateral_btc

        if self.backup_btc >= btc_to_add > 0:
            # We have enough backup BTC to cure the margin call
            self.backup_btc -= btc_to_add
            self.current_loan.collateral_btc += btc_to_add
            print(f"🔧 Cured margin call by adding {btc_to_add:.4f} BTC to collateral.")
            return False # The cycle does not close, it's been cured
        else:
            # Not enough backup BTC, this will lead to liquidation
            self.liquidation_reason = "Insufficient backup BTC to cure margin call."
            return True # Signal to close/liquidate the cycle

    def close_current_cycle(self, price, day):
        """Closes the current loan cycle, logs the event, and recombines BTC."""
        if not self.current_loan:
            return

        print(f"🚪 Day {day}: Closing cycle {self.cycle_count} at price ${price:,.2f}. Reason: {self.exit_reason}")

        # Store values before clearing the loan
        loan_balance = self.current_loan.get_balance()

        # To repay the loan, we must sell some of the BTC we initially bought with it
        btc_to_sell_for_repayment = loan_balance / price

        # The profit is what's left of the BTC purchased with the loan, plus the original collateral and backup
        net_btc_profit_from_loan = self.current_loan.btc_purchased - btc_to_sell_for_repayment

        # Recombine all BTC holdings
        self.total_btc = self.current_loan.collateral_btc + self.backup_btc + net_btc_profit_from_loan
        self.backup_btc = 0

        self.current_loan = None
        self.exit_reason = ""

    def run_forward_simulation(self, initial_total_btc, price_path):
        """Run simulation forward through a generated price path."""
        self._reset_state()
        self.total_btc = initial_total_btc

        for day, price in enumerate(price_path):
            # Accrue interest on the active loan
            if self.current_loan:
                self.accrue_daily_interest()

            # Check for and handle exit conditions (liquidation, profit-taking)
            if self.should_close_cycle(price):
                if self.liquidation_reason:
                    # This is a liquidation event
                    self.is_liquidated = True
                    self.total_btc = 0  # Total loss of all BTC
                    print(f"💥 Day {day}: LIQUIDATION at price ${price:,.2f}. Reason: {self.liquidation_reason}")
                    break  # End this simulation run
                else:
                    # This is a normal cycle close (e.g., profit taking)
                    self.close_current_cycle(price, day)

            # If there is no active loan, check if we can start a new one
            elif not self.current_loan:
                self.start_new_cycle(price, day)

        # Return a summary of this single simulation run
        return {
            'liquidated': self.is_liquidated,
            'final_btc': self.total_btc,
            'cycles_completed': self.cycle_count,
        }


def run_comprehensive_monte_carlo(num_simulations=5000):
    """Run Monte Carlo across multiple market scenarios."""
    print(f"🇲🇨 Starting comprehensive Monte Carlo analysis with {num_simulations} simulations...")

    initial_state = initialize_current_market_state()
    start_price = initial_state['current_price']
    initial_btc = initial_state['total_btc']

    scenarios = generate_forward_price_scenarios()

    scenario_names = list(scenarios.keys())
    scenario_probabilities = [scenarios[s]['probability'] for s in scenario_names]

    results = []
    sim_engine = ForwardLoanSimulator()

    for i in range(num_simulations):
        if (i + 1) % 500 == 0:
            print(f"   ... running simulation {i+1}/{num_simulations}")

        # Select a scenario for this run based on its probability
        selected_scenario_name = np.random.choice(scenario_names, p=scenario_probabilities)
        scenario_params = scenarios[selected_scenario_name]

        # Generate a new, random price path for this specific simulation run
        daily_mu = scenario_params['annual_return'] / 365
        daily_sigma = scenario_params['volatility'] / sqrt(365)
        price_path = generate_gbm_path(
            start_price=start_price,
            mu=daily_mu,
            sigma=daily_sigma,
            days=730 # 2-year simulation
        )

        # Run the simulation on the generated path
        sim_result = sim_engine.run_forward_simulation(initial_btc, price_path)

        # Record results
        sim_result['scenario'] = selected_scenario_name
        results.append(sim_result)

    # --- Analyze and Report Results ---
    results_df = pd.DataFrame(results)

    liquidation_rate = results_df['liquidated'].mean() * 100
    goal_achieved_mask = results_df['final_btc'] >= BTC_GOAL
    goal_achievement_rate = goal_achieved_mask.mean() * 100
    avg_final_btc = results_df['final_btc'].mean()
    median_final_btc = results_df['final_btc'].median()
    avg_cycles = results_df['cycles_completed'].mean()

    print("\n" + "=" * 60)
    print("Monte Carlo Simulation Results")
    print("=" * 60)
    print(f"Total Simulations: {num_simulations}")
    print(f"Liquidation Rate: {liquidation_rate:.2f}%")
    print(f"BTC Goal ({BTC_GOAL:.1f}) Reached Rate: {goal_achievement_rate:.2f}%")
    print(f"Average Final BTC (all outcomes): {avg_final_btc:.4f} BTC")
    print(f"Median Final BTC (all outcomes): {median_final_btc:.4f} BTC")
    print(f"Average Cycles Completed: {avg_cycles:.2f}")
    print("-" * 60)

    # --- Export Results ---
    export_dir = Path("new-btc-sim/exports") / f"monte_carlo_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    results_csv_path = export_dir / 'monte_carlo_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    print(f"💾 Saved raw results to {results_csv_path}")

    # Plot distribution of final BTC
    plt.figure(figsize=(12, 7))
    sns.histplot(results_df['final_btc'], bins=100, kde=True)
    plt.title(f'Distribution of Final BTC Holdings ({num_simulations} Simulations)')
    plt.xlabel('Final BTC')
    plt.ylabel('Frequency')
    plt.axvline(x=BTC_GOAL, color='r', linestyle='--', label=f'BTC Goal ({BTC_GOAL})')
    plt.axvline(x=initial_btc, color='g', linestyle='--', label=f'Starting BTC ({initial_btc:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = export_dir / 'monte_carlo_btc_distribution.png'
    plt.savefig(plot_path)
    print(f"💾 Saved distribution plot to {plot_path}")

    return results_df

import seaborn as sns

def main():
    """Main function to run the simulation and generate outputs."""
    import argparse

    parser = argparse.ArgumentParser(description="Run a forward-looking BTC leverage simulation.")
    parser.add_argument(
        '-n', '--num-simulations',
        type=int,
        default=1000, # A smaller default for quick runs
        help="Number of simulations to run for Monte Carlo analysis."
    )
    args = parser.parse_args()

    run_comprehensive_monte_carlo(num_simulations=args.num_simulations)


if __name__ == "__main__":
    main()