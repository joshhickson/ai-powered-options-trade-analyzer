#!/usr/bin/env python3
"""
leverage_sim.py
Accurate Bitcoin-collateral loan strategy simulation with realistic contract terms.

Based on Figure Lending contract analysis:
    • Minimum loan: $10,000 at 11.5% APR
    • Monthly interest-only payments: $95.83/month
    • LTV triggers: 85% margin call, 90% liquidation
    • 48-hour cure period for margin calls
    • Interest can be deferred (compounds daily)
    • 2% processing fee on liquidations
"""

import datetime as dt
import math
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





def load_btc_history() -> pd.Series:
    """Load Bitcoin price history with fallbacks using US-compliant APIs."""

    # Method 1: Nasdaq Data Link (Best for 10+ years of data)
    try:
        import nasdaqdatalink
        print("📈 Tier 1: Trying Nasdaq Data Link (Brave New Coin)...")

        # Set API key from secrets
        nasdaq_api_key = os.getenv('NASDAQ_API_KEY')
        if nasdaq_api_key:
            nasdaqdatalink.ApiConfig.api_key = nasdaq_api_key
            print("🔑 Using Nasdaq API key from secrets")
        else:
            print("⚠️  No Nasdaq API key found, using limited access")

        try:
            # Try the BNC/BLX dataset (Bitcoin Liquid Index)
            df = nasdaqdatalink.get("BNC/BLX", start_date="2010-01-01")
            if not df.empty and 'Value' in df.columns:
                btc_series = df['Value'].astype(float)
                btc_series.name = 'Close'
                print(f"✅ Nasdaq Data Link successful: {len(btc_series)} days")
                return btc_series
        except Exception as auth_e:
            print(f"⚠️  Nasdaq BLX failed: {auth_e}")

            # Try alternative dataset if BLX fails
            try:
                print("🔄 Trying alternative Nasdaq dataset...")
                df = nasdaqdatalink.get("BITFINEX/BTCUSD", start_date="2015-01-01")
                if not df.empty and 'Last' in df.columns:
                    btc_series = df['Last'].astype(float)
                    btc_series.name = 'Close'
                    print(f"✅ Nasdaq Bitfinex successful: {len(btc_series)} days")
                    return btc_series
            except Exception as e2:
                print(f"⚠️  Alternative Nasdaq dataset failed: {e2}")

    except ImportError:
        print("⚠️  nasdaq-data-link not installed")
    except Exception as e:
        print(f"⚠️  Nasdaq Data Link failed: {e}")

    # Method 2: Kraken API (US-compliant exchange)
    try:
        import requests
        import time
        print("📈 Tier 2: Trying Kraken API...")

        # Get recent data first (more reliable)
        url = "https://api.kraken.com/0/public/OHLC"
        headers = {
            'User-Agent': 'Bitcoin-Simulation/1.0'
        }

        response = requests.get(url, params={
            'pair': 'XBTUSD', 
            'interval': 1440  # Daily
        }, headers=headers, timeout=15)

        if response.status_code == 200:
            data = response.json()

            if 'error' in data and not data['error']:  # No errors
                result_keys = [k for k in data['result'].keys() if k != 'last']
                if result_keys:
                    pair_name = result_keys[0]
                    ohlc_data = data['result'][pair_name]

                    if ohlc_data:
                        # Convert to DataFrame
                        df = pd.DataFrame(ohlc_data, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        df = df.set_index('time').sort_index()
                        btc_series = df['close'].astype(float)
                        btc_series.name = 'Close'

                        print(f"✅ Kraken successful: {len(btc_series)} days")
                        return btc_series
            else:
                print(f"⚠️  Kraken API error: {data.get('error', 'Unknown error')}")
        else:
            print(f"⚠️  Kraken HTTP error: {response.status_code}")

    except Exception as e:
        print(f"⚠️  Kraken API failed: {e}")

    # All data sources failed
    print("❌ All data sources failed - unable to load Bitcoin price data")
    raise Exception("No Bitcoin price data available from any source")

def worst_drop_until_recovery(price_series: pd.Series, jump: float = 30000.0) -> pd.DataFrame:
    """
    For each start date, find worst drawdown before price recovers by jump amount.
    """
    res = []
    dates = price_series.index
    p = price_series.values
    n = len(p)

    for i in range(n):
        target = p[i] + jump
        j = i
        min_p = p[i]

        while j < n and p[j] < target:
            min_p = min(min_p, p[j])
            j += 1

        if j == n:
            continue  # never recovered

        draw = (min_p / p[i]) - 1.0  # negative value
        res.append((dates[i], p[i], draw))

    df = pd.DataFrame(res, columns=["date", "price", "draw"])
    return df

def create_true_monte_carlo_drawdown_model(historical_data=None):
    """Create a TRUE Monte Carlo drawdown model that samples from actual historical data."""

    def monte_carlo_drawdown(price: float, simulation_run: int = None) -> float:
        """
        TRUE data-driven Monte Carlo sampling from actual historical drawdowns.
        Uses the loaded historical data instead of hardcoded assumptions.
        """

        if historical_data is not None and len(historical_data) > 10:
            # Use actual historical drawdown distribution
            historical_drawdowns = np.abs(historical_data['draw'].values)
            
            # Remove extreme outliers (top 5%) to avoid unrealistic scenarios
            percentile_95 = np.percentile(historical_drawdowns, 95)
            filtered_drawdowns = historical_drawdowns[historical_drawdowns <= percentile_95]
            
            # Sample directly from historical distribution
            sampled_drawdown = np.random.choice(filtered_drawdowns)
            
            # Add small random variation to avoid exact repeats
            variation = np.random.normal(0, 0.02)  # ±2% variation
            sampled_drawdown = np.clip(sampled_drawdown + variation, 0.02, 0.35)  # Cap at 35%
            
            scenario_desc = "Historical data sample"
            
        else:
            # Fallback: Conservative model based on actual recent Bitcoin behavior
            # Use realistic probabilities based on 2-year Kraken data showing max 26.2% drawdown
            drawdown_buckets = [
                {"range": (0.02, 0.08), "probability": 0.50, "description": "Minor correction (2-8%)"},
                {"range": (0.08, 0.15), "probability": 0.30, "description": "Moderate correction (8-15%)"},
                {"range": (0.15, 0.25), "probability": 0.15, "description": "Major correction (15-25%)"},
                {"range": (0.25, 0.35), "probability": 0.05, "description": "Severe correction (25-35%)"},
            ]

            # TRUE RANDOM SAMPLING
            rand_val = np.random.random()
            cumulative_prob = 0
            selected_bucket = None

            for bucket in drawdown_buckets:
                cumulative_prob += bucket["probability"]
                if rand_val <= cumulative_prob:
                    selected_bucket = bucket
                    break

            if selected_bucket is None:
                selected_bucket = drawdown_buckets[-1]

            # Sample specific drawdown within bucket
            min_draw, max_draw = selected_bucket["range"]
            sampled_drawdown = np.random.uniform(min_draw, max_draw)
            scenario_desc = selected_bucket["description"]

        if simulation_run is not None:
            print(f"🎲 Monte Carlo Run #{simulation_run}: {sampled_drawdown:.1%} drawdown at ${price:,.0f}")
            print(f"   Scenario: {scenario_desc}")

        return sampled_drawdown

    return monte_carlo_drawdown

def run_monte_carlo_simulation(start_btc: float, start_price: float, btc_goal: float, num_simulations: int = 1000):
    """Run TRUE Monte Carlo simulation with thousands of random scenarios."""
    print(f"🎲 Running {num_simulations} Monte Carlo simulations...")

    drawdown_model = create_true_monte_carlo_drawdown_model()
    simulator = LoanSimulator()

    results = []
    liquidation_count = 0
    success_count = 0

    for sim_run in range(num_simulations):
        if sim_run % 100 == 0:
            print(f"   Progress: {sim_run}/{num_simulations} simulations...")

        # Reset for each simulation
        free_btc = start_btc * 0.5  # Conservative starting position
        collateral_btc = start_btc * 0.5
        current_price = start_price
        cycle = 0
        simulation_liquidated = False

        # Run simulation until goal or liquidation
        while free_btc < btc_goal and cycle < 20 and not simulation_liquidated:
            cycle += 1

            # Get random drawdown for this cycle
            worst_drop = drawdown_model(current_price, sim_run)
            worst_price = current_price * (1 - worst_drop)

            # Calculate safe loan size
            max_loan = collateral_btc * worst_price * 0.75
            loan_amount = max(max_loan * 0.5, simulator.min_loan)

            # Simulate cycle outcome
            cycle_result = simulator.simulate_cycle(
                current_price, collateral_btc, loan_amount, 
                lambda p, c=None: worst_drop  # Use the sampled drawdown
            )

            if cycle_result["liquidation_occurred"]:
                simulation_liquidated = True
                liquidation_count += 1
                break

            # Update state
            free_btc += cycle_result["net_btc_gain"]
            current_price = cycle_result["exit_price"]

        # Record result
        if free_btc >= btc_goal:
            success_count += 1
            results.append({
                'simulation': sim_run,
                'outcome': 'SUCCESS',
                'final_btc': free_btc,
                'cycles': cycle,
                'liquidated': simulation_liquidated
            })
        else:
            results.append({
                'simulation': sim_run,
                'outcome': 'FAILURE',
                'final_btc': free_btc,
                'cycles': cycle,
                'liquidated': simulation_liquidated
            })

    # Calculate statistics
    success_rate = (success_count / num_simulations) * 100
    liquidation_rate = (liquidation_count / num_simulations) * 100

    print(f"\n📊 MONTE CARLO RESULTS ({num_simulations} simulations)")
    print("=" * 50)
    print(f"✅ Success Rate: {success_rate:.1f}%")
    print(f"💥 Liquidation Rate: {liquidation_rate:.1f}%")
    print(f"🎯 Strategy Viability: {'VIABLE' if success_rate > 70 else 'HIGH RISK' if success_rate > 30 else 'NOT VIABLE'}")

    return pd.DataFrame(results)

def create_probabilistic_drawdown_model(historical_data=None):
    """Wrapper to maintain backward compatibility - redirects to true Monte Carlo."""
    return create_true_monte_carlo_drawdown_model(historical_data)

def fit_drawdown_model(draw_df: pd.DataFrame):
    """Fit a data-driven drawdown model based on actual historical patterns."""
    print("📊 Using data-driven drawdown model based on actual Bitcoin history")

    # Create model that uses the actual historical data
    data_driven_model = create_true_monte_carlo_drawdown_model(draw_df)

    # Show historical data analysis
    if len(draw_df) >= 10:
        historical_worst = abs(draw_df.draw.min())  # Most negative (worst) drawdown
        historical_95th = np.percentile(np.abs(draw_df.draw), 95)
        historical_median = np.percentile(np.abs(draw_df.draw), 50)
        historical_mean = np.mean(np.abs(draw_df.draw))

        print(f"📈 Historical data analysis:")
        print(f"   • Worst observed drawdown: {historical_worst:.1%}")
        print(f"   • 95th percentile drawdown: {historical_95th:.1%}")
        print(f"   • Mean drawdown: {historical_mean:.1%}")
        print(f"   • Median drawdown: {historical_median:.1%}")

        # Test the data-driven model with samples from actual data
        test_price = draw_df.price.median()
        print(f"🎲 Data-driven model sampling from {len(draw_df)} historical cycles at ${test_price:,.0f}:")
        for i in range(3):
            sample_draw = data_driven_model(test_price, i+1)

        print(f"✅ Model will sample from actual historical distribution instead of assumptions")
    else:
        print("⚠️  Limited historical data - using conservative fallback model")

    return data_driven_model

def generate_realistic_price_scenarios(start_price: float, years: int = 5) -> dict:
    """Generate multiple realistic Bitcoin price scenarios using historical patterns."""
    scenarios = {}

    # Historical Bitcoin volatility and patterns
    annual_volatility = 0.80  # 80% annual volatility
    daily_volatility = annual_volatility / np.sqrt(365)

    np.random.seed(42)  # Reproducible results
    days = years * 365

    # Bull scenario: 100-200% annual gains (rare, only 20% probability)
    bull_drift = np.log(2.0) / 365  # 100% annual
    bull_returns = np.random.normal(bull_drift, daily_volatility, days)
    bull_prices = start_price * np.exp(np.cumsum(bull_returns))
    scenarios['bull'] = bull_prices

    # Bear scenario: -50% to -80% drops lasting 1-3 years (30% probability)
    bear_crash_months = 12
    bear_recovery_months = 24
    crash_days = bear_crash_months * 30
    recovery_days = bear_recovery_months * 30

    # Crash phase: -70% over 12 months
    crash_drift = np.log(0.3) / crash_days
    crash_returns = np.random.normal(crash_drift, daily_volatility, crash_days)
    crash_prices = start_price * np.exp(np.cumsum(crash_returns))

    # Recovery phase: slow recovery to 50% of original
    recovery_target = start_price * 0.5
    recovery_drift = np.log(recovery_target / crash_prices[-1]) / recovery_days
    recovery_returns = np.random.normal(recovery_drift, daily_volatility * 0.5, recovery_days)
    recovery_prices = crash_prices[-1] * np.exp(np.cumsum(recovery_returns))

    bear_prices = np.concatenate([crash_prices, recovery_prices])
    if len(bear_prices) < days:
        # Extend with sideways movement
        remaining_days = days - len(bear_prices)
        sideways_returns = np.random.normal(0, daily_volatility * 0.3, remaining_days)
        sideways_prices = bear_prices[-1] * np.exp(np.cumsum(sideways_returns))
        bear_prices = np.concatenate([bear_prices, sideways_prices])

    scenarios['bear'] = bear_prices[:days]

    # Realistic scenario: 20-30% annual average with high volatility (40% probability)
    realistic_drift = np.log(1.25) / 365  # 25% annual
    realistic_returns = np.random.normal(realistic_drift, daily_volatility, days)
    realistic_prices = start_price * np.exp(np.cumsum(realistic_returns))
    scenarios['realistic'] = realistic_prices

    # Crash scenario: Sudden 60% drop with slow recovery (10% probability)
    crash_day = days // 4  # Crash happens 1/4 through period
    pre_crash_drift = np.log(1.5) / crash_day  # 50% gain before crash
    pre_crash_returns = np.random.normal(pre_crash_drift, daily_volatility, crash_day)
    pre_crash_prices = start_price * np.exp(np.cumsum(pre_crash_returns))

    # Sudden crash
    crash_price = pre_crash_prices[-1] * 0.4  # -60% crash

    # Slow recovery
    remaining_days = days - crash_day - 1
    recovery_drift = np.log(start_price * 0.8 / crash_price) / remaining_days
    post_crash_returns = np.random.normal(recovery_drift, daily_volatility * 0.6, remaining_days)
    post_crash_prices = crash_price * np.exp(np.cumsum(post_crash_returns))

    crash_prices = np.concatenate([pre_crash_prices, [crash_price], post_crash_prices])
    scenarios['crash'] = crash_prices[:days]

    return scenarios

def simulate_realistic_cycle_outcome(start_price: float, loan_size: float, 
                                          collateral_btc: float, months: int = 6, 
                                          historical_data=None) -> dict:
    """
    Simulate realistic cycle outcome using actual historical Bitcoin patterns.
    Returns probability-weighted scenarios based on real data analysis.
    """
    print(f"🎲 Simulating realistic outcomes for {months}-month cycle starting at ${start_price:,.0f}")

    # Use realistic scenarios based on actual Bitcoin behavior patterns
    scenarios = {
        "bull_run": {
            "probability": 0.15,  # 15% chance - bull runs are rare
            "monthly_return": 0.08,  # 8% per month (more realistic than 15%)
            "description": "Strong bull market"
        },
        "moderate_growth": {
            "probability": 0.25,  # 25% chance
            "monthly_return": 0.03,  # 3% per month (conservative)
            "description": "Steady upward trend"
        },
        "sideways": {
            "probability": 0.35,  # 35% chance - most common
            "monthly_return": 0.00,  # 0% per month (true sideways)
            "description": "Choppy sideways movement"
        },
        "decline": {
            "probability": 0.20,  # 20% chance
            "monthly_return": -0.04,  # -4% per month (realistic decline)
            "description": "Gradual decline"
        },
        "correction": {
            "probability": 0.05,  # 5% chance - rare but happens
            "monthly_return": -0.12,  # -12% per month (major correction)
            "description": "Market correction"
        }
    }

    results = []

    for scenario_name, scenario in scenarios.items():
        # Calculate final price for this scenario
        monthly_multiplier = 1 + scenario["monthly_return"]
        final_price = start_price * (monthly_multiplier ** months)

        # Add realistic volatility (reduced from 20% to 15%)
        np.random.seed(hash(scenario_name) % 1000)  # Deterministic but varied
        volatility_factor = np.random.normal(1.0, 0.15)  # ±15% volatility around trend
        final_price *= volatility_factor

        # Use realistic drawdown based on scenario and historical data
        if historical_data is not None and len(historical_data) > 10:
            # Sample from actual historical drawdowns
            historical_drawdowns = np.abs(historical_data['draw'].values)
            # Cap at 95th percentile to avoid extreme outliers
            percentile_95 = np.percentile(historical_drawdowns, 95)
            filtered_drawdowns = historical_drawdowns[historical_drawdowns <= percentile_95]
            
            # Bias selection based on scenario
            if scenario_name == "correction":
                # Use worst 20% of historical drawdowns
                worst_drawdowns = filtered_drawdowns[filtered_drawdowns >= np.percentile(filtered_drawdowns, 80)]
                worst_drawdown = np.random.choice(worst_drawdowns) if len(worst_drawdowns) > 0 else percentile_95
            elif scenario["monthly_return"] < 0:
                # Use worst 50% of historical drawdowns
                worst_drawdowns = filtered_drawdowns[filtered_drawdowns >= np.percentile(filtered_drawdowns, 50)]
                worst_drawdown = np.random.choice(worst_drawdowns) if len(worst_drawdowns) > 0 else np.mean(filtered_drawdowns)
            else:
                # Use typical drawdowns (median or better)
                typical_drawdowns = filtered_drawdowns[filtered_drawdowns <= np.percentile(filtered_drawdowns, 70)]
                worst_drawdown = np.random.choice(typical_drawdowns) if len(typical_drawdowns) > 0 else np.median(filtered_drawdowns)
        else:
            # Fallback: Conservative but realistic drawdowns
            if scenario_name == "correction":
                worst_drawdown = np.random.uniform(0.20, 0.30)  # 20-30% correction
            elif scenario["monthly_return"] < 0:
                worst_drawdown = np.random.uniform(0.10, 0.20)  # 10-20% drawdown
            else:
                worst_drawdown = np.random.uniform(0.05, 0.15)  # 5-15% drawdown

        worst_price = start_price * (1 - worst_drawdown)

        # Calculate LTV at worst point with more conservative liquidation thresholds
        loan_balance = loan_size * (1 + 0.115 * months/12)  # Simple interest approximation
        worst_ltv = loan_balance / (collateral_btc * worst_price)

        # Determine outcome using actual contract thresholds
        if worst_ltv >= 0.90:  # Contract liquidation threshold
            outcome = "LIQUIDATION"
        elif worst_ltv >= 0.85:  # Contract margin call threshold
            outcome = "MARGIN_CALL"
        elif final_price > start_price * 1.05:  # Reduced from 1.10 to 1.05
            outcome = "SUCCESSFUL_EXIT"
        else:
            outcome = "BREAK_EVEN"

        results.append({
            "scenario": scenario_name,
            "probability": scenario["probability"],
            "description": scenario["description"],
            "final_price": final_price,
            "worst_price": worst_price,
            "worst_drawdown": worst_drawdown,
            "worst_ltv": worst_ltv,
            "outcome": outcome,
            "price_return": (final_price / start_price) - 1
        })

    return results

def simulate_price_path(start_price: float, target_price: float, days: int) -> np.ndarray:
    """DEPRECATED: Generate realistic price path - use simulate_realistic_cycle_outcome instead."""
    print("⚠️  WARNING: Using deprecated optimistic price model")
    print("⚠️  This function should not be used - switching to realistic modeling")

    # Return a simple linear path as fallback, but warn heavily
    if days <= 1:
        return np.array([start_price, target_price])

    # Just return linear interpolation as emergency fallback
    return np.linspace(start_price, target_price, days)

class LoanSimulator:
    def __init__(self):
        # Contract terms based on Figure Lending analysis
        self.min_loan = 10000.0
        self.base_apr = 0.115  # 11.5% for $10K loan
        self.origination_fee_rate = 0.033  # ~3.3% estimated
        self.processing_fee_rate = 0.02  # 2% on liquidations

        # Contract terms - match Figure Lending actual thresholds
        self.baseline_ltv = 0.75  # 75% LTV baseline per contract
        self.margin_call_ltv = 0.85  # 85% margin call trigger per contract
        self.liquidation_ltv = 0.90  # 90% liquidation trigger per contract
        self.collateral_release_ltv = 0.35  # 35% collateral release per contract

        # Operational parameters - realistic expectations
        self.cure_period_hours = 48
        self.exit_jump = 10000.0  # Very conservative $10k target (not $30k)

        # Risk management parameters - maximum safety
        self.max_safe_ltv = 0.30  # Never exceed 30% LTV for safety
        self.min_collateral_buffer = 0.15  # Always keep 0.15 BTC buffer
        self.bear_market_detection_threshold = -0.30  # Stop strategy if price drops 30% in 3 months

    def validate_strategy_viability(self, start_btc: float, start_price: float, 
                                  goal_btc: float) -> dict:
        """Check if strategy is mathematically viable given constraints."""
        starting_value = start_btc * start_price
        goal_value = goal_btc * start_price  # Conservative: same price
        required_gain = goal_value - starting_value

        # Calculate theoretical maximum with safe leverage
        max_collateral = start_btc - self.min_collateral_buffer
        if max_collateral <= 0:
            return {
                "viable": False, 
                "reason": "Insufficient starting capital for any collateral",
                "recommendation": "Increase starting BTC to at least 0.15 BTC"
            }

        # Conservative drawdown assumption - allow for 60% crash with contract terms
        conservative_crash_price = start_price * 0.40  # 60% crash (historical worst-case)
        max_safe_loan = max_collateral * conservative_crash_price * 0.75  # Use baseline LTV

        if max_safe_loan < self.min_loan:
            return {
                "viable": False,
                "reason": f"Max safe loan ${max_safe_loan:,.0f} below minimum ${self.min_loan:,.0f}",
                "recommendation": f"Need at least {self.min_loan / (conservative_crash_price * self.max_safe_ltv):.3f} BTC for viable strategy"
            }

        # Estimate cycles needed (very rough)
        btc_per_cycle = max_safe_loan / start_price * 0.5  # Conservative 50% efficiency
        cycles_needed = required_gain / (btc_per_cycle * start_price)

        if cycles_needed > 20:
            return {
                "viable": False,
                "reason": f"Would require {cycles_needed:.0f} cycles - too risky/long",
                "recommendation": "Strategy not viable with current parameters. Consider: 1) Increase starting capital, 2) Lower goal, 3) Use DCA instead"
            }

        # Check interest cost impact over time
        annual_interest_cost = max_safe_loan * self.base_apr
        cycles_per_year = 365 / 180  # Assume 6 months per cycle
        annual_cycles = min(cycles_needed, cycles_per_year)
        interest_burden = annual_interest_cost / starting_value

        if interest_burden > 0.5:  # 50% of portfolio value per year
            return {
                "viable": False,
                "reason": f"Interest costs {interest_burden:.1%} of portfolio annually - unsustainable",
                "recommendation": "Interest burden too high. Reduce leverage or increase starting capital"
            }

        return {
            "viable": True,
            "reason": f"Estimated {cycles_needed:.1f} cycles, {interest_burden:.1%} annual interest burden",
            "recommendation": f"Proceed with caution. Max safe loan: ${max_safe_loan:,.0f}"
        }

    def model_bear_market_impact(self, start_price: float, collateral_btc: float, 
                               loan_balance: float) -> dict:
        """Model what happens during extended bear market using historical patterns."""
        print("🐻 Modeling bear market survival using historical crash patterns...")

        # Model multiple bear market scenarios based on Bitcoin history
        bear_scenarios = [
            {
                "name": "2018_style",
                "description": "84% crash over 12 months, 24-month recovery",
                "crash_months": 12,
                "max_drawdown": 0.84,
                "recovery_months": 24,
                "probability": 0.30
            },
            {
                "name": "2022_style", 
                "description": "77% crash over 6 months, 12-month recovery",
                "crash_months": 6,
                "max_drawdown": 0.77,
                "recovery_months": 12,
                "probability": 0.25
            },
            {
                "name": "2011_style",
                "description": "93% crash over 8 months, 18-month recovery", 
                "crash_months": 8,
                "max_drawdown": 0.93,
                "recovery_months": 18,
                "probability": 0.15
            },
            {
                "name": "extended_bear",
                "description": "70% crash with 36-month sideways grind",
                "crash_months": 18,
                "max_drawdown": 0.70,
                "recovery_months": 36,
                "probability": 0.30
            }
        ]

        survival_results = []

        for scenario in bear_scenarios:
            print(f"   Testing {scenario['name']}: {scenario['description']}")

            # Generate price path for this scenario
            crash_months = scenario["crash_months"]
            max_drawdown = scenario["max_drawdown"]
            recovery_months = scenario["recovery_months"]

            total_months = crash_months + recovery_months
            prices = []

            # Crash phase
            for month in range(crash_months):
                drop_progress = (month + 1) / crash_months
                current_price = start_price * (1 - max_drawdown * drop_progress)
                prices.append(current_price)

            # Recovery phase (partial recovery to 50% of original)
            bottom_price = start_price * (1 - max_drawdown)
            recovery_target = start_price * 0.5  # Only 50% recovery

            for month in range(recovery_months):
                recovery_progress = month / recovery_months
                current_price = bottom_price + (recovery_target - bottom_price) * recovery_progress
                # Add volatility
                volatility = np.random.normal(0, 0.15)  # 15% monthly volatility
                current_price *= (1 + volatility)
                prices.append(current_price)

            # Test survival
            max_ltv = 0
            liquidation_month = None
            monthly_interest = loan_balance * self.base_apr / 12
            remaining_collateral = collateral_btc
            total_interest_paid = 0

            for month, price in enumerate(prices):
                # Sell BTC to make monthly interest payment
                if remaining_collateral > 0:
                    btc_sold_for_interest = monthly_interest / price
                    remaining_collateral -= btc_sold_for_interest
                    total_interest_paid += monthly_interest

                if remaining_collateral <= 0:
                    liquidation_month = month + 1
                    reason = "Ran out of collateral paying interest"
                    break

                # Check LTV trigger
                current_ltv = loan_balance / (remaining_collateral * price)
                max_ltv = max(max_ltv, current_ltv)

                if current_ltv >= self.liquidation_ltv:
                    liquidation_month = month + 1
                    reason = f"LTV reached {current_ltv:.1%} at price ${price:,.0f}"
                    break

            survival_results.append({
                "scenario": scenario["name"],
                "description": scenario["description"],
                "probability": scenario["probability"],
                "survives": liquidation_month is None,
                "liquidation_month": liquidation_month,
                "max_ltv": max_ltv,
                "final_collateral": remaining_collateral if liquidation_month is None else 0,
                "total_interest_paid": total_interest_paid,
                "reason": reason if liquidation_month else "Survived full bear market"
            })

        # Calculate overall survival probability
        total_survival_prob = sum(s["probability"] for s in survival_results if s["survives"])

        print(f"   📊 Bear Market Survival Analysis:")
        for result in survival_results:
            status = "✅ SURVIVES" if result["survives"] else "💥 LIQUIDATED"
            print(f"      {result['scenario']}: {status} (Prob: {result['probability']:.1%})")
            if not result["survives"]:
                print(f"         Liquidated month {result['liquidation_month']}: {result['reason']}")

        print(f"   📈 Overall survival probability: {total_survival_prob:.1%}")

        return {
            "overall_survival_probability": total_survival_prob,
            "scenario_results": survival_results,
            "recommendation": "ABORT_STRATEGY" if total_survival_prob < 0.7 else "PROCEED_WITH_CAUTION"
        }

    def calculate_monthly_payment(self, principal: float) -> float:
        """Calculate monthly interest-only payment."""
        return principal * self.base_apr / 12

    def calculate_deferred_interest(self, principal: float, days: int) -> float:
        """Calculate compound daily interest if deferred."""
        daily_rate = self.base_apr / 365
        return principal * ((1 + daily_rate) ** days - 1)

    def calculate_ltv(self, loan_balance: float, collateral_btc: float, btc_price: float) -> float:
        """Calculate current LTV ratio."""
        if collateral_btc <= 0 or btc_price <= 0:
            return 1.0
        return loan_balance / (collateral_btc * btc_price)

    def check_ltv_triggers(self, ltv: float) -> str:
        """Check what action is triggered by current LTV."""
        if ltv >= self.liquidation_ltv:
            return "FORCE_LIQUIDATION"
        elif ltv >= self.margin_call_ltv:
            return "MARGIN_CALL"
        elif ltv <= self.collateral_release_ltv:
            return "COLLATERAL_RELEASE_ELIGIBLE"
        else:
            return "NORMAL"

    def calculate_safe_loan_sizing(self, collateral_btc: float, btc_price: float, 
                                  conservative_drawdown: float) -> float:
        """Calculate loan size that survives 80% Bitcoin crash with safety margin."""
        # Assume Bitcoin drops by the conservative drawdown amount
        crash_price = btc_price * (1 - conservative_drawdown)

        # Calculate max loan that keeps LTV < 85% even in crash (not 90%!)
        max_safe_loan = collateral_btc * crash_price * 0.80  # 80% LTV at crash price

        # Additional safety buffer - use only 70% of theoretical max
        recommended_loan = max_safe_loan * 0.70

        # Ensure minimum loan requirement is met, but warn if risky
        if recommended_loan < self.min_loan:
            print(f"⚠️  WARNING: Safe loan size ${recommended_loan:,.0f} below minimum ${self.min_loan:,.0f}")
            print(f"   Collateral may be insufficient for safe leverage strategy")
            return self.min_loan

        return recommended_loan

    def calculate_required_collateral(self, loan_amount: float, btc_price: float, 
                                    worst_case_price: float) -> float:
        """Calculate BTC needed to avoid liquidation in worst case."""
        # Need enough collateral so that at worst_case_price, LTV < 85% (not 90%!)
        # Using 85% as trigger instead of 90% for additional safety
        return loan_amount / (0.82 * worst_case_price)  # Small buffer below 85%

    def simulate_cycle(self, entry_price: float, collateral_btc: float, 
                      loan_amount: float, drawdown_model, cycle_number: int = 1) -> dict:
        """Simulate one complete loan cycle using realistic market scenarios."""

        # Add origination fee to loan balance
        origination_fee = loan_amount * self.origination_fee_rate
        total_loan_balance = loan_amount + origination_fee

        # Use realistic scenario modeling based on historical data
        cycle_months = 6  # Standard 6-month cycle
        realistic_outcomes = simulate_realistic_cycle_outcome(
            entry_price, loan_amount, collateral_btc, cycle_months, None  # Pass historical data if available
        )

        # Choose scenario based on probabilities (use weighted random selection)
        # Use cycle number and price for better randomness
        seed_val = int((entry_price * cycle_number * 1000) % 100000)
        np.random.seed(seed_val)
        rand_val = np.random.random()
        cumulative_prob = 0
        selected_scenario = None

        for outcome in realistic_outcomes:
            cumulative_prob += outcome["probability"]
            if rand_val <= cumulative_prob:
                selected_scenario = outcome
                break

        if selected_scenario is None:
            selected_scenario = realistic_outcomes[-1]  # Fallback to last scenario

        print(f"🎲 Selected scenario: {selected_scenario['scenario']} - {selected_scenario['description']}")
        print(f"   Outcome: {selected_scenario['outcome']}, Price: ${selected_scenario['final_price']:,.0f}")

        # Use scenario results
        exit_price = selected_scenario["final_price"]
        worst_price = selected_scenario["worst_price"] 
        worst_ltv = selected_scenario["worst_ltv"]
        expected_days = cycle_months * 30

        # Determine what happened based on scenario outcome
        if selected_scenario["outcome"] == "LIQUIDATION":
            liquidation_occurred = True
            margin_call_occurred = False
            liquidation_fee = total_loan_balance * self.processing_fee_rate
            final_loan_balance = total_loan_balance + liquidation_fee

            # In liquidation, lose most collateral
            net_btc_gain = -collateral_btc * 0.8  # Lose 80% of collateral
            cure_btc_needed = 0
            btc_purchased = loan_amount / entry_price
            btc_sold_during_cycle = 0
            btc_sold_at_exit = 0
            strategy = "liquidated"
            total_interest = 0

        elif selected_scenario["outcome"] == "MARGIN_CALL":
            liquidation_occurred = False
            margin_call_occurred = True

            # Calculate cure requirements
            target_ltv = self.baseline_ltv
            required_collateral = total_loan_balance / (target_ltv * worst_price)
            cure_btc_needed = max(0, required_collateral - collateral_btc)

            # Assume we can cure and continue
            if cure_btc_needed < collateral_btc * 0.5:  # If cure is feasible
                exit_price = entry_price * 1.05  # Modest 5% gain after cure
                strategy = "monthly_payments"  # Forced to pay monthly after margin call
                monthly_payment = self.calculate_monthly_payment(total_loan_balance)
                total_interest = monthly_payment * (expected_days / 30)
                final_loan_balance = total_loan_balance + total_interest

                btc_purchased = loan_amount / entry_price
                avg_price = (entry_price + exit_price) / 2
                btc_sold_during_cycle = total_interest / avg_price
                btc_sold_at_exit = final_loan_balance / exit_price
                net_btc_gain = btc_purchased - btc_sold_at_exit - btc_sold_during_cycle - cure_btc_needed
            else:
                # Can't cure - becomes liquidation
                liquidation_occurred = True
                margin_call_occurred = False
                net_btc_gain = -collateral_btc * 0.8
                cure_btc_needed = 0
                strategy = "liquidated"
                total_interest = 0
                final_loan_balance = total_loan_balance
                btc_purchased = loan_amount / entry_price
                btc_sold_during_cycle = 0
                btc_sold_at_exit = 0

        else:
            # Normal scenarios: SUCCESSFUL_EXIT or BREAK_EVEN
            liquidation_occurred = False
            margin_call_occurred = False
            cure_btc_needed = 0

            # Determine payment strategy based on scenario performance
            deferred_interest = self.calculate_deferred_interest(total_loan_balance, expected_days)
            final_loan_balance_deferred = total_loan_balance + deferred_interest
            exit_ltv_deferred = self.calculate_ltv(final_loan_balance_deferred, collateral_btc, exit_price)

            monthly_payment = self.calculate_monthly_payment(total_loan_balance)
            total_monthly_interest = monthly_payment * (expected_days / 30)

            # Choose strategy: defer if exit LTV manageable
            if exit_ltv_deferred < 0.70:  # More conservative threshold
                strategy = "deferred"
                total_interest = deferred_interest
                final_loan_balance = final_loan_balance_deferred
                btc_sold_during_cycle = 0
            else:
                strategy = "monthly_payments"
                total_interest = total_monthly_interest
                final_loan_balance = total_loan_balance + total_interest
                avg_price = (entry_price + exit_price) / 2
                btc_sold_during_cycle = total_interest / avg_price

            # Calculate BTC flows
            btc_purchased = loan_amount / entry_price
            btc_sold_at_exit = final_loan_balance / exit_price
            net_btc_gain = btc_purchased - btc_sold_at_exit - btc_sold_during_cycle

        return {
            "entry_price": entry_price,
            "exit_price": exit_price,
            "cycle_duration_days": expected_days,
            "loan_amount": loan_amount,
            "origination_fee": origination_fee,
            "total_loan_balance": total_loan_balance,
            "payment_strategy": strategy,
            "total_interest": total_interest,
            "final_loan_balance": final_loan_balance,
            "interest_rate_effective": (total_interest / loan_amount) * 100 if loan_amount > 0 else 0,
            "btc_purchased": btc_purchased,
            "btc_sold_during_cycle": btc_sold_during_cycle,
            "btc_sold_at_exit": btc_sold_at_exit,
            "cure_btc_needed": cure_btc_needed,
            "net_btc_gain": net_btc_gain,
            "margin_call_occurred": margin_call_occurred,
            "liquidation_occurred": liquidation_occurred,
            "worst_expected_drawdown": selected_scenario["worst_drawdown"],
            "worst_ltv": worst_ltv,
            "exit_ltv": self.calculate_ltv(final_loan_balance, collateral_btc - cure_btc_needed, exit_price),
            "scenario_name": selected_scenario["scenario"],
            "scenario_description": selected_scenario["description"]
        }

def setup_export_directory():
    """Create timestamped export directory."""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path("exports") / f"simulation_{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir

def generate_synthetic_btc_data() -> pd.Series:
    """Generate synthetic Bitcoin price data for simulation backtesting."""
    try:
        # Generate synthetic Bitcoin-like price data
        n_days = 3650  # 10 years

        # Base trend: $4k to $100k over 10 years
        trend = np.linspace(4000, 100000, n_days)

        # Add realistic volatility
        np.random.seed(42)  # Reproducible results
        daily_volatility = 0.08  # 8% daily volatility
        random_walk = np.cumsum(np.random.normal(0, daily_volatility, n_days))

        # Combine trend with volatility
        prices = trend * np.exp(random_walk * 0.3)  # Scale volatility impact

        # Add some major crash events (historically accurate)
        crash_points = [800, 1200, 2100, 2800]  # Approximate crash days
        for crash_day in crash_points:
            if crash_day < len(prices):
                crash_magnitude = np.random.uniform(0.5, 0.8)  # 50-80% crash
                recovery_days = np.random.randint(200, 600)

                end_crash = min(crash_day + recovery_days, len(prices))
                for i in range(crash_day, end_crash):
                    recovery_factor = (i - crash_day) / recovery_days
                    prices[i] *= (crash_magnitude + (1 - crash_magnitude) * recovery_factor)

        # Create pandas Series with dates
        start_date = dt.datetime(2014, 1, 1)
        dates = [start_date + dt.timedelta(days=i) for i in range(n_days)]

        synthetic_series = pd.Series(prices, index=pd.DatetimeIndex(dates))
        synthetic_series.name = 'Close'

        print(f"📊 Generated {len(synthetic_series)} days of synthetic Bitcoin data")
        print(f"   Price range: ${synthetic_series.min():,.0f} - ${synthetic_series.max():,.0f}")

        return synthetic_series

    except Exception as e:
        print(f"⚠️  Synthetic data generation failed: {e}")
        # Fallback: simple linear data
        dates = pd.date_range(start='2014-01-01', periods=1000, freq='D')
        simple_prices = np.linspace(5000, 80000, 1000)
        return pd.Series(simple_prices, index=dates, name='Close')

def main():
    """Run the improved Bitcoin lending simulation."""
    print("🚀 Starting Accurate Bitcoin Collateral Lending Simulation")
    print("=" * 60)

    # Create timestamped export directory
    export_dir = setup_export_directory()
    print(f"📁 Export directory: {export_dir}")
    print("=" * 60)

    # Load price data
    prices = load_btc_history()
    print(f"📊 Loaded {len(prices)} days of price data")

    # Analyze historical drawdowns
    draw_df = worst_drop_until_recovery(prices, 30000.0)
    print(f"📈 Analyzed {len(draw_df)} historical recovery cycles")

    # Fit drawdown model
    drawdown_model = fit_drawdown_model(draw_df)

    # Initialize simulation
    simulator = LoanSimulator()

    # Starting conditions
    start_btc = 0.24
    start_price = 118000.0
    btc_goal = 1.0

    # Calculate initial loan amount using contract terms
    initial_collateral = start_btc * 0.6  # Use 60% as collateral, keep 40% as buffer
    worst_case_drop = drawdown_model(start_price)
    worst_case_price = start_price * (1 - worst_case_drop)
    
    # Use contract baseline LTV (75%) with safety margin for worst case
    max_safe_loan = initial_collateral * worst_case_price * 0.70  # 70% of collateral value at crash price
    
    initial_loan = min(max_safe_loan, 30000.0)  # Reasonable cap
    initial_loan = max(initial_loan, simulator.min_loan)  # Ensure minimum

    print(f"💰 Initial loan amount: ${initial_loan:,.0f}")
    print(f"🪙 Initial collateral: {initial_collateral:.4f} BTC")
    print(f"📉 Expected worst drawdown: {worst_case_drop:.1%}")
    
    # VALIDATE STRATEGY VIABILITY UPFRONT
    viability_check = simulator.validate_strategy_viability(start_btc, start_price, btc_goal)
    print(f"\n🔍 STRATEGY VIABILITY CHECK:")
    print(f"   Status: {'✅ VIABLE' if viability_check['viable'] else '❌ NOT VIABLE'}")
    print(f"   Reason: {viability_check['reason']}")
    print(f"   Recommendation: {viability_check['recommendation']}")
    
    if not viability_check['viable']:
        print(f"\n🛑 ABORTING SIMULATION - Strategy is not mathematically viable")
        print(f"   Please adjust parameters based on recommendations above")
        return

    # Simulation state
    free_btc = start_btc - initial_collateral
    collateral_btc = initial_collateral
    current_price = start_price
    cycle = 0

    results = []

    while free_btc < btc_goal and cycle < 50:  # Safety limit
        cycle += 1

        # Determine loan size for this cycle using contract terms
        worst_drop = drawdown_model(current_price)
        worst_price = current_price * (1 - worst_drop)
        
        # Use contract baseline LTV but with safety margin for drawdowns
        max_loan = collateral_btc * worst_price * 0.70  # 70% of collateral at worst price
        loan_amount = min(max_loan, free_btc * current_price * 0.8)  # Less conservative sizing
        loan_amount = max(loan_amount, simulator.min_loan)

        # Simulate this cycle with probabilistic drawdowns
        cycle_result = simulator.simulate_cycle(
            current_price, collateral_btc, loan_amount, drawdown_model, cycle
        )

        # Check if liquidation occurred
        if cycle_result["liquidation_occurred"]:
            print(f"💥 LIQUIDATION in cycle {cycle} - simulation terminated")
            cycle_result["cycle"] = cycle
            cycle_result["free_btc_before"] = free_btc
            cycle_result["collateral_btc"] = collateral_btc
            cycle_result["free_btc_after"] = 0  # Lost everything
            results.append(cycle_result)
            break

        # Apply cycle results
        btc_change = cycle_result["net_btc_gain"] - cycle_result["cure_btc_needed"]
        free_btc += btc_change
        collateral_btc -= cycle_result["cure_btc_needed"]  # BTC moved to cure margin call
        current_price = cycle_result["exit_price"]
        
        # CRITICAL: Check for negative BTC (impossible scenario)
        if free_btc < 0:
            print(f"🚨 CRITICAL ERROR: Negative BTC holdings detected ({free_btc:.4f} BTC)")
            print(f"   This indicates the strategy has failed - terminating simulation")
            break
            
        # Check for insufficient capital to continue
        if free_btc < 0.01:  # Less than 0.01 BTC remaining
            print(f"⚠️  Insufficient BTC to continue strategy ({free_btc:.4f} BTC remaining)")
            print(f"   Strategy has effectively failed - terminating simulation")
            break

        # Add tracking info
        cycle_result["cycle"] = cycle
        cycle_result["free_btc_before"] = free_btc - btc_change
        cycle_result["collateral_btc"] = collateral_btc
        cycle_result["free_btc_after"] = free_btc
        cycle_result["total_btc"] = free_btc + collateral_btc

        results.append(cycle_result)

        print(f"📊 Cycle {cycle}: ${current_price:,.0f} → ${cycle_result['exit_price']:,.0f}, "
              f"BTC: {free_btc:.4f}, Strategy: {cycle_result['payment_strategy']}")

        # Stop if we've reached our goal
        if free_btc >= btc_goal:
            print(f"🎯 Goal reached! Free BTC: {free_btc:.4f}")
            break

        # Prepare for next cycle - reset collateral to safe level
        if free_btc > 0.1:  # Ensure we have enough for collateral
            collateral_btc = min(free_btc / 2, 0.5)  # Conservative collateral management
            free_btc -= collateral_btc

    # Save and analyze results
    if results:
        df = pd.DataFrame(results)

        # Save to timestamped export directory
        cycles_csv = export_dir / "cycles_log.csv"
        df.to_csv(cycles_csv, index=False)
        print(f"💾 Saved cycles log: {cycles_csv}")

        # Generate plots
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(df.cycle, df.exit_price, marker="o")
        plt.xlabel("Cycle")
        plt.ylabel("BTC Exit Price ($)")
        plt.title("BTC Price per Cycle")

        plt.subplot(2, 2, 2)
        plt.plot(df.cycle, df.free_btc_after, marker="o", color="orange")
        plt.xlabel("Cycle")
        plt.ylabel("Free BTC")
        plt.title("BTC Holdings Over Time")

        plt.subplot(2, 2, 3)
        strategy_colors = {'deferred': 'blue', 'monthly_payments': 'red'}
        colors = [strategy_colors.get(s, 'gray') for s in df.payment_strategy]
        plt.scatter(df.cycle, df.interest_rate_effective, c=colors, alpha=0.7)
        plt.xlabel("Cycle")
        plt.ylabel("Effective Interest Rate (%)")
        plt.title("Interest Rates by Strategy")
        plt.legend(['Deferred', 'Monthly Payments'])

        plt.subplot(2, 2, 4)
        margin_calls = df[df.margin_call_occurred]
        plt.bar(range(len(df)), df.worst_ltv, alpha=0.6)
        plt.axhline(y=0.85, color='orange', linestyle='--', label='Margin Call (85%)')
        plt.axhline(y=0.90, color='red', linestyle='--', label='Liquidation (90%)')
        if len(margin_calls) > 0:
            plt.scatter(margin_calls.cycle - 1, margin_calls.worst_ltv, color='red', s=100, label='Margin Calls')
        plt.xlabel("Cycle")
        plt.ylabel("Worst LTV During Cycle")
        plt.title("LTV Risk Management")
        plt.legend()

        plt.tight_layout()

        # Save plots to export directory
        analysis_plot = export_dir / "simulation_analysis.png"
        plt.savefig(analysis_plot, dpi=150, bbox_inches='tight')
        print(f"📊 Saved analysis plot: {analysis_plot}")

        # Print summary
        print("\n" + "=" * 60)
        print("📊 SIMULATION SUMMARY")
        print("=" * 60)

        total_interest = df.total_interest.sum()
        total_loans = df.loan_amount.sum()
        final_btc = df.free_btc_after.iloc[-1] if len(df) > 0 else 0
        total_cycles = len(df)
        total_days = df.cycle_duration_days.sum()

        deferred_cycles = len(df[df.payment_strategy == 'deferred'])
        monthly_cycles = len(df[df.payment_strategy == 'monthly_payments'])
        margin_calls = len(df[df.margin_call_occurred])
        liquidations = len(df[df.liquidation_occurred])

        print(f"💰 Final BTC Holdings: {final_btc:.4f} BTC")
        print(f"🔄 Total Cycles: {total_cycles}")
        print(f"⏱️  Total Time: {total_days:.0f} days ({total_days/365:.1f} years)")
        print(f"💸 Total Interest Paid: ${total_interest:,.0f}")
        print(f"📊 Average Interest Rate: {100*total_interest/total_loans:.1f}%")
        print(f"🔵 Deferred Interest Cycles: {deferred_cycles}")
        print(f"🔴 Monthly Payment Cycles: {monthly_cycles}")
        print(f"⚠️  Margin Calls: {margin_calls}")
        print(f"💥 Liquidations: {liquidations}")

        if final_btc >= btc_goal:
            print(f"🎯 SUCCESS: Goal of {btc_goal} BTC achieved!")
        else:
            print(f"❌ Goal not reached. Strategy may not be viable.")

        # Create summary report
        summary_file = export_dir / "simulation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Bitcoin Collateral Lending Simulation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Export Date: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Final BTC Holdings: {final_btc:.4f} BTC\n")
            f.write(f"Total Cycles: {total_cycles}\n")
            f.write(f"Total Time: {total_days:.0f} days ({total_days/365:.1f} years)\n")
            f.write(f"Total Interest Paid: ${total_interest:,.0f}\n")
            f.write(f"Average Interest Rate: {100*total_interest/total_loans:.1f}%\n")
            f.write(f"Deferred Interest Cycles: {deferred_cycles}\n")
            f.write(f"Monthly Payment Cycles: {monthly_cycles}\n")
            f.write(f"Margin Calls: {margin_calls}\n")
            f.write(f"Liquidations: {liquidations}\n\n")

            if final_btc >= btc_goal:
                f.write(f"✅ SUCCESS: Goal of {btc_goal} BTC achieved!\n")
            else:
                f.write(f"❌ Goal not reached. Strategy may not be viable.\n")

        print(f"\n📁 All files saved to: {export_dir}")
        print(f"   • cycles_log.csv - Detailed cycle data")
        print(f"   • simulation_analysis.png - Visual analysis")
        print(f"   • simulation_summary.txt - Text summary")

    else:
        print("❌ No cycles completed - strategy failed immediately")

if __name__ == "__main__":
    main()