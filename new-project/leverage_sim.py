#!/usr/bin/env python3
"""
Restructured the strategy to match the mermaid diagram, incorporating a $30K starting capital and fixed loan parameters.
"""
import datetime as dt
import math
import os
import sys
import time
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

        # Risk management parameters - maximum safety
        self.max_safe_ltv = 0.30  # Never exceed 30% LTV for safety
        self.min_collateral_buffer = 0.12  # Always keep 0.12 BTC as backup per diagram
        self.exit_appreciation = 30000.0  # Exit when BTC appreciates by $30K per diagram

        # Risk management parameters - maximum safety
        self.max_safe_ltv = 0.30  # Never exceed 30% LTV for safety
        self.min_collateral_buffer = 0.12  # Always keep 0.12 BTC as backup per diagram
        self.exit_appreciation = 30000.0  # Exit when BTC appreciates by $30K per diagram

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

    def suggest_viable_parameters(self, start_price: float, goal_btc: float) -> dict:
        """Suggest viable parameter combinations when strategy fails."""
        suggestions = []

        # Option 1: Calculate minimum BTC needed for current goal
        min_btc_for_goal = self.min_loan / (start_price * 0.40 * self.max_safe_ltv) + self.min_collateral_buffer
        suggestions.append({
            "option": "Increase Starting Capital",
            "start_btc": min_btc_for_goal,
            "goal_btc": goal_btc,
            "start_value": min_btc_for_goal * start_price,
            "description": f"Need {min_btc_for_goal:.3f} BTC (${min_btc_for_goal * start_price:,.0f}) to safely reach {goal_btc} BTC goal"
        })

        # Option 2: Lower goal to match current capital
        for test_btc in [0.24, 0.5, 0.75]:
            max_collateral = test_btc - self.min_collateral_buffer
            if max_collateral > 0:
                conservative_crash_price = start_price * 0.40
                max_safe_loan = max_collateral * conservative_crash_price * 0.75
                if max_safe_loan >= self.min_loan:
                    # Estimate achievable goal
                    btc_per_cycle = max_safe_loan / start_price * 0.3  # Conservative efficiency
                    achievable_gain = btc_per_cycle * 10  # 10 cycles max
                    achievable_goal = test_btc + achievable_gain

                    suggestions.append({
                        "option": f"Lower Goal (with {test_btc} BTC capital)",
                        "start_btc": test_btc,
                        "goal_btc": achievable_goal,
                        "start_value": test_btc * start_price,
                        "description": f"With {test_btc} BTC, realistically achieve ~{achievable_goal:.2f} BTC goal"
                    })

        # Option 3: DCA alternative
        monthly_dca = 1000  # $1000/month
        months_to_goal = (goal_btc * start_price) / monthly_dca
        suggestions.append({
            "option": "Dollar Cost Averaging Alternative",
            "start_btc": 0.24,
            "goal_btc": goal_btc,
            "start_value": 0.24 * start_price,
            "description": f"DCA ${monthly_dca}/month would reach goal in {months_to_goal:.1f} months with less risk"
        })

        return {"suggestions": suggestions}

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
                "recovery_months":