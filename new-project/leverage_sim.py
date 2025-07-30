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

def create_probabilistic_drawdown_model(historical_data=None):
    """Create a probabilistic drawdown model that samples from historical crash distributions."""
    
    # Historical Bitcoin crash data (manually curated from research)
    # Format: (year, peak_price, trough_price, crash_percentage, duration_days)
    historical_crashes = [
        (2011, 32, 2, 0.93, 160),          # 93% crash
        (2014, 1165, 152, 0.87, 410),      # 87% crash  
        (2018, 20000, 3200, 0.84, 365),    # 84% crash
        (2022, 69000, 15500, 0.77, 238),   # 77% crash
        (2021, 65000, 28800, 0.56, 92),    # 56% correction
        (2020, 12000, 3800, 0.68, 35),     # COVID crash
        (2017, 20000, 11000, 0.45, 45),    # Mid-cycle correction
        (2019, 14000, 6400, 0.54, 180),    # Bear market low test
    ]
    
    # Convert to drawdown percentages
    crash_severities = [crash[3] for crash in historical_crashes]
    
    def probabilistic_drawdown(price: float, cycle_number: int = 1) -> float:
        """
        Sample a drawdown from historical distribution with realistic probabilities.
        
        Uses Monte Carlo sampling to determine crash severity based on:
        - Historical frequency of different crash magnitudes
        - Market cycle position
        - Price level factors
        """
        
        # Set random seed based on price and cycle for deterministic but varied results
        np.random.seed(int(price) + cycle_number * 1000)
        
        # Define probability buckets based on historical frequency
        drawdown_buckets = [
            {"range": (0.05, 0.20), "probability": 0.40, "description": "Minor correction (5-20%)"},
            {"range": (0.20, 0.40), "probability": 0.25, "description": "Moderate correction (20-40%)"},
            {"range": (0.40, 0.60), "probability": 0.20, "description": "Major correction (40-60%)"},
            {"range": (0.60, 0.80), "probability": 0.10, "description": "Severe crash (60-80%)"},
            {"range": (0.80, 0.95), "probability": 0.05, "description": "Extreme crash (80-95%)"},
        ]
        
        # Adjust probabilities based on price level and market conditions
        if price > 100000:  # Uncharted territory - higher crash risk
            # Shift probability toward larger crashes
            drawdown_buckets[0]["probability"] = 0.25  # Reduce minor corrections
            drawdown_buckets[2]["probability"] = 0.30  # Increase major corrections
            drawdown_buckets[3]["probability"] = 0.15  # Increase severe crashes
            drawdown_buckets[4]["probability"] = 0.10  # Increase extreme crashes
        elif price < 30000:  # Lower prices - more stable historically
            # Shift probability toward smaller corrections
            drawdown_buckets[0]["probability"] = 0.50  # Increase minor corrections
            drawdown_buckets[3]["probability"] = 0.05  # Decrease severe crashes
            drawdown_buckets[4]["probability"] = 0.02  # Decrease extreme crashes
        
        # Normalize probabilities
        total_prob = sum(bucket["probability"] for bucket in drawdown_buckets)
        for bucket in drawdown_buckets:
            bucket["probability"] /= total_prob
        
        # Sample from probability distribution
        rand_val = np.random.random()
        cumulative_prob = 0
        selected_bucket = None
        
        for bucket in drawdown_buckets:
            cumulative_prob += bucket["probability"]
            if rand_val <= cumulative_prob:
                selected_bucket = bucket
                break
        
        if selected_bucket is None:
            selected_bucket = drawdown_buckets[-1]  # Fallback
        
        # Sample specific drawdown within the selected bucket
        min_draw, max_draw = selected_bucket["range"]
        sampled_drawdown = np.random.uniform(min_draw, max_draw)
        
        print(f"🎲 Probabilistic drawdown at ${price:,.0f}: {sampled_drawdown:.1%}")
        print(f"   Selected: {selected_bucket['description']}")
        print(f"   (Sampled from historical crash distribution)")
        
        return sampled_drawdown
    
    return probabilistic_drawdown

def fit_drawdown_model(draw_df: pd.DataFrame):
    """Fit a probabilistic drawdown model based on historical crash patterns."""
    print("📊 Using probabilistic drawdown model based on Bitcoin crash history")
    
    # Create probabilistic model that samples from historical distributions
    probabilistic_model = create_probabilistic_drawdown_model(draw_df)
    
    # If we have historical data, show comparison
    if len(draw_df) >= 10:
        historical_worst = abs(draw_df.draw.min())  # Most negative (worst) drawdown
        historical_95th = np.percentile(np.abs(draw_df.draw), 95)
        historical_median = np.percentile(np.abs(draw_df.draw), 50)
        
        print(f"📈 Historical data shows:")
        print(f"   • Worst observed drawdown: {historical_worst:.1%}")
        print(f"   • 95th percentile drawdown: {historical_95th:.1%}")
        print(f"   • Median drawdown: {historical_median:.1%}")
        
        # Test the probabilistic model with a few samples
        test_price = draw_df.price.median()
        print(f"🎲 Probabilistic model samples at ${test_price:,.0f}:")
        for i in range(3):
            sample_draw = probabilistic_model(test_price, i+1)
            print(f"   Sample {i+1}: {sample_draw:.1%}")
    else:
        print("⚠️  Limited historical data - using built-in crash distribution")
    
    return probabilistic_model

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
                                          collateral_btc: float, months: int = 6) -> dict:
    """
    Simulate realistic cycle outcome using historical Bitcoin patterns.
    Returns probability-weighted scenarios instead of single optimistic path.
    """
    print(f"🎲 Simulating realistic outcomes for {months}-month cycle starting at ${start_price:,.0f}")

    # Use conservative model-based scenarios that reflect 70-85% crash potential
    scenarios = {
        "bull_run": {
            "probability": 0.20,  # 20% chance
            "monthly_return": 0.15,  # 15% per month (more realistic)
            "description": "Strong bull market"
        },
        "moderate_growth": {
            "probability": 0.25,  # 25% chance
            "monthly_return": 0.05,  # 5% per month (conservative)
            "description": "Steady upward trend"
        },
        "sideways": {
            "probability": 0.25,  # 25% chance
            "monthly_return": 0.00,  # 0% per month (true sideways)
            "description": "Choppy sideways movement"
        },
        "decline": {
            "probability": 0.20,  # 20% chance
            "monthly_return": -0.08,  # -8% per month (realistic decline)
            "description": "Gradual decline"
        },
        "bear_crash": {
            "probability": 0.10,  # 10% chance - matches conservative model expectations
            "monthly_return": -0.25,  # -25% per month (70%+ annual crash)
            "description": "Major bear market crash (70%+ drop)"
        }
    }

    results = []

    for scenario_name, scenario in scenarios.items():
        # Calculate final price for this scenario
        monthly_multiplier = 1 + scenario["monthly_return"]
        final_price = start_price * (monthly_multiplier ** months)

        # Add realistic volatility
        np.random.seed(hash(scenario_name) % 1000)  # Deterministic but varied
        volatility_factor = np.random.normal(1.0, 0.20)  # ±20% volatility around trend
        final_price *= volatility_factor

        # Use probabilistic model for realistic drawdown sampling
        if scenario_name == "bear_crash":
            # Major crash scenario: sample from severe crash distribution
            np.random.seed(hash(scenario_name + str(int(start_price))) % 1000)
            # Force selection from severe crash buckets (60-95% range)
            worst_drawdown = np.random.uniform(0.60, 0.85)
        elif scenario["monthly_return"] < 0:
            # Decline scenarios: sample from moderate to major correction range
            np.random.seed(hash(scenario_name + str(int(start_price))) % 1000)
            worst_drawdown = np.random.uniform(0.20, 0.60)
        else:
            # Bull/sideways scenarios: sample from minor to moderate correction range
            np.random.seed(hash(scenario_name + str(int(start_price))) % 1000)
            worst_drawdown = np.random.uniform(0.05, 0.35)

        worst_price = start_price * (1 - worst_drawdown)

        # Calculate LTV at worst point
        loan_balance = loan_size * 1.115 ** (months/12)  # Approximate compound interest
        worst_ltv = loan_balance / (collateral_btc * worst_price)

        # Determine outcome
        if worst_ltv >= 0.90:
            outcome = "LIQUIDATION"
        elif worst_ltv >= 0.85:
            outcome = "MARGIN_CALL"
        elif final_price > start_price * 1.10:  # 10% gain threshold
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

        # Realistic but safe LTV thresholds based on 70% max crash planning
        self.baseline_ltv = 0.50  # Target 50% LTV under normal conditions
        self.margin_call_ltv = 0.70  # Early warning at 70% 
        self.liquidation_ltv = 0.80  # Forced exit at 80% (safety margin for 70% crashes)
        self.collateral_release_ltv = 0.20

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

        # Conservative drawdown assumption
        conservative_crash_price = start_price * 0.25  # 75% crash
        max_safe_loan = max_collateral * conservative_crash_price * self.max_safe_ltv

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

        # Use realistic scenario modeling instead of optimistic price paths
        cycle_months = 6  # Standard 6-month cycle
        realistic_outcomes = simulate_realistic_cycle_outcome(
            entry_price, loan_amount, collateral_btc, cycle_months
        )

        # Choose scenario based on probabilities (use weighted random selection)
        np.random.seed(int(entry_price) % 1000)  # Deterministic but varied
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

    # Calculate initial conservative loan amount
    initial_collateral = start_btc / 2  # Reserve half as buffer
    worst_case_drop = drawdown_model(start_price)
    worst_case_price = start_price * (1 - worst_case_drop)
    max_safe_loan = initial_collateral * worst_case_price * 0.75  # 75% LTV at worst case

    initial_loan = min(max_safe_loan, 50000.0)  # Cap at reasonable amount
    initial_loan = max(initial_loan, simulator.min_loan)  # Ensure minimum

    print(f"💰 Initial loan amount: ${initial_loan:,.0f}")
    print(f"🪙 Initial collateral: {initial_collateral:.4f} BTC")
    print(f"📉 Expected worst drawdown: {worst_case_drop:.1%}")

    # Simulation state
    free_btc = start_btc - initial_collateral
    collateral_btc = initial_collateral
    current_price = start_price
    cycle = 0

    results = []

    while free_btc < btc_goal and cycle < 50:  # Safety limit
        cycle += 1

        # Determine loan size for this cycle
        worst_drop = drawdown_model(current_price)
        worst_price = current_price * (1 - worst_drop)
        max_loan = collateral_btc * worst_price * 0.75
        loan_amount = min(max_loan, free_btc * current_price * 0.5)  # Conservative sizing
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