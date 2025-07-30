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

# Try to import yfinance; fall back to CSV if offline
try:
    import yfinance as yf
    ONLINE = True
except ImportError:
    ONLINE = False
    print("⚠️  yfinance not found. Using synthetic data.")

def generate_synthetic_btc_data():
    """Generate synthetic BTC price data for simulation when real data fails."""
    print("🔄 Generating synthetic BTC price data for simulation...")

    # Create 5 years of daily data
    dates = pd.date_range(start='2019-01-01', end='2024-01-01', freq='D')

    # Generate realistic BTC price progression with volatility
    np.random.seed(42)  # For reproducible results
    n_days = len(dates)

    # Start at $4000, trend upward to ~$100k with realistic volatility
    trend = np.linspace(4000, 100000, n_days)
    volatility = np.random.normal(0, 0.04, n_days)  # 4% daily volatility

    # Apply cumulative volatility
    price_changes = np.exp(np.cumsum(volatility))
    prices = trend * price_changes

    # Add some realistic crashes and recoveries
    crash_points = [int(n_days * 0.3), int(n_days * 0.6), int(n_days * 0.8)]
    for crash_idx in crash_points:
        if crash_idx < len(prices):
            crash_magnitude = np.random.uniform(0.3, 0.7)  # 30-70% crash
            recovery_days = 200
            end_idx = min(crash_idx + recovery_days, len(prices))

            # Apply crash and gradual recovery
            for i in range(crash_idx, end_idx):
                recovery_factor = (i - crash_idx) / recovery_days
                prices[i] *= (crash_magnitude + (1 - crash_magnitude) * recovery_factor)

    return pd.Series(prices, index=dates, name='Close')

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

    # Method 3: yfinance (if available)
    if ONLINE:
        try:
            print("📡 Tier 4: Trying yfinance...")
            btc = yf.download("BTC-USD", start="2015-01-01", progress=False)
            if not btc.empty and 'Adj Close' in btc.columns:
                prices = btc["Adj Close"].dropna()
                if len(prices) >= 100:
                    print(f"✅ yfinance successful: {len(prices)} days")
                    return prices
        except Exception as e:
            print(f"⚠️  yfinance failed: {e}")

    # Method 4: Local CSV fallback
    try:
        csv_files = ["btc_history_backup.csv", "btc_history.csv"]
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                print(f"📂 Loading BTC data from {csv_file}...")
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

                # Try different column names
                price_col = None
                for col in ['Close', 'close', 'price', 'Price']:
                    if col in df.columns:
                        price_col = col
                        break

                if price_col:
                    btc_series = df[price_col].astype(float).dropna()
                    btc_series.index = pd.to_datetime(btc_series.index)
                    print(f"✅ Successfully loaded {len(btc_series)} days from CSV")
                    return btc_series

    except Exception as e:
        print(f"⚠️  CSV loading failed: {e}")

    # Method 5: Synthetic data (last resort)
    print("📊 All real data sources failed, using synthetic data...")
    return generate_synthetic_btc_data()

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

def fit_drawdown_model(draw_df: pd.DataFrame):
    """Fit a drawdown model from historical data."""
    if len(draw_df) < 10:
        print("⚠️  Insufficient data for drawdown model, using default")
        return lambda price: max(0.15, min(0.80, 0.5 * (price / 50000) ** (-0.2)))

    # Simple percentile-based model by price ranges
    min_price = draw_df.price.min()
    max_price = draw_df.price.max()

    # Create price bins
    n_bins = min(20, max(5, len(draw_df) // 10))
    bins = np.linspace(min_price, max_price, n_bins)

    draw_df["bin"] = pd.cut(draw_df.price, bins=bins)

    bin_stats = draw_df.groupby("bin", observed=False).agg(
        p=("price", "median"),
        d95=("draw", lambda x: np.percentile(np.abs(x), 95) if len(x) > 0 else 0.3),
        count=("draw", "count")
    ).dropna()

    # Filter bins with sufficient data
    bin_stats = bin_stats[bin_stats['count'] >= 2]

    if len(bin_stats) < 3:
        return lambda price: max(0.15, min(0.80, 0.4 * (price / 50000) ** (-0.15)))

    # Simple interpolation model
    prices = bin_stats.p.values
    drawdowns = bin_stats.d95.values

    def drawdown_model(price: float) -> float:
        if price <= prices.min():
            return drawdowns[0]
        elif price >= prices.max():
            return drawdowns[-1]
        else:
            # Linear interpolation
            idx = np.searchsorted(prices, price)
            if idx == 0:
                return drawdowns[0]
            elif idx >= len(prices):
                return drawdowns[-1]
            else:
                weight = (price - prices[idx-1]) / (prices[idx] - prices[idx-1])
                return drawdowns[idx-1] * (1 - weight) + drawdowns[idx] * weight

    return drawdown_model

def simulate_price_path(start_price: float, target_price: float, days: int) -> np.ndarray:
    """Generate realistic price path using geometric Brownian motion."""
    if days <= 1:
        return np.array([start_price, target_price])

    # Calculate drift to reach target
    total_return = np.log(target_price / start_price)
    drift = total_return / days

    # Generate path
    np.random.seed(42)
    dt = 1.0
    volatility = 0.045  # 4.5% daily volatility
    random_shocks = np.random.normal(0, volatility * np.sqrt(dt), days - 1)

    log_returns = drift + random_shocks
    log_prices = np.log(start_price) + np.cumsum(np.concatenate([[0], log_returns]))
    prices = np.exp(log_prices)

    # Ensure we end at target
    prices[-1] = target_price

    return prices

class LoanSimulator:
    def __init__(self):
        # Contract terms based on Figure Lending analysis
        self.min_loan = 10000.0
        self.base_apr = 0.115  # 11.5% for $10K loan
        self.origination_fee_rate = 0.033  # ~3.3% estimated
        self.processing_fee_rate = 0.02  # 2% on liquidations

        # LTV thresholds
        self.baseline_ltv = 0.75
        self.margin_call_ltv = 0.85
        self.liquidation_ltv = 0.90
        self.collateral_release_ltv = 0.35

        # Operational parameters
        self.cure_period_hours = 48
        self.exit_jump = 30000.0

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

    def calculate_required_collateral(self, loan_amount: float, btc_price: float, 
                                    worst_case_price: float) -> float:
        """Calculate BTC needed to avoid liquidation in worst case."""
        # Need enough collateral so that at worst_case_price, LTV < 90%
        return loan_amount / (0.89 * worst_case_price)  # Small buffer below 90%

    def simulate_cycle(self, entry_price: float, collateral_btc: float, 
                      loan_amount: float, drawdown_model) -> dict:
        """Simulate one complete loan cycle with realistic dynamics."""

        # Add origination fee to loan balance
        origination_fee = loan_amount * self.origination_fee_rate
        total_loan_balance = loan_amount + origination_fee

        # Predict cycle duration (simplified model)
        # Higher prices generally take longer to appreciate by fixed amounts
        base_days = 120  # 4 months base
        price_factor = max(0.5, min(2.0, entry_price / 70000))  # Scale with price level
        expected_days = int(base_days * price_factor)

        # Generate price path
        exit_price = entry_price + self.exit_jump
        price_path = simulate_price_path(entry_price, exit_price, expected_days)

        # Determine payment strategy
        # Use deferred interest if expected LTV at exit remains manageable
        deferred_interest = self.calculate_deferred_interest(total_loan_balance, expected_days)
        final_loan_balance_deferred = total_loan_balance + deferred_interest
        exit_ltv_deferred = self.calculate_ltv(final_loan_balance_deferred, collateral_btc, exit_price)

        # Calculate monthly payment strategy impact
        monthly_payment = self.calculate_monthly_payment(total_loan_balance)
        num_payments = expected_days / 30
        total_monthly_interest = monthly_payment * num_payments

        # Choose strategy: defer if exit LTV < 75%, otherwise pay monthly
        if exit_ltv_deferred < 0.75:
            strategy = "deferred"
            total_interest = deferred_interest
            final_loan_balance = final_loan_balance_deferred
            btc_sold_during_cycle = 0  # No monthly sales
        else:
            strategy = "monthly_payments"
            total_interest = total_monthly_interest
            final_loan_balance = total_loan_balance + total_interest
            # Approximate BTC sold for monthly payments
            avg_price = (entry_price + exit_price) / 2
            btc_sold_during_cycle = total_monthly_interest / avg_price

        # Check for margin calls during cycle
        worst_expected_drawdown = drawdown_model(entry_price)
        worst_price = entry_price * (1 - worst_expected_drawdown)

        margin_call_occurred = False
        liquidation_occurred = False
        cure_btc_needed = 0

        # Check if margin call would occur at worst drawdown
        effective_collateral = collateral_btc - btc_sold_during_cycle / 2  # Average impact
        worst_ltv = self.calculate_ltv(total_loan_balance, effective_collateral, worst_price)

        if worst_ltv >= self.liquidation_ltv:
            liquidation_occurred = True
            liquidation_fee = total_loan_balance * self.processing_fee_rate
            final_loan_balance += liquidation_fee
        elif worst_ltv >= self.margin_call_ltv:
            margin_call_occurred = True
            # Calculate additional BTC needed to cure
            target_ltv = self.baseline_ltv
            required_collateral = total_loan_balance / (target_ltv * worst_price)
            cure_btc_needed = max(0, required_collateral - effective_collateral)

        # Calculate BTC flows
        btc_purchased = loan_amount / entry_price  # Initial purchase with loan proceeds
        btc_sold_at_exit = final_loan_balance / exit_price  # Sell to repay loan

        # Net BTC change
        net_btc_gain = btc_purchased - btc_sold_at_exit - btc_sold_during_cycle - cure_btc_needed

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
            "interest_rate_effective": (total_interest / loan_amount) * 100,
            "btc_purchased": btc_purchased,
            "btc_sold_during_cycle": btc_sold_during_cycle,
            "btc_sold_at_exit": btc_sold_at_exit,
            "cure_btc_needed": cure_btc_needed,
            "net_btc_gain": net_btc_gain,
            "margin_call_occurred": margin_call_occurred,
            "liquidation_occurred": liquidation_occurred,
            "worst_expected_drawdown": worst_expected_drawdown,
            "worst_ltv": worst_ltv,
            "exit_ltv": self.calculate_ltv(final_loan_balance, collateral_btc - cure_btc_needed, exit_price)
        }

def setup_export_directory():
    """Create timestamped export directory."""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path("new-project/exports") / f"simulation_{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir

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

        # Simulate this cycle
        cycle_result = simulator.simulate_cycle(
            current_price, collateral_btc, loan_amount, drawdown_model
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