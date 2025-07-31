
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import logging
import traceback
from pathlib import Path

# Try to import yfinance; fall back to CSV if offline
try:
    import yfinance as yf
    ONLINE = True
except ImportError:
    ONLINE = False
    print("⚠️  yfinance not found. Using synthetic data.")

# Set style for better visuals
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_error_logging():
    """Set up comprehensive error logging system."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamp for log file
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"leverage_sim_errors_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Error logging initialized - Log file: {log_file}")
    return logger

def log_error(logger, operation: str, error: Exception, additional_info: dict = None):
    """Log detailed error information."""
    error_info = {
        'operation': operation,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc()
    }
    
    if additional_info:
        error_info.update(additional_info)
    
    logger.error(f"ERROR in {operation}: {error_info}")
    
    # Also save error to dedicated error log
    error_log_file = "logs/critical_errors.log"
    with open(error_log_file, "a") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"TIMESTAMP: {dt.datetime.now()}\n")
        f.write(f"OPERATION: {operation}\n")
        f.write(f"ERROR: {error_info}\n")
        if additional_info:
            f.write(f"ADDITIONAL INFO: {additional_info}\n")
        f.write(f"{'='*50}\n")

def generate_synthetic_btc_data(logger=None):
    """Generate synthetic BTC price data for simulation when real data fails."""
    try:
        if logger:
            logger.info("Starting synthetic BTC data generation")
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

    if logger:
            logger.info(f"Successfully generated {len(prices)} synthetic price points")
        return pd.Series(prices, index=dates, name='Close')
        
    except Exception as e:
        if logger:
            log_error(logger, "generate_synthetic_btc_data", e, {
                'dates_range': f"{dates[0]} to {dates[-1]}" if 'dates' in locals() else "Unknown",
                'n_days': n_days if 'n_days' in locals() else "Unknown"
            })
        print(f"❌ Error generating synthetic data: {e}")
        # Return minimal fallback data
        fallback_dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        fallback_prices = np.linspace(20000, 100000, len(fallback_dates))
        return pd.Series(fallback_prices, index=fallback_dates, name='Close')

def load_btc_history(logger=None) -> pd.Series:
    """Load Bitcoin price history with fallbacks."""
    if logger:
        logger.info("Starting BTC price history loading")

    # Method 1: yfinance (if available)
    if ONLINE:
        try:
            if logger:
                logger.info("Attempting yfinance data download")
            print("📡 Trying yfinance...")
            btc = yf.download("BTC-USD", start="2015-01-01", progress=False)
            if not btc.empty and 'Adj Close' in btc.columns:
                prices = btc["Adj Close"].dropna()
                if len(prices) >= 100:
                    if logger:
                        logger.info(f"yfinance successful: {len(prices)} days loaded")
                    print(f"✅ yfinance successful: {len(prices)} days")
                    return prices
        except Exception as e:
            if logger:
                log_error(logger, "yfinance_download", e, {
                    'online_status': ONLINE,
                    'symbol': 'BTC-USD',
                    'start_date': '2015-01-01'
                })
            print(f"⚠️  yfinance failed: {e}")

    # Method 2: Local CSV fallback
    try:
        if logger:
            logger.info("Attempting CSV data loading")
        csv_files = ["btc_history_backup.csv", "btc_history.csv"]
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                if logger:
                    logger.info(f"Found CSV file: {csv_file}")
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
                    if logger:
                        logger.info(f"Successfully loaded {len(btc_series)} days from {csv_file}")
                    print(f"✅ Successfully loaded {len(btc_series)} days from CSV")
                    return btc_series

    except Exception as e:
        if logger:
            log_error(logger, "csv_loading", e, {
                'csv_files_checked': csv_files,
                'files_exist': [os.path.exists(f) for f in csv_files]
            })
        print(f"⚠️  CSV loading failed: {e}")

    # Method 3: Synthetic data (last resort)
    if logger:
        logger.warning("All real data sources failed, falling back to synthetic data")
    print("📊 All real data sources failed, using synthetic data...")
    return generate_synthetic_btc_data(logger)

def worst_drop_until_recovery(price_series: pd.Series, jump: float = 30000.0, logger=None) -> pd.DataFrame:
    """
    For each start date, find worst drawdown before price recovers by jump amount.
    """
    try:
        if logger:
            logger.info(f"Starting drawdown analysis for {len(price_series)} price points")
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
        if logger:
            logger.info(f"Completed drawdown analysis: {len(df)} recovery cycles found")
        return df
        
    except Exception as e:
        if logger:
            log_error(logger, "worst_drop_until_recovery", e, {
                'price_series_length': len(price_series) if price_series is not None else "None",
                'jump_amount': jump,
                'series_start': str(price_series.index[0]) if len(price_series) > 0 else "Empty",
                'series_end': str(price_series.index[-1]) if len(price_series) > 0 else "Empty"
            })
        print(f"❌ Error in drawdown analysis: {e}")
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=["date", "price", "draw"])

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

def create_enhanced_visualizations(df: pd.DataFrame, start_btc: float, btc_goal: float):
    """Create comprehensive visual dashboard for simulation results."""
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color palette
    colors = sns.color_palette("husl", 8)
    strategy_colors = {'deferred': colors[0], 'monthly_payments': colors[1]}
    
    # 1. BTC Price Evolution (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df.cycle, df.entry_price, 'o-', label='Entry Price', color=colors[2], linewidth=2, markersize=6)
    ax1.plot(df.cycle, df.exit_price, 's-', label='Exit Price', color=colors[3], linewidth=2, markersize=6)
    ax1.fill_between(df.cycle, df.entry_price, df.exit_price, alpha=0.3, color=colors[2])
    ax1.set_title('Bitcoin Price Evolution Across Cycles', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Cycle Number')
    ax1.set_ylabel('BTC Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 2. BTC Holdings Progress (top right, spans 2 columns)
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.plot(df.cycle, df.free_btc_after, 'o-', color=colors[4], linewidth=3, markersize=8)
    ax2.axhline(y=btc_goal, color='red', linestyle='--', linewidth=2, label=f'Goal: {btc_goal} BTC')
    ax2.axhline(y=start_btc, color='gray', linestyle=':', linewidth=2, label=f'Start: {start_btc} BTC')
    ax2.fill_between(df.cycle, 0, df.free_btc_after, alpha=0.4, color=colors[4])
    ax2.set_title('BTC Holdings Growth Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Cycle Number')
    ax2.set_ylabel('Free BTC Holdings')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Payment Strategy Distribution (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    strategy_counts = df.payment_strategy.value_counts()
    wedges, texts, autotexts = ax3.pie(strategy_counts.values, labels=strategy_counts.index, 
                                       autopct='%1.1f%%', startangle=90,
                                       colors=[strategy_colors[s] for s in strategy_counts.index])
    ax3.set_title('Payment Strategy\nDistribution', fontsize=12, fontweight='bold')
    
    # 4. Interest Rate Analysis (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    for strategy, color in strategy_colors.items():
        strategy_data = df[df.payment_strategy == strategy]
        if len(strategy_data) > 0:
            ax4.scatter(strategy_data.cycle, strategy_data.interest_rate_effective, 
                       c=color, alpha=0.7, s=60, label=strategy.replace('_', ' ').title())
    ax4.set_title('Effective Interest Rates', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Cycle')
    ax4.set_ylabel('Interest Rate (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. LTV Risk Heatmap (middle right, spans 2 columns)
    ax5 = fig.add_subplot(gs[1, 2:])
    ltv_data = df[['cycle', 'worst_ltv', 'exit_ltv']].set_index('cycle')
    im = ax5.imshow(ltv_data.T, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax5.set_title('LTV Risk Heatmap', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Cycle')
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['Worst LTV', 'Exit LTV'])
    
    # Add threshold lines
    for i, threshold in enumerate([0.85, 0.90]):
        ax5.axhline(y=-0.4 + i * 0.1, color='red', linewidth=2, alpha=0.8)
    
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('LTV Ratio')
    
    # 6. Financial Flow Analysis (bottom left, spans 2 columns)
    ax6 = fig.add_subplot(gs[2, :2])
    x = df.cycle
    width = 0.35
    
    ax6.bar(x - width/2, df.loan_amount, width, label='Loan Amount', color=colors[5], alpha=0.8)
    ax6.bar(x + width/2, df.total_interest, width, label='Total Interest', color=colors[6], alpha=0.8)
    
    ax6.set_title('Financial Flows per Cycle', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Cycle')
    ax6.set_ylabel('Amount ($)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 7. Risk Events Timeline (bottom right, spans 2 columns)
    ax7 = fig.add_subplot(gs[2, 2:])
    margin_calls = df[df.margin_call_occurred]
    liquidations = df[df.liquidation_occurred]
    
    # Normal cycles
    normal_cycles = df[~(df.margin_call_occurred | df.liquidation_occurred)]
    ax7.scatter(normal_cycles.cycle, [1]*len(normal_cycles), 
               c='green', s=100, alpha=0.7, label='Normal Cycles')
    
    # Margin calls
    if len(margin_calls) > 0:
        ax7.scatter(margin_calls.cycle, [2]*len(margin_calls), 
                   c='orange', s=150, marker='^', label='Margin Calls')
    
    # Liquidations
    if len(liquidations) > 0:
        ax7.scatter(liquidations.cycle, [3]*len(liquidations), 
                   c='red', s=200, marker='X', label='Liquidations')
    
    ax7.set_title('Risk Events Timeline', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Cycle')
    ax7.set_ylabel('Event Type')
    ax7.set_yticks([1, 2, 3])
    ax7.set_yticklabels(['Normal', 'Margin Call', 'Liquidation'])
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Cumulative Returns (bottom, spans full width)
    ax8 = fig.add_subplot(gs[3, :])
    cumulative_btc = df.free_btc_after.cumsum()
    cumulative_usd = cumulative_btc * df.exit_price
    
    ax8_twin = ax8.twinx()
    
    line1 = ax8.plot(df.cycle, df.free_btc_after, 'o-', color=colors[7], 
                     linewidth=3, markersize=6, label='BTC Holdings')
    line2 = ax8_twin.plot(df.cycle, cumulative_usd, 's-', color=colors[1], 
                          linewidth=3, markersize=6, label='USD Value')
    
    ax8.set_title('Portfolio Value Growth', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Cycle Number')
    ax8.set_ylabel('BTC Holdings', color=colors[7])
    ax8_twin.set_ylabel('USD Value ($)', color=colors[1])
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax8.legend(lines, labels, loc='upper left')
    
    ax8.grid(True, alpha=0.3)
    ax8_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Overall title
    fig.suptitle('Bitcoin Collateral Lending Strategy - Comprehensive Analysis Dashboard', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    return fig

def generate_executive_summary_report(df: pd.DataFrame, start_btc: float, btc_goal: float) -> str:
    """Generate a detailed executive summary report."""
    
    if len(df) == 0:
        return "❌ No simulation data available for report generation."
    
    # Calculate key metrics
    final_btc = df.free_btc_after.iloc[-1] if len(df) > 0 else 0
    total_cycles = len(df)
    total_days = df.cycle_duration_days.sum()
    total_interest = df.total_interest.sum()
    total_loans = df.loan_amount.sum()
    avg_interest_rate = (total_interest / total_loans) * 100 if total_loans > 0 else 0
    
    deferred_cycles = len(df[df.payment_strategy == 'deferred'])
    monthly_cycles = len(df[df.payment_strategy == 'monthly_payments'])
    margin_calls = len(df[df.margin_call_occurred])
    liquidations = len(df[df.liquidation_occurred])
    
    success_rate = ((total_cycles - liquidations) / total_cycles) * 100 if total_cycles > 0 else 0
    btc_growth = ((final_btc - start_btc) / start_btc) * 100 if start_btc > 0 else 0
    
    # Estimate final USD value
    final_price = df.exit_price.iloc[-1] if len(df) > 0 else 0
    final_usd_value = final_btc * final_price
    
    report = f"""
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTIVE SUMMARY REPORT                     │
│            Bitcoin Collateral Lending Strategy Analysis         │
└─────────────────────────────────────────────────────────────────┘

📊 STRATEGY PERFORMANCE OVERVIEW
═══════════════════════════════════════════════════════════════════
🎯 Goal Achievement:     {'✅ SUCCESS' if final_btc >= btc_goal else '❌ INCOMPLETE'}
💰 Final BTC Holdings:   {final_btc:.4f} BTC ({btc_growth:+.1f}% growth)
💵 Estimated USD Value:  ${final_usd_value:,.0f}
⏱️  Time to Completion:   {total_days:.0f} days ({total_days/365:.1f} years)

📈 OPERATIONAL METRICS
═══════════════════════════════════════════════════════════════════
🔄 Total Loan Cycles:    {total_cycles}
💸 Total Interest Paid:  ${total_interest:,.0f}
📊 Average Interest Rate: {avg_interest_rate:.1f}% effective
🎯 Success Rate:         {success_rate:.1f}% (no liquidations)

💡 STRATEGY BREAKDOWN
═══════════════════════════════════════════════════════════════════
🔵 Deferred Interest:    {deferred_cycles} cycles ({deferred_cycles/total_cycles*100:.1f}%)
🔴 Monthly Payments:     {monthly_cycles} cycles ({monthly_cycles/total_cycles*100:.1f}%)

⚠️  RISK EVENTS SUMMARY
═══════════════════════════════════════════════════════════════════
🟡 Margin Calls:        {margin_calls} events
💥 Liquidations:        {liquidations} events
🛡️  Risk Management:     {'Effective' if liquidations == 0 else 'Needs Improvement'}

📊 FINANCIAL EFFICIENCY
═══════════════════════════════════════════════════════════════════
"""
    
    if total_cycles > 0:
        avg_loan_size = df.loan_amount.mean()
        avg_cycle_duration = df.cycle_duration_days.mean()
        avg_btc_gain = df.net_btc_gain.mean()
        
        report += f"""💰 Average Loan Size:    ${avg_loan_size:,.0f}
⏰ Average Cycle Length: {avg_cycle_duration:.0f} days
📈 Average BTC Gain:     {avg_btc_gain:.4f} BTC per cycle

🎯 STRATEGY ASSESSMENT
═══════════════════════════════════════════════════════════════════
"""
        
        if final_btc >= btc_goal:
            report += "✅ RECOMMENDATION: Strategy is VIABLE and achieved target goals.\n"
            report += f"   Successfully accumulated {final_btc:.4f} BTC in {total_days/365:.1f} years.\n"
        elif liquidations > 0:
            report += "❌ RECOMMENDATION: Strategy is HIGH RISK due to liquidation events.\n"
            report += "   Consider lower leverage or larger collateral buffers.\n"
        else:
            report += "⚠️  RECOMMENDATION: Strategy shows promise but needs optimization.\n"
            report += "   Consider extending time horizon or adjusting parameters.\n"
    
    report += f"""
📋 KEY INSIGHTS
═══════════════════════════════════════════════════════════════════
• Interest deferral was {'preferred' if deferred_cycles > monthly_cycles else 'less common'} strategy
• {'No liquidations occurred - good risk management' if liquidations == 0 else f'{liquidations} liquidation(s) - high risk strategy'}
• Average effective interest rate: {avg_interest_rate:.1f}%
• Strategy required {total_cycles} cycles over {total_days/365:.1f} years

⚠️  DISCLAIMERS
═══════════════════════════════════════════════════════════════════
• This is a simulation based on historical patterns and assumptions
• Actual results may vary significantly due to market volatility
• Bitcoin lending involves substantial risk of loss
• Consider consulting financial advisors before implementing

═══════════════════════════════════════════════════════════════════
Report generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report

def main():
    """Run the improved Bitcoin lending simulation with enhanced visualizations."""
    # Set up error logging first
    logger = setup_error_logging()
    
    try:
        logger.info("Starting Enhanced Bitcoin Collateral Lending Simulation")
        print("🚀 Starting Enhanced Bitcoin Collateral Lending Simulation")
        print("=" * 60)

        # Load price data
        prices = load_btc_history(logger)
        print(f"📊 Loaded {len(prices)} days of price data")
        logger.info(f"Loaded {len(prices)} days of price data")

    # Analyze historical drawdowns
        draw_df = worst_drop_until_recovery(prices, 30000.0, logger)
        print(f"📈 Analyzed {len(draw_df)} historical recovery cycles")
        logger.info(f"Analyzed {len(draw_df)} historical recovery cycles")

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
            
            try:
                logger.info(f"Starting cycle {cycle}")
                
                # Determine loan size for this cycle
                worst_drop = drawdown_model(current_price)
                worst_price = current_price * (1 - worst_drop)
                max_loan = collateral_btc * worst_price * 0.75
                loan_amount = min(max_loan, free_btc * current_price * 0.5)  # Conservative sizing
                loan_amount = max(loan_amount, simulator.min_loan)

                logger.info(f"Cycle {cycle} parameters: price=${current_price:,.0f}, loan=${loan_amount:,.0f}")

                # Simulate this cycle
                cycle_result = simulator.simulate_cycle(
                    current_price, collateral_btc, loan_amount, drawdown_model
                )

        # Check if liquidation occurred
                if cycle_result["liquidation_occurred"]:
                    logger.error(f"LIQUIDATION occurred in cycle {cycle}")
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

        logger.info(f"Cycle {cycle} completed successfully")
                print(f"📊 Cycle {cycle}: ${current_price:,.0f} → ${cycle_result['exit_price']:,.0f}, "
                      f"BTC: {free_btc:.4f}, Strategy: {cycle_result['payment_strategy']}")

                # Stop if we've reached our goal
                if free_btc >= btc_goal:
                    logger.info(f"Goal reached! Free BTC: {free_btc:.4f}")
                    print(f"🎯 Goal reached! Free BTC: {free_btc:.4f}")
                    break

                # Prepare for next cycle - reset collateral to safe level
                if free_btc > 0.1:  # Ensure we have enough for collateral
                    collateral_btc = min(free_btc / 2, 0.5)  # Conservative collateral management
                    free_btc -= collateral_btc
                    
            except Exception as e:
                log_error(logger, f"cycle_{cycle}_processing", e, {
                    'cycle': cycle,
                    'current_price': current_price,
                    'free_btc': free_btc,
                    'collateral_btc': collateral_btc,
                    'loan_amount': loan_amount if 'loan_amount' in locals() else "Unknown"
                })
                print(f"❌ Error in cycle {cycle}: {e}")
                break  # Stop simulation on cycle error

    # Generate enhanced outputs
        if results:
            try:
                logger.info("Starting output generation")
                df = pd.DataFrame(results)
                
                # Save detailed CSV
                df.to_csv("detailed_simulation_results.csv", index=False)
                logger.info("Saved detailed CSV results")
                
                # Create enhanced visualizations
                print("🎨 Generating enhanced visualizations...")
                fig = create_enhanced_visualizations(df, start_btc, btc_goal)
                fig.savefig("bitcoin_lending_dashboard.png", dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close(fig)
                logger.info("Generated visualization dashboard")
                
                # Generate executive summary
                print("📝 Generating executive summary report...")
                summary_report = generate_executive_summary_report(df, start_btc, btc_goal)
                
                # Save report to file
                with open("executive_summary_report.txt", "w") as f:
                    f.write(summary_report)
                logger.info("Generated executive summary report")
                
                # Print summary to console
                print(summary_report)
                
                print("\n📁 Enhanced outputs generated:")
                print("   • detailed_simulation_results.csv - Complete data export")
                print("   • bitcoin_lending_dashboard.png - Comprehensive visual dashboard")
                print("   • executive_summary_report.txt - Executive summary report")
                
            except Exception as e:
                log_error(logger, "output_generation", e, {
                    'results_count': len(results),
                    'dataframe_shape': df.shape if 'df' in locals() else "Not created"
                })
                print(f"❌ Error generating outputs: {e}")
            
        else:
            logger.warning("No cycles completed - strategy failed immediately")
            print("❌ No cycles completed - strategy failed immediately")
            
    except Exception as e:
        log_error(logger, "main_simulation", e, {
            'cycle': cycle if 'cycle' in locals() else "Not started",
            'prices_loaded': len(prices) if 'prices' in locals() else "Failed to load"
        })
        print(f"❌ Critical error in main simulation: {e}")
        
    finally:
        logger.info("Simulation completed - check logs for detailed error information")

if __name__ == "__main__":
    main()
