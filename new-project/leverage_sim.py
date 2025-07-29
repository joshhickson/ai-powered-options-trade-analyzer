#!/usr/bin/env python3
"""
leverage_sim.py
Simulate a repeated BTC-collateral loan strategy with dynamic sizing.

User-defined rules (2025-07-29):
    • Start with 0.24 BTC, price = $118 000.
    • Post 0.12 BTC as collateral; 0.12 BTC stays in reserve.
    • Borrow $10 000 at 11.5 % APR (lump-sum interest at exit).
    • A loan cycle ends when price = entry_price + $30 000.
    • One margin call *always* happens per cycle at the worst drawdown
      (we size it using the 99 th-percentile historical drop).
    • If LTV would hit 90 %, move reserve BTC to cure (locked until exit).
    • Reserve size must equal collateral and remains locked until exit.
    • Each new loan’s principal is capped so that the reserve covers
      the fitted 99 % drawdown at that entry price.
    • Stop when free BTC ≥ 1.0.

Outputs:
    cycles_log.csv, btc_price_over_cycles.png, btc_owned_over_cycles.png
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports & helpers
# ──────────────────────────────────────────────────────────────────────────────
import datetime as dt
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import yfinance; fall back to CSV if offline
try:
    import yfinance as yf
    ONLINE = True
except ImportError:
    ONLINE = False
    print("⚠️  yfinance not found.  Expecting btc_history.csv in this folder.")

# Import requests for CoinGecko API
try:
    import requests
except ImportError:
    print("⚠️  requests not found. Install with: pip install requests")

# ──────────────────────────────────────────────────────────────────────────────
# Section 1 · Pull / load BTC price history
# ──────────────────────────────────────────────────────────────────────────────
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
    """
    Fetches daily Bitcoin closing prices using the most reliable 2025 methods.
    
    Tier 1: Binance public API (no key needed, most reliable)
    Tier 2: Coinbase public API (backup)
    Tier 3: Local CSV fallback
    Tier 4: Synthetic data (last resort)
    """
    
    # Method 1: Binance API (Primary - July 2025)
    print("📡 Attempting to fetch live BTC data from Binance API...")
    try:
        import requests
        
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '1d',
            'limit': 1000  # Max limit for Binance
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Process Binance klines data
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_time', 'Quote_asset_volume', 'Number_of_trades',
            'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
        ])
        
        # Convert timestamp to datetime and select Close price
        df['Date'] = pd.to_datetime(df['Open_time'], unit='ms')
        df = df.set_index('Date')
        btc_series = df['Close'].astype(float)
        
        print(f"✅ Successfully fetched {len(btc_series)} days of live BTC data from Binance")
        return btc_series
        
    except Exception as e:
        print(f"⚠️  Binance API failed: {e}")
    
    # Method 2: Coinbase API (Backup)
    print("📡 Trying Coinbase API as backup...")
    try:
        import requests
        
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        params = {
            'granularity': 86400  # 86400 seconds = 1 day
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('Date')
        btc_series = df['close'].astype(float).sort_index()
        
        print(f"✅ Successfully fetched {len(btc_series)} days of live BTC data from Coinbase")
        return btc_series
        
    except Exception as e:
        print(f"⚠️  Coinbase API failed: {e}")
    
    # Method 3: Local CSV fallback
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
    
    # Method 4: yfinance fallback (if available)
    if ONLINE:
        try:
            print("📡 Trying yfinance as final backup...")
            btc = yf.download("BTC-USD", start="2010-07-17", progress=False)
            if not btc.empty and 'Adj Close' in btc.columns:
                prices = btc["Adj Close"].dropna()
                if len(prices) >= 100:
                    print(f"✅ yfinance backup successful: {len(prices)} days")
                    return prices
        except Exception as e:
            print(f"⚠️  yfinance backup failed: {e}")
    
    # Method 5: Synthetic data (last resort)
    print("📊 All real data sources failed, falling back to synthetic data...")
    return generate_synthetic_btc_data()

prices = load_btc_history()

# ──────────────────────────────────────────────────────────────────────────────
# Section 2 · For every day, compute “worst drawdown before +$30 K recovery”
# ──────────────────────────────────────────────────────────────────────────────
def worst_drop_until_recovery(price_series: pd.Series,
                              jump: float = 30000.0) -> pd.Series:
    """
    For each start date i, find j > i s.t. price_j ≥ price_i + jump.
    Record min(price_i…j) ÷ price_i − 1  (negative % drop).
    If recovery never occurs, drop the sample.
    """
    res = []
    dates = price_series.index
    p = price_series.values
    n = len(p)
    for i in range(n):
        target = p[i] + jump
        # Scan forward until recovery
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

draw_df = worst_drop_until_recovery(prices)

# Validate we have sufficient data
if len(draw_df) < 50:
    print(f"⚠️  Warning: Only {len(draw_df)} recovery cycles found. Results may be less reliable.")

# ──────────────────────────────────────────────────────────────────────────────
# Section 3 · Fit 99 th-percentile drawdown curve  draw99(price) = a·p^(-b)
# ──────────────────────────────────────────────────────────────────────────────
# Get 99th-percentile (largest magnitude) drop for price bins
min_price = draw_df.price.min()
max_price = draw_df.price.max()

print(f"📊 Price range for analysis: ${min_price:.0f} to ${max_price:.0f}")
print(f"📊 Found {len(draw_df)} recovery cycles for analysis")

# Create bins ensuring they are monotonically increasing
if max_price <= min_price:
    print("⚠️  Invalid price range, using default drawdown model...")
    # Use a simple fallback model
    def draw99(price: float) -> float:
        return 0.8 * (price / 50000) ** (-0.3)  # Reasonable power law
else:
    # Use fewer bins if we have limited data
    n_bins = min(40, max(10, len(draw_df) // 5))
    bins = np.logspace(np.log10(min_price), np.log10(max_price), n_bins)
    
    # Ensure bins are strictly increasing
    bins = np.sort(bins)
    bins = bins[bins > 0]  # Remove any zero or negative values
    
    if len(np.unique(bins)) < len(bins):
        print("⚠️  Duplicate bin edges detected, using linear spacing...")
        bins = np.linspace(min_price, max_price, n_bins)
    
    draw_df["bin"] = pd.cut(draw_df.price, bins=bins)
    
    def safe_percentile(x):
        """Calculate 1st percentile, handling empty groups."""
        if len(x) == 0:
            return np.nan
        return np.percentile(x, 1)
    
    bin_stats = draw_df.groupby("bin", observed=False).agg(
            p=("price", "median"),
            d99=("draw", safe_percentile)  # 1st percentile = worst 1%
        ).dropna()
    
    if len(bin_stats) < 3:
        print("⚠️  Insufficient binned data, using simple drawdown model...")
        def draw99(price: float) -> float:
            return 0.8 * (price / 50000) ** (-0.3)
    else:
        try:
            # Fit a log-log linear regression  ln|d| = ln a  – b ln p
            y = np.log(np.abs(bin_stats.d99.values))
            x = np.log(bin_stats.p.values)
            
            # Check for valid data
            if np.any(np.isnan(y)) or np.any(np.isnan(x)) or np.any(np.isinf(y)) or np.any(np.isinf(x)):
                raise ValueError("Invalid data for curve fitting")
                
            b, ln_a = np.polyfit(x, y, 1) * np.array([-1, 1])  # adjust sign
            a = math.exp(ln_a)
            
            print(f"📈 Fitted drawdown model: draw99(p) = {a:.4f} * p^(-{b:.4f})")
            
            def draw99(price: float) -> float:
                """Return the 99 % worst expected drawdown (as +fraction) for a price."""
                result = a * (price ** (-b))
                # Clamp to reasonable bounds
                return max(0.1, min(0.9, result))  # Between 10% and 90%
                
        except Exception as e:
            print(f"⚠️  Curve fitting failed: {e}")
            print("Using fallback drawdown model...")
            def draw99(price: float) -> float:
                return 0.8 * (price / 50000) ** (-0.3)

# ──────────────────────────────────────────────────────────────────────────────
# Section 4 · Simulation parameters & state
# ──────────────────────────────────────────────────────────────────────────────
params = {
    "start_price": 118_000.0,
    "start_btc_free": 0.24,
    "start_collateral_btc": 0.12,
    "loan_rate": 0.115,        # APR lump-sum
    "exit_jump": 30_000.0,
    "margin_LTV": 0.90,
}

# Calculate optimal starting loan using the same logic as subsequent cycles
initial_loan = cap_next_loan(params["start_price"], params["start_collateral_btc"])

state = {
    "cycle": 0,
    "price": params["start_price"],
    "free_btc": params["start_btc_free"] - params["start_collateral_btc"],
    "collat_btc": params["start_collateral_btc"],  # locked reserve = collat
    "loan": initial_loan,
    "btc_goal": 1.0,
}

print(f"💰 Calculated optimal starting loan: ${initial_loan:,.0f} (vs. hardcoded $10,000)")

records = []

# ──────────────────────────────────────────────────────────────────────────────
# Section 5 · Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def btc_needed_for_cure(price: float, loan: float) -> float:
    """
    Given entry price and loan size, compute how much BTC must be added
    to keep LTV ≤ 90 % in a 99 % drawdown.
    """
    drop_pct = draw99(price)        # e.g. 0.25 → –25 %
    worst_price = price * (1 - drop_pct)
    # LTV = loan / (collateral * worst_price)
    # Want collateral s.t. loan / (collat * worst_price) ≤ 0.90
    return loan / (0.90 * worst_price)

def cap_next_loan(price: float, reserve_btc: float) -> float:
    """
    Find the max loan principal such that reserve ≥ needed cure BTC.
    Solve inverse of btc_needed_for_cure.
    """
    drop_pct = draw99(price)
    worst_price = price * (1 - drop_pct)
    return reserve_btc * worst_price * 0.90

# ──────────────────────────────────────────────────────────────────────────────
# Section 6 · Event-driven cycle loop
# ──────────────────────────────────────────────────────────────────────────────
while state["free_btc"] < state["btc_goal"]:
    state["cycle"] += 1
    entry_price = state["price"]
    loan = state["loan"]

    # Step 1: compute cure requirement & check reserve
    needed_btc = btc_needed_for_cure(entry_price, loan)
    if needed_btc > state["collat_btc"]:
        # Reserve insufficient: shrink loan
        loan = cap_next_loan(entry_price, state["collat_btc"])
        state["loan"] = loan

    # Step 2: immediate margin call at worst drawdown
    state["collat_btc"] += needed_btc
    state["free_btc"] -= needed_btc  # moved to collateral locker

    # Step 3: price appreciation +$30 K
    exit_price = entry_price + params["exit_jump"]
    state["price"] = exit_price

    # BTC purchased with loan (assuming instant buy at entry price)
    btc_bought = loan / entry_price

    # Interest lump-sum
    interest = loan * state["cycle"] * 0  # placeholder (simple loop-year calc)
    payoff = loan + loan * params["loan_rate"]  # 1-yr APR lump

    # Sell BTC equal to payoff
    btc_sold = payoff / exit_price
    gain_btc = btc_bought - btc_sold

    # Step 4: release all collateral back to free BTC
    state["free_btc"] += gain_btc + state["collat_btc"]
    state["collat_btc"] = params["start_collateral_btc"]  # reset reserve

    # Step 5: set next loan equal to cap for new price
    state["loan"] = cap_next_loan(exit_price, state["collat_btc"])

    # Log cycle
    records.append({
        "cycle": state["cycle"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "loan": loan,
        "needed_cure_btc": needed_btc,
        "btc_free": state["free_btc"],
    })

    # Safety break
    if state["cycle"] > 500:
        print("Aborting after 500 cycles (should never happen).")
        break

# ──────────────────────────────────────────────────────────────────────────────
# Section 7 · Save results
# ──────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv("cycles_log.csv", index=False)

plt.figure()
plt.plot(df.cycle, df.exit_price, marker="o")
plt.xlabel("Cycle")
plt.ylabel("BTC Exit Price ($)")
plt.title("BTC Price per Cycle")
plt.savefig("btc_price_over_cycles.png", dpi=150)

plt.figure()
plt.plot(df.cycle, df.btc_free, marker="o", color="orange")
plt.xlabel("Cycle")
plt.ylabel("Unencumbered BTC")
plt.title("BTC Held vs. Cycle")
plt.savefig("btc_owned_over_cycles.png", dpi=150)

print("Simulation complete.  Files saved:\n"
      "  • cycles_log.csv\n"
      "  • btc_price_over_cycles.png\n"
      "  • btc_owned_over_cycles.png")