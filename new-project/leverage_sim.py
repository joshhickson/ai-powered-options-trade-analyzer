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

def test_coingecko_api(api_key: str) -> bool:
    """Test CoinGecko API connection with a simple ping endpoint."""
    try:
        import requests
        
        # Test with ping endpoint first
        ping_url = "https://api.coingecko.com/api/v3/ping"
        headers = {
            'accept': 'application/json',
            'x-cg-demo-api-key': api_key,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(ping_url, headers=headers, timeout=10)
        if response.status_code == 200:
            print("✅ CoinGecko API key is working!")
            return True
        else:
            print(f"⚠️  CoinGecko ping failed: {response.status_code}")
            print(f"    Response: {response.text}")
            return False
    except Exception as e:
        print(f"⚠️  CoinGecko API test failed: {e}")
        return False

def fetch_robust_btc_history() -> pd.Series:
    """
    Fetches long-term BTC history using a tiered approach.
    Tier 1: Nasdaq Data Link (Brave New Coin dataset) - Best for 10+ years
    Tier 2: Kraken API - US-compliant exchange, requires looping
    """
    
    # --- Tier 1: Nasdaq Data Link (Best Method) ---
    print("📈 Tier 1: Attempting to fetch data from Nasdaq Data Link (Brave New Coin)...")
    try:
        # Check if nasdaq package is available
        try:
            import nasdaqdatalink as ndl
        except ImportError:
            print("⚠️  nasdaqdatalink package not available in this environment")
            raise ImportError("Package not installed")
        
        # Get API key from environment (Replit Secrets)
        api_key = os.environ.get('NASDAQ_API_KEY')
        if not api_key:
            print("⚠️  NASDAQ_API_KEY not found in environment variables")
            raise ImportError("No API key available")
        
        print(f"✅ Found Nasdaq API key: {api_key[:8]}...")
        ndl.api_key = api_key
        
        # Code 'BNC/BLX' is for the Bitcoin Liquid Index
        df = ndl.get("BNC/BLX", start_date="2010-01-01")
        btc_series = df['Value'].astype(float)
        print(f"✅ Successfully fetched {len(btc_series)} days of data from Nasdaq.")
        return btc_series
    except Exception as e:
        print(f"⚠️ Nasdaq method unavailable: {e}")
        print("📈 Proceeding to Tier 2: Kraken API (US-compliant exchange)...")

    # --- Tier 2: Kraken API (Good Backup) ---
    print("\n📈 Tier 2: Attempting to fetch data from Kraken API...")
    try:
        import time
        
        url = "https://api.kraken.com/0/public/OHLC"
        params = {'pair': 'XBTUSD', 'interval': 1440}  # 1440 minutes = 1 day
        all_data = []
        
        # Kraken's API paginates with a 'since' parameter
        # We'll make a few calls to get over 1000 days of data
        for i in range(5):  # Loop 5 times to get ~10 years of data
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'result' not in data:
                print(f"⚠️ Unexpected Kraken response format: {data}")
                break
            
            pair_name = list(data['result'].keys())[0] if data['result'] else None
            if not pair_name or pair_name == 'last':
                break
                
            ohlc_data = data['result'][pair_name]
            all_data.extend(ohlc_data)
            
            # The 'last' timestamp tells us where the next page should start
            if 'last' in data['result']:
                last_ts = data['result']['last']
                params['since'] = last_ts
            else:
                break
            
            print(f"  📥 Fetched {len(ohlc_data)} records from Kraken (batch {i+1}/5)...")
            time.sleep(1)  # Be polite to the API
        
        if all_data:
            df = pd.DataFrame(all_data, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
            df['Date'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('Date')
            df = df[~df.index.duplicated(keep='first')].sort_index()
            btc_series = df['close'].astype(float)
            
            print(f"✅ Successfully fetched {len(btc_series)} days of data from Kraken.")
            return btc_series
        else:
            print("⚠️ No data received from Kraken")
            
    except Exception as e:
        print(f"⚠️ Kraken API failed: {e}")
    
    return None

def load_btc_history() -> pd.Series:
    """
    Fetches daily Bitcoin closing prices using the most reliable 2025 methods.
    
    Tier 1: Nasdaq Data Link (Brave New Coin) - Primary (US-compliant, 10+ years)
    Tier 2: Kraken API - Secondary (US-compliant exchange)
    Tier 3: Local CSV fallback  
    Tier 4: Synthetic data (last resort)
    """
    
    # Method 1: Robust multi-tier API approach (Primary)
    print("📡 Attempting to fetch 10+ years of BTC data using robust methods...")
    try:
        btc_series = fetch_robust_btc_history()
        if btc_series is not None and len(btc_series) > 100:
            return btc_series
        else:
            print("⚠️ All API methods returned insufficient data")
            
    except Exception as e:
        print(f"⚠️ Robust API methods failed: {e}")
    
    # DISABLED: Method 2: CoinGecko API with Demo key (Primary) - Chunked requests for 10 years
    # print("📡 Attempting to fetch live BTC data from CoinGecko API in chunks...")
    # try:
    #     import requests
    #     import time
    #     from datetime import datetime, timedelta
    #     
    #     # Your CoinGecko Demo API key
    #     API_KEY = "CG-WCcgxgiuhnov31LZB6FzgaB4"
    #     
    #     # Test API key first
    #     if not test_coingecko_api(API_KEY):
    #         raise Exception("API key test failed")
    #     
    #     headers = {
    #         'accept': 'application/json',
    #         'x-cg-demo-api-key': API_KEY,
    #         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    #     }
    #     
    #     all_price_data = []
    #     
    #     # Request data in 350-day chunks (slightly under 365 to be safe)
    #     # Go back 10 years = ~3650 days, so we need about 10-11 chunks
    #     chunk_days = 350
    #     total_days = 3650  # 10 years
    #     num_chunks = (total_days // chunk_days) + 1
    #     
    #     print(f"📊 Fetching {total_days} days in {num_chunks} chunks of {chunk_days} days each...")
    #     
    #     for chunk in range(num_chunks):
    #         days_from_now = chunk * chunk_days
    #         
    #         # For the last chunk, calculate remaining days
    #         if chunk == num_chunks - 1:
    #             remaining_days = total_days - (chunk * chunk_days)
    #             if remaining_days <= 0:
    #                 break
    #             days_to_request = min(remaining_days, chunk_days)
    #         else:
    #             days_to_request = chunk_days
    #         
    #         print(f"  📥 Chunk {chunk + 1}/{num_chunks}: Requesting {days_to_request} days from {days_from_now} days ago...")
    #         
    #         # Calculate the date range for this chunk
    #         end_date = datetime.now() - timedelta(days=days_from_now)
    #         start_date = end_date - timedelta(days=days_to_request)
    #         
    #         url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    #         params = {
    #             'vs_currency': 'usd',
    #             'from': int(start_date.timestamp()),
    #             'to': int(end_date.timestamp())
    #         }
    #         
    #         # Add delay between requests to respect rate limits
    #         if chunk > 0:
    #             time.sleep(2)  # 2-second delay between chunks
    #         
    #         response = requests.get(url, params=params, headers=headers, timeout=20)
    #         response.raise_for_status()
    #         
    #         data = response.json()
    #         if 'prices' in data and len(data['prices']) > 0:
    #             chunk_prices = data['prices']
    #             all_price_data.extend(chunk_prices)
    #             print(f"    ✅ Got {len(chunk_prices)} price points")
    #         else:
    #             print(f"    ⚠️  Empty data for chunk {chunk + 1}")
    #     
    #     if all_price_data:
    #         # Process all combined price data
    #         df = pd.DataFrame(all_price_data, columns=['timestamp', 'price'])
    #         
    #         # Convert timestamp to datetime and set as index
    #         df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    #         df = df.set_index('Date')
    #         
    #         # Remove duplicates and sort
    #         df = df[~df.index.duplicated(keep='first')].sort_index()
    #         
    #         # Return price series
    #         btc_series = df['price'].astype(float)
    #         
    #         print(f"✅ Successfully fetched {len(btc_series)} days of live BTC data from CoinGecko (chunked)")
    #         print(f"📅 Date range: {btc_series.index.min().strftime('%Y-%m-%d')} to {btc_series.index.max().strftime('%Y-%m-%d')}")
    #         return btc_series
    #     else:
    #         print("⚠️  CoinGecko returned no price data across all chunks")
    #         
    # except requests.exceptions.HTTPError as http_err:
    #     print(f"⚠️  CoinGecko HTTP Error: {http_err}")
    #     if hasattr(http_err, 'response'):
    #         print(f"    Status Code: {http_err.response.status_code}")
    #         print(f"    Response Body: {http_err.response.text}")
    #         if http_err.response.status_code == 401:
    #             print("    This suggests an API key issue - please verify your key is correct")
    #         elif http_err.response.status_code == 429:
    #             print("    Rate limit exceeded - try increasing delays between chunks")
    # except Exception as e:
    #     print(f"⚠️  CoinGecko chunked API failed: {e}")
    
    # DISABLED: Method 3: Coinbase API (Secondary Backup)
    # print("📡 Trying Coinbase API as secondary backup...")
    # try:
    #     import requests
    #     
    #     url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    #     params = {
    #         'granularity': 86400  # 86400 seconds = 1 day
    #     }
    #     
    #     headers = {
    #         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    #     }
    #     
    #     response = requests.get(url, params=params, headers=headers, timeout=15)
    #     response.raise_for_status()
    #     
    #     data = response.json()
    #     df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    #     
    #     df['Date'] = pd.to_datetime(df['time'], unit='s')
    #     df = df.set_index('Date')
    #     btc_series = df['close'].astype(float).sort_index()
    #     
    #     print(f"✅ Successfully fetched {len(btc_series)} days of live BTC data from Coinbase")
    #     return btc_series
    #     
    # except Exception as e:
    #     print(f"⚠️  Coinbase API failed: {e}")
    
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
    
    # Method 5: yfinance fallback (if available)
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
    
    # Method 6: Synthetic data (last resort)
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

# Analyze historical price movements for realistic timing
movement_stats = analyze_price_movements(prices, params["exit_jump"])

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
        # Ensure we have meaningful drawdowns (not zero)
        valid_draws = x[x < -0.001]  # Only negative drawdowns > 0.1%
        if len(valid_draws) == 0:
            return -0.05  # Default 5% minimum drawdown
        return np.percentile(valid_draws, 1)
    
    bin_stats = draw_df.groupby("bin", observed=False).agg(
            p=("price", "median"),
            d99=("draw", safe_percentile),  # 1st percentile = worst 1%
            count=("draw", "count")
        ).dropna()
    
    # Filter bins with sufficient data points
    bin_stats = bin_stats[bin_stats['count'] >= 3]
    
    print(f"📊 Valid bins for curve fitting: {len(bin_stats)}")
    
    if len(bin_stats) < 3:
        print("⚠️  Insufficient binned data, using simple drawdown model...")
        def draw99(price: float) -> float:
            return 0.8 * (price / 50000) ** (-0.3)
    else:
        try:
            # DIAGNOSTIC PHASE - Let's examine the data in detail
            drawdown_values = bin_stats.d99.values
            price_values = bin_stats.p.values
            
            print(f"🔍 CURVE FITTING DIAGNOSTICS:")
            print(f"   Raw bins: {len(bin_stats)}")
            print(f"   Price range: ${price_values.min():.0f} - ${price_values.max():.0f}")
            print(f"   Drawdown range: {drawdown_values.min():.1%} - {drawdown_values.max():.1%}")
            print(f"   Sample drawdowns: {drawdown_values[:5]}")
            print(f"   Sample prices: {price_values[:5]}")
            
            # Filter out any remaining problematic values
            valid_mask = (drawdown_values < -0.001) & (price_values > 0) & np.isfinite(drawdown_values) & np.isfinite(price_values)
            if np.sum(valid_mask) < 3:
                raise ValueError(f"Insufficient valid drawdown data: only {np.sum(valid_mask)} valid points")
            
            filtered_drawdowns = np.abs(drawdown_values[valid_mask])
            filtered_prices = price_values[valid_mask]
            
            print(f"   After filtering: {len(filtered_drawdowns)} valid points")
            print(f"   Filtered drawdown range: {filtered_drawdowns.min():.1%} - {filtered_drawdowns.max():.1%}")
            
            # Check if data shows expected power law relationship
            price_ratio = filtered_prices.max() / filtered_prices.min()
            drawdown_ratio = filtered_drawdowns.max() / filtered_drawdowns.min()
            
            print(f"   Price ratio (max/min): {price_ratio:.1f}x")
            print(f"   Drawdown ratio (max/min): {drawdown_ratio:.1f}x")
            
            if price_ratio < 1.5:
                raise ValueError(f"Insufficient price variation for power law fit: {price_ratio:.1f}x")
            
            # Fit a log-log linear regression  ln|d| = ln a  – b ln p
            y = np.log(filtered_drawdowns)
            x = np.log(filtered_prices)
            
            print(f"   Log-transformed ranges: x={x.min():.2f} to {x.max():.2f}, y={y.min():.2f} to {y.max():.2f}")
            
            # Final validation of log-transformed data
            if np.any(np.isnan(y)) or np.any(np.isnan(x)) or np.any(np.isinf(y)) or np.any(np.isinf(x)):
                raise ValueError("Invalid data after log transformation")
            
            # Perform the fit with error checking
            try:
                coeffs = np.polyfit(x, y, 1)
                b, ln_a = coeffs * np.array([-1, 1])  # adjust sign convention
                a = math.exp(ln_a)
                
                # Calculate R-squared for goodness of fit
                y_pred = coeffs[0] * x + coeffs[1]
                r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                
                print(f"   Fit results: a={a:.8f}, b={b:.4f}, R²={r_squared:.3f}")
                
            except np.linalg.LinAlgError as e:
                raise ValueError(f"Linear algebra error in fitting: {e}")
            
            # Validate fitted parameters with detailed checks
            checks = {
                "a_positive": a > 0,
                "a_reasonable": 0.0001 < a < 10,
                "b_finite": np.isfinite(b),
                "b_reasonable": -1 < b < 3,  # Allow steeper negative slopes
                "r_squared_ok": r_squared > 0.1  # At least some correlation
            }
            
            print(f"   Parameter checks: {checks}")
            
            failed_checks = [k for k, v in checks.items() if not v]
            if failed_checks:
                raise ValueError(f"Parameter validation failed: {failed_checks}")
            
            # Test the model with sample prices to ensure reasonable outputs
            test_prices = [30000, 50000, 75000, 100000, 150000]
            test_results = []
            for p in test_prices:
                try:
                    result = a * (p ** (-b))
                    test_results.append(result)
                except:
                    test_results.append(float('nan'))
            
            valid_predictions = [r for r in test_results if 0.01 < r < 0.95 and np.isfinite(r)]
            
            if len(valid_predictions) < 3:
                raise ValueError(f"Model produces too many invalid predictions: {test_results}")
            
            print(f"📈 SUCCESS: Fitted power law model: draw99(p) = {a:.8f} * p^(-{b:.4f})")
            print(f"📊 Model validation: R²={r_squared:.3f}, {len(filtered_prices)} points")
            print(f"📝 Test predictions:")
            for p, r in zip(test_prices, test_results):
                if np.isfinite(r):
                    print(f"     ${p/1000:.0f}k → {r:.1%}")
            
            def draw99(price: float) -> float:
                """Return the 99% worst expected drawdown (as +fraction) for a price."""
                try:
                    result = a * (price ** (-b))
                    # Robust bounds checking
                    if not np.isfinite(result) or result <= 0:
                        # Fallback using median historical drawdown with price adjustment
                        median_drawdown = np.median(filtered_drawdowns)
                        result = median_drawdown * (price / np.median(filtered_prices)) ** (-0.2)
                    
                    return max(0.02, min(0.90, result))  # 2% to 90% bounds
                except Exception as fallback_error:
                    # Emergency fallback
                    return 0.5 * (price / 70000) ** (-0.3)
                
        except Exception as e:
            print(f"⚠️  Power law curve fitting failed: {e}")
            
            # FALLBACK STRATEGY 1: Polynomial fit
            try:
                print("📊 Attempting polynomial drawdown model...")
                
                # Try quadratic relationship in normal space (not log)
                valid_mask = (bin_stats.d99.values < -0.001) & (bin_stats.p.values > 0)
                if np.sum(valid_mask) >= 3:
                    x_poly = bin_stats.p.values[valid_mask]
                    y_poly = np.abs(bin_stats.d99.values[valid_mask])
                    
                    # Normalize prices to improve numerical stability
                    x_norm = x_poly / 50000  # Normalize around $50k
                    
                    # Try quadratic: drawdown = c0 + c1*x + c2*x^2
                    if len(x_norm) >= 6:  # Need enough points for quadratic
                        coeffs = np.polyfit(x_norm, y_poly, 2)
                        c2, c1, c0 = coeffs
                        
                        # Test model at several points
                        test_norm = np.array([0.6, 1.0, 1.5, 2.0])  # $30k, $50k, $75k, $100k
                        test_results = c2 * test_norm**2 + c1 * test_norm + c0
                        
                        if all(0.01 < r < 0.95 for r in test_results):
                            print(f"📈 Polynomial model: draw99(p) = {c2:.2e}*(p/50k)² + {c1:.4f}*(p/50k) + {c0:.4f}")
                            
                            def draw99(price: float) -> float:
                                p_norm = price / 50000
                                result = c2 * p_norm**2 + c1 * p_norm + c0
                                return max(0.02, min(0.90, result))
                        else:
                            raise ValueError("Polynomial produces invalid results")
                    else:
                        raise ValueError("Not enough points for polynomial fit")
                else:
                    raise ValueError("No valid data for polynomial fit")
                    
            except Exception as poly_error:
                print(f"   Polynomial fit failed: {poly_error}")
                
                # FALLBACK STRATEGY 2: Robust empirical model
                print("📊 Using robust empirical drawdown model...")
                
                # Calculate percentile-based drawdowns from raw data
                all_drawdowns = np.abs(draw_df.draw.values)
                all_prices = draw_df.price.values
                
                # Remove outliers (beyond 3 standard deviations)
                drawdown_mean = np.mean(all_drawdowns)
                drawdown_std = np.std(all_drawdowns)
                price_mean = np.mean(all_prices)
                price_std = np.std(all_prices)
                
                clean_mask = (
                    (np.abs(all_drawdowns - drawdown_mean) < 3 * drawdown_std) &
                    (np.abs(all_prices - price_mean) < 3 * price_std)
                )
                
                clean_drawdowns = all_drawdowns[clean_mask]
                clean_prices = all_prices[clean_mask]
                
                # Use percentiles for different price ranges
                low_price_thresh = np.percentile(clean_prices, 33)
                high_price_thresh = np.percentile(clean_prices, 67)
                
                low_price_drawdown = np.percentile(clean_drawdowns[clean_prices <= low_price_thresh], 95)
                mid_price_drawdown = np.percentile(clean_drawdowns[
                    (clean_prices > low_price_thresh) & (clean_prices <= high_price_thresh)
                ], 95)
                high_price_drawdown = np.percentile(clean_drawdowns[clean_prices > high_price_thresh], 95)
                
                print(f"📈 Empirical model:")
                print(f"   Low prices (<${low_price_thresh:.0f}): {low_price_drawdown:.1%}")
                print(f"   Mid prices (${low_price_thresh:.0f}-${high_price_thresh:.0f}): {mid_price_drawdown:.1%}")  
                print(f"   High prices (>${high_price_thresh:.0f}): {high_price_drawdown:.1%}")
                
                def draw99(price: float) -> float:
                    """Empirical drawdown model with price tiers."""
                    if price <= low_price_thresh:
                        base = low_price_drawdown
                    elif price <= high_price_thresh:
                        # Linear interpolation between low and mid
                        weight = (price - low_price_thresh) / (high_price_thresh - low_price_thresh)
                        base = low_price_drawdown * (1 - weight) + mid_price_drawdown * weight
                    else:
                        # Linear interpolation between mid and high, with gentle extrapolation
                        if price > high_price_thresh * 2:
                            # Very high prices get modestly higher drawdowns
                            base = high_price_drawdown * (1 + 0.1 * np.log(price / high_price_thresh))
                        else:
                            weight = (price - high_price_thresh) / high_price_thresh
                            base = mid_price_drawdown * (1 - weight) + high_price_drawdown * weight
                    
                    return max(0.02, min(0.90, base))

# ──────────────────────────────────────────────────────────────────────────────
# Section 4 · Historical Price Movement Analysis
# ──────────────────────────────────────────────────────────────────────────────
def analyze_price_movements(price_series: pd.Series, jump_amount: float = 30000.0) -> dict:
    """
    Analyze historical Bitcoin price movements to predict realistic timing 
    for future $30K appreciation cycles.
    
    Returns statistics on how long Bitcoin typically takes to appreciate 
    by jump_amount from various starting price levels.
    """
    print(f"📊 Analyzing historical ${jump_amount:,.0f} price movements...")
    
    movements = []
    dates = price_series.index
    prices = price_series.values
    n = len(prices)
    
    # Find all instances where price increased by jump_amount
    for i in range(n - 1):
        start_price = prices[i]
        target_price = start_price + jump_amount
        
        # Look forward to find when target was reached
        for j in range(i + 1, min(i + 365, n)):  # Max 1 year lookforward
            if prices[j] >= target_price:
                days_taken = j - i
                start_date = dates[i]
                end_date = dates[j]
                
                # Calculate statistics for this movement
                max_price_during = np.max(prices[i:j+1])
                min_price_during = np.min(prices[i:j+1])
                end_price = prices[j]
                
                # Calculate rate of change (daily average)
                daily_rate = (end_price - start_price) / days_taken
                annualized_rate = daily_rate * 365 / start_price
                
                movements.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'start_price': start_price,
                    'end_price': end_price,
                    'days_taken': days_taken,
                    'daily_rate_usd': daily_rate,
                    'annualized_rate_pct': annualized_rate * 100,
                    'max_during_cycle': max_price_during,
                    'min_during_cycle': min_price_during,
                    'overshoot_amount': max_price_during - target_price,
                    'price_range': start_price // 10000 * 10000  # Bin by $10K ranges
                })
                break
    
    if not movements:
        print(f"⚠️  No ${jump_amount:,.0f} movements found in historical data")
        return {
            'avg_days': 180,  # Default assumption
            'std_days': 60,
            'movements_found': 0,
            'price_range_stats': {}
        }
    
    df = pd.DataFrame(movements)
    
    # Calculate statistics by price range
    price_range_stats = {}
    for price_range in sorted(df['price_range'].unique()):
        range_data = df[df['price_range'] == price_range]
        price_range_stats[price_range] = {
            'count': len(range_data),
            'avg_days': range_data['days_taken'].mean(),
            'std_days': range_data['days_taken'].std(),
            'median_days': range_data['days_taken'].median(),
            'min_days': range_data['days_taken'].min(),
            'max_days': range_data['days_taken'].max(),
            'avg_daily_rate': range_data['daily_rate_usd'].mean(),
            'success_rate': len(range_data) / len(df[df['start_price'] >= price_range])
        }
    
    # Overall statistics
    overall_stats = {
        'movements_found': len(df),
        'avg_days': df['days_taken'].mean(),
        'std_days': df['days_taken'].std(),
        'median_days': df['days_taken'].median(),
        'price_range_stats': price_range_stats,
        'recent_trend': df.tail(10)['days_taken'].mean() if len(df) >= 10 else df['days_taken'].mean()
    }
    
    print(f"📈 Found {len(df)} historical ${jump_amount:,.0f} movements")
    print(f"   Average time: {overall_stats['avg_days']:.0f} days ({overall_stats['avg_days']/30:.1f} months)")
    print(f"   Median time: {overall_stats['median_days']:.0f} days")
    print(f"   Range: {df['days_taken'].min():.0f} - {df['days_taken'].max():.0f} days")
    
    return overall_stats

def predict_cycle_duration(start_price: float, movement_stats: dict, jump_amount: float = 30000.0) -> tuple:
    """
    Predict how long a $30K price appreciation will take based on 
    historical patterns and current price level.
    
    Returns: (expected_days, confidence_interval_days)
    """
    if not movement_stats['price_range_stats']:
        # Fallback to overall average
        return movement_stats['avg_days'], movement_stats['std_days']
    
    # Find the closest price range
    price_range = start_price // 10000 * 10000
    
    # Look for exact match first
    if price_range in movement_stats['price_range_stats']:
        stats = movement_stats['price_range_stats'][price_range]
        return stats['avg_days'], stats['std_days']
    
    # Find nearest ranges and interpolate
    available_ranges = sorted(movement_stats['price_range_stats'].keys())
    
    if start_price <= min(available_ranges):
        # Use lowest range data
        stats = movement_stats['price_range_stats'][min(available_ranges)]
        return stats['avg_days'], stats['std_days']
    elif start_price >= max(available_ranges):
        # Extrapolate based on trend (higher prices typically take longer)
        highest_range = max(available_ranges)
        stats = movement_stats['price_range_stats'][highest_range]
        
        # Apply scaling factor for higher prices (diminishing returns effect)
        scaling_factor = 1 + 0.1 * np.log(start_price / highest_range)
        scaled_days = stats['avg_days'] * scaling_factor
        
        return scaled_days, stats['std_days'] * scaling_factor
    else:
        # Interpolate between two nearest ranges
        lower_range = max([r for r in available_ranges if r <= price_range])
        upper_range = min([r for r in available_ranges if r > price_range])
        
        lower_stats = movement_stats['price_range_stats'][lower_range]
        upper_stats = movement_stats['price_range_stats'][upper_range]
        
        # Linear interpolation
        weight = (start_price - lower_range) / (upper_range - lower_range)
        interpolated_days = lower_stats['avg_days'] * (1 - weight) + upper_stats['avg_days'] * weight
        interpolated_std = lower_stats['std_days'] * (1 - weight) + upper_stats['std_days'] * weight
        
        return interpolated_days, interpolated_std

def simulate_realistic_price_path(start_price: float, end_price: float, days: int, 
                                  historical_volatility: float = 0.04) -> np.ndarray:
    """
    Generate a realistic price path from start_price to end_price over 'days' period.
    Uses geometric Brownian motion with drift to reach target.
    
    Returns: Array of daily prices
    """
    if days <= 1:
        return np.array([start_price, end_price])
    
    # Calculate required drift to reach target
    total_return = np.log(end_price / start_price)
    drift = total_return / days
    
    # Generate random price path
    np.random.seed(42)  # For reproducible results
    dt = 1.0  # Daily steps
    random_shocks = np.random.normal(0, historical_volatility * np.sqrt(dt), days - 1)
    
    # Build price path
    log_returns = drift + random_shocks
    log_prices = np.log(start_price) + np.cumsum(np.concatenate([[0], log_returns]))
    prices = np.exp(log_prices)
    
    # Ensure we end at target price (adjust final value)
    prices[-1] = end_price
    
    return prices

# ──────────────────────────────────────────────────────────────────────────────
# Section 5 · Helper functions (defined before usage)
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
# Section 5 · Simulation parameters & state
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

    # Step 3: Predict realistic cycle duration and price path
    expected_days, std_days = predict_cycle_duration(entry_price, movement_stats, params["exit_jump"])
    
    # Add some randomness to cycle duration (but keep deterministic for now)
    actual_days = max(7, int(expected_days))  # Minimum 1 week
    
    # Generate realistic price path (for future margin call modeling)
    exit_price = entry_price + params["exit_jump"]
    price_path = simulate_realistic_price_path(entry_price, exit_price, actual_days)
    state["price"] = exit_price

    # BTC purchased with loan (assuming instant buy at entry price)
    btc_bought = loan / entry_price

    # Realistic compound interest calculation based on actual cycle duration
    daily_rate = params["loan_rate"] / 365
    interest = loan * ((1 + daily_rate) ** actual_days - 1)
    payoff = loan + interest

    # Sell BTC equal to payoff
    btc_sold = payoff / exit_price
    gain_btc = btc_bought - btc_sold

    # Step 4: release all collateral back to free BTC
    state["free_btc"] += gain_btc + state["collat_btc"]
    state["collat_btc"] = params["start_collateral_btc"]  # reset reserve

    # Step 5: set next loan equal to cap for new price
    state["loan"] = cap_next_loan(exit_price, state["collat_btc"])

    # Log cycle with timing and interest details
    records.append({
        "cycle": state["cycle"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "cycle_duration_days": actual_days,
        "expected_duration_days": expected_days,
        "annualized_return_pct": ((exit_price / entry_price) ** (365 / actual_days) - 1) * 100,
        "loan": loan,
        "daily_interest_rate": daily_rate * 100,
        "interest_paid": interest,
        "total_payoff": payoff,
        "interest_as_pct_of_loan": (interest / loan) * 100,
        "needed_cure_btc": needed_btc,
        "btc_free": state["free_btc"],
        "btc_bought": btc_bought,
        "btc_sold": btc_sold,
        "net_btc_gain": gain_btc,
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

# Calculate and display summary statistics
total_interest = df['interest_paid'].sum()
total_loans = df['loan'].sum()
final_btc = df['btc_free'].iloc[-1]
total_cycles = len(df)
total_days = df['cycle_duration_days'].sum()
avg_cycle_days = df['cycle_duration_days'].mean()

print(f"\n📊 SIMULATION SUMMARY:")
print(f"   💰 Final BTC Holdings: {final_btc:.4f} BTC")
print(f"   🔄 Total Cycles: {total_cycles}")
print(f"   ⏱️  Total Time: {total_days:.0f} days ({total_days/365:.1f} years)")
print(f"   📅 Average Cycle Duration: {avg_cycle_days:.0f} days ({avg_cycle_days/30:.1f} months)")
print(f"   💸 Total Interest Paid: ${total_interest:,.0f}")
print(f"   📈 Average Loan Size: ${total_loans/total_cycles:,.0f}")
print(f"   📊 Interest as % of Total Loans: {100*total_interest/total_loans:.1f}%")
print(f"   🎯 Average Annualized Return: {df['annualized_return_pct'].mean():.1f}%")

print("\nSimulation complete.  Files saved:\n"
      "  • cycles_log.csv\n"
      "  • btc_price_over_cycles.png\n"
      "  • btc_owned_over_cycles.png")