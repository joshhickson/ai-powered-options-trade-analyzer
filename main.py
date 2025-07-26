
#!/usr/bin/env python3
"""
Options Trading Analysis System
Pulls data from TastyTrade and Yahoo Finance to screen for profitable trades
"""
import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
from scipy.stats import norm
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure requests session with retries
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Set session for yfinance
yf.pdr_override()

# Main tickers to analyze (AI-focused portfolio from the README)
# Starting with smaller list to avoid rate limits
TICKERS = [
    'NVDA', 'TSLA', 'GOOGL', 'PLTR', 'COIN'
]

# Full list - add more tickers once rate limiting is resolved
# 'LMT', 'ISRG', 'HLX', 'RBLX', 'Z', 'AVAV', 'DE', 'SYM', 'RXRX', 'UPST'

def get_stock_data(ticker):
    """Get current stock price with multiple fallback methods"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Progressive delay strategy
            delay = random.uniform(2, 5) + (attempt * 3)
            print(f"  Waiting {delay:.1f}s before requesting {ticker} (attempt {attempt + 1})...")
            time.sleep(delay)
            
            # Method 1: Try yf.download (often more reliable)
            print(f"  Method 1: Using yf.download for {ticker}...")
            try:
                data = yf.download(ticker, period="1d", progress=False, timeout=15)
                if not data.empty and 'Close' in data.columns:
                    current_price = data['Close'].iloc[-1]
                    print(f"  ✅ yf.download success for {ticker}: ${current_price:.2f}")
                    return {
                        'ticker': ticker,
                        'current_price': current_price,
                        'market_cap': 0,
                        'sector': 'Technology',
                        'industry': 'Unknown'
                    }
            except Exception as e:
                print(f"  Method 1 failed: {e}")
            
            # Method 2: Try longer period
            print(f"  Method 2: Using 5-day period for {ticker}...")
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d", timeout=15)
                if not hist.empty and 'Close' in hist.columns:
                    current_price = hist['Close'].iloc[-1]
                    print(f"  ✅ 5-day history success for {ticker}: ${current_price:.2f}")
                    return {
                        'ticker': ticker,
                        'current_price': current_price,
                        'market_cap': 0,
                        'sector': 'Technology',
                        'industry': 'Unknown'
                    }
            except Exception as e:
                print(f"  Method 2 failed: {e}")
            
            # Method 3: Try with different interval
            print(f"  Method 3: Using hourly data for {ticker}...")
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d", interval="1h", timeout=15)
                if not hist.empty and 'Close' in hist.columns:
                    current_price = hist['Close'].iloc[-1]
                    print(f"  ✅ Hourly data success for {ticker}: ${current_price:.2f}")
                    return {
                        'ticker': ticker,
                        'current_price': current_price,
                        'market_cap': 0,
                        'sector': 'Technology',
                        'industry': 'Unknown'
                    }
            except Exception as e:
                print(f"  Method 3 failed: {e}")
            
            # Method 4: Direct API call as last resort
            print(f"  Method 4: Direct Yahoo API for {ticker}...")
            try:
                url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'chart' in data and data['chart']['result']:
                        result = data['chart']['result'][0]
                        if 'meta' in result and 'regularMarketPrice' in result['meta']:
                            current_price = result['meta']['regularMarketPrice']
                            print(f"  ✅ Direct API success for {ticker}: ${current_price:.2f}")
                            return {
                                'ticker': ticker,
                                'current_price': current_price,
                                'market_cap': 0,
                                'sector': 'Technology',
                                'industry': 'Unknown'
                            }
            except Exception as e:
                print(f"  Method 4 failed: {e}")
            
        except Exception as e:
            print(f"  ⚠️  All methods failed for {ticker} (attempt {attempt + 1}): {e}")
            
        if attempt < max_retries - 1:
            wait_time = 15 + (attempt * 10)  # Longer waits between full retry cycles
            print(f"  All methods failed. Waiting {wait_time}s before full retry...")
            time.sleep(wait_time)
    
    print(f"  ❌ Failed to get data for {ticker} after {max_retries} attempts with all methods")
    return None

def calculate_black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option Greeks"""
    if T <= 0 or sigma <= 0:
        return {
            'delta': 0, 'gamma': 0, 'theta': 0, 
            'vega': 0, 'rho': 0, 'theoretical_price': 0
        }
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        delta = -norm.cdf(-d1)
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
             r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type.lower() == 'call' else norm.cdf(-d2)))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta / 365,  # Per day
        'vega': vega / 100,    # Per 1% IV change
        'rho': rho / 100,      # Per 1% rate change
        'theoretical_price': price
    }

def analyze_portfolio():
    """Analyze the portfolio of tickers"""
    print("🚀 Starting Options Portfolio Analysis")
    print("=" * 50)
    print("⏱️  Adding delays to avoid Yahoo Finance rate limits...")
    print("=" * 50)
    
    portfolio_data = []
    
    for i, ticker in enumerate(TICKERS):
        print(f"Analyzing {ticker}... ({i+1}/{len(TICKERS)})")
        stock_data = get_stock_data(ticker)
        if stock_data:
            portfolio_data.append(stock_data)
            print(f"  ✅ Successfully got data for {ticker}")
        else:
            print(f"  ❌ Failed to get data for {ticker}")
    
    # Create DataFrame
    df = pd.DataFrame(portfolio_data)
    
    # Save to CSV
    df.to_csv('portfolio_analysis.csv', index=False)
    print(f"\n✅ Portfolio analysis saved to portfolio_analysis.csv")
    print(f"📊 Analyzed {len(df)} tickers")
    
    # Display summary
    print("\n📈 PORTFOLIO SUMMARY")
    print("=" * 30)
    for _, row in df.iterrows():
        print(f"{row['ticker']:<6} | ${row['current_price']:<8.2f} | {row['sector']}")
    
    return df

def get_options_data_yahoo(ticker, current_price):
    """Get basic options data from Yahoo Finance with rate limiting"""
    try:
        # Add delay before options request
        time.sleep(random.uniform(2, 4))
        
        stock = yf.Ticker(ticker)
        
        try:
            expirations = stock.options
        except Exception as e:
            print(f"  Cannot get options expirations for {ticker}: {e}")
            return pd.DataFrame()
        
        if not expirations:
            print(f"  No options available for {ticker}")
            return pd.DataFrame()
        
        # Get options for the first few expirations
        all_options = []
        
        for i, exp_date in enumerate(expirations[:2]):  # Reduced to 2 expirations to avoid rate limits
            try:
                print(f"    Getting options for {ticker} exp {exp_date}...")
                time.sleep(random.uniform(1, 2))  # Delay between each expiration
                
                opt_chain = stock.option_chain(exp_date)
                
                # Process calls
                if not opt_chain.calls.empty:
                    calls = opt_chain.calls.copy()
                    calls['option_type'] = 'call'
                    calls['expiration'] = exp_date
                    calls['ticker'] = ticker
                    all_options.append(calls)
                
                # Process puts  
                if not opt_chain.puts.empty:
                    puts = opt_chain.puts.copy()
                    puts['option_type'] = 'put'
                    puts['expiration'] = exp_date
                    puts['ticker'] = ticker
                    all_options.append(puts)
                
            except Exception as e:
                print(f"    Error getting options for {exp_date}: {e}")
                continue
        
        if not all_options:
            print(f"  No options data retrieved for {ticker}")
            return pd.DataFrame()
            
        options_df = pd.concat(all_options, ignore_index=True)
        
        # Calculate days to expiration
        options_df['days_to_expiration'] = pd.to_datetime(options_df['expiration']).apply(
            lambda x: (x - pd.Timestamp.now()).days
        )
        
        # Filter for reasonable options (not too far OTM, decent volume)
        options_df = options_df[
            (options_df['volume'] > 5) &  # Lowered volume requirement
            (options_df['openInterest'] > 20) &  # Lowered OI requirement
            (options_df['days_to_expiration'] > 0) &
            (options_df['days_to_expiration'] < 60)
        ].copy()
        
        print(f"    Found {len(options_df)} viable options for {ticker}")
        return options_df
        
    except Exception as e:
        print(f"  Error fetching options for {ticker}: {e}")
        return pd.DataFrame()

def screen_for_trades(portfolio_df, max_trades=5):
    """Screen for the best trading opportunities"""
    print("\n🔍 SCREENING FOR TRADES")
    print("=" * 30)
    
    all_opportunities = []
    
    for _, stock in portfolio_df.iterrows():
        ticker = stock['ticker']
        current_price = stock['current_price']
        
        print(f"Screening {ticker}...")
        
        # Get options data
        options_df = get_options_data_yahoo(ticker, current_price)
        
        if options_df.empty:
            continue
            
        # Calculate theoretical Greeks for each option
        for _, option in options_df.iterrows():
            try:
                greeks = calculate_black_scholes_greeks(
                    S=current_price,
                    K=option['strike'],
                    T=option['days_to_expiration'] / 365,
                    r=0.03,  # Risk-free rate
                    sigma=option['impliedVolatility'],
                    option_type=option['option_type']
                )
                
                # Calculate trade metrics
                mid_price = (option['bid'] + option['ask']) / 2
                spread = option['ask'] - option['bid']
                spread_pct = spread / mid_price if mid_price > 0 else 999
                
                # Simple scoring system
                score = 0
                if spread_pct < 0.05: score += 2  # Tight spread
                if option['volume'] > 100: score += 2  # Good volume
                if option['openInterest'] > 500: score += 1  # Good OI
                if abs(greeks['delta']) > 0.3: score += 1  # Reasonable delta
                
                all_opportunities.append({
                    'ticker': ticker,
                    'option_type': option['option_type'],
                    'strike': option['strike'],
                    'expiration': option['expiration'],
                    'days_to_exp': option['days_to_expiration'],
                    'bid': option['bid'],
                    'ask': option['ask'],
                    'mid_price': mid_price,
                    'volume': option['volume'],
                    'open_interest': option['openInterest'],
                    'implied_vol': option['impliedVolatility'],
                    'spread_pct': spread_pct,
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'theta': greeks['theta'],
                    'vega': greeks['vega'],
                    'score': score,
                    'sector': stock['sector']
                })
                
            except Exception as e:
                continue
    
    if not all_opportunities:
        print("❌ No trading opportunities found")
        return pd.DataFrame()
    
    # Convert to DataFrame and sort by score
    opps_df = pd.DataFrame(all_opportunities)
    opps_df = opps_df.sort_values('score', ascending=False)
    
    # Save all opportunities
    opps_df.to_csv('trading_opportunities.csv', index=False)
    
    # Get top trades
    top_trades = opps_df.head(max_trades)
    
    print(f"\n🏆 TOP {len(top_trades)} TRADING OPPORTUNITIES")
    print("=" * 60)
    
    for _, trade in top_trades.iterrows():
        print(f"{trade['ticker']:<6} {trade['option_type']:<4} "
              f"${trade['strike']:<7.2f} {trade['expiration']} "
              f"| Mid: ${trade['mid_price']:<6.2f} | Vol: {trade['volume']:<6.0f} "
              f"| Score: {trade['score']}")
    
    return top_trades

def main():
    """Main function to run the analysis"""
    print("🎯 OPTIONS TRADING ANALYSIS SYSTEM")
    print("=" * 50)
    print("⚠️  RISK WARNING: Options trading involves significant risk!")
    print("📚 This is for educational purposes. Start with paper trading!")
    print("=" * 50)
    
    # Analyze portfolio
    portfolio_df = analyze_portfolio()
    
    # Screen for trades
    if not portfolio_df.empty:
        trades_df = screen_for_trades(portfolio_df)
        
        if not trades_df.empty:
            print(f"\n✅ Analysis complete! Check these files:")
            print("  📄 portfolio_analysis.csv - Stock analysis")
            print("  📄 trading_opportunities.csv - All opportunities")
            print("\n💡 Next steps:")
            print("  1. Review the top trades above")
            print("  2. Do your own research on each ticker")
            print("  3. Consider paper trading first")
            print("  4. Start with small position sizes")
        else:
            print("\n❌ No suitable trades found today")
    
    print("\n🚀 Happy trading! (But trade responsibly!)")

if __name__ == '__main__':
    main()
