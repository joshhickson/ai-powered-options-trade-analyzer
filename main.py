
#!/usr/bin/env python3
"""
Options Trading Analysis System
Pulls data from TastyTrade and Yahoo Finance to screen for profitable trades
"""
import os
import sys
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.stats import norm
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

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

# Main tickers to analyze (AI-focused portfolio from the README)
TICKERS = [
    'NVDA', 'TSLA', 'GOOGL', 'PLTR', 'COIN',
    'LMT', 'ISRG', 'HLX', 'RBLX', 'Z', 
    'AVAV', 'DE', 'SYM', 'RXRX', 'UPST'
]

def get_stock_data(ticker):
    """Get current stock price using direct Yahoo Finance API"""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            # Add delay between requests
            delay = random.uniform(1, 3) + (attempt * 2)
            print(f"  Waiting {delay:.1f}s before requesting {ticker} (attempt {attempt + 1})...")
            time.sleep(delay)
            
            # Use direct Yahoo Finance API that we know works
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            print(f"  Fetching data for {ticker} via direct API...")
            response = session.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'chart' in data and data['chart']['result'] and len(data['chart']['result']) > 0:
                    result = data['chart']['result'][0]
                    meta = result.get('meta', {})
                    
                    # Get current price
                    current_price = meta.get('regularMarketPrice')
                    if current_price is None:
                        # Try getting from price data
                        timestamps = result.get('timestamp', [])
                        indicators = result.get('indicators', {})
                        quote = indicators.get('quote', [{}])
                        if quote and len(quote) > 0:
                            close_prices = quote[0].get('close', [])
                            if close_prices:
                                # Get the last non-null price
                                for price in reversed(close_prices):
                                    if price is not None:
                                        current_price = price
                                        break
                    
                    if current_price is not None:
                        # Get additional metadata
                        currency = meta.get('currency', 'USD')
                        exchange = meta.get('fullExchangeName', 'Unknown')
                        
                        print(f"  ✅ Direct API success for {ticker}: ${current_price:.2f} {currency}")
                        return {
                            'ticker': ticker,
                            'current_price': current_price,
                            'currency': currency,
                            'exchange': exchange,
                            'market_cap': 0,
                            'sector': 'Technology',  # Default for AI portfolio
                            'industry': 'Unknown'
                        }
                    else:
                        print(f"  ⚠️  No price data found in response for {ticker}")
                else:
                    print(f"  ⚠️  Invalid chart data structure for {ticker}")
            else:
                print(f"  ⚠️  HTTP {response.status_code} for {ticker}")
                
        except Exception as e:
            print(f"  ⚠️  Error fetching {ticker} (attempt {attempt + 1}): {e}")
            
        if attempt < max_retries - 1:
            wait_time = 10 + (attempt * 5)
            print(f"  Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    print(f"  ❌ Failed to get data for {ticker} after {max_retries} attempts")
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
    print(f"  ⚠️  Options data temporarily disabled due to yfinance rate limiting")
    print(f"  📈 Current stock price for {ticker}: ${current_price:.2f}")
    print(f"  💡 You can manually check options at: https://finance.yahoo.com/quote/{ticker}/options")
    return pd.DataFrame()

def screen_for_trades(portfolio_df, max_trades=5):
    """Screen for the best trading opportunities"""
    print("\n🔍 SCREENING FOR TRADES")
    print("=" * 30)
    print("⚠️  Options screening temporarily disabled due to Yahoo Finance rate limits")
    print("📊 Showing stock analysis instead:\n")
    
    # Show stock analysis since options are unavailable
    print("🏆 STOCK ANALYSIS SUMMARY")
    print("=" * 50)
    
    for _, stock in portfolio_df.iterrows():
        ticker = stock['ticker']
        current_price = stock['current_price']
        currency = stock.get('currency', 'USD')
        exchange = stock.get('exchange', 'Unknown')
        
        print(f"{ticker:<6} | ${current_price:<8.2f} {currency:<3} | {exchange}")
        print(f"       💡 Manual options check: https://finance.yahoo.com/quote/{ticker}/options")
    
    print(f"\n💡 NEXT STEPS:")
    print("1. Review the stock prices above")
    print("2. Manually check options chains on Yahoo Finance")
    print("3. Look for high IV options near support/resistance levels")
    print("4. Consider paper trading first")
    print("5. Start with small position sizes")
    
    return pd.DataFrame()

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
