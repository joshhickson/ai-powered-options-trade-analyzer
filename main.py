
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

# Main tickers to analyze (AI-focused portfolio from the README)
TICKERS = [
    'NVDA', 'LMT', 'ISRG', 'HLX', 'TSLA', 
    'COIN', 'RBLX', 'Z', 'AVAV', 'DE', 
    'SYM', 'RXRX', 'GOOGL', 'PLTR', 'UPST'
]

def get_stock_data(ticker):
    """Get current stock price and basic info"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1d")
        
        current_price = hist['Close'].iloc[-1] if not hist.empty else None
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'market_cap': info.get('marketCap', 0),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown')
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
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
    
    portfolio_data = []
    
    for ticker in TICKERS:
        print(f"Analyzing {ticker}...")
        stock_data = get_stock_data(ticker)
        if stock_data:
            portfolio_data.append(stock_data)
    
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
    """Get basic options data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        
        if not expirations:
            return pd.DataFrame()
        
        # Get options for the first few expirations
        all_options = []
        
        for exp_date in expirations[:3]:  # First 3 expiration dates
            try:
                opt_chain = stock.option_chain(exp_date)
                
                # Process calls
                calls = opt_chain.calls.copy()
                calls['option_type'] = 'call'
                calls['expiration'] = exp_date
                calls['ticker'] = ticker
                
                # Process puts  
                puts = opt_chain.puts.copy()
                puts['option_type'] = 'put'
                puts['expiration'] = exp_date
                puts['ticker'] = ticker
                
                all_options.extend([calls, puts])
                
            except Exception as e:
                print(f"  Error getting options for {exp_date}: {e}")
                continue
        
        if not all_options:
            return pd.DataFrame()
            
        options_df = pd.concat(all_options, ignore_index=True)
        
        # Calculate days to expiration
        options_df['days_to_expiration'] = pd.to_datetime(options_df['expiration']).apply(
            lambda x: (x - pd.Timestamp.now()).days
        )
        
        # Filter for reasonable options (not too far OTM, decent volume)
        options_df = options_df[
            (options_df['volume'] > 10) & 
            (options_df['openInterest'] > 50) &
            (options_df['days_to_expiration'] > 0) &
            (options_df['days_to_expiration'] < 60)
        ].copy()
        
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
