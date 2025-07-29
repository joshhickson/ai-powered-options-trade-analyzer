
# Data Research Request for Bitcoin Loan Simulation (Updated January 2025)

## Development Environment Context
- **Platform**: Replit (Linux/Nix environment)
- **Language**: Python 3.11
- **Libraries**: pandas, numpy, matplotlib, requests
- **Project**: Bitcoin-backed lending strategy simulation

## Current Problem (January 2025)
I'm running a Bitcoin loan simulation (`leverage_sim.py`) that models a sophisticated lending strategy using historical BTC price data. Both primary data sources are now failing:

1. **CoinGecko API**: Returns 401 Unauthorized (they may have changed their free tier policies)
2. **yfinance**: Consistently fails with JSON parsing errors and "symbol may be delisted" messages

## Specific API Issues Encountered
```
⚠️  CoinGecko API failed: 401 Client Error: Unauthorized for url: https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=max&interval=daily
```

## Updated Data Requirements for 2025
Please research current (January 2025) solutions for:

### 1. CoinGecko API Fixes
- Has CoinGecko changed their free tier in 2025?
- Do they now require API keys for historical data?
- Are there working rate limit headers or authentication methods?
- Alternative CoinGecko endpoints that still work without auth?

### 2. Current Working Bitcoin APIs (2025)
Research APIs that are currently working in January 2025:
- **Binance Public API** (historical klines)
- **Coinbase Pro API** (historical rates)
- **Alpha Vantage** (crypto endpoints)
- **Polygon.io** (crypto data)
- **Yahoo Finance alternatives** that work
- Any new crypto data APIs launched in 2024-2025

### 3. Rate Limiting Solutions
- How to properly handle rate limits with delays
- User-Agent headers that work better
- Request session configurations for crypto APIs

### 4. 2025-Specific Python Libraries
Research crypto-specific libraries released or updated in 2024-2025:
- Alternatives to yfinance that focus on crypto
- Libraries that handle multiple exchanges
- Any that specifically work around recent API changes

## Code Requirements
The solution must work with this exact function signature:
```python
def load_btc_history() -> pd.Series:
    """Return a daily Close price series indexed by date."""
    # Need working implementation here
```

## Expected Output
Please provide:
1. **Current working API** (as of January 2025)
2. **Complete Python code** with proper error handling
3. **Rate limiting strategy** to avoid 401/429 errors
4. **Backup methods** if primary API fails
5. **Installation commands** for any new dependencies

## Success Criteria
- Fetches real Bitcoin daily prices from 2015+ (minimum)
- Works in Replit environment (January 2025)
- Handles rate limits gracefully
- No complex authentication if possible
- Returns pandas Series with datetime index

## Additional Context
The simulation needs at least 1,000+ days of historical data to fit the drawdown model properly. Synthetic data works but real data would make the lending strategy analysis much more accurate.

Please prioritize solutions that are currently working as of January 2025 and account for recent API policy changes.
