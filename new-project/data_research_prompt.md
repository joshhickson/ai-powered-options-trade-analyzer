
# Data Research Request for Bitcoin Loan Simulation

## Development Environment Context
- **Platform**: Replit (Linux/Nix environment)
- **Language**: Python 3.11
- **Libraries**: pandas, numpy, matplotlib, yfinance
- **Project**: Bitcoin-backed lending strategy simulation

## Current Problem
I'm running a Bitcoin loan simulation (`leverage_sim.py`) that models a sophisticated lending strategy using historical BTC price data. The simulation currently falls back to synthetic data when yfinance fails to fetch real Bitcoin price history.

## Data Requirements
Please research and provide the following information:

### 1. Bitcoin Price Data Sources
- **Primary need**: Daily Bitcoin prices from 2010-present 
- **Format**: Date, Close price (USD)
- **Alternatives to yfinance** that work reliably in 2024
- **Free APIs** that don't require authentication
- **CSV download sources** as backup options

### 2. Alternative Python Libraries
Research Python libraries that can fetch Bitcoin historical data:
- Libraries that work better than yfinance for crypto
- Any that specifically handle Bitcoin/crypto data
- Installation commands for Nix/pip environments

### 3. Direct API Solutions
Find direct REST APIs that provide:
- Bitcoin historical price data
- No authentication required (or simple API key)
- JSON format responses
- Rate limits that allow reasonable usage

### 4. Backup Data Sources
- Reliable websites offering Bitcoin price CSV downloads
- Historical data going back to at least 2015
- Daily granularity minimum

## Code Integration Requirements
The solution should work with this existing code structure:
```python
def load_btc_history() -> pd.Series:
    """Return a daily Close price series indexed by date."""
    # Your research should help improve this function
```

## Expected Output Format
Please provide:
1. **Recommended approach** (library/API/method)
2. **Complete Python code example** showing how to fetch the data
3. **Installation instructions** for any new dependencies
4. **Fallback options** if the primary method fails
5. **Error handling strategies** for network issues

## Success Criteria
- Fetches real Bitcoin daily prices from 2010+ 
- Returns pandas Series with datetime index
- Handles network failures gracefully
- Works in Replit's Nix environment
- No complex authentication required

Please prioritize solutions that are simple, reliable, and don't require API keys if possible.
