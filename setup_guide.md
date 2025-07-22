
# 📚 Options Trading Analysis - Setup Guide

## ⚠️ IMPORTANT DISCLAIMERS
- **Options trading is risky** and you can lose money quickly
- This is for **educational purposes only**
- **Start with paper trading** before using real money
- **Do your own research** - don't blindly follow any signals
- Only trade with money you can afford to lose

## 🚀 Quick Start

### 1. Install Dependencies
The system will automatically install required packages when you run it.

### 2. Run the Analysis
```bash
python3 main.py
```

### 3. Review Results
The system generates two CSV files:
- `portfolio_analysis.csv` - Stock analysis for each ticker
- `trading_opportunities.csv` - All options opportunities found

## 📊 What the System Does

1. **Analyzes 15 AI-focused stocks** from the predefined list
2. **Gets real-time stock prices** from Yahoo Finance
3. **Fetches options chains** for near-term expirations
4. **Calculates Greeks** (Delta, Gamma, Theta, Vega, Rho)
5. **Scores trading opportunities** based on:
   - Bid-ask spread (tighter is better)
   - Trading volume (higher is better)  
   - Open interest (higher is better)
   - Delta exposure (reasonable levels)

## 🎯 Understanding the Output

### Portfolio Analysis
Shows current stock prices, sectors, and basic info for each ticker.

### Top Trading Opportunities
The system shows the highest-scored options with:
- **Ticker**: Stock symbol
- **Type**: Call or Put
- **Strike**: Option strike price
- **Expiration**: When the option expires
- **Mid Price**: Average of bid/ask
- **Volume**: Daily trading volume
- **Score**: Internal ranking (higher = better)

## 📈 Trading Strategies to Consider

### For Beginners:
1. **Buy Calls** on stocks you think will go up
2. **Buy Puts** on stocks you think will go down
3. **Sell options close to expiration** to capture time decay
4. **Focus on liquid options** (high volume/open interest)

### Key Metrics to Watch:
- **Delta**: How much option price changes per $1 stock move
- **Theta**: How much you lose per day due to time decay
- **Implied Volatility**: Higher = more expensive options
- **Volume/Open Interest**: Higher = easier to trade

## 🛡️ Risk Management Rules

1. **Never risk more than 2-5%** of your account on one trade
2. **Set stop losses** before entering trades
3. **Take profits** when you have them
4. **Diversify** across different stocks and strategies
5. **Keep a trading journal** to learn from mistakes

## 🔧 Customization

You can modify `TICKERS` in `main.py` to analyze different stocks:

```python
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'
    # Add your preferred tickers here
]
```

## 📞 Getting Help

- Review the generated CSV files for detailed data
- Start with paper trading platforms like ThinkOrSwim
- Join trading communities for education (r/options, etc.)
- Consider taking a formal options trading course

## 🚨 Remember
**This tool helps you find opportunities, but YOU make the trading decisions. Always do your own research and understand the risks involved.**

Happy (and safe) trading! 🎯
