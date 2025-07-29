
# Nasdaq Data Link API Setup Guide

## Quick Start (2 minutes)

1. **Get your free API key:**
   - Go to [data.nasdaq.com](https://data.nasdaq.com)
   - Click "Sign Up" (it's free)
   - Verify your email
   - Go to Account Settings → API Key

2. **Add to Replit Secrets:**
   - In Replit, click the lock icon (🔒) in the sidebar
   - Add a new secret:
     - Key: `NASDAQ_API_KEY`
     - Value: Your API key from step 1

3. **Run your simulation:**
   - The code will automatically use your API key
   - You'll get 10+ years of Bitcoin data from 2010-present

## What You Get
- **Bitcoin Liquid Index (BLX)** from Brave New Coin
- Daily prices from 2010 to present
- No geo-blocking (US-compliant)
- 50,000 API calls per day (free tier)

## Backup Method
If Nasdaq fails, the code automatically tries Kraken API (also US-compliant) as a backup.
