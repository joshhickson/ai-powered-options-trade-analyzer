
#!/usr/bin/env python3
"""
Diagnostic script to test Yahoo Finance API issues
"""
import yfinance as yf
import requests
import json
import time

def test_basic_connection():
    """Test basic internet connectivity"""
    print("🔍 Testing basic connectivity...")
    try:
        response = requests.get("https://httpbin.org/get", timeout=10)
        print(f"✅ Internet connection: {response.status_code}")
    except Exception as e:
        print(f"❌ Internet connection failed: {e}")

def test_yahoo_directly():
    """Test Yahoo Finance URL directly"""
    print("\n🔍 Testing Yahoo Finance URL directly...")
    try:
        # This is similar to what yfinance uses internally
        url = "https://query2.finance.yahoo.com/v8/finance/chart/NVDA"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Content length: {len(response.text)}")
        print(f"First 200 chars: {response.text[:200]}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("✅ Valid JSON response received")
                return True
            except json.JSONDecodeError:
                print("❌ Invalid JSON in response")
                return False
        else:
            print(f"❌ Bad status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Direct Yahoo test failed: {e}")
        return False

def test_yfinance_simple():
    """Test simple yfinance call"""
    print("\n🔍 Testing simple yfinance call...")
    try:
        ticker = yf.Ticker("NVDA")
        
        # Test info (often fails first)
        print("Testing .info...")
        try:
            info = ticker.info
            print(f"✅ Info available: {info.get('symbol', 'Unknown')}")
        except Exception as e:
            print(f"❌ Info failed: {e}")
        
        # Test history (more reliable)
        print("Testing .history()...")
        hist = ticker.history(period="1d")
        print(f"History shape: {hist.shape}")
        print(f"History empty: {hist.empty}")
        
        if not hist.empty:
            print(f"✅ Latest close: ${hist['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ No history data")
            return False
            
    except Exception as e:
        print(f"❌ yfinance test failed: {e}")
        return False

def test_alternative_approach():
    """Test alternative data sources"""
    print("\n🔍 Testing alternative approaches...")
    
    # Try different period and interval
    try:
        ticker = yf.Ticker("NVDA")
        hist = ticker.history(period="5d", interval="1d")
        if not hist.empty:
            print(f"✅ 5-day history works: ${hist['Close'].iloc[-1]:.2f}")
            return True
    except Exception as e:
        print(f"❌ 5-day history failed: {e}")
    
    # Try downloading directly
    try:
        import yfinance as yf
        data = yf.download("NVDA", period="1d", progress=False)
        if not data.empty:
            print(f"✅ yf.download works: ${data['Close'].iloc[-1]:.2f}")
            return True
    except Exception as e:
        print(f"❌ yf.download failed: {e}")
    
    return False

def main():
    print("🚀 Yahoo Finance Diagnostic Tests")
    print("=" * 40)
    
    test_basic_connection()
    yahoo_works = test_yahoo_directly()
    yf_works = test_yfinance_simple()
    alt_works = test_alternative_approach()
    
    print("\n📊 SUMMARY")
    print("=" * 20)
    print(f"Direct Yahoo API: {'✅' if yahoo_works else '❌'}")
    print(f"yfinance library: {'✅' if yf_works else '❌'}")
    print(f"Alternative methods: {'✅' if alt_works else '❌'}")
    
    if not any([yahoo_works, yf_works, alt_works]):
        print("\n🚨 ALL TESTS FAILED")
        print("Possible causes:")
        print("- Yahoo Finance is blocking this IP/region")
        print("- Rate limiting is very aggressive")
        print("- Network connectivity issues")
        print("- Yahoo Finance API changes")
    else:
        print("\n✅ At least one method works!")

if __name__ == "__main__":
    main()
