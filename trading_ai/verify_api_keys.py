#!/usr/bin/env python3
"""
Quick API Key Verification Script

Tests that all API keys in .env are working correctly.
Run this before starting Phase 2 to catch any issues early.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 70)
print("API KEY VERIFICATION")
print("=" * 70)
print()

# Track results
results = {}

# ============================================================================
# 1. FRED API (Macro Data)
# ============================================================================
print("Testing FRED API (Macro Data)...")
try:
    from fredapi import Fred
    fred_key = os.getenv('FRED_API_KEY')
    if fred_key and fred_key != 'your_key_here':
        fred = Fred(api_key=fred_key)
        # Try to fetch VIX data
        vix = fred.get_series('VIXCLS', observation_start='2024-01-01')
        if len(vix) > 0:
            print(f"  ✅ FRED API: Working! (Fetched {len(vix)} VIX data points)")
            results['FRED'] = 'PASS'
        else:
            print(f"  ⚠️  FRED API: Connected but no data returned")
            results['FRED'] = 'WARN'
    else:
        print(f"  ⚠️  FRED API: Key not set")
        results['FRED'] = 'SKIP'
except Exception as e:
    print(f"  ❌ FRED API: Error - {str(e)}")
    results['FRED'] = 'FAIL'

print()

# ============================================================================
# 2. Alpha Vantage API (Fundamentals)
# ============================================================================
print("Testing Alpha Vantage API (Fundamentals)...")
try:
    import requests
    av_key = os.getenv('ALPHA_VANTAGE_KEY')
    if av_key and av_key != 'your_key_here':
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol=AAPL&apikey={av_key}"
        response = requests.get(url, timeout=10)
        data = response.json()
        if 'Symbol' in data:
            print(f"  ✅ Alpha Vantage API: Working! (Fetched data for {data['Symbol']})")
            results['AlphaVantage'] = 'PASS'
        elif 'Note' in data:
            print(f"  ⚠️  Alpha Vantage API: Rate limit hit (25 req/day on free tier)")
            results['AlphaVantage'] = 'RATE_LIMIT'
        else:
            print(f"  ❌ Alpha Vantage API: Unexpected response - {data}")
            results['AlphaVantage'] = 'FAIL'
    else:
        print(f"  ⚠️  Alpha Vantage API: Key not set")
        results['AlphaVantage'] = 'SKIP'
except Exception as e:
    print(f"  ❌ Alpha Vantage API: Error - {str(e)}")
    results['AlphaVantage'] = 'FAIL'

print()

# ============================================================================
# 3. Finnhub API (Sentiment & Insider Trading)
# ============================================================================
print("Testing Finnhub API (Sentiment)...")
try:
    import requests
    finnhub_key = os.getenv('FINNHUB_API_KEY')
    if finnhub_key and finnhub_key != 'your_key_here':
        url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={finnhub_key}"
        response = requests.get(url, timeout=10)
        data = response.json()
        if 'c' in data:  # 'c' is current price
            print(f"  ✅ Finnhub API: Working! (AAPL price: ${data['c']:.2f})")
            results['Finnhub'] = 'PASS'
        elif 'error' in data:
            print(f"  ❌ Finnhub API: Error - {data['error']}")
            results['Finnhub'] = 'FAIL'
        else:
            print(f"  ❌ Finnhub API: Unexpected response - {data}")
            results['Finnhub'] = 'FAIL'
    else:
        print(f"  ⚠️  Finnhub API: Key not set")
        results['Finnhub'] = 'SKIP'
except Exception as e:
    print(f"  ❌ Finnhub API: Error - {str(e)}")
    results['Finnhub'] = 'FAIL'

print()

# ============================================================================
# 4. NewsAPI (Headlines)
# ============================================================================
print("Testing NewsAPI (Headlines)...")
try:
    import requests
    news_key = os.getenv('NEWS_API_KEY')
    if news_key and news_key != 'your_key_here':
        from datetime import datetime, timedelta
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        url = f"https://newsapi.org/v2/everything?q=AAPL&from={yesterday}&sortBy=publishedAt&apiKey={news_key}"
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get('status') == 'ok':
            article_count = data.get('totalResults', 0)
            print(f"  ✅ NewsAPI: Working! (Found {article_count} articles)")
            results['NewsAPI'] = 'PASS'
        elif 'code' in data:
            print(f"  ❌ NewsAPI: Error - {data.get('message', 'Unknown error')}")
            results['NewsAPI'] = 'FAIL'
        else:
            print(f"  ❌ NewsAPI: Unexpected response - {data}")
            results['NewsAPI'] = 'FAIL'
    else:
        print(f"  ⚠️  NewsAPI: Key not set")
        results['NewsAPI'] = 'SKIP'
except Exception as e:
    print(f"  ❌ NewsAPI: Error - {str(e)}")
    results['NewsAPI'] = 'FAIL'

print()

# ============================================================================
# 5. yfinance (No API key needed)
# ============================================================================
print("Testing yfinance (Recent Market Data)...")
try:
    import yfinance as yf
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="5d")
    if len(hist) > 0:
        latest_price = hist['Close'].iloc[-1]
        print(f"  ✅ yfinance: Working! (AAPL latest: ${latest_price:.2f})")
        results['yfinance'] = 'PASS'
    else:
        print(f"  ❌ yfinance: No data returned")
        results['yfinance'] = 'FAIL'
except Exception as e:
    print(f"  ❌ yfinance: Error - {str(e)}")
    results['yfinance'] = 'FAIL'

print()

# ============================================================================
# 6. Anthropic API (Claude) - REQUIRED for Phase 6
# ============================================================================
print("Testing Anthropic API (Claude)...")
try:
    import anthropic
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    if anthropic_key and anthropic_key != 'your_anthropic_key_here':
        client = anthropic.Anthropic(api_key=anthropic_key)
        # Simple test message
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'API test successful' in exactly 3 words."}]
        )
        reply = response.content[0].text
        print(f"  ✅ Anthropic API: Working! (Response: {reply})")
        results['Anthropic'] = 'PASS'
    else:
        print(f"  ⚠️  Anthropic API: Key not set (needed for Phase 6)")
        results['Anthropic'] = 'SKIP'
except Exception as e:
    print(f"  ❌ Anthropic API: Error - {str(e)}")
    results['Anthropic'] = 'FAIL'

print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)

pass_count = sum(1 for v in results.values() if v == 'PASS')
fail_count = sum(1 for v in results.values() if v == 'FAIL')
skip_count = sum(1 for v in results.values() if v in ['SKIP', 'WARN', 'RATE_LIMIT'])

print(f"\n✅ Passed: {pass_count}")
print(f"❌ Failed: {fail_count}")
print(f"⚠️  Skipped/Warnings: {skip_count}")
print()

if fail_count > 0:
    print("❌ Some APIs failed. Please check the keys in your .env file.")
    sys.exit(1)
elif pass_count >= 4:  # At least 4 working APIs (enough for Phase 2-5)
    print("✅ Core APIs are working! You're ready to start Phase 2.")
    print()
    print("📝 Note:")
    if results.get('Anthropic') == 'SKIP':
        print("   - Anthropic API not set yet (needed for Phase 6)")
    print("   - Make sure Kaggle data files are in data/raw/kaggle/")
    print()
    sys.exit(0)
else:
    print("⚠️  Some APIs are not set up yet.")
    print("   - You can still start Phase 2 with the working APIs")
    print("   - Set up remaining keys before their respective phases")
    sys.exit(0)
