#!/usr/bin/env python3
import yfinance as yf
import time
import pandas as pd

print("🚀 YFINANCE PERFORMANCE DEMONSTRATION")
print("=" * 50)

# Test 1: Data fetching speed
start = time.time()
ticker = yf.Ticker("AAPL")
data = ticker.history(period="30d", interval="1h")
fetch_time = time.time() - start

print(f"📊 Data Fetching Performance:")
print(f"   - Symbol: AAPL")
print(f"   - Period: 30 days, 1-hour intervals")
print(f"   - Data points: {len(data)}")
print(f"   - Fetch time: {fetch_time:.2f} seconds")
print(f"   - Data per second: {len(data)/fetch_time:.0f} points/sec")

# Test 2: Data quality
print(f"\n🔍 Data Quality Analysis:")
completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
print(f"   - Completeness: {completeness:.1f}%")
print(f'   - Latest price: ${data["Close"].iloc[-1]:.2f}')
print(f'   - Price range: ${data["Low"].min():.2f} - ${data["High"].max():.2f}')
print(f'   - Volume range: {data["Volume"].min():,} - {data["Volume"].max():,}')

# Test 3: Multiple symbols
symbols = ["AAPL", "GOOGL", "MSFT"]
start = time.time()
multi_data = {}
for symbol in symbols:
    multi_data[symbol] = yf.Ticker(symbol).history(period="7d", interval="1h")
multi_time = time.time() - start

print(f"\n📈 Multi-Symbol Performance:")
print(f"   - Symbols: {symbols}")
print(f"   - Total data points: {sum(len(d) for d in multi_data.values())}")
print(f"   - Total fetch time: {multi_time:.2f} seconds")
print(f"   - Average per symbol: {multi_time/len(symbols):.2f} seconds")

print(f"\n✅ yfinance: PERFECT for TFT development!")
