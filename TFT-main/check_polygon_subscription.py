#!/usr/bin/env python3
"""
🔍 POLYGON.IO SUBSCRIPTION CHECKER
Verify Polygon.io API access and subscription status
"""

import os
import requests
from typing import Dict, Any
import json
from datetime import datetime, timedelta


class PolygonSubscriptionChecker:
    """Check Polygon.io API subscription and usage status"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self.base_url = "https://api.polygon.io"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def check_subscription_status(self) -> Dict[str, Any]:
        """Check Polygon.io subscription status and limits"""

        if not self.api_key:
            return {
                "status": "NO_API_KEY",
                "message": "No Polygon.io API key found",
                "configured": False,
                "subscription_active": False,
            }

        try:
            # Test basic API access
            response = requests.get(
                f"{self.base_url}/v3/reference/tickers",
                headers=self.headers,
                params={"limit": 1},
            )

            if response.status_code == 401:
                return {
                    "status": "INVALID_KEY",
                    "message": "API key is invalid or expired",
                    "configured": True,
                    "subscription_active": False,
                    "api_key_preview": (
                        f"{self.api_key[:8]}...{self.api_key[-4:]}"
                        if len(self.api_key) > 12
                        else "Too short"
                    ),
                }

            elif response.status_code == 429:
                return {
                    "status": "RATE_LIMITED",
                    "message": "API rate limit exceeded",
                    "configured": True,
                    "subscription_active": True,
                    "rate_limit_hit": True,
                }

            elif response.status_code == 200:
                # Get subscription details
                data = response.json()
                headers = response.headers

                return {
                    "status": "ACTIVE",
                    "message": "Polygon.io subscription is active",
                    "configured": True,
                    "subscription_active": True,
                    "api_key_preview": f"{self.api_key[:8]}...{self.api_key[-4:]}",
                    "rate_limits": {
                        "requests_per_minute": headers.get(
                            "X-RateLimit-RequestsPerMinute"
                        ),
                        "remaining_today": headers.get("X-RateLimit-Remaining"),
                        "reset_time": headers.get("X-RateLimit-Reset"),
                    },
                    "sample_data": data.get("results", [])[:1],  # First ticker
                }

            else:
                return {
                    "status": "ERROR",
                    "message": f"Unexpected response: {response.status_code}",
                    "configured": True,
                    "subscription_active": False,
                    "response_text": response.text,
                }

        except Exception as e:
            return {
                "status": "CONNECTION_ERROR",
                "message": f"Failed to connect to Polygon.io: {str(e)}",
                "configured": True,
                "subscription_active": False,
            }

    def test_market_data_access(self) -> Dict[str, Any]:
        """Test access to market data endpoints"""

        if not self.api_key:
            return {"error": "No API key available"}

        try:
            # Test AAPL data access
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

            response = requests.get(
                f"{self.base_url}/v2/aggs/ticker/AAPL/range/1/day/{start_date}/{end_date}",
                headers=self.headers,
                params={"adjusted": "true", "sort": "asc"},
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "market_data_access": True,
                    "sample_data_points": len(data.get("results", [])),
                    "latest_price": (
                        data.get("results", [{}])[-1].get("c")
                        if data.get("results")
                        else None
                    ),
                    "data_range": f"{start_date} to {end_date}",
                }
            else:
                return {
                    "market_data_access": False,
                    "error": f"Status {response.status_code}: {response.text}",
                }

        except Exception as e:
            return {"market_data_access": False, "error": str(e)}

    def generate_integration_code(self) -> str:
        """Generate code to integrate Polygon.io with TFT system"""

        return '''
# TFT System - Polygon.io Integration Code
# Add this to your local_tft_demo.py to use Polygon instead of yfinance

import requests
from datetime import datetime, timedelta

class PolygonDataProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.headers = {'Authorization': f'Bearer {api_key}'}
    
    def get_market_data(self, symbol: str, period: str = "30d") -> pd.DataFrame:
        """Get market data from Polygon.io"""
        
        # Convert period to dates
        end_date = datetime.now()
        if period == "30d":
            start_date = end_date - timedelta(days=30)
        elif period == "7d":
            start_date = end_date - timedelta(days=7)
        else:
            start_date = end_date - timedelta(days=365)  # Default 1 year
        
        # Polygon aggregates endpoint
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/hour/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        response = requests.get(url, headers=self.headers, params={
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        })
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            # Convert to pandas DataFrame (yfinance compatible)
            df = pd.DataFrame([
                {
                    'Open': r['o'],
                    'High': r['h'], 
                    'Low': r['l'],
                    'Close': r['c'],
                    'Volume': r['v'],
                    'VWAP': r.get('vw', r['c'])  # Polygon provides VWAP
                } for r in results
            ])
            
            df.index = pd.to_datetime([r['t'] for r in results], unit='ms')
            return df
        else:
            raise Exception(f"Polygon API error: {response.status_code} - {response.text}")

# Usage in TFT system:
# Replace yfinance calls with:
# polygon_provider = PolygonDataProvider(api_key="YOUR_API_KEY")  
# data = polygon_provider.get_market_data("AAPL", "30d")
        '''


def main():
    print("🔍 POLYGON.IO SUBSCRIPTION CHECKER")
    print("=" * 50)

    # Check for API key in various locations
    api_key_sources = [
        ("Environment Variable", os.getenv("POLYGON_API_KEY")),
        ("Config File", ""),  # Would need to read from config
        (
            "Manual Input",
            input("Enter your Polygon.io API key (or press Enter to skip): ").strip(),
        ),
    ]

    api_key = None
    for source, key in api_key_sources:
        if key and key.strip():
            print(f"✅ Found API key from: {source}")
            api_key = key.strip()
            break

    if not api_key:
        print("\n❌ NO POLYGON.IO API KEY FOUND")
        print("\n💡 To get started:")
        print("1. Visit https://polygon.io/dashboard")
        print("2. Sign up for a free account (2 API calls/minute)")
        print("3. Or upgrade to paid plan for higher limits")
        print("4. Copy your API key and add it to your environment:")
        print("   export POLYGON_API_KEY='your_api_key_here'")
        return

    # Check subscription status
    checker = PolygonSubscriptionChecker(api_key)
    status = checker.check_subscription_status()

    print(f"\n📊 SUBSCRIPTION STATUS: {status['status']}")
    print(f"Message: {status['message']}")

    if status["subscription_active"]:
        print(f"✅ API Key: {status['api_key_preview']}")

        if "rate_limits" in status:
            limits = status["rate_limits"]
            print(f"📈 Rate Limits:")
            print(
                f"   - Requests per minute: {limits.get('requests_per_minute', 'N/A')}"
            )
            print(f"   - Remaining today: {limits.get('remaining_today', 'N/A')}")

        # Test market data access
        print(f"\n🧪 Testing market data access...")
        market_test = checker.test_market_data_access()

        if market_test.get("market_data_access"):
            print(f"✅ Market data access confirmed")
            print(f"   - Sample data points: {market_test['sample_data_points']}")
            print(f"   - Latest AAPL price: ${market_test.get('latest_price', 'N/A')}")
            print(f"   - Data range: {market_test['data_range']}")

            print(f"\n🔧 INTEGRATION READY!")
            print(f"You can now replace yfinance with Polygon.io in your TFT system")

        else:
            print(f"❌ Market data access failed: {market_test.get('error')}")

    else:
        print(f"\n💳 SUBSCRIPTION NOT ACTIVE")
        if status["status"] == "INVALID_KEY":
            print("Your API key appears to be invalid. Please check:")
            print("1. Copy the key correctly from polygon.io dashboard")
            print("2. Ensure no extra spaces or characters")
            print("3. Check if the key is still valid")


if __name__ == "__main__":
    main()
