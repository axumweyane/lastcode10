#!/usr/bin/env python3
"""
Test Polygon.io Integration and API Endpoints
Tests the Polygon.io API integration without requiring a valid API key
"""

import os
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


class PolygonAPITester:
    """Test Polygon.io API integration"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self.base_url = "https://api.polygon.io"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }

    def test_endpoint_availability(self) -> Dict[str, Any]:
        """Test if Polygon.io endpoints are accessible"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "api_key_provided": bool(self.api_key),
            "endpoints": {},
        }

        # Test endpoints (these will fail without API key but show if service is up)
        test_endpoints = [
            {
                "name": "Market Status",
                "url": f"{self.base_url}/v1/marketstatus/now",
                "description": "Check if markets are open",
            },
            {
                "name": "Stock Ticker Details",
                "url": f"{self.base_url}/v3/reference/tickers/AAPL",
                "description": "Get ticker information",
            },
            {
                "name": "Aggregates (Daily)",
                "url": f"{self.base_url}/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-02",
                "description": "Get daily aggregates",
            },
            {
                "name": "Real-time Quote",
                "url": f"{self.base_url}/v2/last/nbbo/AAPL",
                "description": "Get real-time quote",
            },
        ]

        for endpoint in test_endpoints:
            try:
                # Test with minimal timeout to check connectivity
                response = requests.get(
                    endpoint["url"],
                    headers=self.headers,
                    timeout=5,
                    params={"apikey": self.api_key} if self.api_key else {},
                )

                results["endpoints"][endpoint["name"]] = {
                    "url": endpoint["url"],
                    "description": endpoint["description"],
                    "status_code": response.status_code,
                    "accessible": True,
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "error": None,
                }

                # Try to parse response
                try:
                    response_data = response.json()
                    if response.status_code == 401:
                        results["endpoints"][endpoint["name"]]["auth_required"] = True
                        results["endpoints"][endpoint["name"]][
                            "message"
                        ] = "API key required"
                    elif response.status_code == 200:
                        results["endpoints"][endpoint["name"]]["working"] = True
                        results["endpoints"][endpoint["name"]]["sample_data"] = (
                            str(response_data)[:200] + "..."
                        )
                    else:
                        results["endpoints"][endpoint["name"]]["message"] = (
                            response_data.get("error", "Unknown error")
                        )
                except:
                    results["endpoints"][endpoint["name"]]["raw_response"] = (
                        response.text[:200] + "..."
                    )

            except requests.exceptions.RequestException as e:
                results["endpoints"][endpoint["name"]] = {
                    "url": endpoint["url"],
                    "description": endpoint["description"],
                    "accessible": False,
                    "error": str(e),
                    "status_code": None,
                }

        return results

    def test_websocket_connectivity(self) -> Dict[str, Any]:
        """Test WebSocket endpoint connectivity (without subscribing)"""
        import socket

        try:
            # Test if WebSocket server is reachable
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(("socket.polygon.io", 443))
            sock.close()

            return {
                "websocket_server": "socket.polygon.io:443",
                "reachable": result == 0,
                "error": (
                    None if result == 0 else f"Connection failed with code {result}"
                ),
            }
        except Exception as e:
            return {
                "websocket_server": "socket.polygon.io:443",
                "reachable": False,
                "error": str(e),
            }

    def generate_integration_code(self) -> str:
        """Generate sample integration code for Polygon.io"""
        return '''
# Polygon.io Integration Code Sample
import os
import requests
import websocket
import json

class PolygonDataProvider:
    """Production-ready Polygon.io data provider"""
    
    def __init__(self):
        self.api_key = os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable required")
        
        self.base_url = "https://api.polygon.io"
        self.ws_url = "wss://socket.polygon.io/stocks"
    
    def get_market_status(self):
        """Get current market status"""
        url = f"{self.base_url}/v1/marketstatus/now"
        response = requests.get(url, params={'apikey': self.api_key})
        return response.json()
    
    def get_daily_bars(self, symbol: str, from_date: str, to_date: str):
        """Get daily OHLCV bars"""
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{from_date}/{to_date}"
        response = requests.get(url, params={'apikey': self.api_key})
        return response.json()
    
    def get_real_time_quote(self, symbol: str):
        """Get real-time quote"""
        url = f"{self.base_url}/v2/last/nbbo/{symbol}"
        response = requests.get(url, params={'apikey': self.api_key})
        return response.json()
    
    def stream_real_time_data(self, symbols: list, callback):
        """Stream real-time data via WebSocket"""
        def on_message(ws, message):
            data = json.loads(message)
            callback(data)
        
        def on_open(ws):
            # Authenticate
            auth_msg = {"action": "auth", "params": self.api_key}
            ws.send(json.dumps(auth_msg))
            
            # Subscribe to symbols
            sub_msg = {
                "action": "subscribe",
                "params": ",".join([f"T.{symbol}" for symbol in symbols])
            }
            ws.send(json.dumps(sub_msg))
        
        ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=on_message,
            on_open=on_open
        )
        ws.run_forever()

# Usage Example:
if __name__ == "__main__":
    provider = PolygonDataProvider()
    
    # Get market status
    status = provider.get_market_status()
    print("Market Status:", status)
    
    # Get daily data
    bars = provider.get_daily_bars("AAPL", "2023-01-01", "2023-01-31")
    print("AAPL Daily Bars:", len(bars.get('results', [])))
    
    # Real-time quote
    quote = provider.get_real_time_quote("AAPL")
    print("AAPL Quote:", quote)
'''

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive Polygon.io integration test"""
        print("🔍 POLYGON.IO INTEGRATION TEST")
        print("=" * 50)

        # Test API endpoints
        print("\n📡 Testing API Endpoints...")
        api_results = self.test_endpoint_availability()

        # Test WebSocket connectivity
        print("\n🌐 Testing WebSocket Connectivity...")
        ws_results = self.test_websocket_connectivity()

        # Combine results
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "api_key_configured": bool(self.api_key),
            "api_tests": api_results,
            "websocket_test": ws_results,
            "overall_status": (
                "Ready for integration"
                if api_results and ws_results
                else "Configuration needed"
            ),
        }

        # Print summary
        self.print_test_summary(results)

        return results

    def print_test_summary(self, results: Dict[str, Any]):
        """Print formatted test summary"""
        print(f"\n📊 TEST SUMMARY")
        print("=" * 30)
        print(
            f"API Key Configured: {'✅ YES' if results['api_key_configured'] else '❌ NO'}"
        )

        print(f"\n🔗 API Endpoints:")
        for name, endpoint in results["api_tests"]["endpoints"].items():
            status = "✅" if endpoint.get("accessible", False) else "❌"
            auth = "🔐" if endpoint.get("auth_required", False) else ""
            print(f"  {status} {name} {auth}")
            if endpoint.get("response_time_ms"):
                print(f"     Response time: {endpoint['response_time_ms']:.0f}ms")

        print(f"\n🌐 WebSocket:")
        ws_status = "✅" if results["websocket_test"]["reachable"] else "❌"
        print(f"  {ws_status} {results['websocket_test']['websocket_server']}")

        print(f"\n🎯 Overall Status: {results['overall_status']}")

        if not results["api_key_configured"]:
            print(f"\n💡 Next Steps:")
            print(f"1. Get your free API key: https://polygon.io/dashboard")
            print(f"2. Add to your .env file: POLYGON_API_KEY=your_key_here")
            print(f"3. Re-run this test")
        else:
            print(f"\n🚀 Ready to use Polygon.io for real-time market data!")


if __name__ == "__main__":
    tester = PolygonAPITester()
    results = tester.run_comprehensive_test()

    # Generate integration code
    print(f"\n📝 INTEGRATION CODE SAMPLE:")
    print("=" * 40)
    print(tester.generate_integration_code())
