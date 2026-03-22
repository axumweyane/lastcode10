#!/usr/bin/env python3
"""
End-to-End TFT Trading System Demo with Local AAPL Data
=====================================================
This script demonstrates the complete workflow using local data
"""

import pandas as pd
import numpy as np
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import yfinance as yf
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalTFTDemo:
    def __init__(self):
        self.symbol = "AAPL"
        self.raw_data = None
        self.features = None
        self.sentiment_data = None
        self.predictions = None
        self.trading_signals = None
        self.portfolio = {"cash": 10000, "positions": {}}

    def step1_data_ingestion(self):
        """Step 1: Data Ingestion - Collect AAPL data locally"""
        print("\n🌊 STEP 1: DATA INGESTION")
        print("=" * 50)

        try:
            # Download AAPL data using yfinance (local alternative to Polygon.io)
            print("📈 Downloading AAPL market data...")
            ticker = yf.Ticker(self.symbol)

            # Get historical data (last 60 days for context)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)

            self.raw_data = ticker.history(
                start=start_date, end=end_date, interval="1d", prepost=True
            )

            print(f"✅ Downloaded {len(self.raw_data)} days of {self.symbol} data")
            print(
                f"   - Date range: {self.raw_data.index[0].date()} to {self.raw_data.index[-1].date()}"
            )
            print(f"   - Latest close: ${self.raw_data['Close'].iloc[-1]:.2f}")

            # Simulate real-time data structure
            market_data = {
                "symbol": self.symbol,
                "timestamp": datetime.now().isoformat(),
                "data_points": len(self.raw_data),
                "latest_price": float(self.raw_data["Close"].iloc[-1]),
                "volume": int(self.raw_data["Volume"].iloc[-1]),
                "status": "success",
            }

            print(f"📊 Market Data Summary:")
            print(f"   - Current Price: ${market_data['latest_price']:.2f}")
            print(f"   - Today's Volume: {market_data['volume']:,}")

            return market_data

        except Exception as e:
            print(f"❌ Data ingestion failed: {e}")
            return None

    def step2_sentiment_analysis(self):
        """Step 2: Sentiment Analysis - Generate mock sentiment data"""
        print("\n💭 STEP 2: SENTIMENT ANALYSIS")
        print("=" * 50)

        try:
            # Simulate Reddit sentiment data (in production, this comes from Reddit API)
            print("🤖 Generating sentiment analysis for AAPL...")

            # Create realistic sentiment patterns
            np.random.seed(42)  # For reproducible results

            # Generate mock Reddit comments data
            comments_data = []
            sentiment_scores = []

            # Simulate 100 recent comments
            for i in range(100):
                # Create sentiment bias based on recent price movement
                price_change = (
                    self.raw_data["Close"].iloc[-1] - self.raw_data["Close"].iloc[-2]
                ) / self.raw_data["Close"].iloc[-2]

                # Sentiment tends to follow price momentum with some noise
                base_sentiment = np.tanh(
                    price_change * 10
                )  # Scale and bound to [-1, 1]
                noise = np.random.normal(0, 0.3)
                sentiment = np.clip(base_sentiment + noise, -1, 1)

                comment = {
                    "comment_id": f"reddit_comment_{i}",
                    "ticker": self.symbol,
                    "sentiment_score": round(sentiment, 3),
                    "confidence": round(np.random.uniform(0.6, 0.95), 2),
                    "upvotes": np.random.randint(1, 100),
                    "timestamp": datetime.now()
                    - timedelta(minutes=np.random.randint(0, 1440)),
                }

                comments_data.append(comment)
                sentiment_scores.append(sentiment)

            # Calculate aggregated sentiment metrics
            sentiment_scores = np.array(sentiment_scores)
            bullish_count = np.sum(sentiment_scores > 0.3)
            bearish_count = np.sum(sentiment_scores < -0.3)
            neutral_count = len(sentiment_scores) - bullish_count - bearish_count

            self.sentiment_data = {
                "symbol": self.symbol,
                "total_comments": len(comments_data),
                "avg_sentiment": round(np.mean(sentiment_scores), 3),
                "sentiment_std": round(np.std(sentiment_scores), 3),
                "bullish_count": int(bullish_count),
                "bearish_count": int(bearish_count),
                "neutral_count": int(neutral_count),
                "bullish_pct": round((bullish_count / len(sentiment_scores)) * 100, 1),
                "bearish_pct": round((bearish_count / len(sentiment_scores)) * 100, 1),
                "neutral_pct": round((neutral_count / len(sentiment_scores)) * 100, 1),
                "sentiment_momentum": round(
                    np.mean(sentiment_scores[-10:])
                    - np.mean(sentiment_scores[-30:-10]),
                    3,
                ),
                "timestamp": datetime.now().isoformat(),
            }

            print(f"📱 Sentiment Analysis Results:")
            print(
                f"   - Total Comments Analyzed: {self.sentiment_data['total_comments']}"
            )
            print(
                f"   - Average Sentiment: {self.sentiment_data['avg_sentiment']:+.3f}"
            )
            print(f"   - Bullish: {self.sentiment_data['bullish_pct']}%")
            print(f"   - Bearish: {self.sentiment_data['bearish_pct']}%")
            print(f"   - Neutral: {self.sentiment_data['neutral_pct']}%")
            print(f"   - Momentum: {self.sentiment_data['sentiment_momentum']:+.3f}")

            return self.sentiment_data

        except Exception as e:
            print(f"❌ Sentiment analysis failed: {e}")
            return None

    def step3_feature_engineering(self):
        """Step 3: Feature Engineering - Create TFT model features"""
        print("\n🔧 STEP 3: FEATURE ENGINEERING")
        print("=" * 50)

        try:
            print("⚙️ Engineering features for TFT model...")

            # Create a copy for feature engineering
            df = self.raw_data.copy()

            # Technical indicators
            print("   - Calculating technical indicators...")

            # Price-based features
            df["returns"] = df["Close"].pct_change()
            df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
            df["volatility"] = df["returns"].rolling(window=20).std()

            # Moving averages
            df["sma_10"] = df["Close"].rolling(window=10).mean()
            df["sma_20"] = df["Close"].rolling(window=20).mean()
            df["sma_50"] = df["Close"].rolling(window=min(50, len(df))).mean()

            # RSI
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df["Close"].ewm(span=12).mean()
            exp2 = df["Close"].ewm(span=26).mean()
            df["macd"] = exp1 - exp2
            df["macd_signal"] = df["macd"].ewm(span=9).mean()

            # Bollinger Bands
            df["bb_middle"] = df["Close"].rolling(window=20).mean()
            bb_std = df["Close"].rolling(window=20).std()
            df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
            df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
            df["bb_position"] = (df["Close"] - df["bb_lower"]) / (
                df["bb_upper"] - df["bb_lower"]
            )

            # Volume features
            df["volume_sma"] = df["Volume"].rolling(window=20).mean()
            df["volume_ratio"] = df["Volume"] / df["volume_sma"]

            # Price position features
            df["high_low_ratio"] = df["High"] / df["Low"]
            df["close_open_ratio"] = df["Close"] / df["Open"]

            # Add sentiment features (broadcast to all rows)
            if self.sentiment_data:
                df["sentiment_score"] = self.sentiment_data["avg_sentiment"]
                df["sentiment_momentum"] = self.sentiment_data["sentiment_momentum"]
                df["bullish_pct"] = self.sentiment_data["bullish_pct"] / 100
                df["bearish_pct"] = self.sentiment_data["bearish_pct"] / 100
            else:
                df["sentiment_score"] = 0
                df["sentiment_momentum"] = 0
                df["bullish_pct"] = 0.33
                df["bearish_pct"] = 0.33

            # Select features for model
            feature_columns = [
                "returns",
                "log_returns",
                "volatility",
                "sma_10",
                "sma_20",
                "sma_50",
                "rsi",
                "macd",
                "macd_signal",
                "bb_position",
                "volume_ratio",
                "high_low_ratio",
                "close_open_ratio",
                "sentiment_score",
                "sentiment_momentum",
                "bullish_pct",
                "bearish_pct",
            ]

            # Create feature matrix
            self.features = df[feature_columns].fillna(method="ffill").fillna(0)

            print(f"✅ Feature engineering completed:")
            print(f"   - Features created: {len(feature_columns)}")
            print(f"   - Data points: {len(self.features)}")
            print(f"   - Feature names: {', '.join(feature_columns[:5])}...")

            # Show latest feature values
            latest_features = self.features.iloc[-1]
            print(f"\n📊 Latest Feature Values (for prediction):")
            for col in ["returns", "rsi", "macd", "sentiment_score", "volatility"]:
                if col in latest_features:
                    print(f"   - {col}: {latest_features[col]:.4f}")

            return self.features

        except Exception as e:
            print(f"❌ Feature engineering failed: {e}")
            return None

    def step4_tft_prediction(self):
        """Step 4: TFT Model Prediction - Generate price predictions"""
        print("\n🤖 STEP 4: TFT MODEL PREDICTION")
        print("=" * 50)

        try:
            print("🧠 Running TFT model inference...")

            if self.features is None:
                print("❌ No features available for prediction")
                return None

            # Simulate TFT model prediction (in production, this uses actual PyTorch model)
            print("   - Loading model artifacts...")
            print("   - Processing feature tensor...")
            print("   - Running multi-horizon inference...")

            # Get latest features for prediction
            latest_features = self.features.iloc[
                -10:
            ].values  # Use last 10 days as context
            current_price = float(self.raw_data["Close"].iloc[-1])

            # Simulate realistic predictions based on features
            np.random.seed(42)

            # Base prediction influenced by technical indicators and sentiment
            base_momentum = self.features["returns"].iloc[-5:].mean()
            rsi_signal = (
                50 - self.features["rsi"].iloc[-1]
            ) / 50  # Normalized RSI signal
            macd_signal = np.sign(self.features["macd"].iloc[-1])
            sentiment_signal = self.features["sentiment_score"].iloc[-1]

            # Combine signals
            combined_signal = (
                base_momentum * 0.3
                + rsi_signal * 0.2
                + macd_signal * 0.1
                + sentiment_signal * 0.2
            ) + np.random.normal(
                0, 0.01
            )  # Add noise

            # Generate multi-horizon predictions
            horizons = {"1h": 1, "4h": 4, "24h": 24}
            predictions = {}

            for horizon, hours in horizons.items():
                # Scale prediction by horizon
                time_decay = np.sqrt(hours)  # Uncertainty increases with time
                predicted_return = combined_signal / time_decay
                predicted_price = current_price * (1 + predicted_return)

                # Confidence decreases with horizon
                confidence = max(0.5, 0.9 - (hours * 0.05))

                predictions[horizon] = {
                    "predicted_price": round(predicted_price, 2),
                    "predicted_return": round(
                        predicted_return * 100, 2
                    ),  # As percentage
                    "confidence": round(confidence, 2),
                    "horizon_hours": hours,
                }

            self.predictions = {
                "symbol": self.symbol,
                "current_price": current_price,
                "prediction_time": datetime.now().isoformat(),
                "model_version": "v1.0.0",
                "predictions": predictions,
                "feature_importance": {
                    "technical_momentum": 0.3,
                    "rsi_signal": 0.2,
                    "sentiment": 0.2,
                    "macd": 0.1,
                    "volume": 0.1,
                    "volatility": 0.1,
                },
            }

            print(f"🎯 TFT Predictions Generated:")
            print(f"   - Current Price: ${current_price:.2f}")

            for horizon, pred in predictions.items():
                direction = "📈" if pred["predicted_return"] > 0 else "📉"
                print(
                    f"   - {horizon} forecast: ${pred['predicted_price']:.2f} ({pred['predicted_return']:+.2f}%) {direction}"
                )
                print(f"     Confidence: {pred['confidence']:.1%}")

            return self.predictions

        except Exception as e:
            print(f"❌ TFT prediction failed: {e}")
            return None

    def step5_trading_signals(self):
        """Step 5: Trading Signal Generation"""
        print("\n💰 STEP 5: TRADING SIGNAL GENERATION")
        print("=" * 50)

        try:
            print("⚡ Generating trading signals...")

            if not self.predictions:
                print("❌ No predictions available for signal generation")
                return None

            # Extract key metrics for decision making
            current_price = self.predictions["current_price"]
            predictions = self.predictions["predictions"]

            # Get 1-hour and 4-hour predictions (most reliable)
            pred_1h = predictions["1h"]
            pred_4h = predictions["4h"]

            # Decision thresholds
            min_confidence = 0.7
            min_return_threshold = 0.5  # 0.5% minimum expected return

            # Signal generation logic
            signal = "HOLD"
            reason = "No clear signal"
            position_size = 0

            # Check confidence thresholds
            if (
                pred_1h["confidence"] >= min_confidence
                and pred_4h["confidence"] >= min_confidence
            ):

                avg_predicted_return = (
                    pred_1h["predicted_return"] + pred_4h["predicted_return"]
                ) / 2

                if avg_predicted_return > min_return_threshold:
                    signal = "BUY"
                    reason = f"Strong bullish signal: {avg_predicted_return:+.2f}% expected return"
                    # Position size based on available cash (buy what we can afford)
                    max_affordable = int(
                        self.portfolio["cash"] * 0.9 / current_price
                    )  # Use 90% of cash
                    position_size = max_affordable

                elif avg_predicted_return < -min_return_threshold:
                    signal = "SELL"
                    reason = f"Strong bearish signal: {avg_predicted_return:+.2f}% expected return"
                    # Sell existing position if we have one
                    position_size = self.portfolio["positions"].get(self.symbol, 0)

            # Add sentiment confirmation
            sentiment_score = (
                self.sentiment_data["avg_sentiment"] if self.sentiment_data else 0
            )
            sentiment_confirmation = ""

            if signal == "BUY" and sentiment_score > 0.1:
                sentiment_confirmation = " + Positive sentiment support"
            elif signal == "BUY" and sentiment_score < -0.1:
                sentiment_confirmation = " - Negative sentiment warning"
                position_size = int(position_size * 0.5)  # Reduce position
            elif signal == "SELL" and sentiment_score < -0.1:
                sentiment_confirmation = " + Negative sentiment confirmation"

            # Risk management checks
            risk_checks = []

            # Volatility check
            current_volatility = self.features["volatility"].iloc[-1]
            if current_volatility > self.features["volatility"].quantile(0.8):
                risk_checks.append("High volatility detected")
                if signal == "BUY":
                    position_size = int(
                        position_size * 0.7
                    )  # Reduce position in high vol

            # RSI check
            current_rsi = self.features["rsi"].iloc[-1]
            if signal == "BUY" and current_rsi > 80:
                risk_checks.append("Overbought RSI warning")
            elif signal == "SELL" and current_rsi < 20:
                risk_checks.append("Oversold RSI warning")

            self.trading_signals = {
                "symbol": self.symbol,
                "signal": signal,
                "reason": reason + sentiment_confirmation,
                "position_size": position_size,
                "current_price": current_price,
                "target_price": pred_4h["predicted_price"],
                "expected_return": (
                    pred_1h["predicted_return"] + pred_4h["predicted_return"]
                )
                / 2,
                "confidence": (pred_1h["confidence"] + pred_4h["confidence"]) / 2,
                "risk_checks": risk_checks,
                "sentiment_score": sentiment_score,
                "timestamp": datetime.now().isoformat(),
            }

            print(f"🎯 Trading Signal Generated:")
            print(f"   - Signal: {signal} 📊")
            print(f"   - Reason: {reason}")
            if sentiment_confirmation:
                print(f"   - Sentiment: {sentiment_confirmation}")
            print(
                f"   - Position Size: {position_size:,} shares (${position_size * current_price:,.0f})"
            )
            print(
                f"   - Expected Return: {self.trading_signals['expected_return']:+.2f}%"
            )
            print(f"   - Confidence: {self.trading_signals['confidence']:.1%}")

            if risk_checks:
                print(f"   - Risk Warnings: {', '.join(risk_checks)}")

            return self.trading_signals

        except Exception as e:
            print(f"❌ Trading signal generation failed: {e}")
            return None

    def step6_portfolio_management(self):
        """Step 6: Portfolio Management & Risk Control"""
        print("\n📊 STEP 6: PORTFOLIO MANAGEMENT")
        print("=" * 50)

        try:
            print("🛡️ Executing portfolio management and risk controls...")

            if not self.trading_signals:
                print("❌ No trading signals available")
                return None

            signal = self.trading_signals["signal"]
            position_size = self.trading_signals["position_size"]
            current_price = self.trading_signals["current_price"]

            # Current portfolio state
            current_cash = self.portfolio["cash"]
            current_position = self.portfolio["positions"].get(self.symbol, 0)
            current_portfolio_value = current_cash + (current_position * current_price)

            print(f"💼 Current Portfolio:")
            print(f"   - Cash: ${current_cash:,.2f}")
            print(
                f"   - {self.symbol} Position: {current_position:,} shares (${current_position * current_price:,.2f})"
            )
            print(f"   - Total Value: ${current_portfolio_value:,.2f}")

            # Execute trading decision
            trade_executed = False
            trade_details = {}

            if signal == "BUY" and position_size > 0:
                trade_cost = position_size * current_price

                # Check if we have enough cash
                if trade_cost <= current_cash:
                    # Execute buy order
                    self.portfolio["cash"] -= trade_cost
                    self.portfolio["positions"][self.symbol] = (
                        current_position + position_size
                    )

                    trade_executed = True
                    trade_details = {
                        "action": "BUY",
                        "shares": position_size,
                        "price": current_price,
                        "total_cost": trade_cost,
                        "new_position": self.portfolio["positions"][self.symbol],
                    }

                    print(f"✅ BUY ORDER EXECUTED:")
                    print(
                        f"   - Purchased: {position_size:,} shares at ${current_price:.2f}"
                    )
                    print(f"   - Total Cost: ${trade_cost:,.2f}")
                    print(f"   - Remaining Cash: ${self.portfolio['cash']:,.2f}")

                else:
                    print(f"❌ Insufficient cash for buy order")
                    print(
                        f"   - Required: ${trade_cost:,.2f}, Available: ${current_cash:,.2f}"
                    )

            elif signal == "SELL" and current_position > 0:
                # Determine shares to sell
                shares_to_sell = (
                    min(position_size, current_position)
                    if position_size > 0
                    else current_position
                )
                trade_proceeds = shares_to_sell * current_price

                # Execute sell order
                self.portfolio["cash"] += trade_proceeds
                self.portfolio["positions"][self.symbol] = (
                    current_position - shares_to_sell
                )

                trade_executed = True
                trade_details = {
                    "action": "SELL",
                    "shares": shares_to_sell,
                    "price": current_price,
                    "total_proceeds": trade_proceeds,
                    "new_position": self.portfolio["positions"][self.symbol],
                }

                print(f"✅ SELL ORDER EXECUTED:")
                print(f"   - Sold: {shares_to_sell:,} shares at ${current_price:.2f}")
                print(f"   - Total Proceeds: ${trade_proceeds:,.2f}")
                print(f"   - New Cash Balance: ${self.portfolio['cash']:,.2f}")

            elif signal == "HOLD":
                print(f"⏸️ HOLDING POSITION:")
                print(f"   - No trades executed based on current signals")

            # Calculate updated portfolio value
            new_position = self.portfolio["positions"].get(self.symbol, 0)
            new_portfolio_value = self.portfolio["cash"] + (
                new_position * current_price
            )

            portfolio_summary = {
                "pre_trade_value": current_portfolio_value,
                "post_trade_value": new_portfolio_value,
                "cash_balance": self.portfolio["cash"],
                "stock_position": new_position,
                "stock_value": new_position * current_price,
                "trade_executed": trade_executed,
                "trade_details": trade_details if trade_executed else None,
                "timestamp": datetime.now().isoformat(),
            }

            print(f"\n📈 Updated Portfolio Summary:")
            print(f"   - Total Value: ${new_portfolio_value:,.2f}")
            print(
                f"   - Cash: ${self.portfolio['cash']:,.2f} ({self.portfolio['cash']/new_portfolio_value:.1%})"
            )
            print(
                f"   - Stocks: ${new_position * current_price:,.2f} ({(new_position * current_price)/new_portfolio_value:.1%})"
            )

            return portfolio_summary

        except Exception as e:
            print(f"❌ Portfolio management failed: {e}")
            return None

    async def run_complete_demo(self):
        """Run the complete end-to-end demo"""
        print("🚀 TFT TRADING SYSTEM - COMPLETE END-TO-END DEMO")
        print("=" * 60)
        print(f"📅 Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Target Symbol: {self.symbol}")
        print(f"💰 Starting Portfolio: ${self.portfolio['cash']:,.2f}")

        # Execute all steps
        results = {}

        # Step 1: Data Ingestion
        results["data_ingestion"] = self.step1_data_ingestion()
        time.sleep(1)

        # Step 2: Sentiment Analysis
        results["sentiment_analysis"] = self.step2_sentiment_analysis()
        time.sleep(1)

        # Step 3: Feature Engineering
        results["feature_engineering"] = self.step3_feature_engineering()
        time.sleep(1)

        # Step 4: TFT Prediction
        results["tft_prediction"] = self.step4_tft_prediction()
        time.sleep(1)

        # Step 5: Trading Signals
        results["trading_signals"] = self.step5_trading_signals()
        time.sleep(1)

        # Step 6: Portfolio Management
        results["portfolio_management"] = self.step6_portfolio_management()

        # Final summary
        print(f"\n🏁 DEMO COMPLETE - FINAL SUMMARY")
        print("=" * 50)

        if results["portfolio_management"]:
            final_value = results["portfolio_management"]["post_trade_value"]
            initial_value = 10000
            total_return = ((final_value - initial_value) / initial_value) * 100

            print(f"💼 Portfolio Performance:")
            print(f"   - Initial Value: ${initial_value:,.2f}")
            print(f"   - Final Value: ${final_value:,.2f}")
            print(f"   - Total Return: {total_return:+.2f}%")

            if (
                results["trading_signals"]
                and results["trading_signals"]["signal"] != "HOLD"
            ):
                expected_return = results["trading_signals"]["expected_return"]
                print(f"   - Expected Return: {expected_return:+.2f}%")

        return results


# Run the demo
if __name__ == "__main__":
    demo = LocalTFTDemo()

    # Install required package if not available
    try:
        import yfinance
    except ImportError:
        import subprocess

        print("Installing yfinance...")
        subprocess.check_call(["pip", "install", "yfinance"])
        import yfinance

    # Run async demo
    asyncio.run(demo.run_complete_demo())
