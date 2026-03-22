"""
Automated Training and Prediction Scheduler
"""

"""
# COPILOT PROMPT: Create daily job to:
# 1. Fetch all S&P 500 symbols from Polygon reference API
# 2. Update OHLCV data in parallel threads (10 concurrent)
# 3. Validate corporate action adjustments using splits/dividends API
# EXPECTED OUTPUT: Robust batch processing system with parallel execution
# POLYGON INTEGRATION: Reference API, parallel processing, corporate actions
"""


import schedule
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any

# Local imports
from data_pipeline import DataPipeline
from data_preprocessing import StockDataPreprocessor
from tft_model import TFTTrainingPipeline
from stock_ranking import StockRankingSystem, PortfolioConstructor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/scheduler.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class AutomatedTFTSystem:
    """
    Automated TFT system for scheduled training and prediction
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()

        # Initialize components
        self.data_pipeline = DataPipeline(self.config["db_path"])

    def execute_live_trades(self):
        """
        ENHANCED COPILOT PROMPT: Professional live trading execution with Alpaca integration
        Implements institutional risk controls and compliance logging
        """
        logger.info("🚀 Starting live trading execution...")

        try:
            # Initialize Alpaca trading client with environment credentials
            import alpaca_trade_api as tradeapi
            import os

            # Connect to Alpaca API (paper trading for safety)
            api = tradeapi.REST(
                key_id=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY"),
                base_url="https://paper-api.alpaca.markets",  # Paper trading
            )

            # Fetch latest TFT predictions from API
            import requests

            predictions_response = requests.get(
                "http://localhost:8001/predict",
                json={"symbols": ["AAPL", "MSFT", "GOOGL", "TSLA"]},
            )
            predictions = predictions_response.json()

            # Get current portfolio value for position sizing
            account = api.get_account()
            portfolio_value = float(account.portfolio_value)
            max_position_size = portfolio_value * 0.05  # 5% max per position

            execution_summary = {
                "trades_executed": [],
                "total_value_traded": 0,
                "successful_orders": 0,
                "failed_orders": 0,
                "timestamp": datetime.now().isoformat(),
            }

            # Process each prediction signal
            for symbol, prediction_data in predictions.get("predictions", {}).items():
                try:
                    # Risk checks
                    confidence = prediction_data.get("confidence", 0)
                    predicted_return = prediction_data.get("predicted_return", 0)

                    # Skip low-confidence predictions
                    if confidence < 0.7:
                        logger.info(
                            f"Skipping {symbol}: Low confidence ({confidence:.2f})"
                        )
                        continue

                    # Check market conditions (simplified VIX check)
                    # In production, fetch real VIX data
                    market_volatility_ok = True  # Placeholder

                    if not market_volatility_ok:
                        logger.warning(f"Skipping {symbol}: High market volatility")
                        continue

                    # Calculate position size
                    current_price_response = api.get_latest_trade(symbol)
                    current_price = current_price_response.price
                    shares_to_trade = int(max_position_size / current_price)

                    if shares_to_trade < 1:
                        logger.info(
                            f"Skipping {symbol}: Insufficient funds for 1 share"
                        )
                        continue

                    # Determine trade direction
                    side = "buy" if predicted_return > 0 else "sell"

                    # Calculate risk management levels
                    stop_loss_price = (
                        current_price * 0.98 if side == "buy" else current_price * 1.02
                    )
                    take_profit_price = (
                        current_price * 1.04 if side == "buy" else current_price * 0.96
                    )

                    # Execute market order with OCO (One-Cancels-Other) for risk management
                    order = api.submit_order(
                        symbol=symbol,
                        qty=shares_to_trade,
                        side=side,
                        type="market",
                        time_in_force="day",
                        order_class="oco",
                        stop_loss={"stop_price": stop_loss_price},
                        take_profit={"limit_price": take_profit_price},
                    )

                    trade_info = {
                        "symbol": symbol,
                        "side": side,
                        "quantity": shares_to_trade,
                        "entry_price": current_price,
                        "stop_loss": stop_loss_price,
                        "take_profit": take_profit_price,
                        "confidence": confidence,
                        "predicted_return": predicted_return,
                        "order_id": order.id,
                    }

                    execution_summary["trades_executed"].append(trade_info)
                    execution_summary["total_value_traded"] += (
                        shares_to_trade * current_price
                    )
                    execution_summary["successful_orders"] += 1

                    logger.info(
                        f"✅ Executed {side.upper()} order for {shares_to_trade} shares of {symbol} at ${current_price:.2f}"
                    )

                except Exception as trade_error:
                    logger.error(
                        f"❌ Failed to execute trade for {symbol}: {trade_error}"
                    )
                    execution_summary["failed_orders"] += 1

            logger.info("✅ Live trading execution completed")
            logger.info(
                f"📊 Summary: {execution_summary['successful_orders']} successful, {execution_summary['failed_orders']} failed"
            )

            return execution_summary

        except Exception as e:
            logger.error(f"❌ Live trading execution failed: {e}")
            raise
            return execution_summary

        except Exception as e:
            logger.error(f"❌ Live trading execution failed: {e}")
            raise
        self.preprocessor = StockDataPreprocessor()
        self.model = None
        self.ranking_system = StockRankingSystem()
        self.portfolio_constructor = PortfolioConstructor()

        # Create necessary directories
        Path("logs").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("predictions").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)

    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for automated system"""
        return {
            "db_path": "data/stock_data.db",
            "model_path": "models/tft_model.pth",
            "symbols": None,  # Use S&P 500 if None
            "training_schedule": {
                "frequency": "weekly",  # daily, weekly, monthly
                "day": "sunday",
                "time": "02:00",
            },
            "prediction_schedule": {"frequency": "daily", "time": "06:00"},
            "data_update_schedule": {"frequency": "daily", "time": "18:00"},
            "model_config": {
                "max_encoder_length": 63,
                "max_prediction_length": 5,
                "batch_size": 64,
                "max_epochs": 50,
            },
            "retrain_threshold_days": 7,
            "prediction_horizon": 5,
        }

    def daily_data_update(self):
        """Run daily data update"""
        try:
            logger.info("Starting daily data update...")

            symbols = self.config["symbols"]
            records_updated = self.data_pipeline.run_daily_update(symbols)

            logger.info(
                f"Daily data update completed. {records_updated} records updated."
            )

            # Save update log
            update_log = {
                "timestamp": datetime.now().isoformat(),
                "records_updated": records_updated,
                "status": "success",
            }

            self._save_update_log(update_log)

        except Exception as e:
            logger.error(f"Daily data update failed: {e}")
            update_log = {
                "timestamp": datetime.now().isoformat(),
                "records_updated": 0,
                "status": "failed",
                "error": str(e),
            }
            self._save_update_log(update_log)

    def weekly_model_training(self):
        """Run weekly model training"""
        try:
            logger.info("Starting weekly model training...")

            # Load training data
            symbols = self.config["symbols"]
            df = self.data_pipeline.collector.load_data_from_db(
                symbols=symbols, start_date="2020-01-01"
            )

            if df.empty:
                logger.error("No training data available")
                return

            # Preprocess data
            logger.info("Preprocessing training data...")
            processed_df = self.preprocessor.fit_transform(
                df,
                target_type="returns",
                target_horizon=self.config["prediction_horizon"],
            )

            # Train model
            logger.info("Training TFT model...")
            pipeline = TFTTrainingPipeline(self.config["model_config"])

            self.model = pipeline.run_pipeline(
                processed_df,
                validation_split=0.2,
                optimize_hyperparams=False,  # Set to True for production
            )

            # Save model
            self.model.save_model(self.config["model_path"])

            logger.info("Weekly model training completed successfully")

            # Save training log
            training_log = {
                "timestamp": datetime.now().isoformat(),
                "data_shape": processed_df.shape,
                "training_config": self.config["model_config"],
                "status": "success",
            }

            self._save_training_log(training_log)

        except Exception as e:
            logger.error(f"Weekly model training failed: {e}")

            training_log = {
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e),
            }

            self._save_training_log(training_log)

    def daily_predictions(self):
        """Generate daily predictions"""
        try:
            logger.info("Starting daily predictions...")

            # Load model if not already loaded
            if self.model is None:
                from tft_model import EnhancedTFTModel

                self.model = EnhancedTFTModel()
                try:
                    self.model.load_model(self.config["model_path"])
                except FileNotFoundError:
                    logger.error("No trained model found. Please train model first.")
                    return

            # Load recent data for prediction
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")

            symbols = self.config["symbols"]
            df = self.data_pipeline.collector.load_data_from_db(
                symbols=symbols, start_date=start_date, end_date=end_date
            )

            if df.empty:
                logger.error("No data available for predictions")
                return

            # Preprocess data
            processed_df = self.preprocessor.transform(df)

            # Generate predictions
            logger.info("Generating predictions...")
            predictions = self.model.predict(self.model.validation_dataset)

            # Process predictions into signals
            unique_symbols = processed_df["symbol"].unique()
            predictions_df = self.ranking_system.process_predictions(
                predictions, unique_symbols, prediction_type="quantile"
            )

            # Generate trading signals
            signals = self.ranking_system.generate_trading_signals(
                predictions_df, method="quintile"
            )

            # Construct portfolio
            portfolio = self.portfolio_constructor.construct_portfolio(signals)

            # Save predictions
            self._save_predictions(signals, portfolio, predictions_df)

            logger.info(
                f"Daily predictions completed. Generated {len(signals['long'])} long and {len(signals['short'])} short signals."
            )

        except Exception as e:
            logger.error(f"Daily predictions failed: {e}")

    def generate_performance_report(self):
        """Generate weekly performance report"""
        try:
            logger.info("Generating performance report...")

            # Load recent predictions and actual returns
            # This would compare predictions vs actual performance
            # Implementation depends on your specific requirements

            report = {
                "timestamp": datetime.now().isoformat(),
                "period": "weekly",
                "metrics": {
                    "predictions_generated": 0,  # Would be calculated from logs
                    "accuracy": 0.0,
                    "returns": 0.0,
                    "sharpe_ratio": 0.0,
                },
            }

            # Save report
            report_file = (
                f"reports/performance_report_{datetime.now().strftime('%Y%m%d')}.json"
            )
            import json

            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Performance report saved to {report_file}")

        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")

    def _save_update_log(self, log_data: Dict):
        """Save data update log"""
        log_file = f"logs/data_updates_{datetime.now().strftime('%Y%m')}.json"
        self._append_to_json_log(log_file, log_data)

    def _save_training_log(self, log_data: Dict):
        """Save training log"""
        log_file = f"logs/training_{datetime.now().strftime('%Y%m')}.json"
        self._append_to_json_log(log_file, log_data)

    def _save_predictions(
        self, signals: Dict, portfolio: Dict, predictions_df: pd.DataFrame
    ):
        """Save daily predictions"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save signals
        signals_file = f"predictions/signals_{timestamp}.json"
        signals_data = {
            "timestamp": datetime.now().isoformat(),
            "long_signals": [
                {
                    "symbol": s.symbol,
                    "predicted_return": s.predicted_return,
                    "confidence": s.confidence,
                    "rank": s.rank,
                }
                for s in signals["long"]
            ],
            "short_signals": [
                {
                    "symbol": s.symbol,
                    "predicted_return": s.predicted_return,
                    "confidence": s.confidence,
                    "rank": s.rank,
                }
                for s in signals["short"]
            ],
        }

        import json

        with open(signals_file, "w") as f:
            json.dump(signals_data, f, indent=2)

        # Save portfolio
        portfolio_file = f"predictions/portfolio_{timestamp}.json"
        portfolio_data = {
            "timestamp": datetime.now().isoformat(),
            "long_portfolio": {
                symbol: {
                    "weight": pos["weight"],
                    "predicted_return": pos["predicted_return"],
                    "confidence": pos["confidence"],
                }
                for symbol, pos in portfolio["long_portfolio"].items()
            },
            "short_portfolio": {
                symbol: {
                    "weight": pos["weight"],
                    "predicted_return": pos["predicted_return"],
                    "confidence": pos["confidence"],
                }
                for symbol, pos in portfolio["short_portfolio"].items()
            },
            "stats": portfolio["portfolio_stats"],
        }

        with open(portfolio_file, "w") as f:
            json.dump(portfolio_data, f, indent=2)

        # Save raw predictions CSV
        predictions_csv = f"predictions/raw_predictions_{timestamp}.csv"
        predictions_df.to_csv(predictions_csv, index=False)

        logger.info(
            f"Predictions saved: {signals_file}, {portfolio_file}, {predictions_csv}"
        )

    def _append_to_json_log(self, log_file: str, log_data: Dict):
        """Append data to JSON log file"""
        import json

        # Read existing log
        try:
            with open(log_file, "r") as f:
                logs = json.load(f)
        except FileNotFoundError:
            logs = []

        # Append new log
        logs.append(log_data)

        # Write back
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)

    def setup_scheduler(self):
        """Setup automated scheduling"""
        logger.info("Setting up automated scheduler...")

        # Data update schedule
        if self.config["data_update_schedule"]["frequency"] == "daily":
            schedule.every().day.at(self.config["data_update_schedule"]["time"]).do(
                self.daily_data_update
            )

        # Training schedule
        if self.config["training_schedule"]["frequency"] == "weekly":
            day = self.config["training_schedule"]["day"]
            time = self.config["training_schedule"]["time"]
            getattr(schedule.every(), day.lower()).at(time).do(
                self.weekly_model_training
            )

        # Prediction schedule
        if self.config["prediction_schedule"]["frequency"] == "daily":
            schedule.every().day.at(self.config["prediction_schedule"]["time"]).do(
                self.daily_predictions
            )

        # Performance report (weekly on Sundays)
        schedule.every().sunday.at("08:00").do(self.generate_performance_report)

        logger.info("Scheduler setup completed")
        logger.info("Scheduled jobs:")
        for job in schedule.jobs:
            logger.info(f"  - {job}")

    def run_scheduler(self):
        """Run the automated scheduler"""
        logger.info("Starting automated TFT system...")

        self.setup_scheduler()

        logger.info("Scheduler is running. Press Ctrl+C to stop.")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")

    def run_manual_task(self, task: str):
        """Run a specific task manually"""
        tasks = {
            "data_update": self.daily_data_update,
            "training": self.weekly_model_training,
            "predictions": self.daily_predictions,
            "report": self.generate_performance_report,
        }

        if task in tasks:
            logger.info(f"Running manual task: {task}")
            tasks[task]()
        else:
            logger.error(f"Unknown task: {task}. Available tasks: {list(tasks.keys())}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated TFT Stock Prediction System"
    )
    parser.add_argument(
        "--mode",
        choices=["scheduler", "manual"],
        default="scheduler",
        help="Run mode: scheduler (automated) or manual (one-time task)",
    )
    parser.add_argument(
        "--task",
        choices=["data_update", "training", "predictions", "report"],
        help="Manual task to run (only for manual mode)",
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        import json

        with open(args.config, "r") as f:
            config = json.load(f)

    # Initialize system
    system = AutomatedTFTSystem(config)

    if args.mode == "scheduler":
        system.run_scheduler()
    elif args.mode == "manual":
        if args.task:
            system.run_manual_task(args.task)
        else:
            print("Manual mode requires --task argument")
