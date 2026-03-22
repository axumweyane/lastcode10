#!/usr/bin/env python3
"""
Prediction Script for Trained TFT Model
Generate predictions using a trained model
"""

import argparse
import logging
import json
import pickle
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Local imports
from data_pipeline import DataPipeline
from data_preprocessing import StockDataPreprocessor
from tft_model import EnhancedTFTModel
from stock_ranking import StockRankingSystem, PortfolioConstructor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main prediction pipeline"""

    parser = argparse.ArgumentParser(description="Generate TFT Stock Predictions")

    # Model arguments
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--preprocessor-path",
        type=str,
        help="Path to preprocessor (default: same dir as model)",
    )

    # Data arguments
    parser.add_argument(
        "--data-source",
        choices=["db", "csv", "api"],
        default="db",
        help="Data source for predictions",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/stock_data.db",
        help="Path to data file or database",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Specific symbols to predict (default: all available)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=100,
        help="Number of days to look back for features",
    )

    # Prediction arguments
    parser.add_argument(
        "--prediction-method",
        choices=["quintile", "threshold", "top_bottom"],
        default="quintile",
        help="Signal generation method",
    )
    parser.add_argument(
        "--max-positions", type=int, default=20, help="Maximum positions per side"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum confidence threshold for signals",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="predictions",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "csv", "both"],
        default="both",
        help="Output format for predictions",
    )
    parser.add_argument(
        "--include-portfolio",
        action="store_true",
        help="Generate portfolio construction",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    log_file = (
        output_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info("=== TFT Stock Prediction Generation ===")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # Step 1: Load Model
        logger.info("Step 1: Loading trained model...")

        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        model = EnhancedTFTModel()
        model.load_model(str(model_path))
        logger.info(f"Model loaded from {model_path}")

        # Step 2: Load Preprocessor
        logger.info("Step 2: Loading preprocessor...")

        if args.preprocessor_path:
            preprocessor_path = Path(args.preprocessor_path)
        else:
            preprocessor_path = model_path.parent / "preprocessor.pkl"

        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")

        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)
        logger.info(f"Preprocessor loaded from {preprocessor_path}")

        # Step 3: Load Data
        logger.info("Step 3: Loading prediction data...")

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.lookback_days)).strftime(
            "%Y-%m-%d"
        )

        if args.data_source == "db":
            pipeline = DataPipeline(args.data_path)
            df = pipeline.collector.load_data_from_db(
                symbols=args.symbols, start_date=start_date, end_date=end_date
            )
        elif args.data_source == "csv":
            df = pd.read_csv(args.data_path)
            # Filter by date and symbols if specified
            if args.symbols:
                df = df[df["symbol"].isin(args.symbols)]
            df = df[df["timestamp"] >= start_date]
        elif args.data_source == "api":
            pipeline = DataPipeline()
            symbols = (
                args.symbols or pipeline.collector.get_sp500_symbols()[:50]
            )  # Limit for demo
            df = pipeline.collector.fetch_stock_data(
                symbols=symbols, start_date=start_date, end_date=end_date
            )

        if df.empty:
            raise ValueError("No data loaded for prediction")

        logger.info(f"Loaded dataset shape: {df.shape}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Number of symbols: {df['symbol'].nunique()}")

        # Step 4: Preprocess Data
        logger.info("Step 4: Preprocessing data...")

        processed_df = preprocessor.transform(df)
        logger.info(f"Processed dataset shape: {processed_df.shape}")

        # Step 5: Generate Predictions
        logger.info("Step 5: Generating predictions...")

        # Use the validation dataset structure for prediction
        if model.validation_dataset is None:
            raise ValueError("Model validation dataset not available")

        predictions = model.predict(model.validation_dataset, mode="prediction")
        logger.info(f"Generated predictions shape: {predictions.shape}")

        # Step 6: Process Predictions into Signals
        logger.info("Step 6: Processing predictions into trading signals...")

        ranking_system = StockRankingSystem(
            max_positions=args.max_positions,
            confidence_threshold=args.confidence_threshold,
        )

        unique_symbols = processed_df["symbol"].unique()

        # Convert predictions to DataFrame
        predictions_df = ranking_system.process_predictions(
            predictions, unique_symbols, prediction_type="quantile"
        )

        # Apply liquidity filter if volume data is available
        liquidity_filter = None
        if "volume" in processed_df.columns:
            liquidity_filter = ranking_system.calculate_liquidity_filter(processed_df)

        # Generate trading signals
        signals = ranking_system.generate_trading_signals(
            predictions_df,
            liquidity_filter=liquidity_filter,
            method=args.prediction_method,
        )

        logger.info(f"Generated {len(signals['long'])} long signals")
        logger.info(f"Generated {len(signals['short'])} short signals")

        # Step 7: Portfolio Construction (Optional)
        portfolio = None
        if args.include_portfolio:
            logger.info("Step 7: Constructing portfolio...")

            portfolio_constructor = PortfolioConstructor()
            portfolio = portfolio_constructor.construct_portfolio(signals)

            logger.info(f"Portfolio stats: {portfolio['portfolio_stats']}")

        # Step 8: Save Results
        logger.info("Step 8: Saving results...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw predictions
        if args.output_format in ["csv", "both"]:
            predictions_csv = output_dir / f"raw_predictions_{timestamp}.csv"
            predictions_df.to_csv(predictions_csv, index=False)
            logger.info(f"Raw predictions saved to {predictions_csv}")

        # Save signals
        signals_data = {
            "timestamp": datetime.now().isoformat(),
            "model_path": str(model_path),
            "prediction_method": args.prediction_method,
            "parameters": {
                "max_positions": args.max_positions,
                "confidence_threshold": args.confidence_threshold,
                "lookback_days": args.lookback_days,
            },
            "data_info": {
                "symbols_count": df["symbol"].nunique(),
                "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                "total_records": len(df),
            },
            "long_signals": [
                {
                    "symbol": s.symbol,
                    "predicted_return": float(s.predicted_return),
                    "confidence": float(s.confidence),
                    "rank": s.rank,
                    "signal_strength": s.signal_strength,
                }
                for s in signals["long"]
            ],
            "short_signals": [
                {
                    "symbol": s.symbol,
                    "predicted_return": float(s.predicted_return),
                    "confidence": float(s.confidence),
                    "rank": s.rank,
                    "signal_strength": s.signal_strength,
                }
                for s in signals["short"]
            ],
            "neutral_signals": [
                {
                    "symbol": s.symbol,
                    "predicted_return": float(s.predicted_return),
                    "confidence": float(s.confidence),
                    "rank": s.rank,
                    "signal_strength": s.signal_strength,
                }
                for s in signals["neutral"]
            ],
        }

        if args.output_format in ["json", "both"]:
            signals_json = output_dir / f"signals_{timestamp}.json"
            with open(signals_json, "w") as f:
                json.dump(signals_data, f, indent=2)
            logger.info(f"Signals saved to {signals_json}")

        if args.output_format in ["csv", "both"]:
            # Convert signals to CSV format
            all_signals = []
            for signal_type, signal_list in signals.items():
                for signal in signal_list:
                    all_signals.append(
                        {
                            "symbol": signal.symbol,
                            "signal_type": signal_type,
                            "predicted_return": signal.predicted_return,
                            "confidence": signal.confidence,
                            "rank": signal.rank,
                            "signal_strength": signal.signal_strength,
                            "timestamp": signal.timestamp.isoformat(),
                        }
                    )

            signals_csv = output_dir / f"signals_{timestamp}.csv"
            pd.DataFrame(all_signals).to_csv(signals_csv, index=False)
            logger.info(f"Signals CSV saved to {signals_csv}")

        # Save portfolio if generated
        if portfolio:
            portfolio_data = {
                "timestamp": datetime.now().isoformat(),
                "long_portfolio": {
                    symbol: {
                        "weight": float(pos["weight"]),
                        "predicted_return": float(pos["predicted_return"]),
                        "confidence": float(pos["confidence"]),
                        "rank": pos["rank"],
                        "signal_strength": pos["signal_strength"],
                        "sector": pos["sector"],
                    }
                    for symbol, pos in portfolio["long_portfolio"].items()
                },
                "short_portfolio": {
                    symbol: {
                        "weight": float(pos["weight"]),
                        "predicted_return": float(pos["predicted_return"]),
                        "confidence": float(pos["confidence"]),
                        "rank": pos["rank"],
                        "signal_strength": pos["signal_strength"],
                        "sector": pos["sector"],
                    }
                    for symbol, pos in portfolio["short_portfolio"].items()
                },
                "portfolio_stats": {
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in portfolio["portfolio_stats"].items()
                },
            }

            if args.output_format in ["json", "both"]:
                portfolio_json = output_dir / f"portfolio_{timestamp}.json"
                with open(portfolio_json, "w") as f:
                    json.dump(portfolio_data, f, indent=2)
                logger.info(f"Portfolio saved to {portfolio_json}")

            if args.output_format in ["csv", "both"]:
                # Convert portfolio to CSV
                portfolio_records = []
                for side in ["long_portfolio", "short_portfolio"]:
                    for symbol, pos in portfolio_data[side].items():
                        portfolio_records.append(
                            {
                                "symbol": symbol,
                                "side": side.replace("_portfolio", ""),
                                **pos,
                            }
                        )

                portfolio_csv = output_dir / f"portfolio_{timestamp}.csv"
                pd.DataFrame(portfolio_records).to_csv(portfolio_csv, index=False)
                logger.info(f"Portfolio CSV saved to {portfolio_csv}")

        # Step 9: Generate Summary Report
        logger.info("=== Prediction Summary ===")
        logger.info(f"Model: {model_path}")
        logger.info(f"Data period: {start_date} to {end_date}")
        logger.info(f"Symbols analyzed: {df['symbol'].nunique()}")
        logger.info(f"Long signals: {len(signals['long'])}")
        logger.info(f"Short signals: {len(signals['short'])}")
        logger.info(f"Neutral signals: {len(signals['neutral'])}")

        if portfolio:
            stats = portfolio["portfolio_stats"]
            logger.info(
                f"Portfolio expected return: {stats.get('net_expected_return', 'N/A'):.4f}"
            )
            logger.info(f"Total positions: {stats.get('total_positions', 'N/A')}")
            logger.info(f"Average confidence: {stats.get('avg_confidence', 'N/A'):.3f}")

        logger.info(f"Results saved to: {output_dir}")
        logger.info("Prediction generation completed successfully!")

        # Print top signals
        if signals["long"]:
            logger.info("\nTop 5 Long Signals:")
            for i, signal in enumerate(signals["long"][:5]):
                logger.info(
                    f"  {i+1}. {signal.symbol}: {signal.predicted_return:.4f} (confidence: {signal.confidence:.3f})"
                )

        if signals["short"]:
            logger.info("\nTop 5 Short Signals:")
            for i, signal in enumerate(signals["short"][:5]):
                logger.info(
                    f"  {i+1}. {signal.symbol}: {signal.predicted_return:.4f} (confidence: {signal.confidence:.3f})"
                )

        return True

    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
