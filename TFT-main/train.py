#!/usr/bin/env python3
"""
Main TFT Training Script
Complete pipeline for training the Temporal Fusion Transformer
"""

import argparse
import logging
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

# Local imports
from data_pipeline import DataPipeline
from data_preprocessing import StockDataPreprocessor
from tft_model import TFTTrainingPipeline
from stock_ranking import StockRankingSystem, PortfolioConstructor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""

    parser = argparse.ArgumentParser(
        description="TFT Stock Prediction Training Pipeline"
    )

    # Data arguments
    parser.add_argument(
        "--data-source",
        choices=["db", "csv", "api"],
        default="db",
        help="Data source for training",
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
        help="Specific symbols to train on (default: S&P 500)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Training data start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Training data end date (YYYY-MM-DD, default: today)",
    )

    # Model arguments
    parser.add_argument(
        "--target-type",
        choices=["returns", "classification", "quintile"],
        default="returns",
        help="Type of target variable",
    )
    parser.add_argument(
        "--target-horizon", type=int, default=1, help="Prediction horizon in days"
    )
    parser.add_argument(
        "--max-encoder-length",
        type=int,
        default=63,
        help="Maximum encoder length (lookback period)",
    )
    parser.add_argument(
        "--max-prediction-length",
        type=int,
        default=5,
        help="Maximum prediction length (forecast horizon)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="Maximum training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=64, help="Hidden size for TFT"
    )

    # Training arguments
    parser.add_argument(
        "--validation-split", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--optimize-hyperparams",
        action="store_true",
        help="Run hyperparameter optimization",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of hyperparameter optimization trials",
    )
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for training")

    # Output arguments
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/tft_model.pth",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory for results"
    )
    parser.add_argument(
        "--generate-predictions",
        action="store_true",
        help="Generate predictions after training",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info("=== TFT Stock Prediction Training Pipeline ===")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # Step 1: Load Data
        logger.info("Step 1: Loading data...")

        if args.data_source == "db":
            pipeline = DataPipeline(args.data_path)
            df = pipeline.collector.load_data_from_db(
                symbols=args.symbols, start_date=args.start_date, end_date=args.end_date
            )
        elif args.data_source == "csv":
            df = pd.read_csv(args.data_path)
        elif args.data_source == "api":
            # Fetch fresh data from API
            pipeline = DataPipeline()
            symbols = args.symbols or pipeline.collector.get_sp500_symbols()
            df = pipeline.collector.fetch_stock_data(
                symbols=symbols, start_date=args.start_date, end_date=args.end_date
            )

        if df.empty:
            raise ValueError(
                "No data loaded. Please check your data source and parameters."
            )

        logger.info(f"Loaded dataset shape: {df.shape}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Number of symbols: {df['symbol'].nunique()}")

        # Step 2: Preprocess Data
        logger.info("Step 2: Preprocessing data...")

        preprocessor = StockDataPreprocessor()
        processed_df = preprocessor.fit_transform(
            df, target_type=args.target_type, target_horizon=args.target_horizon
        )

        logger.info(f"Processed dataset shape: {processed_df.shape}")
        logger.info(f"Features: {list(processed_df.columns)}")

        # Save preprocessor
        import pickle

        preprocessor_path = output_dir / "preprocessor.pkl"
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)
        logger.info(f"Preprocessor saved to {preprocessor_path}")

        # Step 3: Configure Model
        logger.info("Step 3: Configuring model...")

        model_config = {
            "max_encoder_length": args.max_encoder_length,
            "max_prediction_length": args.max_prediction_length,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "max_epochs": args.max_epochs,
            "target_type": args.target_type,
            "target_horizon": args.target_horizon,
        }

        # Step 4: Train Model
        logger.info("Step 4: Training model...")

        training_pipeline = TFTTrainingPipeline(model_config)
        model = training_pipeline.run_pipeline(
            processed_df,
            validation_split=args.validation_split,
            optimize_hyperparams=args.optimize_hyperparams,
            n_trials=args.n_trials,
        )

        # Step 5: Save Model
        logger.info("Step 5: Saving model...")

        model_dir = Path(args.model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        model.save_model(args.model_path)
        logger.info(f"Model saved to {args.model_path}")

        # Save model config
        config_path = output_dir / "model_config.json"
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)

        # Step 6: Generate Predictions (Optional)
        if args.generate_predictions:
            logger.info("Step 6: Generating predictions...")

            # Initialize ranking system
            ranking_system = StockRankingSystem()
            portfolio_constructor = PortfolioConstructor()

            # Generate predictions on validation set
            predictions = model.predict(model.validation_dataset)

            # Get unique symbols from validation set
            unique_symbols = processed_df["symbol"].unique()

            # Process predictions
            predictions_df = ranking_system.process_predictions(
                predictions, unique_symbols, prediction_type="quantile"
            )

            # Generate signals
            signals = ranking_system.generate_trading_signals(
                predictions_df, method="quintile"
            )

            # Construct portfolio
            portfolio = portfolio_constructor.construct_portfolio(signals)

            # Save results
            predictions_path = output_dir / "predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)

            signals_path = output_dir / "signals.json"
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

            with open(signals_path, "w") as f:
                json.dump(signals_data, f, indent=2)

            portfolio_path = output_dir / "portfolio.json"
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

            with open(portfolio_path, "w") as f:
                json.dump(portfolio_data, f, indent=2)

            logger.info(f"Predictions saved to {predictions_path}")
            logger.info(f"Signals saved to {signals_path}")
            logger.info(f"Portfolio saved to {portfolio_path}")

            # Print summary
            logger.info(
                f"Generated {len(signals['long'])} long signals and {len(signals['short'])} short signals"
            )
            logger.info(
                f"Portfolio expected return: {portfolio['portfolio_stats'].get('net_expected_return', 'N/A')}"
            )

        # Step 7: Training Summary
        logger.info("=== Training Summary ===")
        logger.info(f"Dataset: {df.shape[0]} records, {df['symbol'].nunique()} symbols")
        logger.info(f"Training period: {args.start_date} to {args.end_date or 'today'}")
        logger.info(f"Model config: {model_config}")
        logger.info(f"Model saved: {args.model_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("Training completed successfully!")

        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
