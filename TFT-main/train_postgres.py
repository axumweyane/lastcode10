"""
PostgreSQL-based Training Script for TFT Stock Prediction System
Enhanced training script that uses PostgreSQL as the data source
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import with fallback
try:
    from tft_postgres_model import TFTPostgresModel
    from postgres_data_pipeline import PostgresDataPipeline
    from postgres_data_loader import PostgresDataLoader

    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML libraries not available: {e}")
    ML_AVAILABLE = False

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/training_postgres.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories"""
    directories = ["models", "logs", "reports", "data", "output"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info("Directories created")


def load_config():
    """Load configuration from environment variables"""
    load_dotenv()

    db_config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "database": os.getenv("POSTGRES_DB", "stock_trading_analysis"),
        "user": os.getenv("POSTGRES_USER", "trading_user"),
        "password": os.environ["POSTGRES_PASSWORD"],
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "schema": os.getenv("POSTGRES_SCHEMA", "public"),
    }

    model_config = {
        "max_encoder_length": int(os.getenv("TFT_ENCODER_LENGTH", 30)),
        "max_prediction_length": int(os.getenv("TFT_PREDICTION_LENGTH", 5)),
        "batch_size": int(os.getenv("TFT_BATCH_SIZE", 64)),
        "learning_rate": float(os.getenv("TFT_LEARNING_RATE", 0.001)),
        "hidden_size": int(os.getenv("TFT_HIDDEN_SIZE", 64)),
        "attention_head_size": int(os.getenv("TFT_ATTENTION_HEADS", 4)),
        "dropout": float(os.getenv("TFT_DROPOUT", 0.2)),
        "max_epochs": int(os.getenv("TFT_MAX_EPOCHS", 100)),
        "early_stopping_patience": int(os.getenv("TFT_EARLY_STOPPING", 10)),
        "validation_split": float(os.getenv("TFT_VALIDATION_SPLIT", 0.2)),
        "target_type": "returns",
        "prediction_horizon": 5,
        "hidden_continuous_size": 32,
        "lstm_layers": 2,
        "quantiles": [0.1, 0.5, 0.9],
    }

    return db_config, model_config


def get_symbols_from_db(db_config, limit=None):
    """Get available symbols from database"""
    try:
        loader = PostgresDataLoader(db_config)
        symbols = loader.get_available_symbols()

        if limit:
            symbols = symbols[:limit]

        logger.info(f"Found {len(symbols)} symbols in database")
        return symbols
    except Exception as e:
        logger.error(f"Failed to get symbols from database: {e}")
        return []


def validate_data_quality(db_config, symbols, start_date, end_date):
    """Validate data quality before training"""
    logger.info("Validating data quality...")

    try:
        loader = PostgresDataLoader(db_config)
        validation_results = loader.validate_data_quality(symbols, start_date, end_date)

        # Log validation results
        good_symbols = []
        for symbol, metrics in validation_results.items():
            completeness = metrics["data_completeness"]
            record_count = metrics["total_records"]

            if (
                completeness > 0.9 and record_count > 100
            ):  # 90% completeness, min 100 records
                good_symbols.append(symbol)
                logger.info(
                    f"{symbol}: {completeness:.2%} complete, {record_count} records"
                )
            else:
                logger.warning(
                    f"{symbol}: {completeness:.2%} complete, {record_count} records - EXCLUDED"
                )

        logger.info(
            f"Data validation: {len(good_symbols)}/{len(symbols)} symbols passed quality checks"
        )
        return good_symbols

    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return symbols  # Return original list if validation fails


def train_model(args):
    """Main training function"""
    if not ML_AVAILABLE:
        logger.error("ML libraries not available. Please install required packages.")
        return False

    # Load configuration
    db_config, model_config = load_config()

    # Update model config with command line arguments
    if args.target_type:
        model_config["target_type"] = args.target_type
    if args.prediction_horizon:
        model_config["prediction_horizon"] = args.prediction_horizon
    if args.max_epochs:
        model_config["max_epochs"] = args.max_epochs
    if args.batch_size:
        model_config["batch_size"] = args.batch_size
    if args.learning_rate:
        model_config["learning_rate"] = args.learning_rate

    logger.info(f"Model configuration: {model_config}")

    # Get symbols
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = get_symbols_from_db(db_config, limit=args.symbol_limit)

    if not symbols:
        logger.error("No symbols available for training")
        return False

    logger.info(f"Training with {len(symbols)} symbols: {symbols}")

    # Set date range
    if args.end_date:
        end_date = args.end_date
    else:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if args.start_date:
        start_date = args.start_date
    else:
        # Default to 2 years of data
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    logger.info(f"Training date range: {start_date} to {end_date}")

    # Validate data quality
    if args.validate_data:
        symbols = validate_data_quality(db_config, symbols, start_date, end_date)
        if not symbols:
            logger.error("No symbols passed data quality validation")
            return False

    try:
        # Initialize TFT model
        logger.info("Initializing TFT model...")
        tft_model = TFTPostgresModel(db_config, model_config)

        # Train model
        logger.info("Starting model training...")
        training_results = tft_model.train(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            optimize_hyperparams=args.optimize_hyperparams,
            n_trials=args.n_trials,
            include_fundamentals=args.include_fundamentals,
            include_sentiment=args.include_sentiment,
            include_earnings=args.include_earnings,
        )

        # Save training results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"reports/training_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(
                {
                    "training_results": training_results,
                    "config": model_config,
                    "symbols": symbols,
                    "date_range": {"start": start_date, "end": end_date},
                    "args": vars(args),
                },
                f,
                indent=2,
            )

        logger.info(f"Training results saved to {results_file}")

        # Generate predictions if requested
        if args.generate_predictions:
            logger.info("Generating predictions with trained model...")

            prediction_date = end_date
            predictions = tft_model.predict(
                symbols=symbols, prediction_date=prediction_date, lookback_days=90
            )

            # Save predictions
            predictions_file = f"output/predictions_{timestamp}.json"
            with open(predictions_file, "w") as f:
                json.dump(predictions, f, indent=2)

            logger.info(f"Predictions saved to {predictions_file}")

        # Evaluate model if requested
        if args.run_evaluation:
            logger.info("Running model evaluation...")

            # Use last 3 months for evaluation
            eval_start = (
                datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=90)
            ).strftime("%Y-%m-%d")

            metrics = tft_model.evaluate(
                symbols=symbols, start_date=eval_start, end_date=end_date
            )

            # Save evaluation results
            eval_file = f"reports/evaluation_{timestamp}.json"
            with open(eval_file, "w") as f:
                json.dump(
                    {
                        "evaluation_metrics": metrics,
                        "eval_period": {"start": eval_start, "end": end_date},
                        "symbols": symbols,
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Evaluation results saved to {eval_file}")
            logger.info(f"Key metrics: {metrics}")

        logger.info("Training completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train TFT model using PostgreSQL data"
    )

    # Data arguments
    parser.add_argument(
        "--symbols", nargs="+", help="List of stock symbols to train on"
    )
    parser.add_argument(
        "--symbol-limit",
        type=int,
        default=50,
        help="Limit number of symbols from database",
    )
    parser.add_argument(
        "--start-date", type=str, help="Training start date (YYYY-MM-DD)"
    )
    parser.add_argument("--end-date", type=str, help="Training end date (YYYY-MM-DD)")
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Validate data quality before training",
    )

    # Model arguments
    parser.add_argument(
        "--target-type",
        type=str,
        choices=["returns", "classification", "quintile"],
        default="returns",
        help="Type of target variable",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=5,
        help="Number of days to predict ahead",
    )
    parser.add_argument("--max-epochs", type=int, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")

    # Feature arguments
    parser.add_argument(
        "--include-fundamentals",
        action="store_true",
        default=True,
        help="Include fundamental data",
    )
    parser.add_argument(
        "--include-sentiment",
        action="store_true",
        default=True,
        help="Include sentiment data",
    )
    parser.add_argument(
        "--include-earnings",
        action="store_true",
        default=True,
        help="Include earnings calendar",
    )

    # Optimization arguments
    parser.add_argument(
        "--optimize-hyperparams", action="store_true", help="Optimize hyperparameters"
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Number of optimization trials"
    )

    # Output arguments
    parser.add_argument(
        "--generate-predictions",
        action="store_true",
        help="Generate predictions after training",
    )
    parser.add_argument(
        "--run-evaluation",
        action="store_true",
        help="Run model evaluation after training",
    )

    # System arguments
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create directories
    setup_directories()

    # Log startup information
    logger.info("=" * 60)
    logger.info("TFT PostgreSQL Model Training")
    logger.info("=" * 60)
    logger.info(f"Arguments: {vars(args)}")

    # Check ML libraries
    if not ML_AVAILABLE:
        logger.error("ML libraries not available. Install required packages:")
        logger.error(
            "pip install torch pytorch-lightning pytorch-forecasting scikit-learn"
        )
        return

    # Run training
    success = train_model(args)

    if success:
        logger.info("Training completed successfully!")
        sys.exit(0)
    else:
        logger.error("Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
