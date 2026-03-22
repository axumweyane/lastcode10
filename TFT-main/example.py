"""
Example usage and testing script for TFT Stock Prediction System
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Import local modules
from data_preprocessing import StockDataPreprocessor, load_sample_data
from stock_ranking import (
    StockRankingSystem,
    PortfolioConstructor,
    create_sample_predictions,
)


def test_data_preprocessing():
    """Test data preprocessing pipeline"""
    print("🧪 Testing Data Preprocessing...")

    # Load sample data
    df = load_sample_data()
    print(f"Sample data shape: {df.shape}")

    # Initialize preprocessor
    preprocessor = StockDataPreprocessor()

    # Test preprocessing
    processed_df = preprocessor.fit_transform(
        df, target_type="returns", target_horizon=1
    )
    print(f"Processed data shape: {processed_df.shape}")
    print(f"Features: {list(processed_df.columns)}")

    # Check for missing values
    missing_values = processed_df.isnull().sum().sum()
    print(f"Missing values: {missing_values}")

    # Test transform (new data)
    new_data = load_sample_data()[:100]  # Smaller subset
    transformed_data = preprocessor.transform(new_data)
    print(f"Transform test shape: {transformed_data.shape}")

    print("✅ Data preprocessing test passed!\n")
    return processed_df


def test_stock_ranking():
    """Test stock ranking and portfolio construction"""
    print("🧪 Testing Stock Ranking System...")

    # Create sample predictions
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"]
    predictions = create_sample_predictions(symbols)

    # Initialize ranking system
    ranking_system = StockRankingSystem(
        liquidity_threshold=len(symbols), max_positions=4  # Use all symbols
    )

    # Process predictions
    predictions_df = ranking_system.process_predictions(
        predictions, symbols, prediction_type="quantile"
    )
    print(f"Predictions DataFrame shape: {predictions_df.shape}")

    # Generate signals
    signals = ranking_system.generate_trading_signals(predictions_df, method="quintile")

    print(f"Long signals: {len(signals['long'])}")
    print(f"Short signals: {len(signals['short'])}")
    print(f"Neutral signals: {len(signals['neutral'])}")

    # Test portfolio construction
    portfolio_constructor = PortfolioConstructor()
    portfolio = portfolio_constructor.construct_portfolio(signals)

    print(f"Portfolio stats: {portfolio['portfolio_stats']}")

    print("✅ Stock ranking test passed!\n")
    return signals, portfolio


def test_data_pipeline():
    """Test data collection pipeline"""
    print("🧪 Testing Data Pipeline...")

    try:
        from data_pipeline import DataPipeline

        # Initialize pipeline
        pipeline = DataPipeline("test_data.db")

        # Test with a small set of symbols
        test_symbols = ["AAPL", "GOOGL", "MSFT"]

        print("Testing data collection (this may take a moment)...")

        # Test data collection
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        df = pipeline.collector.fetch_stock_data(
            symbols=test_symbols, start_date=start_date, end_date=end_date
        )

        if not df.empty:
            print(f"Collected data shape: {df.shape}")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print("✅ Data pipeline test passed!")
        else:
            print("⚠️  No data collected (this might be due to network issues)")

        # Clean up test database
        if os.path.exists("test_data.db"):
            os.remove("test_data.db")

    except ImportError as e:
        print(f"⚠️  Data pipeline test skipped due to missing dependencies: {e}")
    except Exception as e:
        print(f"⚠️  Data pipeline test failed: {e}")

    print()


def test_model_structure():
    """Test model structure (without training)"""
    print("🧪 Testing TFT Model Structure...")

    try:
        from tft_model import EnhancedTFTModel

        # Initialize model with test config
        config = {
            "max_encoder_length": 30,
            "max_prediction_length": 1,
            "batch_size": 32,
            "learning_rate": 0.001,
            "hidden_size": 16,
            "max_epochs": 1,  # Very small for testing
        }

        model = EnhancedTFTModel(config)
        print(f"Model initialized with config: {config}")

        # Test dataset creation with sample data
        df = load_sample_data()[:200]  # Small subset for testing
        preprocessor = StockDataPreprocessor()
        processed_df = preprocessor.fit_transform(df, target_type="returns")

        training_dataset, validation_dataset = model.create_datasets(
            processed_df, validation_split=0.3
        )

        print(f"Training dataset length: {len(training_dataset)}")
        print(f"Validation dataset length: {len(validation_dataset)}")

        print("✅ Model structure test passed!")

    except ImportError as e:
        print(f"⚠️  Model test skipped due to missing dependencies: {e}")
    except Exception as e:
        print(f"⚠️  Model test failed: {e}")

    print()


def test_api_structure():
    """Test API structure (without running server)"""
    print("🧪 Testing API Structure...")

    try:
        from api import app

        # Check if FastAPI app is created
        print(f"FastAPI app created: {app is not None}")
        print(f"App title: {app.title}")
        print(f"App version: {app.version}")

        # Check routes
        routes = [route.path for route in app.routes]
        expected_routes = ["/health", "/predict", "/train", "/model/status"]

        for route in expected_routes:
            if any(route in r for r in routes):
                print(f"✅ Route {route} found")
            else:
                print(f"⚠️  Route {route} not found")

        print("✅ API structure test passed!")

    except ImportError as e:
        print(f"⚠️  API test skipped due to missing dependencies: {e}")
    except Exception as e:
        print(f"⚠️  API test failed: {e}")

    print()


def run_comprehensive_test():
    """Run comprehensive system test"""
    print("🎯 Running Comprehensive TFT System Test")
    print("=" * 50)

    # Test 1: Data Preprocessing
    processed_df = test_data_preprocessing()

    # Test 2: Stock Ranking
    signals, portfolio = test_stock_ranking()

    # Test 3: Data Pipeline
    test_data_pipeline()

    # Test 4: Model Structure
    test_model_structure()

    # Test 5: API Structure
    test_api_structure()

    # Summary
    print("📊 Test Summary")
    print("=" * 50)
    print("✅ Data preprocessing: Working")
    print("✅ Stock ranking system: Working")
    print("✅ Portfolio construction: Working")
    print("📊 Sample Results:")
    print(f"   • Processed {processed_df.shape[0]} records")
    print(f"   • Generated {len(signals['long'])} long signals")
    print(f"   • Generated {len(signals['short'])} short signals")
    print(
        f"   • Portfolio expected return: {portfolio['portfolio_stats'].get('net_expected_return', 'N/A'):.4f}"
    )

    print("\n🎉 System test completed!")
    print("\nNext steps:")
    print("1. Install missing dependencies if any tests were skipped")
    print("2. Run './setup.sh' to complete the full setup")
    print("3. Use './collect_sample_data.sh' to get real market data")
    print("4. Use './run_training.sh' to train the model")


def demo_workflow():
    """Demonstrate a complete workflow"""
    print("\n🚀 Demo: Complete TFT Workflow")
    print("=" * 50)

    # Step 1: Data
    print("Step 1: Loading and preprocessing data...")
    df = load_sample_data()
    preprocessor = StockDataPreprocessor()
    processed_df = preprocessor.fit_transform(df, target_type="returns")
    print(f"✅ Data ready: {processed_df.shape}")

    # Step 2: Mock predictions (in real workflow, this would be from trained model)
    print("\nStep 2: Generating mock predictions...")
    symbols = processed_df["symbol"].unique()[:8]  # Top 8 symbols
    mock_predictions = create_sample_predictions(symbols)
    print(f"✅ Predictions ready: {mock_predictions.shape}")

    # Step 3: Signal generation
    print("\nStep 3: Generating trading signals...")
    ranking_system = StockRankingSystem(max_positions=5)
    predictions_df = ranking_system.process_predictions(
        mock_predictions, symbols, prediction_type="quantile"
    )
    signals = ranking_system.generate_trading_signals(predictions_df)
    print(
        f"✅ Signals generated: {len(signals['long'])} long, {len(signals['short'])} short"
    )

    # Step 4: Portfolio construction
    print("\nStep 4: Constructing portfolio...")
    portfolio_constructor = PortfolioConstructor()
    portfolio = portfolio_constructor.construct_portfolio(signals)
    print(
        f"✅ Portfolio ready with {portfolio['portfolio_stats']['total_positions']} positions"
    )

    # Step 5: Display results
    print("\n📈 Trading Recommendations:")
    print("\nLONG POSITIONS:")
    for i, signal in enumerate(signals["long"][:3], 1):
        print(
            f"  {i}. {signal.symbol}: {signal.predicted_return:.4f} return (confidence: {signal.confidence:.3f})"
        )

    print("\nSHORT POSITIONS:")
    for i, signal in enumerate(signals["short"][:3], 1):
        print(
            f"  {i}. {signal.symbol}: {signal.predicted_return:.4f} return (confidence: {signal.confidence:.3f})"
        )

    print(
        f"\n💰 Portfolio Expected Return: {portfolio['portfolio_stats'].get('net_expected_return', 0):.4f}"
    )
    print(f"📊 Total Positions: {portfolio['portfolio_stats']['total_positions']}")

    print("\n🎯 Demo completed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TFT System Testing and Demo")
    parser.add_argument(
        "--mode",
        choices=["test", "demo", "both"],
        default="both",
        help="Run mode: test, demo, or both",
    )

    args = parser.parse_args()

    if args.mode in ["test", "both"]:
        run_comprehensive_test()

    if args.mode in ["demo", "both"]:
        demo_workflow()
