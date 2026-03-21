#!/bin/bash

# TFT PostgreSQL Setup Script
# Sets up the complete TFT system with PostgreSQL integration

set -e

echo "=========================================="
echo "TFT PostgreSQL System Setup"
echo "=========================================="

# Create directories
echo "Creating directories..."
mkdir -p data models logs predictions reports output
echo "✓ Directories created"

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "⚠️  PostgreSQL not found. Please install PostgreSQL first:"
    echo "   Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib"
    echo "   macOS: brew install postgresql"
    echo "   Windows: Download from https://www.postgresql.org/download/"
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ PostgreSQL and Python 3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "tft_env" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv tft_env
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source tft_env/bin/activate

# Install essential packages first
echo "Installing essential Python packages..."
pip install --upgrade pip
pip install python-dotenv psycopg2-binary pandas numpy requests

# Try to install ML packages (may fail if dependencies not available)
echo "Installing ML packages (this may take a while)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || echo "⚠️  PyTorch installation failed - you may need to install manually"
pip install pytorch-lightning pytorch-forecasting scikit-learn || echo "⚠️  Some ML packages failed to install"

# Install API packages
echo "Installing API packages..."
pip install fastapi uvicorn pydantic || echo "⚠️  API packages failed to install"

# Install additional packages
echo "Installing additional packages..."
pip install -r requirements.txt || echo "⚠️  Some packages from requirements.txt failed to install"

echo "✓ Package installation completed"

# Set up .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env configuration file..."
    cat > .env << EOF
# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_DB=stock_trading_analysis
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=CHANGE_ME_REQUIRED
POSTGRES_PORT=5432
POSTGRES_SCHEMA=public

# Polygon.io API (for enhanced data collection)
POLYGON_API_KEY=your_polygon_api_key_here

# Alpaca Trading API
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Reddit API (sentiment analysis)
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=TFT_Trading_Bot_v1.0

# OpenAI API (for sentiment analysis)
OPENAI_API_KEY=sk-proj-3HJ9Jg7bVm6vYSAzE6J3m4nCGpU9kLqPtR2FWBX8

# VIX-based Dynamic Thresholds
VIX_LOW_THRESHOLD=20
VIX_MEDIUM_THRESHOLD=40
VIX_LOW_CONFIDENCE_THRESHOLD=0.15
VIX_MEDIUM_CONFIDENCE_THRESHOLD=0.08
VIX_HIGH_CONFIDENCE_THRESHOLD=0.05

# Model Configuration
TFT_ENCODER_LENGTH=63
TFT_PREDICTION_LENGTH=5
TFT_BATCH_SIZE=64
TFT_LEARNING_RATE=0.001
TFT_HIDDEN_SIZE=64
TFT_ATTENTION_HEADS=4
TFT_DROPOUT=0.2
TFT_MAX_EPOCHS=100
TFT_EARLY_STOPPING=10

# API Configuration
API_PORT=8000
TFT_MODEL_PATH=models/tft_postgres_model.pth

# Logging
LOG_LEVEL=INFO
EOF
    echo "✓ .env file created with default configuration"
    echo "⚠️  Please update .env with your actual API keys and database credentials"
else
    echo "✓ .env file already exists"
fi

# Database setup
echo ""
echo "=========================================="
echo "Database Setup"
echo "=========================================="

# Prompt for database setup
read -p "Do you want to set up the PostgreSQL database now? (y/n): " setup_db

if [[ $setup_db =~ ^[Yy]$ ]]; then
    echo "Setting up PostgreSQL database..."
    
    # Check if database exists
    if psql -lqt | cut -d \| -f 1 | grep -qw stock_trading_analysis; then
        echo "✓ Database 'stock_trading_analysis' already exists"
    else
        echo "Creating database..."
        createdb stock_trading_analysis || echo "⚠️  Failed to create database. You may need to create it manually."
    fi
    
    # Run schema setup
    echo "Setting up database schema..."
    python postgres_schema.py || echo "⚠️  Schema setup failed. You may need to run it manually."
    
    echo "✓ Database setup completed"
else
    echo "⚠️  Database setup skipped. Run 'python postgres_schema.py' later to set up the schema."
fi

# Test the installation
echo ""
echo "=========================================="
echo "Testing Installation"
echo "=========================================="

echo "Testing PostgreSQL connection..."
python -c "
try:
    from postgres_data_loader import PostgresDataLoader
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'database': os.getenv('POSTGRES_DB', 'stock_trading_analysis'),
        'user': os.getenv('POSTGRES_USER', 'trading_user'),
        'password': os.environ['POSTGRES_PASSWORD'],
        'port': int(os.getenv('POSTGRES_PORT', 5432))
    }
    
    loader = PostgresDataLoader(db_config)
    print('✓ PostgreSQL connection successful')
except Exception as e:
    print(f'⚠️  PostgreSQL connection failed: {e}')
" || echo "❌ PostgreSQL connection test failed"

echo "Testing ML libraries..."
python -c "
try:
    import torch
    import pytorch_lightning
    import pytorch_forecasting
    print('✓ ML libraries available')
except ImportError as e:
    print(f'⚠️  Some ML libraries missing: {e}')
" || echo "⚠️  ML libraries test failed"

echo "Testing API libraries..."
python -c "
try:
    import fastapi
    import uvicorn
    import pydantic
    print('✓ API libraries available')
except ImportError as e:
    print(f'⚠️  Some API libraries missing: {e}')
" || echo "⚠️  API libraries test failed"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo "1. Update .env file with your actual API keys and database credentials"
echo "2. Load your stock data into PostgreSQL (see postgres_schema.py for table structure)"
echo "3. Train your first model:"
echo "   python train_postgres.py --symbols AAPL MSFT GOOGL --validate-data"
echo "4. Start the API server:"
echo "   python -m uvicorn api_postgres:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "Documentation: Check README.md for detailed usage instructions"
echo "Logs: Check logs/ directory for detailed logs"
echo ""
echo "🚀 TFT PostgreSQL system is ready!"

# Deactivate virtual environment
deactivate 2>/dev/null || true
