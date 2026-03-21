# APEX — Multi-Strategy Algorithmic Trading Platform

A production-grade trading platform built around the Temporal Fusion Transformer (TFT), with a 10-model ensemble spanning equities, FX, and options. The system combines 6 independent alpha strategies through a Bayesian regime-adaptive ensemble combiner, with paper trading execution via Alpaca and real-time signal distribution via Redis pub/sub.

## Features

### Core Capabilities
- **10-Model Ensemble**: TFT stocks/forex/volatility + Kronos foundation model + Deep Surrogates + TDGF PDE solver
- **6 Alpha Strategies**: Cross-sectional momentum, pairs trading, FX carry+trend, Kronos forecasting, neural surrogate pricing, American option mispricing
- **Bayesian Ensemble**: Regime-adaptive signal combination (60% Sharpe-based + 40% regime-based weights)
- **Paper Trading**: Automated daily pipeline with Alpaca execution, PostgreSQL logging, Discord reports
- **Multi-horizon Forecasting**: Predict 1-30 days ahead with uncertainty quantification
- **Portfolio Construction**: Risk-constrained long/short portfolio with position limits, sector caps, target volatility
- **Real-time Signals**: Redis pub/sub for dashboard consumers + Kafka event bus for microservices
- **Live Dashboard**: HTML dashboard at port 8010 with per-strategy signal panels and tail risk gauge

### Data Processing
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Volume ratios
- **Temporal Features**: Cyclical encoding, market structure indicators
- **Sentiment Integration**: Ready for news/social media sentiment feeds
- **Multi-source Data**: Polygon.io, fundamental data, economic indicators

### Model Architecture
- **TFT Core**: PyTorch Forecasting with quantile loss, attention, multi-horizon output
- **Kronos Foundation Model**: Pre-trained K-line transformer (HuggingFace `NeoQuasar/Kronos-base`)
- **Deep Surrogates**: Neural option pricing with Heston parameter calibration and tail risk index
- **TDGF PDE Solver**: Time Deep Gradient Flow for American option pricing
- **Hyperparameter Optimization**: Optuna-based automated tuning
- **GPU Acceleration**: Mixed-precision training support

### Trading Features
- **Signal Generation**: Cross-sectional z-scoring, direction assignment post-normalization
- **Risk Management**: Position sizing, sector limits, turnover constraints, tail risk alerts
- **Performance Tracking**: Rolling Sharpe (21d/63d), per-strategy diagnostics
- **Backtesting Ready**: Historical signal generation and validation

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd TFT

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data models logs predictions reports output

# Set up PostgreSQL database (optional - can also use SQLite)
python postgres_schema.py
```

### PostgreSQL Setup (Recommended)

The system now supports PostgreSQL as the primary data source for production deployments:

```bash
# 1. Set up PostgreSQL database
createdb stock_trading_analysis

# 2. Configure environment variables
cp .env.example .env
# Edit .env with your PostgreSQL credentials

# 3. Create database schema
python postgres_schema.py

# 4. Load your data into PostgreSQL tables:
# - ohlcv (price data)
# - fundamentals (financial metrics)  
# - sentiment (news/social media sentiment)
# - earnings (earnings calendar)
```

### Basic Usage

#### 1. PostgreSQL-based Training (Recommended)

```bash
# Train model using PostgreSQL data source
python train_postgres.py \
    --symbols AAPL GOOGL MSFT TSLA AMZN \
    --start-date 2022-01-01 \
    --target-type returns \
    --max-epochs 50 \
    --validate-data \
    --generate-predictions \
    --run-evaluation
```

#### 2. Legacy File-based Training

```bash
# Collect historical data and train model
python train.py \
    --data-source api \
    --symbols AAPL GOOGL MSFT TSLA AMZN \
    --start-date 2020-01-01 \
    --target-type returns \
    --max-epochs 50 \
    --generate-predictions
```

#### 3. Generate Predictions

```bash
# Generate predictions with trained model
python predict.py \
    --model-path models/tft_model.pth \
    --symbols AAPL GOOGL MSFT TSLA AMZN \
    --prediction-method quintile \
    --include-portfolio \
    --output-format both
```

#### 4. Run PostgreSQL API Server

```bash
# Start prediction API server with PostgreSQL backend
python -m uvicorn api_postgres:app --host 0.0.0.0 --port 8000 --reload
```

#### 5. Legacy API Server

```bash
# Start prediction API server (file-based)
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

#### 6. Automated Scheduling

```bash
# Run automated system
python scheduler.py --mode scheduler

# Or run specific tasks manually
python scheduler.py --mode manual --task training
```

#### 7. Paper Trader (Multi-Strategy Ensemble)

```bash
# Start the paper trading service
cd paper-trader
python -m uvicorn main:app --host 0.0.0.0 --port 8010 --reload

# Dashboard: http://localhost:8010/dashboard
# Health:    http://localhost:8010/health
# Manual trigger: POST http://localhost:8010/run-now
```

## Multi-Strategy Ensemble Architecture

### Model Layer

10 models managed by `ModelManager` (`models/manager.py`) with graceful fallback — missing models return empty predictions:

| Model | File | Asset Class | Source |
|-------|------|-------------|--------|
| TFT Stocks | `models/stocks_adapter.py` | Equities | Trained locally |
| TFT Forex | `models/forex_model.py` | FX | Trained locally |
| TFT Volatility | `models/volatility_model.py` | Volatility | Trained locally |
| Kronos (#12) | `models/kronos_model.py` | Stocks + FX | HuggingFace `NeoQuasar/Kronos-base` |
| Deep Surrogates (#13) | `models/deep_surrogate_model.py` | Options/Vol | `/opt/deep_surrogate` repo |
| TDGF (#14) | `models/tdgf_model.py` | Options | `/opt/tdgf` repo |

### Strategy Layer

Each strategy extends `BaseStrategy` (`strategies/base.py`) and produces `StrategyOutput` containing `List[AlphaScore]`:

| Strategy | File | Alpha Source |
|----------|------|-------------|
| Cross-Sectional Momentum | `strategies/momentum/cross_sectional.py` | Factor-based momentum signals |
| Pairs Trading (StatArb) | `strategies/statarb/pairs.py` | Cointegration mean-reversion |
| FX Carry + Trend | `strategies/fx/carry_trend.py` | Interest rate carry + trend following |
| Kronos Forecasting | `strategies/kronos/strategy.py` | Foundation model price forecasts |
| Deep Surrogates | `strategies/deep_surrogates/strategy.py` | Tail risk + IV surface anomalies |
| TDGF American Options | `strategies/tdgf/strategy.py` | Option mispricing (model vs market) |

### Ensemble Pipeline

```
Data Fetch (yfinance) → Regime Detection (4-state) → All Strategies → Bayesian Combiner → Portfolio Optimizer → Alpaca Execution
                                                                            ↓
                                                                    Redis Pub/Sub (optional)
                                                                    PostgreSQL Logging
                                                                    Discord Reports
```

1. **Regime Detection**: Classifies market into 4 states (calm/volatile x trending/choppy)
2. **Signal Generation**: Each enabled strategy produces z-scored alpha scores with confidence
3. **Bayesian Combination**: 60% Sharpe-based weights + 40% regime-based weights, clamped and renormalized
4. **Portfolio Optimization**: Position limits, leverage constraints, target volatility
5. **Execution**: Alpaca paper trading with PostgreSQL trade logging

### Paper Trader

FastAPI service on port 8010 (`paper-trader/main.py`):

| Endpoint | Description |
|----------|-------------|
| `/health` | Model status, Redis stats, regime, ensemble weights |
| `/run-now` | Manual pipeline trigger |
| `/positions` | Current portfolio positions |
| `/history` | Trade history |
| `/weights` | Current ensemble weights |
| `/dashboard` | HTML dashboard with per-strategy panels |

Database: PostgreSQL port **5432**, database **`tft_trading`** (tables: `paper_trades`, `paper_daily_snapshots`, `paper_strategy_signals`).

### Redis Signal Channels

Optional fire-and-forget pub/sub layer (`strategies/signals/publisher.py`):

| Channel | Content |
|---------|---------|
| `apex:signals:stock` | Equity ensemble signals |
| `apex:signals:forex` | FX pair signals |
| `apex:signals:options` | Options/volatility signals |
| `apex:signals:risk` | Composite tail risk index |

### Strategy Activation

All strategies disabled by default. Enable via `.env`:
```bash
STRATEGY_MOMENTUM_ENABLED=true
STRATEGY_STATARB_ENABLED=true
STRATEGY_FX_ENABLED=true
STRATEGY_KRONOS_ENABLED=true
STRATEGY_DEEP_SURROGATES_ENABLED=true
STRATEGY_TDGF_ENABLED=true
```

New model repos must be cloned to `/opt/` (see `scripts/setup_kronos.sh`, `scripts/setup_deep_surrogate.sh`, `scripts/setup_tdgf.sh`). If repos are missing, strategies gracefully return empty signals and the ensemble skips them.

## Project Structure

```
TFT/
├── models/                        # Model layer
│   ├── base.py                    # BaseTFTModel ABC, ModelPrediction
│   ├── manager.py                 # ModelManager — unified loader
│   ├── stocks_adapter.py          # TFT stocks model
│   ├── forex_model.py             # TFT forex model
│   ├── volatility_model.py        # TFT volatility model
│   ├── kronos_model.py            # Kronos foundation model
│   ├── deep_surrogate_model.py    # Deep Surrogates + Heston calibration
│   └── tdgf_model.py              # TDGF PDE solver
├── strategies/                    # Strategy layer
│   ├── base.py                    # BaseStrategy ABC, AlphaScore, StrategyOutput
│   ├── config.py                  # All strategy configs (env-loaded)
│   ├── momentum/                  # Cross-sectional momentum
│   ├── statarb/                   # Pairs trading
│   ├── fx/                        # FX carry + trend
│   ├── kronos/                    # Kronos strategy wrapper
│   ├── deep_surrogates/           # Deep Surrogates strategy wrapper
│   ├── tdgf/                      # TDGF strategy wrapper
│   ├── ensemble/                  # Bayesian combiner + portfolio optimizer
│   │   ├── combiner.py            # EnsembleCombiner, TFTAdapter
│   │   └── portfolio_optimizer.py # Risk-constrained optimization
│   ├── regime/                    # Market regime detection
│   ├── risk/                      # Portfolio risk management
│   └── signals/                   # Redis pub/sub publisher
├── paper-trader/                  # Execution service
│   └── main.py                    # FastAPI app, daily pipeline, dashboard
├── scripts/                       # Setup scripts for external models
├── postgres_data_loader.py        # PostgreSQL data connector
├── postgres_data_pipeline.py      # PostgreSQL data processing
├── tft_postgres_model.py          # TFT + PostgreSQL integration
├── api_postgres.py                # FastAPI with PostgreSQL backend
├── train_postgres.py              # PostgreSQL training script
├── postgres_schema.py             # Database schema
├── data_preprocessing.py          # Legacy preprocessing
├── tft_model.py                   # Legacy TFT model
├── stock_ranking.py               # Signal generation + portfolio construction
├── data_pipeline.py               # Legacy data collection
├── api.py                         # Legacy FastAPI service
├── scheduler.py                   # Automated scheduling
├── train.py                       # Legacy training
├── predict.py                     # Prediction script
├── config_manager.py              # Configuration management
├── requirements.txt               # Dependencies
├── .env.template                  # Full env config template
└── .env.example                   # Minimal env config
```

## Configuration

### Model Configuration

```python
config = {
    'max_encoder_length': 63,      # ~3 months lookback
    'max_prediction_length': 5,    # 5-day forecast
    'batch_size': 64,
    'learning_rate': 0.001,
    'hidden_size': 64,
    'lstm_layers': 2,
    'attention_head_size': 4,
    'dropout': 0.2,
    'quantiles': [0.1, 0.5, 0.9],  # Prediction intervals
    'max_epochs': 100
}
```

### PostgreSQL Configuration

The system requires the following PostgreSQL tables for full functionality:

```sql
-- Core price data
ohlcv (symbol, date, open, high, low, close, volume, adj_*)

-- Company information  
symbols (symbol, company_name, sector, industry, exchange)

-- Financial metrics
fundamentals (symbol, date, market_cap, pe_ratio, eps, etc.)

-- Sentiment analysis
sentiment (symbol, date, sentiment_score, news_count, etc.)

-- Earnings calendar
earnings (symbol, earnings_date, earnings_estimate, etc.)

-- Economic indicators
economic_indicators (indicator_name, date, value)

-- VIX data for market regime detection
vix_data (date, vix_close, etc.)
```

Environment variables for PostgreSQL:
```bash
POSTGRES_HOST=localhost
POSTGRES_DB=stock_trading_analysis  
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=trading_password
POSTGRES_PORT=5432
POSTGRES_SCHEMA=public
```

### Trading Configuration

```python
ranking_config = {
    'liquidity_threshold': 500,     # Top N liquid stocks
    'confidence_threshold': 0.1,    # Minimum confidence
    'max_positions': 20,           # Max positions per side
    'max_position_size': 0.05,     # Max 5% per position
    'sector_limit': 0.3,           # Max 30% per sector
    'turnover_limit': 0.5          # Max 50% daily turnover
}
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Generate Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "symbols": ["AAPL", "GOOGL", "MSFT"],
       "historical_data": {
         "AAPL": [
           {
             "timestamp": "2024-01-15",
             "open": 185.0,
             "high": 187.5,
             "low": 184.0,
             "close": 186.5,
             "volume": 50000000,
             "sentiment": 0.2
           }
         ],
         ...
       },
       "prediction_horizon": 5,
       "include_portfolio": true
     }'
```

### Train Model
```bash
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{
       "data_path": "data/stock_data.db",
       "target_type": "returns",
       "validation_split": 0.2,
       "optimize_hyperparams": false
     }'
```

## Advanced Features

### Hyperparameter Optimization

```bash
python train.py \
    --optimize-hyperparams \
    --n-trials 50 \
    --max-epochs 30
```

### Custom Target Variables

```bash
# Binary classification (up/down)
python train.py --target-type classification

# Cross-sectional ranking
python train.py --target-type quintile

# Multi-day returns
python train.py --target-horizon 5
```

### Portfolio Backtesting

```python
from stock_ranking import StockRankingSystem, PortfolioConstructor

# Initialize systems
ranking_system = StockRankingSystem(max_positions=20)
portfolio_constructor = PortfolioConstructor()

# Generate historical signals
for date in backtest_dates:
    predictions = model.predict(date)
    signals = ranking_system.generate_trading_signals(predictions)
    portfolio = portfolio_constructor.construct_portfolio(signals)
    # Calculate returns, metrics, etc.
```

## Performance Monitoring

### Key Metrics
- **Prediction Accuracy**: Hit rate, directional accuracy
- **Risk-Adjusted Returns**: Sharpe ratio, maximum drawdown
- **Portfolio Metrics**: Turnover, concentration, sector exposure
- **Model Quality**: Calibration, confidence intervals

### Logging
All components include comprehensive logging:
- Training progress and metrics
- Prediction generation and quality
- API request/response times
- System errors and warnings

## Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
```bash
export TFT_DB_PATH="data/stock_data.db"
export TFT_MODEL_PATH="models/tft_model.pth"
export TFT_LOG_LEVEL="INFO"
export TFT_GPU_ENABLED="true"
```

### Monitoring
- Health checks at `/health`
- Metrics collection via logging
- Model performance tracking
- Automated alerting on failures

## Research and Development

### Experiment Tracking
- MLflow integration for experiment tracking
- Hyperparameter optimization with Optuna
- Model versioning and artifact storage

### Extensions
- Multi-asset class support
- Alternative data integration (satellite, credit card, etc.)
- Ensemble methods with multiple models
- Reinforcement learning for dynamic portfolio allocation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## 🚀 GitHub Copilot Validation Prompts for Polygon.io Integration

### 1. Data Pipeline Optimization

```python
# In polygon_data_loader.py
# Implement batch OHLCV fetcher for 100+ symbols with Polygon.io
# Features: auto-throttling (5 req/min), retry on 429, and PostgreSQL caching
# Return: Dict of DataFrames with {symbol: df}
```

### 2. Advanced Feature Engineering

```python
# In data_preprocessing.py
# Calculate technical indicators from Polygon's vwap:
# - Volume-weighted RSI(14)
# - MACD(12,26,9) using vwap
# - Bollinger %B(20,2)
# Handle corporate actions via Polygon's 'adjusted' flag
```

### 3. Real-Time Prediction Endpoint

```python
# In api_postgres.py
# Create FastAPI endpoint: /polygon/realtime-predict
# Input: List of Polygon-formatted symbols (e.g., 'O:SPY230818C00325000')
# Output: Predictions with Polygon's native symbol format
# Use Polygon's WebSocket client for live data
```

### 4. Intelligent Rate Limit Handling

```python
# In polygon_data_loader.py
# Implement decorator for Polygon API calls:
# - Exponential backoff on 429 errors
# - Automatic 60s cooldown after 5 requests
# - Fallback to cached PostgreSQL data
```

### 5. News Sentiment Integration

```python
# In enhanced_data_pipeline.py
# Process Polygon news into trading features:
# 1. Calculate sentiment polarity score per article
# 2. Compute daily sentiment momentum (3-day change)
# 3. Merge with OHLCV using Polygon's news timestamp
```

### 6. Fundamental Data Processing

```python
# In polygon_data_loader.py
# Fetch and normalize Polygon fundamental data:
# - Map 'marketCap' to size quantiles
# - Convert 'peRatio' to z-scores
# - Handle missing values with sector averages
```

### 7. WebSocket Real-Time Integration

```python
# In realtime_handler.py
# Implement Polygon WebSocket client that:
# 1. Subscribes to specified symbols
# 2. Updates PostgreSQL every 15 seconds
# 3. Triggers predictions on volume spikes
```

### 8. Automated Batch Processing

```python
# In scheduler.py
# Create daily job to:
# 1. Fetch all S&P 500 symbols from Polygon
# 2. Update OHLCV in parallel threads
# 3. Validate corporate action adjustments
```

### 9. Production Error Handling

```python
# In polygon_data_loader.py
# Implement fault tolerance for:
# - Polygon API downtime (switch to cached data)
# - Symbol delistings (auto-purge from DB)
# - Data gaps (linear interpolation)
```

### 10. Model Feature Optimization

```python
# In tft_postgres_model.py
# Add Polygon-specific features to TFT:
# - vwap relative to close
# - News sentiment momentum
# - Fundamental z-scores
# Quantize model for faster Polygon real-time predictions
```

### 🔥 Critical Options Trading Validation

```python
# In api_postgres.py
# Create endpoint that:
# 1. Accepts Polygon options symbols (e.g., 'O:SPY230818C00325000')
# 2. Fetches underlying equity data
# 3. Generates volatility-adjusted predictions
# 4. Returns in Polygon's options response format
```

### 🧠 Copilot Integration Standards

**GitHub Copilot Should Understand:**
- ✅ Polygon symbol formats (`C:AAPL`, `O:TSLA240118C00500000`)
- ✅ Handling adjusted data (splits/dividends)
- ✅ Polygon's rate limits and error codes
- ✅ Real-time WebSocket streaming for price/sentiment updates
- ✅ Financial signal engineering with volume-weighted price logic

### Implementation Guidelines

1. **File Header Prompts**: Add relevant prompts as comments at the top of each file
2. **Function-Level Guidance**: Include specific prompts before complex functions
3. **Type Hints**: Use Polygon-specific data types in function signatures
4. **Error Handling**: Include Polygon API error codes in exception handling

### Validation Testing

```bash
# Test Copilot understanding with these files
python test_copilot_polygon.py --validate-prompts
```

## 🧠 How to Use GitHub Copilot with Your Optimized Toolkit

### 💡 1. File-Specific Prompt Headers

Each file now starts with rich contextual prompts (via `devtools/insert_copilot_headers.py`):

```python
# In polygon_data_loader.py
"""
# COPILOT PROMPT: Implement batch OHLCV fetcher for 100+ symbols with Polygon.io
# Features: auto-throttling (5 req/min), retry on 429, and PostgreSQL caching
# Return: Dict of DataFrames with {symbol: df}
"""

# Now just start typing your function:
def fetch_polygon_ohlcv_batch(...):
    # ✅ Copilot will complete using your domain-specific prompt!
```

### 🔧 2. Contextual In-Function Prompts

Inside functions, use specific prompts for complex logic:

```python
def process_market_data(symbols):
    # Add retry logic for HTTP 429 errors using exponential backoff
    
    # Switch to cached PostgreSQL data if Polygon API fails
    
    # Calculate VWAP-based RSI using 14-period window
```

### 📚 3. Use Your Prompt Library

Reference `devtools/copilot_prompts_polygon.md` directly:

```python
# Copy from your prompt library:
# Add Polygon-specific features to TFT:
# - vwap_ratio: vwap relative to close price
# - news_sentiment_momentum: 3-day sentiment change
# - fundamental_zscore: sector-adjusted fundamental metrics

class TFTWithPolygonFeatures(TemporalFusionTransformer):
    # ✅ Copilot understands exactly what "Polygon-specific features" means
```

### 🧪 4. Validate Copilot Output

After Copilot generates code, immediately validate:

```bash
# Test specific functionality
./devtools/prompt_runner.sh --test rate_limiting
./devtools/prompt_runner.sh --test websocket

# Run complete validation
./devtools/prompt_runner.sh --test-all
```

### 🚀 5. Advanced Developer Prompts

Your toolkit enables sophisticated prompts:

```python
# Optimize this function to support 500 symbol batch with caching and retry

# Convert Polygon options symbol to underlying equity and compute implied volatility

# Implement WebSocket reconnection with exponential backoff for production use

# Create sector-neutral portfolio with risk parity weighting
```

## 🎯 Optimal Copilot Workflow

### **Step-by-Step Process**

1. **Open Target File**: e.g., `polygon_data_loader.py`
2. **Insert/Use Prompt**: From your toolkit or `copilot_prompts_polygon.md`
3. **Start Function**: Let Copilot complete implementation
4. **Validate**: Run `./devtools/prompt_runner.sh --test-all`
5. **Refine**: Based on test results and requirements
6. **Commit**: Save validated, production-ready code

### **Sample Prompts for Immediate Use**

#### 🔹 Real-Time Data Handler
```python
# In realtime_handler.py
# Use Polygon WebSocket to subscribe to 20 symbols and trigger TFT predictions on volume spikes
# Handle reconnection, error recovery, and PostgreSQL batch updates every 15 seconds

class PolygonWebSocketHandler:
```

#### 🔹 Stock Ranking System
```python
# In stock_ranking.py
# Rank stocks based on TFT predicted returns and confidence intervals
# Apply sector caps (max 30%), liquidity filters (min $10M volume), and position limits

def generate_ranked_signals(predictions_df):
```

#### 🔹 PostgreSQL Training Pipeline
```python
# In train_postgres.py
# Train TFT model using PostgreSQL-backed OHLCV + sentiment + fundamentals
# Enable quantile loss, mixed precision, and automatic hyperparameter tuning

def train_tft_with_postgres(symbols, start_date):
```

#### 🔹 Options Trading API
```python
# In api_postgres.py
# Create FastAPI endpoint: /predict/options
# Accepts Polygon options symbols (O:SPY230818C00325000) and returns volatility-adjusted signals
# Include Greeks calculation and underlying equity correlation

@app.post("/predict/options")
async def predict_options_signals(request: OptionsRequest):
```

### **Domain-Specific Copilot Enhancements**

Your toolkit teaches Copilot about:

- **Financial Markets**: OHLCV, VWAP, technical indicators, options Greeks
- **Polygon.io API**: Symbol formats, rate limits, WebSocket streaming
- **Production Requirements**: Error handling, caching, batch processing
- **TFT Models**: Quantile forecasting, attention mechanisms, feature engineering

### **Quality Assurance Integration**

```bash
# After each Copilot completion, run:
./devtools/prompt_runner.sh --quality-check    # Syntax + import validation
./devtools/prompt_runner.sh --test performance # Performance requirements
./devtools/prompt_runner.sh --report          # Generate status summary
```

### **Advanced Usage Patterns**

#### **Ensemble Prompting**
```python
# Combine multiple prompt patterns:
# 1. Use Polygon VWAP for technical indicators
# 2. Apply sector-neutral position sizing
# 3. Implement exponential backoff for API calls
# 4. Cache results in PostgreSQL with 15-minute TTL

def advanced_signal_generator():
```

#### **Error-First Development**
```python
# Handle these Polygon.io error scenarios:
# - 429 rate limit (exponential backoff)
# - 401 invalid API key (log and fallback)
# - Network timeout (retry with cached data)
# - Symbol delisting (auto-remove from active list)

def robust_polygon_client():
```

## 🌐 Next-Level Integration Options

### **GitHub Actions CI/CD**
Add automated validation to your workflow:

```yaml
# .github/workflows/copilot-validation.yml
- name: Validate Copilot Prompts
  run: ./devtools/prompt_runner.sh --full
```

### **VS Code Extension**
Convert your toolkit into a VS Code snippet extension for instant prompt access.

### **Copilot Plugin Development**
Transform your domain knowledge into a reusable Copilot plugin for financial ML projects.

### **Team Onboarding**
Use your toolkit as a standard for onboarding new developers to Copilot-optimized financial modeling.

---

**🏆 Result**: Your Copilot now acts like a senior quantitative developer with deep domain expertise in financial markets, Polygon.io APIs, and production ML systems.

## Disclaimer

This software is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with qualified financial advisors before making investment decisions.

---

**Key Features Summary:**
- 10-model ensemble spanning equities, FX, and options
- 6 independent alpha strategies with Bayesian regime-adaptive combination
- Kronos foundation model, Deep Surrogates neural pricing, TDGF PDE solver
- Tail risk monitoring with composite index and per-symbol alerts
- Heston model calibration with multi-start optimization and Feller condition validation
- Automated paper trading with Alpaca execution and PostgreSQL logging
- Real-time signal distribution via Redis pub/sub
- Live HTML dashboard with per-strategy panels and tail risk gauge
- Graceful degradation — missing models/strategies silently skipped
- Polygon.io API integration for market data
