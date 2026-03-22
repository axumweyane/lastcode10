"""
Enhanced Configuration Management for Kironix TFT Trading System
Integrates advanced trading features with TFT model predictions
"""

"""
# COPILOT PROMPT: Configuration management with:
# - Environment variable handling
# - Polygon.io API key management
# - Database connection settings
# EXPECTED OUTPUT: Secure, flexible configuration system
"""


import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class TFTConfig:
    """TFT Model Configuration"""

    max_encoder_length: int = 63
    max_prediction_length: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001
    hidden_size: int = 64
    lstm_layers: int = 2
    attention_head_size: int = 4
    dropout: float = 0.2
    max_epochs: int = 100
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    target_type: str = "returns"
    target_horizon: int = 1


@dataclass
class TradingConfig:
    """Advanced Trading Configuration"""

    # Basic Parameters
    liquidity_threshold: int = 500
    confidence_threshold: float = 0.1
    max_positions: int = 20
    max_position_size: float = 0.05
    sector_limit: float = 0.3
    turnover_limit: float = 0.5

    # VIX-based Dynamic Thresholds
    dynamic_thresholds_enabled: bool = True
    vix_low_regime_max: float = 20
    vix_medium_regime_max: float = 40
    vix_high_regime_min: float = 40

    # Sentiment Integration
    bullish_threshold_vix_low: float = 0.65
    bullish_threshold_vix_medium: float = 0.75
    bullish_threshold_vix_high: float = 0.85
    bearish_threshold_vix_low: float = -0.65
    bearish_threshold_vix_medium: float = -0.75
    bearish_threshold_vix_high: float = -0.85

    # Risk Management
    circuit_breaker_enabled: bool = True
    vix_spike_threshold_percent: float = 50
    max_portfolio_drawdown_percent: float = 5
    trading_halt_cooldown_minutes: int = 30

    # Liquidity Filters
    min_daily_volume_usd: float = 5000000
    max_bid_ask_spread_percent: float = 0.3
    liquidity_check_enabled: bool = True


@dataclass
class SentimentConfig:
    """Sentiment Analysis Configuration"""

    # Model Weights
    finbert_weight: float = 0.60
    vader_weight: float = 0.30
    wsb_weight: float = 0.10

    # Emotional Thresholds
    euphoric_threshold: float = 0.8
    optimistic_threshold: float = 0.5
    panic_threshold: float = -0.5
    despair_threshold: float = -0.8

    # Monitoring Configuration
    monitor_subreddits: List[str] = field(
        default_factory=lambda: [
            "stocks",
            "investing",
            "SecurityAnalysis",
            "ValueInvesting",
            "StockMarket",
            "wallstreetbets",
            "options",
            "pennystocks",
        ]
    )
    monitor_twitter_accounts: List[str] = field(
        default_factory=lambda: [
            "@DeItaone",
            "@unusual_whales",
            "@zerohedge",
            "@GoldmanSachs",
            "@JPMorgan",
            "@MorganStanley",
        ]
    )
    posts_per_subreddit: int = 25
    rate_limit_seconds: float = 0.5


@dataclass
class DataConfig:
    """Data Source Configuration"""

    # API Keys
    polygon_api_key: str = ""
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    openai_api_key: str = ""

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "stock_trading_analysis"
    db_user: str = ""
    db_password: str = ""

    # TFT Specific
    tft_db_path: str = "data/stock_data.db"
    cache_dir: str = "data/cache"

    # Sector ETFs for analysis
    sector_etfs: List[str] = field(
        default_factory=lambda: [
            "QQQ",
            "SPY",
            "XLF",
            "XLE",
            "XLK",
            "XLV",
            "XLI",
            "XLP",
            "XLU",
            "XLY",
            "XLB",
            "XLRE",
        ]
    )


class KironixConfigManager:
    """
    Comprehensive configuration manager for Kironix TFT Trading System
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/default_config.json"
        self.tft_config = TFTConfig()
        self.trading_config = TradingConfig()
        self.sentiment_config = SentimentConfig()
        self.data_config = DataConfig()

        self._load_configurations()

    def _load_configurations(self):
        """Load configurations from environment and JSON files"""
        self._load_from_env()
        self._load_from_json()
        self._validate_configurations()

    def _load_from_env(self):
        """Load configuration from environment variables"""
        # TFT Configuration
        self.tft_config.max_encoder_length = int(
            os.getenv("TFT_MAX_ENCODER_LENGTH", 63)
        )
        self.tft_config.max_prediction_length = int(
            os.getenv("TFT_MAX_PREDICTION_LENGTH", 5)
        )
        self.tft_config.batch_size = int(os.getenv("TFT_BATCH_SIZE", 64))
        self.tft_config.learning_rate = float(os.getenv("TFT_LEARNING_RATE", 0.001))
        self.tft_config.hidden_size = int(os.getenv("TFT_HIDDEN_SIZE", 64))
        self.tft_config.max_epochs = int(os.getenv("TFT_MAX_EPOCHS", 100))

        # Trading Configuration
        self.trading_config.liquidity_threshold = int(
            os.getenv("TFT_LIQUIDITY_THRESHOLD", 500)
        )
        self.trading_config.confidence_threshold = float(
            os.getenv("TFT_CONFIDENCE_THRESHOLD", 0.1)
        )
        self.trading_config.max_positions = int(os.getenv("TFT_MAX_POSITIONS", 20))
        self.trading_config.max_position_size = float(
            os.getenv("TFT_MAX_POSITION_SIZE", 0.05)
        )

        # VIX-based thresholds
        self.trading_config.dynamic_thresholds_enabled = (
            os.getenv("DYNAMIC_THRESHOLDS_ENABLED", "true").lower() == "true"
        )
        self.trading_config.vix_low_regime_max = float(
            os.getenv("VIX_LOW_REGIME_MAX", 20)
        )
        self.trading_config.vix_medium_regime_max = float(
            os.getenv("VIX_MEDIUM_REGIME_MAX", 40)
        )
        self.trading_config.vix_high_regime_min = float(
            os.getenv("VIX_HIGH_REGIME_MIN", 40)
        )

        # Sentiment thresholds
        self.trading_config.bullish_threshold_vix_low = float(
            os.getenv("BULLISH_THRESHOLD_VIX_LOW", 0.65)
        )
        self.trading_config.bullish_threshold_vix_medium = float(
            os.getenv("BULLISH_THRESHOLD_VIX_MEDIUM", 0.75)
        )
        self.trading_config.bullish_threshold_vix_high = float(
            os.getenv("BULLISH_THRESHOLD_VIX_HIGH", 0.85)
        )

        # Risk management
        self.trading_config.circuit_breaker_enabled = (
            os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
        )
        self.trading_config.vix_spike_threshold_percent = float(
            os.getenv("VIX_SPIKE_THRESHOLD_PERCENT", 50)
        )
        self.trading_config.max_portfolio_drawdown_percent = float(
            os.getenv("MAX_PORTFOLIO_DRAWDOWN_PERCENT", 5)
        )

        # Sentiment Configuration
        self.sentiment_config.finbert_weight = float(
            os.getenv("SENTIMENT_MODEL_WEIGHTS_FINBERT", 0.60)
        )
        self.sentiment_config.vader_weight = float(
            os.getenv("SENTIMENT_MODEL_WEIGHTS_VADER", 0.30)
        )
        self.sentiment_config.wsb_weight = float(
            os.getenv("SENTIMENT_MODEL_WEIGHTS_WSB", 0.10)
        )

        self.sentiment_config.euphoric_threshold = float(
            os.getenv("EUPHORIC_THRESHOLD", 0.8)
        )
        self.sentiment_config.optimistic_threshold = float(
            os.getenv("OPTIMISTIC_THRESHOLD", 0.5)
        )
        self.sentiment_config.panic_threshold = float(
            os.getenv("PANIC_THRESHOLD", -0.5)
        )
        self.sentiment_config.despair_threshold = float(
            os.getenv("DESPAIR_THRESHOLD", -0.8)
        )

        # Data Configuration
        self.data_config.polygon_api_key = os.getenv("POLYGON_API_KEY", "")
        self.data_config.alpaca_api_key = os.getenv("ALPACA_API_KEY", "")
        self.data_config.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.data_config.reddit_client_id = os.getenv("REDDIT_CLIENT_ID", "")
        self.data_config.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        self.data_config.openai_api_key = os.getenv("OPENAI_API_KEY", "")

        self.data_config.db_host = os.getenv("DB_HOST", "localhost")
        self.data_config.db_port = int(os.getenv("DB_PORT", 5432))
        self.data_config.db_name = os.getenv("DB_NAME", "stock_trading_analysis")
        self.data_config.db_user = os.getenv("DB_USER", "")
        self.data_config.db_password = os.getenv("DB_PASSWORD", "")

        self.data_config.tft_db_path = os.getenv("TFT_DB_PATH", "data/stock_data.db")

        # Parse list configurations
        subreddits = os.getenv("MONITOR_SUBREDDITS", "")
        if subreddits:
            self.sentiment_config.monitor_subreddits = [
                s.strip() for s in subreddits.split(",")
            ]

        twitter_accounts = os.getenv("MONITOR_TWITTER_ACCOUNTS", "")
        if twitter_accounts:
            self.sentiment_config.monitor_twitter_accounts = [
                s.strip() for s in twitter_accounts.split(",")
            ]

        sector_etfs = os.getenv("SECTOR_ETFS", "")
        if sector_etfs:
            self.data_config.sector_etfs = [s.strip() for s in sector_etfs.split(",")]

    def _load_from_json(self):
        """Load additional configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    json_config = json.load(f)

                # Update configurations from JSON if not already set by environment
                self._update_config_from_json(json_config)

        except Exception as e:
            logger.warning(f"Could not load JSON config from {self.config_path}: {e}")

    def _update_config_from_json(self, json_config: Dict[str, Any]):
        """Update configuration objects from JSON data"""
        # Update TFT config
        if "model" in json_config:
            model_config = json_config["model"]
            for key, value in model_config.items():
                if hasattr(self.tft_config, key):
                    setattr(self.tft_config, key, value)

        # Update trading config
        if "trading" in json_config:
            trading_config = json_config["trading"]
            for key, value in trading_config.items():
                if hasattr(self.trading_config, key):
                    setattr(self.trading_config, key, value)

    def _validate_configurations(self):
        """Validate configuration values"""
        # Validate API keys
        required_keys = {
            "Polygon API": self.data_config.polygon_api_key,
            "Alpaca API": self.data_config.alpaca_api_key,
            "Reddit Client ID": self.data_config.reddit_client_id,
        }

        missing_keys = [name for name, key in required_keys.items() if not key]
        if missing_keys:
            logger.warning(f"Missing API keys: {', '.join(missing_keys)}")

        # Validate model parameters
        if self.tft_config.max_encoder_length <= 0:
            raise ValueError("max_encoder_length must be positive")

        if not (0 < self.tft_config.learning_rate < 1):
            raise ValueError("learning_rate must be between 0 and 1")

        # Validate trading parameters
        if not (0 < self.trading_config.max_position_size <= 1):
            raise ValueError("max_position_size must be between 0 and 1")

        if not (0 < self.trading_config.sector_limit <= 1):
            raise ValueError("sector_limit must be between 0 and 1")

    def get_vix_regime_thresholds(self, vix_value: float) -> Dict[str, float]:
        """
        Get dynamic thresholds based on VIX regime
        """
        if not self.trading_config.dynamic_thresholds_enabled:
            return {"bullish_threshold": 0.3, "bearish_threshold": -0.3}

        if vix_value <= self.trading_config.vix_low_regime_max:
            return {
                "bullish_threshold": self.trading_config.bullish_threshold_vix_low,
                "bearish_threshold": self.trading_config.bearish_threshold_vix_low,
                "regime": "low_volatility",
            }
        elif vix_value <= self.trading_config.vix_medium_regime_max:
            return {
                "bullish_threshold": self.trading_config.bullish_threshold_vix_medium,
                "bearish_threshold": self.trading_config.bearish_threshold_vix_medium,
                "regime": "medium_volatility",
            }
        else:
            return {
                "bullish_threshold": self.trading_config.bullish_threshold_vix_high,
                "bearish_threshold": self.trading_config.bearish_threshold_vix_high,
                "regime": "high_volatility",
            }

    def get_sentiment_weights(self) -> Dict[str, float]:
        """Get sentiment model weights"""
        return {
            "finbert": self.sentiment_config.finbert_weight,
            "vader": self.sentiment_config.vader_weight,
            "wsb": self.sentiment_config.wsb_weight,
        }

    def get_emotional_thresholds(self) -> Dict[str, float]:
        """Get emotional gradient thresholds"""
        return {
            "euphoric": self.sentiment_config.euphoric_threshold,
            "optimistic": self.sentiment_config.optimistic_threshold,
            "panic": self.sentiment_config.panic_threshold,
            "despair": self.sentiment_config.despair_threshold,
        }

    def should_halt_trading(self, current_drawdown: float, vix_spike: float) -> bool:
        """
        Determine if trading should be halted based on circuit breaker rules
        """
        if not self.trading_config.circuit_breaker_enabled:
            return False

        # Check portfolio drawdown
        if current_drawdown >= self.trading_config.max_portfolio_drawdown_percent:
            logger.warning(
                f"Trading halted: Portfolio drawdown {current_drawdown}% exceeds limit"
            )
            return True

        # Check VIX spike
        if vix_spike >= self.trading_config.vix_spike_threshold_percent:
            logger.warning(f"Trading halted: VIX spike {vix_spike}% exceeds threshold")
            return True

        return False

    def get_database_connection_string(self) -> str:
        """Get database connection string"""
        return (
            f"postgresql://{self.data_config.db_user}:{self.data_config.db_password}"
            f"@{self.data_config.db_host}:{self.data_config.db_port}/{self.data_config.db_name}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert all configurations to dictionary"""
        return {
            "tft": self.tft_config.__dict__,
            "trading": self.trading_config.__dict__,
            "sentiment": self.sentiment_config.__dict__,
            "data": self.data_config.__dict__,
        }

    def save_config(self, output_path: str = None):
        """Save current configuration to JSON file"""
        output_path = output_path or "config/current_config.json"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        logger.info(f"Configuration saved to {output_path}")


# Global configuration instance
config_manager = KironixConfigManager()


def get_config() -> KironixConfigManager:
    """Get the global configuration manager instance"""
    return config_manager


def reload_config():
    """Reload configuration from sources"""
    global config_manager
    config_manager = KironixConfigManager()
    return config_manager


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()

    print("🔧 Kironix TFT Configuration Loaded")
    print("=" * 50)

    print(f"TFT Model Config:")
    print(f"  Max Encoder Length: {config.tft_config.max_encoder_length}")
    print(f"  Batch Size: {config.tft_config.batch_size}")
    print(f"  Learning Rate: {config.tft_config.learning_rate}")

    print(f"\nTrading Config:")
    print(f"  Max Positions: {config.trading_config.max_positions}")
    print(f"  Dynamic Thresholds: {config.trading_config.dynamic_thresholds_enabled}")
    print(f"  Circuit Breaker: {config.trading_config.circuit_breaker_enabled}")

    print(f"\nSentiment Config:")
    print(f"  FinBERT Weight: {config.sentiment_config.finbert_weight}")
    print(f"  Monitor Subreddits: {len(config.sentiment_config.monitor_subreddits)}")

    print(f"\nData Sources:")
    print(f"  Polygon API: {'✅' if config.data_config.polygon_api_key else '❌'}")
    print(f"  Alpaca API: {'✅' if config.data_config.alpaca_api_key else '❌'}")
    print(f"  Reddit API: {'✅' if config.data_config.reddit_client_id else '❌'}")

    # Test VIX regime thresholds
    print(f"\nVIX Regime Examples:")
    for vix in [15, 25, 45]:
        thresholds = config.get_vix_regime_thresholds(vix)
        print(f"  VIX {vix}: {thresholds}")

    # Save configuration
    config.save_config("config/test_config.json")
    print(f"\n✅ Configuration test completed!")
