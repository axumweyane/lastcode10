"""
Stock Ranking and Portfolio Construction System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


@dataclass
class TradingSignal:
    """Trading signal data structure"""

    symbol: str
    predicted_return: float
    confidence: float
    rank: int
    signal_strength: str
    timestamp: pd.Timestamp


class StockRankingSystem:
    """
    Advanced stock ranking system for generating trading signals
    from TFT predictions
    """

    def __init__(
        self,
        liquidity_threshold: int = 500,
        confidence_threshold: float = 0.1,
        max_positions: int = 20,
    ):
        self.liquidity_threshold = liquidity_threshold
        self.confidence_threshold = confidence_threshold
        self.max_positions = max_positions

    def calculate_liquidity_filter(
        self, df: pd.DataFrame, window: int = 20
    ) -> List[str]:
        """Filter stocks by liquidity (average volume)"""

        # Calculate average volume per symbol
        avg_volume = (
            df.groupby("symbol")["volume"]
            .rolling(window=window, min_periods=10)
            .mean()
            .groupby("symbol")
            .last()
        )

        # Get top liquid symbols
        liquid_symbols = avg_volume.nlargest(self.liquidity_threshold).index.tolist()

        print(
            f"Selected {len(liquid_symbols)} liquid symbols from {df['symbol'].nunique()} total"
        )
        return liquid_symbols

    def process_predictions(
        self,
        predictions: np.ndarray,
        symbols: List[str],
        prediction_type: str = "quantile",
    ) -> pd.DataFrame:
        """Process model predictions into DataFrame"""

        if prediction_type == "quantile":
            # Assume predictions are [10th percentile, median, 90th percentile]
            predictions_df = pd.DataFrame(
                {
                    "symbol": symbols,
                    "predicted_return": predictions[:, 1],  # Median prediction
                    "lower_bound": predictions[:, 0],  # 10th percentile
                    "upper_bound": predictions[:, 2],  # 90th percentile
                }
            )

            # Calculate confidence as the width of prediction interval
            predictions_df["confidence"] = (
                predictions_df["upper_bound"] - predictions_df["lower_bound"]
            )

        elif prediction_type == "point":
            # Single point prediction
            predictions_df = pd.DataFrame(
                {
                    "symbol": symbols,
                    "predicted_return": predictions.flatten(),
                    "confidence": np.ones(len(symbols)),  # Default confidence
                }
            )

        elif prediction_type == "classification":
            # Binary classification probabilities
            predictions_df = pd.DataFrame(
                {
                    "symbol": symbols,
                    "predicted_return": predictions[:, 1]
                    - 0.5,  # Convert to return-like
                    "confidence": np.abs(predictions[:, 1] - 0.5)
                    * 2,  # Distance from 50%
                }
            )

        return predictions_df

    def generate_trading_signals(
        self,
        predictions_df: pd.DataFrame,
        liquidity_filter: List[str] = None,
        method: str = "quintile",
    ) -> Dict[str, List[TradingSignal]]:
        """Generate trading signals from predictions"""

        # Apply liquidity filter
        if liquidity_filter:
            predictions_df = predictions_df[
                predictions_df["symbol"].isin(liquidity_filter)
            ].copy()

        # Apply confidence filter
        predictions_df = predictions_df[
            predictions_df["confidence"] >= self.confidence_threshold
        ].copy()

        if len(predictions_df) == 0:
            print("Warning: No symbols passed filters")
            return {"long": [], "short": [], "neutral": []}

        current_time = pd.Timestamp.now()

        if method == "quintile":
            # Quintile-based ranking
            predictions_df["quintile"] = pd.qcut(
                predictions_df["predicted_return"], 5, labels=False, duplicates="drop"
            )

            # Create signals
            long_signals = []
            short_signals = []
            neutral_signals = []

            for _, row in predictions_df.iterrows():
                signal = TradingSignal(
                    symbol=row["symbol"],
                    predicted_return=row["predicted_return"],
                    confidence=row["confidence"],
                    rank=int(row["quintile"]),
                    signal_strength=self._get_signal_strength(row["quintile"]),
                    timestamp=current_time,
                )

                if row["quintile"] == 4:  # Top quintile
                    long_signals.append(signal)
                elif row["quintile"] == 0:  # Bottom quintile
                    short_signals.append(signal)
                else:
                    neutral_signals.append(signal)

            return {
                "long": long_signals,
                "short": short_signals,
                "neutral": neutral_signals,
            }

        elif method == "threshold":
            # Threshold-based ranking
            predictions_df = predictions_df.sort_values(
                "predicted_return", ascending=False
            )

            # Top performers for long positions
            long_threshold = predictions_df["predicted_return"].quantile(0.8)
            short_threshold = predictions_df["predicted_return"].quantile(0.2)

            long_signals = []
            short_signals = []
            neutral_signals = []

            for i, (_, row) in enumerate(predictions_df.iterrows()):
                signal = TradingSignal(
                    symbol=row["symbol"],
                    predicted_return=row["predicted_return"],
                    confidence=row["confidence"],
                    rank=i + 1,
                    signal_strength=self._get_signal_strength_threshold(
                        row["predicted_return"], long_threshold, short_threshold
                    ),
                    timestamp=current_time,
                )

                if (
                    row["predicted_return"] >= long_threshold
                    and len(long_signals) < self.max_positions
                ):
                    long_signals.append(signal)
                elif (
                    row["predicted_return"] <= short_threshold
                    and len(short_signals) < self.max_positions
                ):
                    short_signals.append(signal)
                else:
                    neutral_signals.append(signal)

            return {
                "long": long_signals,
                "short": short_signals,
                "neutral": neutral_signals,
            }

        elif method == "top_bottom":
            # Simple top/bottom N approach
            predictions_df = predictions_df.sort_values(
                "predicted_return", ascending=False
            )

            n_positions = min(self.max_positions, len(predictions_df) // 2)

            long_signals = []
            short_signals = []

            # Top N for long
            for i, (_, row) in enumerate(predictions_df.head(n_positions).iterrows()):
                long_signals.append(
                    TradingSignal(
                        symbol=row["symbol"],
                        predicted_return=row["predicted_return"],
                        confidence=row["confidence"],
                        rank=i + 1,
                        signal_strength="strong" if i < n_positions // 2 else "medium",
                        timestamp=current_time,
                    )
                )

            # Bottom N for short
            for i, (_, row) in enumerate(predictions_df.tail(n_positions).iterrows()):
                short_signals.append(
                    TradingSignal(
                        symbol=row["symbol"],
                        predicted_return=row["predicted_return"],
                        confidence=row["confidence"],
                        rank=len(predictions_df) - n_positions + i + 1,
                        signal_strength="strong" if i >= n_positions // 2 else "medium",
                        timestamp=current_time,
                    )
                )

            return {"long": long_signals, "short": short_signals, "neutral": []}

    def _get_signal_strength(self, quintile: int) -> str:
        """Get signal strength from quintile"""
        if quintile in [0, 4]:
            return "strong"
        elif quintile in [1, 3]:
            return "medium"
        else:
            return "weak"

    def _get_signal_strength_threshold(
        self, return_val: float, long_thresh: float, short_thresh: float
    ) -> str:
        """Get signal strength from threshold method"""
        if return_val >= long_thresh or return_val <= short_thresh:
            return "strong"
        elif return_val >= long_thresh * 0.8 or return_val <= short_thresh * 0.8:
            return "medium"
        else:
            return "weak"


class PortfolioConstructor:
    """
    Portfolio construction and optimization
    """

    def __init__(
        self,
        max_position_size: float = 0.05,
        sector_limit: float = 0.3,
        turnover_limit: float = 0.5,
    ):
        self.max_position_size = max_position_size
        self.sector_limit = sector_limit
        self.turnover_limit = turnover_limit

    def construct_portfolio(
        self,
        signals: Dict[str, List[TradingSignal]],
        current_positions: Dict[str, float] = None,
        sector_mapping: Dict[str, str] = None,
    ) -> Dict[str, Dict]:
        """Construct optimized portfolio from signals"""

        current_positions = current_positions or {}

        # Long portfolio
        long_portfolio = self._construct_long_short_portfolio(
            signals["long"], "long", current_positions, sector_mapping
        )

        # Short portfolio
        short_portfolio = self._construct_long_short_portfolio(
            signals["short"], "short", current_positions, sector_mapping
        )

        # Calculate portfolio statistics
        portfolio_stats = self._calculate_portfolio_stats(
            long_portfolio, short_portfolio, signals
        )

        return {
            "long_portfolio": long_portfolio,
            "short_portfolio": short_portfolio,
            "portfolio_stats": portfolio_stats,
            "execution_timestamp": pd.Timestamp.now(),
        }

    def _construct_long_short_portfolio(
        self,
        signals: List[TradingSignal],
        side: str,
        current_positions: Dict[str, float],
        sector_mapping: Dict[str, str] = None,
    ) -> Dict[str, Dict]:
        """Construct long or short portfolio"""

        portfolio = {}
        sector_exposure = {}
        total_weight = 0.0

        # Sort signals by predicted return (descending for long, ascending for short)
        sorted_signals = sorted(
            signals, key=lambda x: x.predicted_return, reverse=(side == "long")
        )

        for signal in sorted_signals:
            # Check position size limit
            if total_weight >= 1.0:
                break

            # Get sector
            sector = (
                sector_mapping.get(signal.symbol, "Unknown")
                if sector_mapping
                else "Unknown"
            )

            # Check sector limit
            if sector_exposure.get(sector, 0) >= self.sector_limit:
                continue

            # Calculate position size
            base_weight = min(self.max_position_size, (1.0 - total_weight))

            # Adjust for confidence
            confidence_adjustment = min(signal.confidence, 1.0)
            position_weight = base_weight * confidence_adjustment

            # Check turnover limit
            current_weight = current_positions.get(signal.symbol, 0.0)
            turnover = abs(position_weight - current_weight)

            if turnover > self.turnover_limit:
                position_weight = (
                    current_weight
                    + np.sign(position_weight - current_weight) * self.turnover_limit
                )

            # Add to portfolio
            if position_weight > 0.01:  # Minimum position size
                portfolio[signal.symbol] = {
                    "weight": position_weight,
                    "predicted_return": signal.predicted_return,
                    "confidence": signal.confidence,
                    "rank": signal.rank,
                    "signal_strength": signal.signal_strength,
                    "sector": sector,
                    "side": side,
                }

                total_weight += position_weight
                sector_exposure[sector] = (
                    sector_exposure.get(sector, 0) + position_weight
                )

        return portfolio

    def _calculate_portfolio_stats(
        self, long_portfolio: Dict, short_portfolio: Dict, signals: Dict
    ) -> Dict:
        """Calculate portfolio statistics"""

        stats = {}

        # Long portfolio stats
        if long_portfolio:
            long_returns = [pos["predicted_return"] for pos in long_portfolio.values()]
            long_weights = [pos["weight"] for pos in long_portfolio.values()]

            stats["long_expected_return"] = np.average(
                long_returns, weights=long_weights
            )
            stats["long_positions"] = len(long_portfolio)
            stats["long_total_weight"] = sum(long_weights)
        else:
            stats.update(
                {"long_expected_return": 0, "long_positions": 0, "long_total_weight": 0}
            )

        # Short portfolio stats
        if short_portfolio:
            short_returns = [
                pos["predicted_return"] for pos in short_portfolio.values()
            ]
            short_weights = [pos["weight"] for pos in short_portfolio.values()]

            stats["short_expected_return"] = np.average(
                short_returns, weights=short_weights
            )
            stats["short_positions"] = len(short_portfolio)
            stats["short_total_weight"] = sum(short_weights)
        else:
            stats.update(
                {
                    "short_expected_return": 0,
                    "short_positions": 0,
                    "short_total_weight": 0,
                }
            )

        # Overall stats
        stats["total_positions"] = stats["long_positions"] + stats["short_positions"]
        stats["net_expected_return"] = (
            stats["long_expected_return"] - stats["short_expected_return"]
        )

        # Signal quality stats
        all_signals = signals["long"] + signals["short"] + signals["neutral"]
        if all_signals:
            stats["avg_confidence"] = np.mean([s.confidence for s in all_signals])
            stats["signal_coverage"] = len(all_signals)
        else:
            stats.update({"avg_confidence": 0, "signal_coverage": 0})

        return stats


def create_sample_predictions(symbols: List[str], n_samples: int = None) -> np.ndarray:
    """Create sample predictions for testing"""
    if n_samples is None:
        n_samples = len(symbols)

    np.random.seed(42)

    # Generate quantile predictions [10th, 50th, 90th percentiles]
    predictions = np.random.normal(0, 0.02, (n_samples, 3))

    # Ensure proper ordering of quantiles
    predictions = np.sort(predictions, axis=1)

    return predictions


if __name__ == "__main__":
    # Test the ranking system
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"]

    # Create sample predictions
    predictions = create_sample_predictions(symbols)

    # Initialize ranking system
    ranking_system = StockRankingSystem(
        liquidity_threshold=len(symbols), max_positions=4  # Use all symbols for testing
    )

    # Process predictions
    predictions_df = ranking_system.process_predictions(
        predictions, symbols, prediction_type="quantile"
    )

    print("Predictions DataFrame:")
    print(predictions_df)

    # Generate trading signals
    signals = ranking_system.generate_trading_signals(predictions_df, method="quintile")

    print(f"\nGenerated {len(signals['long'])} long signals:")
    for signal in signals["long"]:
        print(
            f"  {signal.symbol}: {signal.predicted_return:.4f} (confidence: {signal.confidence:.3f})"
        )

    print(f"\nGenerated {len(signals['short'])} short signals:")
    for signal in signals["short"]:
        print(
            f"  {signal.symbol}: {signal.predicted_return:.4f} (confidence: {signal.confidence:.3f})"
        )

    # Construct portfolio
    portfolio_constructor = PortfolioConstructor()
    portfolio = portfolio_constructor.construct_portfolio(signals)

    print("\nPortfolio Statistics:")
    for key, value in portfolio["portfolio_stats"].items():
        print(f"  {key}: {value}")

    print(f"\nLong Portfolio ({len(portfolio['long_portfolio'])} positions):")
    for symbol, pos in portfolio["long_portfolio"].items():
        print(
            f"  {symbol}: {pos['weight']:.3f} weight, {pos['predicted_return']:.4f} return"
        )
