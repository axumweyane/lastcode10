"""
Strategy 4: Volatility Arbitrage — trade the IV-RV spread.

Edge: implied volatility exceeds realized volatility ~85% of the time.
This persistent overpricing (the "variance risk premium") is one of the
most documented anomalies in options markets. Selling options when IV >> RV
and buying when IV << RV captures this spread.

Signal:
  - IV - RV > 5 vol pts → SELL vol (short straddle/strangle on the symbol)
  - RV - IV > 3 vol pts → BUY vol (long straddle for gamma scalping)

Enhancement: use TFT's 5-day volatility forecast to time entries.
If TFT predicts low vol ahead and IV is high → strong sell signal.
If TFT predicts high vol ahead and IV is low → strong buy signal.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from strategies.base import (
    AlphaScore,
    BaseStrategy,
    SignalDirection,
    StrategyOutput,
    StrategyPerformance,
)
from strategies.options.config import VolArbConfig
from strategies.options.infrastructure.vol_monitor import VolMonitor

logger = logging.getLogger(__name__)


class VolatilityArbitrage(BaseStrategy):

    def __init__(self, config: Optional[VolArbConfig] = None):
        self.config = config or VolArbConfig.from_env()
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._vol_monitor = VolMonitor(lookback_days=self.config.garch_lookback_days)
        self._initialized = False

    @property
    def name(self) -> str:
        return "vol_arb"

    @property
    def description(self) -> str:
        return "Trade the implied-vs-realized volatility spread with GARCH + TFT timing"

    def initialize(self, historical_data: pd.DataFrame) -> None:
        self._initialized = True

    def generate_signals(
        self,
        data: pd.DataFrame,
        tft_vol_forecasts: Optional[dict] = None,
    ) -> StrategyOutput:
        """
        Args:
            data: price data [symbol, timestamp, close, volume]
            tft_vol_forecasts: optional {symbol: predicted_vol} from TFT model
        """
        if not self._initialized:
            self.initialize(data)

        scores: List[AlphaScore] = []
        symbols = data["symbol"].unique()

        for symbol in symbols:
            sym_data = data[data["symbol"] == symbol].sort_values("timestamp")
            if len(sym_data) < 100:
                continue

            close = sym_data["close"]
            returns = close.pct_change().dropna()
            rv_21d = float(returns.tail(21).std() * np.sqrt(252))

            # Estimate IV (in production, get from chain; here use rv * premium)
            estimated_iv = rv_21d * 1.15
            metrics = self._vol_monitor.compute(symbol, sym_data, estimated_iv)

            iv_rv_spread = metrics.iv_rv_spread
            garch_forecast = metrics.garch_forecast

            # TFT enhancement: if TFT predicts vol, blend with GARCH
            if tft_vol_forecasts and symbol in tft_vol_forecasts:
                tft_vol = tft_vol_forecasts[symbol]
                # Blend: 60% GARCH + 40% TFT
                blended_rv = 0.6 * garch_forecast + 0.4 * tft_vol
            else:
                blended_rv = garch_forecast

            # Adjusted spread: IV vs blended RV forecast
            adjusted_spread = estimated_iv - blended_rv

            # Determine signal
            if adjusted_spread > self.config.iv_rv_entry_threshold:
                # IV overpriced → SELL vol (short options)
                direction = SignalDirection.SHORT
                raw_score = -adjusted_spread / 0.10  # normalize: 10 vol pts → score -1
                trade_type = "sell_vol"
            elif adjusted_spread < -self.config.iv_rv_entry_threshold:
                # IV underpriced → BUY vol (long straddle)
                direction = SignalDirection.LONG
                raw_score = -adjusted_spread / 0.10  # positive score for buy
                trade_type = "buy_vol"
            else:
                continue  # no signal

            confidence = min(abs(adjusted_spread) / 0.10, 0.95)

            scores.append(
                AlphaScore(
                    symbol=symbol,
                    score=raw_score,
                    raw_score=adjusted_spread,
                    confidence=confidence,
                    direction=direction,
                    metadata={
                        "strategy_type": "vol_arb",
                        "trade_type": trade_type,
                        "iv_estimate": round(estimated_iv, 4),
                        "rv_21d": round(rv_21d, 4),
                        "garch_forecast": round(garch_forecast, 4),
                        "blended_rv": round(blended_rv, 4),
                        "iv_rv_spread": round(iv_rv_spread, 4),
                        "adjusted_spread": round(adjusted_spread, 4),
                        "iv_rank": round(metrics.iv_rank, 1),
                        "vol_regime": metrics.vol_regime,
                    },
                )
            )

        scores.sort(key=lambda s: abs(s.score), reverse=True)
        sells = [s for s in scores if s.direction == SignalDirection.SHORT]
        buys = [s for s in scores if s.direction == SignalDirection.LONG]

        logger.info(
            "%s: %d signals (%d sell_vol, %d buy_vol)",
            self.name,
            len(scores),
            len(sells),
            len(buys),
        )

        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=scores,
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
        )

    def get_performance(self) -> StrategyPerformance:
        return self._performance
