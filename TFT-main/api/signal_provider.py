"""
Public-facing signal provider REST API.

Exposes APEX ensemble signals with API key authentication, rate limiting,
and ETag caching.  Mounted as a sub-application under ``/api/v1/`` in the
paper trader.

Endpoints:
    GET /signals             — all current signals
    GET /signals/{symbol}    — signal for one symbol
    GET /signals/history/{symbol}?days=7  — historical signals from DB
    GET /signals/regime      — current + recent regime state
    GET /signals/weights     — strategy weights (fixed + Bayesian)
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ── Rate limiter ─────────────────────────────────────────────────────────────

class RateLimiter:
    """In-memory per-key rate limiter.  Resets every ``window_s`` seconds."""

    def __init__(self, max_requests: int = 100, window_s: int = 60):
        self.max_requests = max_requests
        self.window_s = window_s
        self._buckets: Dict[str, list] = {}   # key → [count, window_start]

    def check(self, key: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        now = time.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None or now - bucket[1] >= self.window_s:
            self._buckets[key] = [1, now]
            return True
        if bucket[0] >= self.max_requests:
            return False
        bucket[0] += 1
        return True

    def remaining(self, key: str) -> int:
        bucket = self._buckets.get(key)
        if bucket is None:
            return self.max_requests
        now = time.monotonic()
        if now - bucket[1] >= self.window_s:
            return self.max_requests
        return max(0, self.max_requests - bucket[0])


# ── Signal cache ─────────────────────────────────────────────────────────────

@dataclass
class SignalCache:
    """In-memory cache for current pipeline signals."""
    signals: List[Dict[str, Any]] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    bayesian_weights: Optional[Dict[str, float]] = None
    bayesian_state: Optional[List[Dict]] = None
    regime: Optional[str] = None
    regime_detail: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    etag: str = ""

    def refresh(
        self,
        signals: List[Dict],
        weights: Dict[str, float],
        regime: Optional[str],
        regime_detail: Optional[Dict] = None,
        bayesian_weights: Optional[Dict[str, float]] = None,
        bayesian_state: Optional[List[Dict]] = None,
    ) -> None:
        self.signals = signals
        self.weights = weights
        self.regime = regime
        self.regime_detail = regime_detail or {}
        self.bayesian_weights = bayesian_weights
        self.bayesian_state = bayesian_state
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.etag = self._compute_etag()

    def _compute_etag(self) -> str:
        content = f"{self.timestamp}:{len(self.signals)}:{self.regime}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def get_signal(self, symbol: str) -> Optional[Dict]:
        for s in self.signals:
            if s.get("symbol", "").upper() == symbol.upper():
                return s
        return None


# ── Factory ──────────────────────────────────────────────────────────────────

def create_signal_api(
    api_key: str,
    cache: SignalCache,
    db_query_fn: Optional[Callable] = None,
    rate_limit: int = 100,
) -> FastAPI:
    """
    Create the signal provider FastAPI sub-application.

    Args:
        api_key: Required API key for authentication.
        cache: Shared SignalCache instance (refreshed by the paper trader).
        db_query_fn: Callable(query, params) → list of dicts for history queries.
        rate_limit: Max requests per minute per key.
    """
    app = FastAPI(title="APEX Signal Provider", version="1.0.0")
    limiter = RateLimiter(max_requests=rate_limit)

    # ── Auth + rate limit middleware ──────────────────────────────────────

    @app.middleware("http")
    async def auth_and_rate_limit(request: Request, call_next):
        # Skip docs endpoints
        if request.url.path in ("/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # API key check
        provided_key = request.headers.get("x-api-key", "")
        if not provided_key:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing X-API-Key header"},
            )
        if provided_key != api_key:
            return JSONResponse(
                status_code=403,
                content={"error": "Invalid API key"},
            )

        # Rate limit
        if not limiter.check(provided_key):
            remaining = limiter.remaining(provided_key)
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "retry_after_s": limiter.window_s},
                headers={"Retry-After": str(limiter.window_s)},
            )

        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(limiter.remaining(provided_key))

        return response

    # ── Helpers ───────────────────────────────────────────────────────────

    def _etag_response(data: Any, response: Response) -> Any:
        """Add ETag header and check If-None-Match."""
        if cache.etag:
            response.headers["ETag"] = f'"{cache.etag}"'
        return data

    def _signal_response(signal: Dict) -> Dict:
        """Standardize a signal dict to the public schema."""
        return {
            "timestamp": cache.timestamp,
            "symbol": signal.get("symbol", ""),
            "direction": signal.get("direction", "neutral"),
            "score": round(signal.get("combined_score", 0.0), 6),
            "confidence": round(signal.get("confidence", 0.0), 4),
            "regime": cache.regime,
            "strategies": signal.get("contributing_strategies", {}),
            "metadata": {
                k: v for k, v in signal.items()
                if k not in ("symbol", "direction", "combined_score",
                             "confidence", "contributing_strategies")
            },
        }

    # ── Endpoints ─────────────────────────────────────────────────────────

    @app.get("/signals")
    async def get_signals(response: Response):
        """Return all current signals with scores, regime, and timestamp."""
        result = {
            "timestamp": cache.timestamp,
            "regime": cache.regime,
            "signal_count": len(cache.signals),
            "signals": [_signal_response(s) for s in cache.signals],
        }
        return _etag_response(result, response)

    @app.get("/signals/regime")
    async def get_regime(response: Response):
        """Return current regime state and recent history."""
        result = {
            "timestamp": cache.timestamp,
            "current_regime": cache.regime,
            "detail": cache.regime_detail,
        }
        return _etag_response(result, response)

    @app.get("/signals/weights")
    async def get_weights(response: Response):
        """Return current strategy weights (fixed + Bayesian if enabled)."""
        result = {
            "timestamp": cache.timestamp,
            "weights": cache.weights,
            "bayesian_enabled": cache.bayesian_weights is not None,
        }
        if cache.bayesian_weights is not None:
            result["bayesian_weights"] = cache.bayesian_weights
        if cache.bayesian_state is not None:
            result["bayesian_state"] = cache.bayesian_state
        return _etag_response(result, response)

    @app.get("/signals/history/{symbol}")
    async def get_signal_history(
        symbol: str,
        response: Response,
        days: int = Query(default=7, ge=1, le=90),
    ):
        """Return signal history for a symbol from the database."""
        if db_query_fn is None:
            raise HTTPException(status_code=503, detail="Database not available")

        start_date = date.today() - timedelta(days=days)
        try:
            rows = db_query_fn(
                """SELECT signal_date, strategy_name, symbol, score,
                          confidence, direction, metadata, created_at
                   FROM paper_strategy_signals
                   WHERE symbol = %s AND signal_date >= %s
                   ORDER BY signal_date DESC, strategy_name""",
                (symbol.upper(), start_date),
            )
        except Exception as e:
            logger.error("Signal history query failed: %s", e)
            raise HTTPException(status_code=500, detail="Database query failed")

        signals = []
        for row in rows:
            signals.append({
                "date": str(row[0]),
                "strategy": row[1],
                "symbol": row[2],
                "score": row[3],
                "confidence": row[4],
                "direction": row[5],
                "metadata": row[6] if row[6] else {},
            })

        result = {
            "symbol": symbol.upper(),
            "days": days,
            "start_date": str(start_date),
            "count": len(signals),
            "signals": signals,
        }
        return _etag_response(result, response)

    @app.get("/signals/{symbol}")
    async def get_signal(symbol: str, response: Response):
        """Return signal for a specific symbol with full metadata."""
        signal = cache.get_signal(symbol)
        if signal is None:
            raise HTTPException(
                status_code=404,
                detail=f"No signal found for {symbol.upper()}",
            )
        result = _signal_response(signal)
        return _etag_response(result, response)

    return app
