"""
LLM-powered signal analyst agent.

Interprets ensemble output after each daily pipeline run and generates
plain-language explanations using a local Ollama instance (default model:
qwen2.5:32b).  Detects patterns — consensus, conflict, weight shifts,
regime changes — and feeds them to the LLM for richer analysis.

Usage::

    analyst = SignalAnalyst()
    analysis = await analyst.analyze(
        signals=combined_signals,
        weights=strategy_weights,
        regime="calm_trending",
        risk_summary=risk_report.to_dict(),
    )
    print(analysis.summary)  # 3-sentence plain-language summary
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:32b"
DEFAULT_TIMEOUT_S = 60
CONSENSUS_THRESHOLD = 0.80
CONFLICT_THRESHOLD_LOW = 0.40
CONFLICT_THRESHOLD_HIGH = 0.60
WEIGHT_SHIFT_THRESHOLD = 0.10


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class PatternFlags:
    """Detected patterns in the ensemble output."""

    strong_consensus: bool = False
    consensus_direction: Optional[str] = None  # "long" or "short"
    consensus_pct: float = 0.0
    conflicted: bool = False
    weight_shifts: List[Dict[str, Any]] = field(default_factory=list)
    regime_changed: bool = False
    prior_regime: Optional[str] = None
    current_regime: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "strong_consensus": self.strong_consensus,
            "consensus_direction": self.consensus_direction,
            "consensus_pct": round(self.consensus_pct, 3),
            "conflicted": self.conflicted,
            "weight_shifts": self.weight_shifts,
            "regime_changed": self.regime_changed,
            "prior_regime": self.prior_regime,
            "current_regime": self.current_regime,
        }

    def describe(self) -> str:
        parts = []
        if self.strong_consensus:
            parts.append(
                f"Strong consensus: {self.consensus_pct:.0%} of strategies "
                f"agree on {self.consensus_direction}"
            )
        if self.conflicted:
            parts.append("Ensemble is conflicted — near 50/50 directional split")
        if self.weight_shifts:
            shifted = ", ".join(
                f"{s['strategy']} ({s['change']:+.1%})" for s in self.weight_shifts
            )
            parts.append(f"Significant weight shifts: {shifted}")
        if self.regime_changed:
            parts.append(f"Regime changed: {self.prior_regime} → {self.current_regime}")
        return "; ".join(parts) if parts else "No unusual patterns detected"


@dataclass
class SignalAnalysis:
    """Structured output from the LLM signal analyst."""

    timestamp: str
    summary: str  # 3-sentence market summary
    patterns: str  # unusual pattern description
    confidence: str  # "low", "medium", or "high"
    flags: PatternFlags = field(default_factory=PatternFlags)
    regime: str = ""
    top_signals: List[Dict] = field(default_factory=list)
    raw_llm_response: str = ""
    model_used: str = ""
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "summary": self.summary,
            "patterns": self.patterns,
            "confidence": self.confidence,
            "flags": self.flags.to_dict(),
            "regime": self.regime,
            "top_signals": self.top_signals,
            "model_used": self.model_used,
            "latency_s": round(self.latency_s, 2),
        }

    def to_report_line(self) -> str:
        """One-paragraph version for Discord reports."""
        flag_str = self.flags.describe()
        return (
            f"**LLM Analysis** ({self.confidence} confidence, {self.model_used}):\n"
            f"{self.summary}\n"
            f"Flags: {flag_str}"
        )


# ── Pattern detection ──────────────────────────────────────────────────────────


def detect_patterns(
    signals: List[Dict],
    weights: Dict[str, float],
    prior_weights: Optional[Dict[str, float]],
    regime: str,
    prior_regime: Optional[str],
) -> PatternFlags:
    """
    Detect notable patterns in ensemble output.

    Args:
        signals: List of CombinedSignal-like dicts with 'direction' key.
        weights: Current strategy weights {name: weight}.
        prior_weights: Previous day's weights (None on first run).
        regime: Current regime string.
        prior_regime: Previous day's regime string.
    """
    flags = PatternFlags()

    # 1. Consensus / conflict detection
    if signals:
        directions = [s.get("direction", "neutral") for s in signals]
        long_count = sum(1 for d in directions if d == "long")
        short_count = sum(1 for d in directions if d == "short")
        total_directional = long_count + short_count

        if total_directional > 0:
            long_pct = long_count / total_directional
            short_pct = short_count / total_directional

            if long_pct >= CONSENSUS_THRESHOLD:
                flags.strong_consensus = True
                flags.consensus_direction = "long"
                flags.consensus_pct = long_pct
            elif short_pct >= CONSENSUS_THRESHOLD:
                flags.strong_consensus = True
                flags.consensus_direction = "short"
                flags.consensus_pct = short_pct

            if CONFLICT_THRESHOLD_LOW <= long_pct <= CONFLICT_THRESHOLD_HIGH:
                flags.conflicted = True

    # 2. Weight shift detection
    if prior_weights and weights:
        for name, current_w in weights.items():
            prior_w = prior_weights.get(name, current_w)
            change = current_w - prior_w
            if abs(change) > WEIGHT_SHIFT_THRESHOLD:
                flags.weight_shifts.append(
                    {
                        "strategy": name,
                        "prior": round(prior_w, 4),
                        "current": round(current_w, 4),
                        "change": round(change, 4),
                    }
                )

    # 3. Regime change detection
    flags.current_regime = regime
    flags.prior_regime = prior_regime
    if prior_regime is not None and regime != prior_regime:
        flags.regime_changed = True

    return flags


# ── Prompt construction ────────────────────────────────────────────────────────


def build_prompt(
    signals: List[Dict],
    weights: Dict[str, float],
    regime: str,
    risk_summary: Dict[str, Any],
    flags: PatternFlags,
) -> str:
    """
    Construct the LLM prompt with all context.

    Returns a structured prompt asking for: 3-sentence summary, unusual
    patterns, and confidence assessment.
    """
    # Top 5 signals by absolute score
    sorted_signals = sorted(
        signals, key=lambda s: abs(s.get("combined_score", 0)), reverse=True
    )
    top5 = sorted_signals[:5]

    signals_text = "\n".join(
        f"  {i+1}. {s.get('symbol', '?')}: score={s.get('combined_score', 0):.3f}, "
        f"direction={s.get('direction', '?')}, confidence={s.get('confidence', 0):.2f}, "
        f"strategies={s.get('contributing_strategies', {})}"
        for i, s in enumerate(top5)
    )

    weights_text = "\n".join(
        f"  {name}: {w:.1%}" for name, w in sorted(weights.items(), key=lambda x: -x[1])
    )

    risk_text = (
        f"  Drawdown: {risk_summary.get('portfolio_drawdown', 0):.2%}\n"
        f"  VaR-99: {risk_summary.get('var_99', 0):.4f}\n"
        f"  CVaR-95: {risk_summary.get('cvar_95', 0):.4f}\n"
        f"  Sharpe (21d): {risk_summary.get('sharpe_21d', 0):.2f}\n"
        f"  Killed strategies: {risk_summary.get('killed_strategies', [])}\n"
        f"  Correlation alerts: {risk_summary.get('correlation_alerts', [])}"
    )

    flags_text = flags.describe()

    prompt = f"""You are a quantitative trading analyst reviewing today's ensemble signal output.

MARKET REGIME: {regime}

TOP 5 SIGNALS (by absolute score):
{signals_text}

STRATEGY WEIGHTS:
{weights_text}

RISK SUMMARY:
{risk_text}

DETECTED PATTERNS:
{flags_text}

Based on this data, provide your analysis in EXACTLY this JSON format:
{{
  "summary": "<exactly 3 sentences: what the ensemble is saying, why, and what to watch>",
  "patterns": "<describe any unusual patterns or notable observations>",
  "confidence": "<one of: low, medium, high>"
}}

Rules:
- Summary must be exactly 3 sentences
- Confidence reflects how actionable the signals are: high = clear consensus with good risk metrics, medium = mixed signals or elevated risk, low = conflicted or degraded conditions
- Be specific about ticker names and directions
- Reference the regime and risk metrics in your analysis"""

    return prompt


# ── Response parsing ───────────────────────────────────────────────────────────


def parse_llm_response(raw: str) -> Dict[str, str]:
    """
    Parse the LLM JSON response, with fallback for malformed output.

    Returns dict with keys: summary, patterns, confidence.
    """
    # Try to extract JSON from the response
    # The LLM may wrap it in markdown code blocks
    cleaned = raw.strip()
    if "```json" in cleaned:
        start = cleaned.index("```json") + 7
        end = cleaned.index("```", start)
        cleaned = cleaned[start:end].strip()
    elif "```" in cleaned:
        start = cleaned.index("```") + 3
        end = cleaned.index("```", start)
        cleaned = cleaned[start:end].strip()

    try:
        parsed = json.loads(cleaned)
        return {
            "summary": str(parsed.get("summary", "Analysis unavailable.")),
            "patterns": str(parsed.get("patterns", "None detected.")),
            "confidence": _normalize_confidence(
                str(parsed.get("confidence", "medium"))
            ),
        }
    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse LLM JSON response, using raw text")
        # Fallback: use the raw text as summary
        summary = raw[:500].strip() if raw else "LLM response could not be parsed."
        return {
            "summary": summary,
            "patterns": "Parse error — raw response used as summary.",
            "confidence": "low",
        }


def _normalize_confidence(value: str) -> str:
    value = value.lower().strip()
    if value in ("low", "medium", "high"):
        return value
    if "high" in value:
        return "high"
    if "low" in value:
        return "low"
    return "medium"


# ── Ollama client ──────────────────────────────────────────────────────────────


class OllamaClient:
    """Async HTTP client for local Ollama API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ):
        self.base_url = (
            base_url or os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        ).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        self.timeout_s = timeout_s

    async def generate(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to Ollama and return the response text.

        Returns None if Ollama is unreachable or errors.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning(
                            "Ollama returned %d: %s", resp.status, body[:200]
                        )
                        return None
                    data = await resp.json()
                    return data.get("response", "")
        except aiohttp.ClientError as e:
            logger.warning("Ollama unreachable at %s: %s", self.base_url, e)
            return None
        except asyncio.TimeoutError:
            logger.warning("Ollama request timed out after %ds", self.timeout_s)
            return None
        except Exception as e:
            logger.warning("Ollama client error: %s", e)
            return None


# Need asyncio for TimeoutError handling
import asyncio

# ── Signal Analyst ─────────────────────────────────────────────────────────────


class SignalAnalyst:
    """
    Orchestrates LLM-based signal analysis.

    Detects patterns, constructs prompts, calls Ollama, parses responses,
    and caches the latest analysis.
    """

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        allow_manual: bool = False,
    ):
        self.client = client or OllamaClient()
        self.allow_manual = allow_manual or os.getenv(
            "LLM_ANALYST_ON_MANUAL", "false"
        ).lower() in ("true", "1", "yes")
        self._last_analysis: Optional[SignalAnalysis] = None
        self._prior_weights: Optional[Dict[str, float]] = None
        self._prior_regime: Optional[str] = None
        self._run_count: int = 0

    async def analyze(
        self,
        signals: List[Dict],
        weights: Dict[str, float],
        regime: str,
        risk_summary: Dict[str, Any],
        is_manual: bool = False,
    ) -> Optional[SignalAnalysis]:
        """
        Run a full signal analysis cycle.

        Args:
            signals: List of CombinedSignal-like dicts.
            weights: Strategy weights {name: float}.
            regime: Current regime string.
            risk_summary: Risk report dict (from RiskReport.to_dict()).
            is_manual: True if triggered via /run-now.

        Returns:
            SignalAnalysis or None if skipped (rate limit / manual block).
        """
        # Rate limiting: skip manual triggers unless allowed
        if is_manual and not self.allow_manual:
            logger.info(
                "LLM analyst skipping manual trigger (set LLM_ANALYST_ON_MANUAL=true)"
            )
            return self._last_analysis

        self._run_count += 1

        start = time.monotonic()

        # 1. Detect patterns
        flags = detect_patterns(
            signals=signals,
            weights=weights,
            prior_weights=self._prior_weights,
            regime=regime,
            prior_regime=self._prior_regime,
        )

        # 2. Build prompt
        prompt = build_prompt(
            signals=signals,
            weights=weights,
            regime=regime,
            risk_summary=risk_summary,
            flags=flags,
        )

        # 3. Call LLM
        raw_response = await self.client.generate(prompt)

        latency = time.monotonic() - start

        if raw_response is None:
            # Graceful fallback — produce analysis from pattern detection alone
            analysis = SignalAnalysis(
                timestamp=datetime.now(timezone.utc).isoformat(),
                summary="LLM unavailable. " + flags.describe(),
                patterns=flags.describe(),
                confidence="low",
                flags=flags,
                regime=regime,
                top_signals=sorted(
                    signals, key=lambda s: abs(s.get("combined_score", 0)), reverse=True
                )[:5],
                raw_llm_response="",
                model_used=self.client.model,
                latency_s=latency,
            )
        else:
            parsed = parse_llm_response(raw_response)
            analysis = SignalAnalysis(
                timestamp=datetime.now(timezone.utc).isoformat(),
                summary=parsed["summary"],
                patterns=parsed["patterns"],
                confidence=parsed["confidence"],
                flags=flags,
                regime=regime,
                top_signals=sorted(
                    signals, key=lambda s: abs(s.get("combined_score", 0)), reverse=True
                )[:5],
                raw_llm_response=raw_response,
                model_used=self.client.model,
                latency_s=latency,
            )

        # 4. Cache and update priors
        self._last_analysis = analysis
        self._prior_weights = dict(weights)
        self._prior_regime = regime

        logger.info(
            "Signal analysis complete: confidence=%s, latency=%.1fs, model=%s",
            analysis.confidence,
            analysis.latency_s,
            analysis.model_used,
        )

        return analysis

    @property
    def last_analysis(self) -> Optional[SignalAnalysis]:
        return self._last_analysis

    @property
    def run_count(self) -> int:
        return self._run_count
