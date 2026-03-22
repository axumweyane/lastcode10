"""Tests for the LLM signal analyst agent."""

import asyncio
import json
import os
import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.signal_analyst import (
    SignalAnalyst,
    SignalAnalysis,
    PatternFlags,
    OllamaClient,
    detect_patterns,
    build_prompt,
    parse_llm_response,
    _normalize_confidence,
    CONSENSUS_THRESHOLD,
    WEIGHT_SHIFT_THRESHOLD,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_signals(directions: list) -> list:
    """Create signal dicts with given directions."""
    return [
        {
            "symbol": f"SYM{i}",
            "combined_score": 0.5 if d == "long" else -0.5,
            "confidence": 0.7,
            "direction": d,
            "contributing_strategies": {"strat_a": 0.3},
        }
        for i, d in enumerate(directions)
    ]


def _make_weights() -> dict:
    return {"momentum": 0.3, "pairs": 0.25, "tft": 0.25, "mean_reversion": 0.2}


def _make_risk_summary() -> dict:
    return {
        "portfolio_drawdown": 0.05,
        "var_99": 0.02,
        "cvar_95": 0.03,
        "sharpe_21d": 1.2,
        "killed_strategies": [],
        "correlation_alerts": [],
    }


class FakeOllamaClient:
    """Mock Ollama client that returns a configurable response."""

    def __init__(self, response: str = None, fail: bool = False):
        self.model = "test-model"
        self._response = response or json.dumps(
            {
                "summary": "Markets are bullish. Momentum strategies dominate. Watch for reversal signals.",
                "patterns": "Strong consensus among strategies.",
                "confidence": "high",
            }
        )
        self._fail = fail
        self.call_count = 0
        self.last_prompt = None

    async def generate(self, prompt: str):
        self.call_count += 1
        self.last_prompt = prompt
        if self._fail:
            return None
        return self._response


# ── 1. Prompt construction ───────────────────────────────────────────────────


class TestPromptConstruction:
    """Test that prompts include all required fields."""

    def test_prompt_includes_regime(self):
        signals = _make_signals(["long", "long", "short"])
        prompt = build_prompt(
            signals,
            _make_weights(),
            "calm_trending",
            _make_risk_summary(),
            PatternFlags(),
        )
        assert "calm_trending" in prompt

    def test_prompt_includes_top_signals(self):
        signals = _make_signals(["long", "short", "long", "long", "short", "long"])
        prompt = build_prompt(
            signals, _make_weights(), "volatile", _make_risk_summary(), PatternFlags()
        )
        assert "SYM0" in prompt
        # Should include up to 5 signals
        assert "SYM4" in prompt

    def test_prompt_includes_strategy_weights(self):
        weights = {"momentum": 0.4, "pairs": 0.3, "tft": 0.3}
        prompt = build_prompt(
            _make_signals(["long"]),
            weights,
            "calm",
            _make_risk_summary(),
            PatternFlags(),
        )
        assert "momentum" in prompt
        assert "40.0%" in prompt

    def test_prompt_includes_risk_metrics(self):
        risk = {
            "portfolio_drawdown": 0.12,
            "var_99": 0.035,
            "cvar_95": 0.045,
            "sharpe_21d": 0.8,
            "killed_strategies": ["bad_strat"],
            "correlation_alerts": [{"a": "s1", "b": "s2", "corr": 0.9}],
        }
        prompt = build_prompt(
            _make_signals(["long"]), _make_weights(), "calm", risk, PatternFlags()
        )
        assert "12.00%" in prompt
        assert "bad_strat" in prompt

    def test_prompt_includes_pattern_flags(self):
        flags = PatternFlags(
            strong_consensus=True, consensus_direction="long", consensus_pct=0.9
        )
        prompt = build_prompt(
            _make_signals(["long"]),
            _make_weights(),
            "calm",
            _make_risk_summary(),
            flags,
        )
        assert "Strong consensus" in prompt
        assert "long" in prompt

    def test_prompt_asks_for_json_format(self):
        prompt = build_prompt(
            _make_signals(["long"]),
            _make_weights(),
            "calm",
            _make_risk_summary(),
            PatternFlags(),
        )
        assert '"summary"' in prompt
        assert '"patterns"' in prompt
        assert '"confidence"' in prompt

    def test_prompt_requests_three_sentences(self):
        prompt = build_prompt(
            _make_signals(["long"]),
            _make_weights(),
            "calm",
            _make_risk_summary(),
            PatternFlags(),
        )
        assert "3 sentences" in prompt or "three sentences" in prompt.lower()


# ── 2. Graceful fallback ────────────────────────────────────────────────────


class TestGracefulFallback:
    """Test behavior when Ollama is unreachable."""

    def test_returns_analysis_when_ollama_fails(self):
        client = FakeOllamaClient(fail=True)
        analyst = SignalAnalyst(client=client)
        analysis = _run(
            analyst.analyze(
                signals=_make_signals(["long", "short"]),
                weights=_make_weights(),
                regime="volatile",
                risk_summary=_make_risk_summary(),
            )
        )
        assert analysis is not None
        assert "LLM unavailable" in analysis.summary
        assert analysis.confidence == "low"

    def test_fallback_includes_pattern_description(self):
        client = FakeOllamaClient(fail=True)
        analyst = SignalAnalyst(client=client)
        # Set prior regime to trigger regime change flag
        analyst._prior_regime = "calm_trending"
        analysis = _run(
            analyst.analyze(
                signals=_make_signals(["long"] * 10),
                weights=_make_weights(),
                regime="volatile_choppy",
                risk_summary=_make_risk_summary(),
            )
        )
        assert (
            "Regime changed" in analysis.summary
            or "Regime changed" in analysis.patterns
        )

    def test_fallback_still_populates_top_signals(self):
        client = FakeOllamaClient(fail=True)
        analyst = SignalAnalyst(client=client)
        analysis = _run(
            analyst.analyze(
                signals=_make_signals(["long", "short", "long"]),
                weights=_make_weights(),
                regime="calm",
                risk_summary=_make_risk_summary(),
            )
        )
        assert len(analysis.top_signals) > 0

    def test_ollama_client_returns_none_on_connection_error(self):
        """Real OllamaClient against non-existent server."""
        client = OllamaClient(base_url="http://127.0.0.1:1", timeout_s=1)
        result = _run(client.generate("test"))
        assert result is None


# ── 3. Pattern detection ────────────────────────────────────────────────────


class TestPatternDetection:
    """Test pattern detection flags."""

    def test_strong_consensus_long(self):
        signals = _make_signals(["long"] * 9 + ["short"])  # 90% long
        flags = detect_patterns(signals, _make_weights(), None, "calm", None)
        assert flags.strong_consensus is True
        assert flags.consensus_direction == "long"
        assert flags.consensus_pct >= CONSENSUS_THRESHOLD

    def test_strong_consensus_short(self):
        signals = _make_signals(["short"] * 9 + ["long"])  # 90% short
        flags = detect_patterns(signals, _make_weights(), None, "calm", None)
        assert flags.strong_consensus is True
        assert flags.consensus_direction == "short"

    def test_no_consensus(self):
        signals = _make_signals(["long"] * 5 + ["short"] * 5)  # 50/50
        flags = detect_patterns(signals, _make_weights(), None, "calm", None)
        assert flags.strong_consensus is False

    def test_conflicted_near_5050(self):
        signals = _make_signals(["long"] * 5 + ["short"] * 5)  # exact 50/50
        flags = detect_patterns(signals, _make_weights(), None, "calm", None)
        assert flags.conflicted is True

    def test_not_conflicted_when_clear(self):
        signals = _make_signals(["long"] * 8 + ["short"] * 2)  # 80/20
        flags = detect_patterns(signals, _make_weights(), None, "calm", None)
        assert flags.conflicted is False

    def test_weight_shift_detected(self):
        prior = {"momentum": 0.3, "pairs": 0.25, "tft": 0.25, "mean_reversion": 0.2}
        current = {"momentum": 0.45, "pairs": 0.20, "tft": 0.25, "mean_reversion": 0.10}
        flags = detect_patterns([], current, prior, "calm", "calm")
        assert len(flags.weight_shifts) >= 1
        shifted_names = [s["strategy"] for s in flags.weight_shifts]
        assert "momentum" in shifted_names  # +0.15 > threshold

    def test_no_weight_shift_when_stable(self):
        w = {"momentum": 0.3, "pairs": 0.25}
        flags = detect_patterns([], w, w, "calm", "calm")
        assert len(flags.weight_shifts) == 0

    def test_regime_change_detected(self):
        flags = detect_patterns(
            [], _make_weights(), None, "volatile_choppy", "calm_trending"
        )
        assert flags.regime_changed is True
        assert flags.prior_regime == "calm_trending"
        assert flags.current_regime == "volatile_choppy"

    def test_no_regime_change_same(self):
        flags = detect_patterns([], _make_weights(), None, "calm", "calm")
        assert flags.regime_changed is False

    def test_no_regime_change_first_run(self):
        flags = detect_patterns([], _make_weights(), None, "calm", None)
        assert flags.regime_changed is False

    def test_empty_signals_no_crash(self):
        flags = detect_patterns([], _make_weights(), None, "calm", None)
        assert flags.strong_consensus is False
        assert flags.conflicted is False

    def test_neutral_signals_excluded_from_directional_count(self):
        signals = _make_signals(["long"] * 4 + ["neutral"] * 6)
        # Only 4 directional signals, all long = 100% consensus
        flags = detect_patterns(signals, _make_weights(), None, "calm", None)
        assert flags.strong_consensus is True

    def test_flags_to_dict_serializable(self):
        flags = PatternFlags(
            strong_consensus=True,
            consensus_direction="long",
            consensus_pct=0.9,
            weight_shifts=[{"strategy": "x", "change": 0.15}],
        )
        d = flags.to_dict()
        json.dumps(d)  # must be JSON-serializable

    def test_flags_describe(self):
        flags = PatternFlags(
            strong_consensus=True,
            consensus_direction="long",
            consensus_pct=0.85,
            regime_changed=True,
            prior_regime="calm",
            current_regime="volatile",
        )
        desc = flags.describe()
        assert "Strong consensus" in desc
        assert "Regime changed" in desc


# ── 4. Response parsing ─────────────────────────────────────────────────────


class TestResponseParsing:
    """Test LLM response parsing into structured fields."""

    def test_valid_json_parsed(self):
        raw = json.dumps(
            {
                "summary": "The market is calm. TFT signals are strong. Watch VIX.",
                "patterns": "Momentum and TFT agree on AAPL.",
                "confidence": "high",
            }
        )
        result = parse_llm_response(raw)
        assert (
            result["summary"]
            == "The market is calm. TFT signals are strong. Watch VIX."
        )
        assert result["confidence"] == "high"

    def test_json_in_markdown_code_block(self):
        raw = '```json\n{"summary": "test", "patterns": "none", "confidence": "medium"}\n```'
        result = parse_llm_response(raw)
        assert result["summary"] == "test"
        assert result["confidence"] == "medium"

    def test_json_in_plain_code_block(self):
        raw = '```\n{"summary": "test2", "patterns": "none", "confidence": "low"}\n```'
        result = parse_llm_response(raw)
        assert result["summary"] == "test2"

    def test_malformed_json_fallback(self):
        raw = "This is not JSON at all, just text from the LLM."
        result = parse_llm_response(raw)
        assert "This is not JSON" in result["summary"]
        assert result["confidence"] == "low"

    def test_empty_response_fallback(self):
        result = parse_llm_response("")
        assert result["confidence"] == "low"

    def test_confidence_normalization(self):
        assert _normalize_confidence("HIGH") == "high"
        assert _normalize_confidence("  Low  ") == "low"
        assert _normalize_confidence("Medium confidence") == "medium"
        assert _normalize_confidence("very high") == "high"
        assert _normalize_confidence("somewhat low") == "low"
        assert _normalize_confidence("unclear") == "medium"


# ── 5. Rate limiting ────────────────────────────────────────────────────────


class TestRateLimiting:
    """Test that analysis runs at most once per pipeline call."""

    def test_manual_trigger_blocked_by_default(self):
        client = FakeOllamaClient()
        analyst = SignalAnalyst(client=client, allow_manual=False)
        # First call: scheduled
        result1 = _run(
            analyst.analyze(
                _make_signals(["long"]),
                _make_weights(),
                "calm",
                _make_risk_summary(),
                is_manual=False,
            )
        )
        assert result1 is not None
        assert client.call_count == 1

        # Second call: manual — should return cached, not call LLM again
        result2 = _run(
            analyst.analyze(
                _make_signals(["short"]),
                _make_weights(),
                "calm",
                _make_risk_summary(),
                is_manual=True,
            )
        )
        assert client.call_count == 1  # no new LLM call
        assert result2 is result1  # returns cached analysis

    def test_manual_trigger_allowed_when_enabled(self):
        client = FakeOllamaClient()
        analyst = SignalAnalyst(client=client, allow_manual=True)
        _run(
            analyst.analyze(
                _make_signals(["long"]),
                _make_weights(),
                "calm",
                _make_risk_summary(),
                is_manual=True,
            )
        )
        assert client.call_count == 1  # LLM was called

    def test_run_count_incremented(self):
        client = FakeOllamaClient()
        analyst = SignalAnalyst(client=client)
        assert analyst.run_count == 0
        _run(
            analyst.analyze(
                _make_signals(["long"]),
                _make_weights(),
                "calm",
                _make_risk_summary(),
            )
        )
        assert analyst.run_count == 1
        _run(
            analyst.analyze(
                _make_signals(["long"]),
                _make_weights(),
                "calm",
                _make_risk_summary(),
            )
        )
        assert analyst.run_count == 2

    def test_last_analysis_cached(self):
        client = FakeOllamaClient()
        analyst = SignalAnalyst(client=client)
        assert analyst.last_analysis is None
        result = _run(
            analyst.analyze(
                _make_signals(["long"]),
                _make_weights(),
                "calm",
                _make_risk_summary(),
            )
        )
        assert analyst.last_analysis is result


# ── 6. SignalAnalysis dataclass ──────────────────────────────────────────────


class TestSignalAnalysis:
    """Test the SignalAnalysis dataclass."""

    def test_to_dict_serializable(self):
        analysis = SignalAnalysis(
            timestamp="2026-03-21T10:00:00Z",
            summary="Test summary.",
            patterns="No patterns.",
            confidence="medium",
            regime="calm",
            model_used="test-model",
        )
        d = analysis.to_dict()
        json.dumps(d)
        assert d["confidence"] == "medium"

    def test_to_report_line(self):
        analysis = SignalAnalysis(
            timestamp="2026-03-21T10:00:00Z",
            summary="Markets are bullish. Momentum leads. Watch VIX.",
            patterns="Strong consensus.",
            confidence="high",
            model_used="qwen2.5:32b",
        )
        line = analysis.to_report_line()
        assert "LLM Analysis" in line
        assert "high confidence" in line
        assert "bullish" in line

    def test_full_pipeline_integration(self):
        """End-to-end: analyst produces valid, serializable analysis."""
        client = FakeOllamaClient()
        analyst = SignalAnalyst(client=client)
        analysis = _run(
            analyst.analyze(
                signals=_make_signals(["long"] * 8 + ["short"] * 2),
                weights=_make_weights(),
                regime="calm_trending",
                risk_summary=_make_risk_summary(),
            )
        )
        assert analysis.confidence == "high"
        assert len(analysis.summary) > 0
        assert analysis.regime == "calm_trending"
        # Verify the prompt was sent to the client
        assert client.last_prompt is not None
        assert "calm_trending" in client.last_prompt
        # Verify serialization roundtrip
        d = analysis.to_dict()
        json.dumps(d)


# ── 7. Paper-trader structural tests ─────────────────────────────────────────


class TestPaperTraderAnalystWiring:
    """Verify signal analyst is wired into paper-trader."""

    def _read_source(self):
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper-trader",
            "main.py",
        )
        with open(main_path) as f:
            return f.read()

    def test_signal_analyst_import(self):
        source = self._read_source()
        assert "from agents.signal_analyst import SignalAnalyst" in source

    def test_llm_analyst_enabled_env_var(self):
        source = self._read_source()
        assert "LLM_ANALYST_ENABLED" in source

    def test_signal_analyst_global(self):
        source = self._read_source()
        assert "signal_analyst:" in source or "signal_analyst =" in source

    def test_signal_analyst_initialized(self):
        source = self._read_source()
        assert "SignalAnalyst()" in source

    def test_analyze_called_in_pipeline(self):
        source = self._read_source()
        assert (
            "signal_analyst.analyze(" in source
            or "await signal_analyst.analyze(" in source
        )

    def test_log_signal_analysis_called(self):
        source = self._read_source()
        assert "log_signal_analysis" in source

    def test_analysis_latest_endpoint(self):
        source = self._read_source()
        assert '"/analysis/latest"' in source

    def test_signal_analyses_table_in_schema(self):
        source = self._read_source()
        assert "paper_signal_analyses" in source

    def test_signal_analyses_table_in_postgres_schema(self):
        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "postgres_schema.py",
        )
        with open(schema_path) as f:
            source = f.read()
        assert "paper_signal_analyses" in source

    def test_llm_analysis_in_discord_report(self):
        source = self._read_source()
        assert "llm_line" in source or "llm_analysis" in source

    def test_env_template_has_llm_vars(self):
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".env.template",
        )
        with open(template_path) as f:
            source = f.read()
        assert "LLM_ANALYST_ENABLED" in source
        assert "LLM_MODEL" in source

    def test_is_manual_parameter_in_pipeline(self):
        source = self._read_source()
        assert "is_manual" in source

    def test_run_now_passes_is_manual(self):
        source = self._read_source()
        assert "is_manual=True" in source
