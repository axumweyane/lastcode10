"""Tests for SentimentModel."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import pytest
from models.sentiment_model import SentimentModel
from models.base import ModelPrediction


@pytest.fixture
def model():
    m = SentimentModel()
    m._is_loaded = True
    m._use_vader_fallback = True
    return m


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "text": [
                "Apple reports record quarterly revenue beating all estimates",
                "Apple stock soars on strong earnings guidance",
                "Microsoft cloud growth slows as enterprise spending cuts deepen",
                "MSFT faces regulatory headwinds in EU markets",
            ],
        }
    )


def test_returns_model_prediction_type(model, sample_data):
    preds = model.predict(sample_data)
    assert all(isinstance(p, ModelPrediction) for p in preds)


def test_handles_empty_data(model):
    preds = model.predict(pd.DataFrame())
    assert preds == []


def test_handles_missing_columns(model):
    preds = model.predict(pd.DataFrame({"symbol": ["AAPL"]}))
    assert preds == []


def test_output_has_required_metadata(model, sample_data):
    preds = model.predict(sample_data)
    if preds:
        for p in preds:
            assert "sentiment_score" in p.metadata
            assert "sentiment_momentum" in p.metadata
            assert "article_count" in p.metadata
            assert "source" in p.metadata


def test_sentiment_score_range(model, sample_data):
    preds = model.predict(sample_data)
    for p in preds:
        assert -1.0 <= p.predicted_value <= 1.0


def test_confidence_range(model, sample_data):
    preds = model.predict(sample_data)
    for p in preds:
        assert 0.0 <= p.confidence <= 1.0


def test_model_name(model):
    assert model.name == "sentiment"


def test_asset_class(model):
    assert model.asset_class == "cross_asset"


def test_predictions_per_symbol(model, sample_data):
    preds = model.predict(sample_data)
    symbols = {p.symbol for p in preds}
    assert symbols <= {"AAPL", "MSFT"}
