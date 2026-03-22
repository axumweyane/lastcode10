"""
Model 7: Sentiment Model — FinBERT-based financial text sentiment scoring.

Uses the pre-trained ProsusAI/finbert model for financial text classification.
Input: news headlines, social media text.
Output: per-symbol sentiment score [-1, +1] as ModelPrediction.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from models.base import BaseTFTModel, ModelInfo, ModelPrediction

logger = logging.getLogger(__name__)


class SentimentModel(BaseTFTModel):
    """
    FinBERT-based sentiment scoring model.

    Processes news/social text per symbol and produces sentiment scores.
    Falls back to VADER if FinBERT is unavailable.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self._model_name = model_name
        self._tokenizer = None
        self._model = None
        self._is_loaded = False
        self._use_vader_fallback = False

    @property
    def name(self) -> str:
        return "sentiment"

    @property
    def asset_class(self) -> str:
        return "cross_asset"

    def prepare_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        required = {"symbol", "text"}
        if not required.issubset(raw_data.columns):
            return pd.DataFrame()
        return raw_data[
            raw_data["text"].notna() & (raw_data["text"].str.len() > 0)
        ].copy()

    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        return {
            "status": "pretrained",
            "note": "FinBERT is pre-trained, no fine-tuning needed",
        }

    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        if data.empty or "text" not in data.columns or "symbol" not in data.columns:
            return []

        prepared = self.prepare_features(data)
        if prepared.empty:
            return []

        predictions = []
        for symbol in prepared["symbol"].unique():
            sym_texts = prepared[prepared["symbol"] == symbol]["text"].tolist()
            if not sym_texts:
                continue

            sentiments = [self._score_text(t) for t in sym_texts]
            sentiments = [s for s in sentiments if s is not None]
            if not sentiments:
                continue

            scores = np.array(sentiments)
            mean_sentiment = float(np.mean(scores))
            recent_scores = scores[-3:] if len(scores) >= 3 else scores
            sentiment_momentum = float(np.mean(recent_scores) - np.mean(scores))
            sentiment_dispersion = float(np.std(scores)) if len(scores) > 1 else 0.0

            confidence = min(
                0.3 + len(sym_texts) * 0.05 + (1.0 - sentiment_dispersion) * 0.3, 1.0
            )

            predictions.append(
                ModelPrediction(
                    symbol=symbol,
                    predicted_value=mean_sentiment,
                    lower_bound=max(mean_sentiment - sentiment_dispersion, -1.0),
                    upper_bound=min(mean_sentiment + sentiment_dispersion, 1.0),
                    confidence=confidence,
                    horizon_days=1,
                    model_name=self.name,
                    metadata={
                        "sentiment_score": round(mean_sentiment, 4),
                        "sentiment_momentum": round(sentiment_momentum, 4),
                        "sentiment_dispersion": round(sentiment_dispersion, 4),
                        "article_count": len(sym_texts),
                        "source": (
                            "finbert" if not self._use_vader_fallback else "vader"
                        ),
                    },
                )
            )

        return predictions

    def _score_text(self, text: str) -> Optional[float]:
        if not text or len(text.strip()) < 5:
            return None

        if self._is_loaded and not self._use_vader_fallback:
            return self._score_finbert(text)
        elif self._use_vader_fallback:
            return self._score_vader(text)
        return None

    def _score_finbert(self, text: str) -> Optional[float]:
        try:
            import torch

            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # FinBERT classes: positive, negative, neutral
            positive = float(probs[0][0])
            negative = float(probs[0][1])
            return positive - negative  # [-1, +1]
        except Exception as e:
            logger.warning("FinBERT scoring failed: %s", e)
            return self._score_vader(text)

    def _score_vader(self, text: str) -> Optional[float]:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            return float(scores["compound"])
        except ImportError:
            return None
        except Exception:
            return None

    def save(self, path: str) -> None:
        pass  # Pre-trained model, no saving needed

    def load(self, path: str) -> bool:
        # Try FinBERT first
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name
            )
            self._model.eval()
            self._is_loaded = True
            self._use_vader_fallback = False
            logger.info("SentimentModel loaded FinBERT from %s", self._model_name)
            return True
        except Exception as e:
            logger.info("FinBERT not available (%s), trying VADER fallback", e)

        # Try VADER fallback
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            SentimentIntensityAnalyzer()  # validate it works
            self._is_loaded = True
            self._use_vader_fallback = True
            logger.info("SentimentModel using VADER fallback")
            return True
        except ImportError:
            logger.warning("Neither FinBERT nor VADER available")
            return False

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            asset_class=self.asset_class,
            version="1.0",
            is_loaded=self._is_loaded,
        )
