"""
Coverage tests for models/manager.py.
All model implementations are mocked.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from models.base import BaseTFTModel, ModelInfo, ModelPrediction
from models.manager import ModelManager, ManagerStatus

# ---------- Mock model ----------


def _make_mock_model(name="test_model", asset_class="stocks", is_loaded=True):
    model = MagicMock(spec=BaseTFTModel)
    model.name = name
    model.asset_class = asset_class
    model.get_info.return_value = ModelInfo(
        name=name,
        asset_class=asset_class,
        version="1.0",
        is_loaded=is_loaded,
    )
    model.load.return_value = is_loaded
    model.predict.return_value = []
    return model


def _make_prediction(symbol="AAPL", value=0.05):
    return ModelPrediction(
        symbol=symbol,
        predicted_value=value,
        lower_bound=value - 0.02,
        upper_bound=value + 0.02,
        confidence=0.8,
        horizon_days=5,
        model_name="test",
    )


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"] * 10,
            "timestamp": pd.date_range("2025-01-01", periods=20),
            "close": [150 + i for i in range(20)],
            "volume": [1000000] * 20,
        }
    )


# ---------- Constructor ----------


class TestConstructor:
    def test_creates_10_models(self):
        with patch("models.manager.TFTStocksAdapter"), patch(
            "models.manager.TFTForexModel"
        ), patch("models.manager.TFTVolatilityModel"), patch(
            "models.manager.KronosModel"
        ), patch(
            "models.manager.DeepSurrogateModel"
        ), patch(
            "models.manager.TDGFModel"
        ), patch(
            "models.manager.SentimentModel"
        ), patch(
            "models.manager.MeanReversionModel"
        ), patch(
            "models.manager.MacroRegimeModel"
        ), patch(
            "models.manager.MicrostructureModel"
        ):
            mgr = ModelManager()
            assert len(mgr._models) == 10

    def test_custom_paths(self):
        custom_paths = {"tft_stocks": "/custom/path.pth"}
        with patch("models.manager.TFTStocksAdapter") as mock_adapter, patch(
            "models.manager.TFTForexModel"
        ), patch("models.manager.TFTVolatilityModel"), patch(
            "models.manager.KronosModel"
        ), patch(
            "models.manager.DeepSurrogateModel"
        ), patch(
            "models.manager.TDGFModel"
        ), patch(
            "models.manager.SentimentModel"
        ), patch(
            "models.manager.MeanReversionModel"
        ), patch(
            "models.manager.MacroRegimeModel"
        ), patch(
            "models.manager.MicrostructureModel"
        ):
            mgr = ModelManager(model_paths=custom_paths)
            mock_adapter.assert_called_once_with("/custom/path.pth")


# ---------- load_all ----------


class TestLoadAll:
    def test_all_load_successfully(self):
        mgr = MagicMock(spec=ModelManager)
        mgr._models = {
            "m1": _make_mock_model("m1", is_loaded=True),
            "m2": _make_mock_model("m2", is_loaded=True),
        }
        mgr._paths = {"m1": "p1", "m2": "p2"}

        # Call real load_all
        status = ModelManager.load_all(mgr)
        assert status.models_loaded == 2
        assert status.models_failed == 0

    def test_partial_load_failure(self):
        mgr = MagicMock(spec=ModelManager)
        ok_model = _make_mock_model("ok", is_loaded=True)
        fail_model = _make_mock_model("fail", is_loaded=False)
        fail_model.load.return_value = False
        mgr._models = {"ok": ok_model, "fail": fail_model}
        mgr._paths = {"ok": "p1", "fail": "p2"}

        status = ModelManager.load_all(mgr)
        assert status.models_loaded == 1
        assert status.models_failed == 1

    def test_load_exception_counted_as_failure(self):
        mgr = MagicMock(spec=ModelManager)
        err_model = _make_mock_model("err")
        err_model.load.side_effect = Exception("crash")
        mgr._models = {"err": err_model}
        mgr._paths = {"err": "p"}

        status = ModelManager.load_all(mgr)
        assert status.models_failed == 1


# ---------- predict_* methods ----------


class TestPredictMethods:
    @pytest.fixture
    def mgr(self):
        m = MagicMock(spec=ModelManager)
        models = {}
        for name in [
            "tft_stocks",
            "tft_forex",
            "tft_volatility",
            "kronos",
            "deep_surrogates",
            "tdgf",
            "sentiment",
            "mean_reversion",
            "macro_regime",
            "microstructure",
        ]:
            models[name] = _make_mock_model(name)
        m._models = models
        return m

    def test_predict_stocks(self, mgr, sample_data):
        ModelManager.predict_stocks(mgr, sample_data)
        mgr._models["tft_stocks"].predict.assert_called_once_with(sample_data)

    def test_predict_forex(self, mgr, sample_data):
        ModelManager.predict_forex(mgr, sample_data)
        mgr._models["tft_forex"].predict.assert_called_once()

    def test_predict_volatility(self, mgr, sample_data):
        ModelManager.predict_volatility(mgr, sample_data)
        mgr._models["tft_volatility"].predict.assert_called_once()

    def test_predict_kronos(self, mgr, sample_data):
        ModelManager.predict_kronos(mgr, sample_data)
        mgr._models["kronos"].predict.assert_called_once()

    def test_predict_deep_surrogates(self, mgr, sample_data):
        ModelManager.predict_deep_surrogates(mgr, sample_data)
        mgr._models["deep_surrogates"].predict.assert_called_once()

    def test_predict_tdgf(self, mgr, sample_data):
        ModelManager.predict_tdgf(mgr, sample_data)
        mgr._models["tdgf"].predict.assert_called_once()

    def test_predict_sentiment(self, mgr, sample_data):
        ModelManager.predict_sentiment(mgr, sample_data)
        mgr._models["sentiment"].predict.assert_called_once()

    def test_predict_mean_reversion(self, mgr, sample_data):
        ModelManager.predict_mean_reversion(mgr, sample_data)
        mgr._models["mean_reversion"].predict.assert_called_once()

    def test_predict_macro(self, mgr, sample_data):
        ModelManager.predict_macro(mgr, sample_data)
        mgr._models["macro_regime"].predict.assert_called_once()

    def test_predict_microstructure(self, mgr, sample_data):
        ModelManager.predict_microstructure(mgr, sample_data)
        mgr._models["microstructure"].predict.assert_called_once()


# ---------- predict_all ----------


class TestPredictAll:
    def test_stock_data_only(self):
        mgr = MagicMock(spec=ModelManager)
        models = {}
        for name in [
            "tft_stocks",
            "tft_forex",
            "tft_volatility",
            "kronos",
            "deep_surrogates",
            "tdgf",
            "sentiment",
            "mean_reversion",
            "macro_regime",
            "microstructure",
        ]:
            m = _make_mock_model(name)
            m.predict.return_value = [_make_prediction()]
            models[name] = m
        mgr._models = models
        mgr.predict_stocks = lambda d: models["tft_stocks"].predict(d)
        mgr.predict_volatility = lambda d: models["tft_volatility"].predict(d)
        mgr.predict_kronos = lambda d: models["kronos"].predict(d)
        mgr.predict_forex = lambda d: models["tft_forex"].predict(d)
        mgr.predict_deep_surrogates = lambda d: models["deep_surrogates"].predict(d)
        mgr.predict_tdgf = lambda d: models["tdgf"].predict(d)

        data = pd.DataFrame({"symbol": ["AAPL"], "close": [150]})
        results = ModelManager.predict_all(mgr, stock_data=data)
        assert "stocks" in results
        assert "volatility" in results
        assert "kronos_stocks" in results

    def test_fx_data(self):
        mgr = MagicMock(spec=ModelManager)
        models = {}
        for name in ["tft_forex", "kronos"]:
            m = _make_mock_model(name)
            m.predict.return_value = []
            models[name] = m
        mgr._models = models
        mgr.predict_forex = lambda d: models["tft_forex"].predict(d)
        mgr.predict_kronos = lambda d: models["kronos"].predict(d)
        mgr.predict_stocks = lambda d: []
        mgr.predict_volatility = lambda d: []

        fx = pd.DataFrame({"symbol": ["EURUSD"], "close": [1.08]})
        results = ModelManager.predict_all(mgr, fx_data=fx)
        assert "forex" in results

    def test_options_data(self):
        mgr = MagicMock(spec=ModelManager)
        models = {}
        for name in ["deep_surrogates", "tdgf"]:
            m = _make_mock_model(name)
            m.predict.return_value = []
            models[name] = m
        mgr._models = models
        mgr.predict_deep_surrogates = lambda d: models["deep_surrogates"].predict(d)
        mgr.predict_tdgf = lambda d: models["tdgf"].predict(d)

        opts = pd.DataFrame({"symbol": ["AAPL_C_150"], "close": [5.0]})
        results = ModelManager.predict_all(mgr, options_data=opts)
        assert "deep_surrogates" in results
        assert "tdgf" in results

    def test_no_data_returns_empty(self):
        mgr = MagicMock(spec=ModelManager)
        mgr._models = {}
        results = ModelManager.predict_all(mgr)
        assert results == {}


# ---------- get_model ----------


class TestGetModel:
    def test_existing(self):
        mgr = MagicMock(spec=ModelManager)
        mock_model = _make_mock_model("sentiment")
        mgr._models = {"sentiment": mock_model}
        result = ModelManager.get_model(mgr, "sentiment")
        assert result == mock_model

    def test_missing(self):
        mgr = MagicMock(spec=ModelManager)
        mgr._models = {}
        result = ModelManager.get_model(mgr, "nonexistent")
        assert result is None


# ---------- is_model_loaded ----------


class TestIsModelLoaded:
    def test_loaded(self):
        mgr = MagicMock(spec=ModelManager)
        mgr._models = {"m1": _make_mock_model("m1", is_loaded=True)}
        assert ModelManager.is_model_loaded(mgr, "m1") is True

    def test_not_loaded(self):
        mgr = MagicMock(spec=ModelManager)
        mgr._models = {"m1": _make_mock_model("m1", is_loaded=False)}
        assert ModelManager.is_model_loaded(mgr, "m1") is False

    def test_missing_model(self):
        mgr = MagicMock(spec=ModelManager)
        mgr._models = {}
        assert ModelManager.is_model_loaded(mgr, "missing") is False


# ---------- get_status ----------


class TestGetStatus:
    def test_status(self):
        mgr = MagicMock(spec=ModelManager)
        mgr._models = {
            "a": _make_mock_model("a", is_loaded=True),
            "b": _make_mock_model("b", is_loaded=False),
            "c": _make_mock_model("c", is_loaded=True),
        }
        status = ModelManager.get_status(mgr)
        assert status.models_registered == 3
        assert status.models_loaded == 2
        assert status.models_failed == 1


# ---------- predictions_to_dict ----------


class TestPredictionsToDict:
    def test_converts(self):
        mgr = MagicMock(spec=ModelManager)
        preds = [_make_prediction("AAPL", 0.05), _make_prediction("MSFT", -0.02)]
        result = ModelManager.predictions_to_dict(mgr, preds)
        assert result == {"AAPL": 0.05, "MSFT": -0.02}

    def test_empty(self):
        mgr = MagicMock(spec=ModelManager)
        assert ModelManager.predictions_to_dict(mgr, []) == {}


# ---------- predictions_to_dataframe ----------


class TestPredictionsToDataframe:
    def test_converts(self):
        mgr = MagicMock(spec=ModelManager)
        preds = [_make_prediction("AAPL", 0.05), _make_prediction("MSFT", -0.02)]
        df = ModelManager.predictions_to_dataframe(mgr, preds)
        assert len(df) == 2
        assert "symbol" in df.columns
        assert "predicted_return" in df.columns
        assert df.iloc[0]["symbol"] == "AAPL"

    def test_empty(self):
        mgr = MagicMock(spec=ModelManager)
        df = ModelManager.predictions_to_dataframe(mgr, [])
        assert len(df) == 0
        assert "symbol" in df.columns
