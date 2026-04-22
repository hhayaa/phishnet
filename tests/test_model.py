"""Tests for model training and evaluation."""
import os
import numpy as np
import pytest
from phishnet.config import PhishnetConfig
from phishnet.model import train_model, evaluate_model, export_to_onnx
from phishnet.features import NUM_FEATURES

@pytest.fixture
def small_dataset():
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, NUM_FEATURES)
    X[:n//2, 0] += 2
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y

class TestTrainModel:
    def test_trains(self, small_dataset):
        X, y = small_dataset
        config = PhishnetConfig()
        config.model.n_estimators = 20
        model = train_model(X, y, config)
        assert hasattr(model, "predict_proba")
    def test_probabilities(self, small_dataset):
        X, y = small_dataset
        config = PhishnetConfig()
        config.model.n_estimators = 20
        probs = train_model(X, y, config).predict_proba(X)
        assert probs.shape == (len(X), 2)

class TestEvaluateModel:
    def test_metrics(self, small_dataset):
        X, y = small_dataset
        config = PhishnetConfig()
        config.model.n_estimators = 20
        model = train_model(X, y, config)
        m = evaluate_model(model, X, y)
        assert 0 <= m["accuracy"] <= 1
        assert 0 <= m["roc_auc"] <= 1

class TestONNXExport:
    def test_export(self, small_dataset, tmp_path):
        X, y = small_dataset
        config = PhishnetConfig()
        config.model.n_estimators = 20
        model = train_model(X, y, config)
        path = str(tmp_path / "model.onnx")
        export_to_onnx(model, NUM_FEATURES, path)
        assert os.path.exists(path) and os.path.getsize(path) > 0
