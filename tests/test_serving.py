"""Tests for the serving endpoint."""
import os
import numpy as np
import pytest
from phishnet.serving import ONNXPredictor
from phishnet.features import NUM_FEATURES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "phishnet.onnx")

class TestONNXPredictor:
    def test_load_missing(self, tmp_path):
        assert not ONNXPredictor(str(tmp_path / "x.onnx")).load()
    def test_load_success(self):
        if not os.path.exists(MODEL_PATH): pytest.skip("No model")
        p = ONNXPredictor(MODEL_PATH)
        assert p.load() and p.is_loaded
    def test_predict(self):
        if not os.path.exists(MODEL_PATH): pytest.skip("No model")
        p = ONNXPredictor(MODEL_PATH); p.load()
        r = p.predict(np.random.randn(NUM_FEATURES))
        assert r["predictions"]["label"] in ("phishing", "legitimate")
    def test_batch(self):
        if not os.path.exists(MODEL_PATH): pytest.skip("No model")
        p = ONNXPredictor(MODEL_PATH); p.load()
        r = p.predict(np.random.randn(5, NUM_FEATURES))
        assert len(r["predictions"]) == 5
