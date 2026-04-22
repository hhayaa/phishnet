"""Integration tests: URL to prediction."""
import os
import numpy as np
import pytest
from phishnet.features import extract_features_from_url, extract_features_batch
from phishnet.serving import ONNXPredictor

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "phishnet.onnx")

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="No model")
class TestEndToEnd:
    @pytest.fixture
    def predictor(self):
        p = ONNXPredictor(MODEL_PATH, threshold=0.5)
        p.load()
        return p
    def test_legit_classified(self, predictor):
        f = extract_features_from_url("https://www.google.com/search?q=python")
        assert predictor.predict(f)["predictions"]["label"] == "legitimate"
    def test_phishing_classified(self, predictor):
        f = extract_features_from_url("http://192.168.1.1/secure-login/verify.html")
        assert predictor.predict(f)["predictions"]["label"] == "phishing"
    def test_batch_mixed(self, predictor):
        urls = ["https://github.com/user/repo", "http://go0gle.tk/password-reset",
                "https://stackoverflow.com/q", "http://paypal.login.abc123.xyz/verify"]
        r = predictor.predict(extract_features_batch(urls))
        labels = [p["label"] for p in r["predictions"]]
        assert "phishing" in labels and "legitimate" in labels
    def test_latency(self, predictor):
        f = extract_features_from_url("https://www.google.com")
        predictor.predict(f)  # warm up
        assert predictor.predict(f)["latency_ms"] < 50.0
