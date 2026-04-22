"""
FastAPI serving endpoint for real-time phishing URL classification.
"""

import logging
import os
import time
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ONNXPredictor:
    """Wraps ONNX Runtime for low-latency inference."""

    def __init__(self, model_path: str, threshold: float = 0.5):
        self.model_path = model_path
        self.threshold = threshold
        self._session = None
        self._input_name = None
        self._request_count = 0
        self._total_latency_ms = 0.0

    def load(self) -> bool:
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self._session = ort.InferenceSession(
                self.model_path, opts, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            # Log output names for debugging
            out_names = [o.name for o in self._session.get_outputs()]
            logger.info(f"ONNX model loaded. Inputs: {self._input_name}, Outputs: {out_names}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def predict(self, features: np.ndarray) -> dict:
        if self._session is None:
            raise RuntimeError("Model not loaded")

        x = features.reshape(-1, features.shape[-1]).astype(np.float32)

        start = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: x})
        latency_ms = (time.perf_counter() - start) * 1000.0

        self._request_count += 1
        self._total_latency_ms += latency_ms

        # Handle both skl2onnx and onnxmltools output formats:
        # onnxmltools: outputs[0]=labels, outputs[1]=probabilities (array or dict list)
        labels_out = outputs[0]
        probs_out = outputs[1]

        results = []
        for i in range(len(x)):
            # Extract probability for class 1 (phishing)
            prob_row = probs_out[i]
            if isinstance(prob_row, dict):
                prob = float(prob_row.get(1, prob_row.get("1", 0.0)))
            elif isinstance(prob_row, np.ndarray) and prob_row.ndim >= 1:
                prob = float(prob_row[1]) if len(prob_row) > 1 else float(prob_row[0])
            elif isinstance(prob_row, (list, tuple)):
                prob = float(prob_row[1]) if len(prob_row) > 1 else float(prob_row[0])
            else:
                prob = float(prob_row)

            label = "phishing" if prob >= self.threshold else "legitimate"
            results.append({
                "label": label,
                "probability": round(prob, 6),
                "is_phishing": prob >= self.threshold,
            })

        return {
            "predictions": results if len(results) > 1 else results[0],
            "latency_ms": round(latency_ms, 3),
        }

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def avg_latency_ms(self) -> float:
        if self._request_count == 0:
            return 0.0
        return self._total_latency_ms / self._request_count


def create_app(model_path: str, threshold: float = 0.5):
    """Create FastAPI application with pre-loaded model."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    app = FastAPI(title="phishnet", version="0.3.0")
    predictor = ONNXPredictor(model_path, threshold)

    # Load immediately instead of relying on startup event
    if os.path.exists(model_path):
        predictor.load()

    class URLRequest(BaseModel):
        url: str

    class BatchURLRequest(BaseModel):
        urls: List[str] = Field(..., max_length=64)

    @app.get("/health")
    async def health():
        return {
            "status": "healthy" if predictor.is_loaded else "degraded",
            "model_loaded": predictor.is_loaded,
            "requests_served": predictor._request_count,
            "avg_latency_ms": round(predictor.avg_latency_ms, 3),
        }

    @app.post("/predict")
    async def predict(request: URLRequest):
        if not predictor.is_loaded:
            raise HTTPException(503, "Model not loaded")
        from phishnet.features import extract_features_from_url
        features = extract_features_from_url(request.url)
        result = predictor.predict(features)
        pred = result["predictions"]
        return {
            "url": request.url,
            "label": pred["label"],
            "probability": pred["probability"],
            "is_phishing": pred["is_phishing"],
            "latency_ms": result["latency_ms"],
        }

    @app.post("/predict/batch")
    async def predict_batch(request: BatchURLRequest):
        if not predictor.is_loaded:
            raise HTTPException(503, "Model not loaded")
        from phishnet.features import extract_features_batch
        features = extract_features_batch(request.urls)
        result = predictor.predict(features)
        preds = result["predictions"]
        if not isinstance(preds, list):
            preds = [preds]
        return {
            "predictions": [
                {"url": u, "label": p["label"], "probability": p["probability"], "is_phishing": p["is_phishing"]}
                for u, p in zip(request.urls, preds)
            ],
            "total_latency_ms": result["latency_ms"],
            "count": len(request.urls),
        }

    @app.get("/metrics")
    async def metrics():
        return {
            "requests_total": predictor._request_count,
            "avg_latency_ms": round(predictor.avg_latency_ms, 3),
        }

    return app
