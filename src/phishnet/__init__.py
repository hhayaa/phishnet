"""
phishnet - Real-Time Phishing URL Classifier with Drift Detection

A production ML pipeline that classifies URLs as phishing or legitimate
using gradient-boosted models served through FastAPI with ONNX Runtime,
monitored for data drift via Evidently AI, and versioned with MLflow.
"""

__version__ = "0.3.0"
__author__ = "Your Name"
