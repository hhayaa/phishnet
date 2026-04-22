"""
Data drift detection using Evidently AI.

Monitors feature distributions in production data against a reference
dataset (training data). When drift is detected above a configurable
threshold, triggers alerts and generates diagnostic reports.

Supports two modes:
  - Batch: compare a window of recent predictions to the reference
  - Continuous: maintain a rolling buffer and check periodically
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Monitors for data drift between reference and production feature
    distributions using Evidently AI.
    """

    def __init__(
        self,
        feature_names: List[str],
        reference_data: Optional[np.ndarray] = None,
        drift_threshold: float = 0.05,
        report_dir: str = "reports",
    ):
        self.feature_names = feature_names
        self.drift_threshold = drift_threshold
        self.report_dir = report_dir
        self._reference_df = None
        self._buffer: List[np.ndarray] = []
        self._check_count = 0

        if reference_data is not None:
            self.set_reference(reference_data)

    def set_reference(self, data: np.ndarray):
        """Set the reference dataset (typically training features)."""
        self._reference_df = pd.DataFrame(data, columns=self.feature_names)
        logger.info(f"Reference data set: {len(self._reference_df)} samples")

    def add_sample(self, features: np.ndarray):
        """Add a single feature vector to the production buffer."""
        self._buffer.append(features.flatten())

    def add_batch(self, features: np.ndarray):
        """Add a batch of feature vectors to the production buffer."""
        for row in features:
            self._buffer.append(row.flatten())

    def check_drift(self, window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Run drift detection on buffered production data.

        Returns a report dict with per-feature drift statistics.
        """
        if self._reference_df is None:
            raise ValueError("Reference data not set. Call set_reference() first.")

        if not self._buffer:
            return {"drifted": False, "reason": "No production data in buffer"}

        # Use the most recent window_size samples
        if window_size and len(self._buffer) > window_size:
            samples = self._buffer[-window_size:]
        else:
            samples = self._buffer

        current_df = pd.DataFrame(samples, columns=self.feature_names)

        self._check_count += 1

        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=self._reference_df, current_data=current_df)

            report_dict = report.as_dict()

            # Extract drift results
            drift_results = self._parse_evidently_report(report_dict)

            # Save HTML report
            os.makedirs(self.report_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = os.path.join(self.report_dir, f"drift_report_{timestamp}.html")
            report.save_html(html_path)
            drift_results["report_path"] = html_path

            logger.info(
                f"Drift check #{self._check_count}: "
                f"{'DRIFT DETECTED' if drift_results['dataset_drift'] else 'No drift'} "
                f"({drift_results['num_drifted_features']}/{len(self.feature_names)} features)"
            )

            return drift_results

        except ImportError:
            logger.warning("Evidently not installed, using statistical fallback")
            return self._statistical_drift_check(current_df)

    def _parse_evidently_report(self, report_dict: Dict) -> Dict[str, Any]:
        """Parse Evidently report dict into a clean summary."""
        try:
            metrics = report_dict.get("metrics", [])
            for metric in metrics:
                result = metric.get("result", {})
                if "drift_by_columns" in result:
                    drift_by_col = result["drift_by_columns"]
                    num_drifted = result.get("number_of_drifted_columns", 0)
                    dataset_drift = result.get("dataset_drift", False)

                    feature_drift = {}
                    for col_name, col_data in drift_by_col.items():
                        feature_drift[col_name] = {
                            "drifted": col_data.get("drift_detected", False),
                            "p_value": col_data.get("drift_score", 1.0),
                            "stattest": col_data.get("stattest_name", "unknown"),
                        }

                    return {
                        "dataset_drift": dataset_drift,
                        "num_drifted_features": num_drifted,
                        "total_features": len(self.feature_names),
                        "drift_share": num_drifted / max(len(self.feature_names), 1),
                        "feature_drift": feature_drift,
                        "check_number": self._check_count,
                        "buffer_size": len(self._buffer),
                        "timestamp": datetime.now().isoformat(),
                    }
        except Exception as e:
            logger.warning(f"Failed to parse Evidently report: {e}")

        return {"dataset_drift": False, "error": "Failed to parse report"}

    def _statistical_drift_check(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback drift detection using KS test when Evidently is unavailable."""
        from scipy import stats

        feature_drift = {}
        num_drifted = 0

        for col in self.feature_names:
            if col in self._reference_df.columns and col in current_df.columns:
                stat, p_value = stats.ks_2samp(
                    self._reference_df[col].values,
                    current_df[col].values,
                )
                drifted = p_value < self.drift_threshold
                if drifted:
                    num_drifted += 1
                feature_drift[col] = {
                    "drifted": drifted,
                    "p_value": float(p_value),
                    "stattest": "ks_2samp",
                    "statistic": float(stat),
                }

        return {
            "dataset_drift": num_drifted > len(self.feature_names) * 0.3,
            "num_drifted_features": num_drifted,
            "total_features": len(self.feature_names),
            "drift_share": num_drifted / max(len(self.feature_names), 1),
            "feature_drift": feature_drift,
            "check_number": self._check_count,
            "buffer_size": len(self._buffer),
            "method": "ks_2samp_fallback",
            "timestamp": datetime.now().isoformat(),
        }

    def clear_buffer(self):
        """Reset the production data buffer."""
        self._buffer.clear()

    def save_reference(self, path: str):
        """Persist reference data to Parquet."""
        if self._reference_df is not None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._reference_df.to_parquet(path, index=False)
            logger.info(f"Reference data saved to {path}")

    def load_reference(self, path: str):
        """Load reference data from Parquet."""
        self._reference_df = pd.read_parquet(path)
        self.feature_names = list(self._reference_df.columns)
        logger.info(f"Reference data loaded: {len(self._reference_df)} samples")
