"""
Configuration management for phishnet.

Loads YAML config with environment variable overrides.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    algorithm: str = "lightgbm"
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 7
    num_leaves: int = 63
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    classification_threshold: float = 0.5


@dataclass
class ServingConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "models/phishnet.onnx"
    max_batch_size: int = 64
    request_timeout_ms: int = 100


@dataclass
class DriftConfig:
    reference_data_path: str = "data/reference_features.parquet"
    window_size: int = 1000
    drift_threshold: float = 0.05
    report_output_dir: str = "reports"


@dataclass
class PhishnetConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    data_dir: str = "data"
    log_level: str = "INFO"


def load_config(path: Optional[str] = None) -> PhishnetConfig:
    """Load configuration from YAML with env overrides."""
    config = PhishnetConfig()

    if path and os.path.exists(path):
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        section_map = {
            "model": config.model,
            "serving": config.serving,
            "drift": config.drift,
        }
        for section_name, obj in section_map.items():
            if section_name in raw:
                for key, val in raw[section_name].items():
                    if hasattr(obj, key):
                        setattr(obj, key, val)

        if "data_dir" in raw:
            config.data_dir = raw["data_dir"]
        if "log_level" in raw:
            config.log_level = raw["log_level"]

    # Environment overrides
    env_map = {
        "PHISHNET_MODEL_PATH": ("serving", "model_path"),
        "PHISHNET_PORT": ("serving", "port"),
        "PHISHNET_THRESHOLD": ("model", "classification_threshold"),
        "PHISHNET_LOG_LEVEL": ("", "log_level"),
    }
    for env_var, (section, key) in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            target = getattr(config, section) if section else config
            current = getattr(target, key)
            if isinstance(current, int):
                val = int(val)
            elif isinstance(current, float):
                val = float(val)
            setattr(target, key, val)

    return config
