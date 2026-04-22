"""
Model training, evaluation, and export pipeline.

Trains a LightGBM classifier on URL features, evaluates with proper
metrics, exports to ONNX for production serving, and logs everything
to MLflow for reproducibility and versioning.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Any,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Any:
    """
    Train a LightGBM classifier.

    Uses early stopping on validation set if provided, otherwise
    trains for the full n_estimators.
    """
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": config.model.n_estimators,
        "learning_rate": config.model.learning_rate,
        "max_depth": config.model.max_depth,
        "num_leaves": config.model.num_leaves,
        "min_child_samples": config.model.min_child_samples,
        "subsample": config.model.subsample,
        "colsample_bytree": config.model.colsample_bytree,
        "reg_alpha": config.model.reg_alpha,
        "reg_lambda": config.model.reg_lambda,
        "random_state": config.model.random_state,
        "verbose": -1,
        "n_jobs": -1,
    }

    model = lgb.LGBMClassifier(**params)

    fit_params = {}
    if X_val is not None and y_val is not None:
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["callbacks"] = [
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ]

    logger.info(f"Training LightGBM with {X_train.shape[0]} samples, {X_train.shape[1]} features")
    start = time.time()
    model.fit(X_train, y_train, **fit_params)
    elapsed = time.time() - start
    logger.info(f"Training completed in {elapsed:.2f}s ({model.n_estimators_} iterations)")

    return model


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.

    Returns metrics dict with accuracy, precision, recall, F1, AUC,
    confusion matrix, and per-class statistics.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(y, y_pred, output_dict=True),
        "num_samples": len(y),
        "positive_rate": float(y.mean()),
    }

    # ROC curve data (for plotting)
    fpr, tpr, roc_thresholds = roc_curve(y, y_prob)
    metrics["roc_curve"] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }

    # Precision-Recall curve data
    prec, rec, pr_thresholds = precision_recall_curve(y, y_prob)
    metrics["pr_curve"] = {
        "precision": prec.tolist(),
        "recall": rec.tolist(),
    }

    return metrics


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    config: Any,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """
    Stratified k-fold cross-validation with aggregated metrics.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.model.random_state)

    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = train_model(X_train, y_train, config, X_val, y_val)
        metrics = evaluate_model(model, X_val, y_val, config.model.classification_threshold)

        fold_metrics.append({
            "fold": fold_idx + 1,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        })
        logger.info(
            f"  Fold {fold_idx+1}/{n_folds}: "
            f"AUC={metrics['roc_auc']:.4f} F1={metrics['f1']:.4f}"
        )

    # Aggregate
    agg = {}
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        values = [m[key] for m in fold_metrics]
        agg[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    return {"folds": fold_metrics, "aggregated": agg}


def export_to_onnx(model: Any, num_features: int, output_path: str) -> str:
    """Export LightGBM model to ONNX via onnxmltools."""
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    initial_type = [("input", FloatTensorType([None, num_features]))]
    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=initial_type,
        target_opset=9,
    )

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info(f"ONNX model saved to {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
    return output_path


def get_feature_importance(model: Any, feature_names: list) -> Dict[str, float]:
    """Extract and rank feature importances."""
    importance = model.feature_importances_
    ranked = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True,
    )
    return {name: float(imp) for name, imp in ranked}
