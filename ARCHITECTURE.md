# Architecture

## Request flow

```
Client Request (URL string)
        |
   +----v----+
   | FastAPI  |  POST /predict
   +----+-----+
        |
   +----v-----------+
   | Feature Engine  |  25 lexical features, <1ms
   +----+------------+
        |
   +----v-----------+
   | ONNX Runtime    |  LightGBM inference, <5ms
   +----+------------+
        |
   +----v-----------+
   | Response        |  {label, probability, latency}
   +----+------------+
        |
   +----v-----------+
   | Drift Monitor   |  Buffer features, periodic check
   +----------------+
```

## Module responsibilities

| Module | Responsibility |
|--------|---------------|
| `features.py` | Extract 25 features from URL strings |
| `model.py` | Train LightGBM, evaluate, export ONNX, cross-validate |
| `serving.py` | FastAPI app with ONNX inference |
| `drift.py` | Evidently-based drift detection |
| `config.py` | YAML + environment config loading |

## Training pipeline

1. Generate/load URL dataset (balanced legitimate + phishing)
2. Extract features from all URLs
3. Train/test split (80/20, stratified)
4. 5-fold cross-validation for model selection
5. Train final model on full training set
6. Evaluate on held-out test set
7. Export to ONNX format
8. Save reference features for drift baseline
9. Log everything to MLflow
