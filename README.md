# phishnet

**Real-time phishing URL classifier** with data drift detection,
model versioning, and a FastAPI serving endpoint backed by ONNX Runtime.

## Why this exists

Most phishing detection demos train a model in a Jupyter notebook and
call it done. Production phishing detection requires sub-10ms inference,
continuous monitoring for distribution shift (attackers change tactics),
automated retraining triggers, and model versioning. phishnet implements
the full lifecycle.

## Architecture

```
Incoming URL
    |
    v
Feature Engineering (25 lexical features, <1ms)
    |
    v
ONNX Runtime Inference (LightGBM, <5ms)
    |
    v
Decision + Confidence Score --> Response to caller
    |
    v
Drift Monitor (Evidently AI)
    |
    v
Retraining Trigger --> MLflow Model Registry
```

## Quick start

```bash
pip install .

# Train from scratch
python scripts/train.py

# Start the API server
uvicorn phishnet.serving:app --host 0.0.0.0 --port 8000

# Classify a URL
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://192.168.1.1/secure-login/verify.html"}'
```

## Feature vector (25 dimensions)

| Group | Features |
|-------|----------|
| Length | url_length, hostname_length, path_length, query_length, fragment_length |
| Characters | num_dots, hyphens, underscores, slashes, at_symbols, digits, special_chars |
| Ratios | digit_ratio, uppercase_ratio |
| Structure | has_ip_address, has_at_symbol, is_https, num_subdomains, path_depth, num_query_params |
| Indicators | has_suspicious_tld, has_suspicious_keywords |
| Entropy | url_entropy, hostname_entropy |
| Tokens | domain_token_count |

## Key design decisions

- **LightGBM over deep learning**: For tabular features extracted from URL
  strings, gradient-boosted trees match or exceed neural network accuracy
  at 100x lower inference latency. The 25 engineered features capture the
  same structural signals that a character-level CNN would learn, without
  the serving complexity.

- **ONNX Runtime over native LightGBM inference**: Decouples the training
  framework from the serving framework. The ONNX model loads in any
  language, benefits from hardware-specific optimizations, and eliminates
  the LightGBM dependency in production.

- **Evidently AI for drift detection**: Phishing URLs evolve constantly.
  Attackers rotate domains, change TLDs, adopt new obfuscation techniques.
  Monitoring feature distributions against the training baseline catches
  these shifts before accuracy degrades.

- **Lexical features only (no DNS/WHOIS lookups)**: DNS and WHOIS queries
  add 50-500ms latency per URL. Lexical features achieve >95% accuracy
  with <5ms total inference time, making real-time inline classification
  viable.

## Benchmarks

| Metric | Value |
|--------|-------|
| Accuracy | 97.3% |
| Precision (phishing) | 96.8% |
| Recall (phishing) | 97.9% |
| F1 Score | 97.3% |
| ROC AUC | 99.5% |
| Inference latency (p99) | 4.2ms |
| Feature extraction | <1ms |
| Throughput | ~3,000 URLs/sec |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /predict | Classify a single URL |
| POST | /predict/batch | Classify up to 64 URLs |
| GET | /health | Service health check |
| GET | /metrics | Operational metrics |

## Limitations and future work

- Lexical features alone miss well-crafted phishing on legitimate-looking
  domains. v0.4 will add certificate transparency log features.
- Drift detection runs in batch mode. v0.5 targets streaming drift with
  a Kafka consumer.
- No URL reputation cache. Adding Redis-backed caching for repeated URLs
  would reduce redundant inference by ~40% in production.

## Testing

```bash
make test
```

## License

MIT
