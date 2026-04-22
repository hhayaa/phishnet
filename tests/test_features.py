"""Tests for URL feature extraction."""
import numpy as np
import pytest
from phishnet.features import (
    extract_features_from_url, extract_features_batch,
    FEATURE_NAMES, NUM_FEATURES, _shannon_entropy,
)

class TestShannonEntropy:
    def test_empty(self):
        assert _shannon_entropy("") == 0.0
    def test_uniform(self):
        assert _shannon_entropy("aaaa") == 0.0
    def test_high_entropy(self):
        assert _shannon_entropy("abcdefghij") > 3.0

class TestExtractFeatures:
    def test_output_shape(self):
        f = extract_features_from_url("https://www.google.com/search?q=test")
        assert f.shape == (NUM_FEATURES,)
    def test_https_detected(self):
        f = extract_features_from_url("https://google.com")
        assert f[FEATURE_NAMES.index("is_https")] == 1.0
    def test_ip_address_detected(self):
        f = extract_features_from_url("http://192.168.1.1/login")
        assert f[FEATURE_NAMES.index("has_ip_address")] == 1.0
    def test_at_symbol_detected(self):
        f = extract_features_from_url("http://legit.com@evil.com/phish")
        assert f[FEATURE_NAMES.index("has_at_symbol")] == 1.0
    def test_suspicious_tld(self):
        f = extract_features_from_url("http://malware.tk/login")
        assert f[FEATURE_NAMES.index("has_suspicious_tld")] == 1.0
    def test_suspicious_keyword(self):
        f = extract_features_from_url("http://secure-login.com/verify")
        assert f[FEATURE_NAMES.index("has_suspicious_keywords")] == 1.0
    def test_batch(self):
        result = extract_features_batch(["https://google.com", "http://evil.tk/login"])
        assert result.shape == (2, NUM_FEATURES)
    def test_empty_url(self):
        assert extract_features_from_url("").shape == (NUM_FEATURES,)
