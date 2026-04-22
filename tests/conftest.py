"""Shared test fixtures."""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phishnet.features import extract_features_from_url, NUM_FEATURES

@pytest.fixture
def legit_urls():
    return [
        "https://www.google.com/search?q=test",
        "https://github.com/user/repo",
        "https://en.wikipedia.org/wiki/Python",
        "https://stackoverflow.com/questions/12345",
        "https://www.amazon.com/dp/B08N5WRWNW",
    ]

@pytest.fixture
def phishing_urls():
    return [
        "http://192.168.1.1/secure-login/verify.html",
        "http://go0gle.tk/password-reset",
        "http://paypal.login.abc123.xyz/verify",
        "http://secure-login.verify-account.ml/auth/signin.php",
        "http://facebook.com@evil-456.gq/phish",
    ]

@pytest.fixture
def sample_features():
    return np.random.randn(100, NUM_FEATURES).astype(np.float64)
