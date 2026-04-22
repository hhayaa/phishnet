"""
URL feature extraction for phishing detection.

Extracts 25 lexical and structural features from a URL string. These
features capture patterns that distinguish phishing URLs from legitimate
ones without requiring DNS lookups or page content analysis, enabling
real-time classification at <10ms per URL.

Feature groups:
  - Length features (URL, hostname, path, query)
  - Character distribution (dots, hyphens, digits, special chars)
  - Structural features (subdomain count, path depth, has IP, has @)
  - Security indicators (HTTPS, suspicious TLD, known brand in subdomain)
  - Entropy features (URL entropy, hostname entropy)
  - Suspicious keyword presence
"""

import math
import re
import string
from collections import Counter
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import numpy as np

FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "fragment_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_at_symbols",
    "num_digits",
    "num_special_chars",
    "digit_ratio",
    "uppercase_ratio",
    "has_ip_address",
    "has_at_symbol",
    "is_https",
    "num_subdomains",
    "path_depth",
    "num_query_params",
    "has_suspicious_tld",
    "has_suspicious_keywords",
    "url_entropy",
    "hostname_entropy",
    "domain_token_count",
]

NUM_FEATURES = len(FEATURE_NAMES)

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work",
    "buzz", "fit", "click", "link", "info", "online", "site",
    "win", "bid", "stream", "download", "racing", "review",
}

SUSPICIOUS_KEYWORDS = {
    "login", "signin", "verify", "account", "update", "secure",
    "banking", "confirm", "password", "credential", "suspend",
    "authenticate", "wallet", "paypal", "ebay", "amazon",
    "apple", "microsoft", "google", "facebook", "netflix",
    "helpdesk", "support", "service", "alert", "notification",
}

_IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)


def _shannon_entropy(text: str) -> float:
    """Compute Shannon entropy of a string."""
    if not text:
        return 0.0
    freq = Counter(text)
    length = len(text)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in freq.values()
        if count > 0
    )


def _count_special_chars(text: str) -> int:
    """Count non-alphanumeric, non-standard URL characters."""
    special = set("!$%^*()+=[]{}|;',<>?~`")
    return sum(1 for c in text if c in special)


def _has_ip_address(hostname: str) -> bool:
    """Check if the hostname is an IP address."""
    return bool(_IP_PATTERN.match(hostname))


def _extract_tld(hostname: str) -> str:
    """Extract the top-level domain from a hostname."""
    parts = hostname.rstrip(".").split(".")
    if len(parts) >= 2:
        return parts[-1].lower()
    return ""


def extract_features_from_url(url: str) -> np.ndarray:
    """
    Extract a feature vector from a single URL string.

    Args:
        url: Raw URL string (with or without scheme)

    Returns:
        numpy array of shape (NUM_FEATURES,)
    """
    # Normalize: add scheme if missing
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        parsed = urlparse(url)
    except Exception:
        # Return zero vector for unparseable URLs
        return np.zeros(NUM_FEATURES, dtype=np.float64)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""
    full_url = url.lower()

    # Length features
    url_length = len(url)
    hostname_length = len(hostname)
    path_length = len(path)
    query_length = len(query)
    fragment_length = len(fragment)

    # Character counts
    num_dots = url.count(".")
    num_hyphens = url.count("-")
    num_underscores = url.count("_")
    num_slashes = url.count("/")
    num_at_symbols = url.count("@")
    num_digits = sum(c.isdigit() for c in url)
    num_special = _count_special_chars(url)

    # Ratios
    digit_ratio = num_digits / max(url_length, 1)
    uppercase_ratio = sum(c.isupper() for c in url) / max(url_length, 1)

    # Structural features
    has_ip = float(_has_ip_address(hostname))
    has_at = float(num_at_symbols > 0)
    is_https = float(parsed.scheme == "https")

    # Subdomain analysis
    hostname_parts = hostname.split(".")
    num_subdomains = max(len(hostname_parts) - 2, 0)

    # Path depth
    path_segments = [s for s in path.split("/") if s]
    path_depth = len(path_segments)

    # Query parameters
    try:
        num_query_params = len(parse_qs(query))
    except Exception:
        num_query_params = 0

    # TLD check
    tld = _extract_tld(hostname)
    has_suspicious_tld = float(tld in SUSPICIOUS_TLDS)

    # Keyword check
    has_suspicious_kw = float(
        any(kw in full_url for kw in SUSPICIOUS_KEYWORDS)
    )

    # Entropy
    url_entropy = _shannon_entropy(url)
    hostname_entropy = _shannon_entropy(hostname)

    # Domain token count (split hostname on dots and hyphens)
    domain_tokens = re.split(r"[.\-]", hostname)
    domain_token_count = len([t for t in domain_tokens if t])

    features = np.array([
        url_length,
        hostname_length,
        path_length,
        query_length,
        fragment_length,
        num_dots,
        num_hyphens,
        num_underscores,
        num_slashes,
        num_at_symbols,
        num_digits,
        num_special,
        digit_ratio,
        uppercase_ratio,
        has_ip,
        has_at,
        is_https,
        num_subdomains,
        path_depth,
        num_query_params,
        has_suspicious_tld,
        has_suspicious_kw,
        url_entropy,
        hostname_entropy,
        domain_token_count,
    ], dtype=np.float64)

    return features


def extract_features_batch(urls: List[str]) -> np.ndarray:
    """
    Extract features from a list of URLs.

    Returns:
        numpy array of shape (len(urls), NUM_FEATURES)
    """
    return np.array([extract_features_from_url(u) for u in urls])


def get_feature_names() -> List[str]:
    """Return ordered list of feature names."""
    return list(FEATURE_NAMES)
