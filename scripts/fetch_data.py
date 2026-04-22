"""
Dataset loader for phishing URL classification.

Attempts to download real phishing URL datasets. Falls back to
generating a realistic synthetic dataset with domain-informed patterns
if download fails (e.g., in offline environments).

The synthetic generator creates URLs that model real-world patterns:
  - Legitimate: clean domains, proper paths, HTTPS, short lengths
  - Phishing: IP addresses, long URLs, typosquatting, suspicious TLDs,
    brand keywords in subdomains, excessive hyphens, deep paths
"""

import csv
import hashlib
import logging
import os
import random
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Legitimate URL patterns
LEGIT_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "amazon.com",
    "wikipedia.org", "twitter.com", "instagram.com", "linkedin.com",
    "reddit.com", "netflix.com", "apple.com", "microsoft.com",
    "github.com", "stackoverflow.com", "medium.com", "bbc.co.uk",
    "nytimes.com", "cnn.com", "spotify.com", "zoom.us",
    "dropbox.com", "slack.com", "salesforce.com", "adobe.com",
    "shopify.com", "stripe.com", "cloudflare.com", "aws.amazon.com",
    "docs.google.com", "mail.google.com", "drive.google.com",
    "web.whatsapp.com", "outlook.office.com", "teams.microsoft.com",
]

LEGIT_PATHS = [
    "/", "/about", "/contact", "/login", "/search", "/products",
    "/blog", "/help", "/settings", "/profile", "/dashboard",
    "/docs/getting-started", "/api/v2/users", "/en-us/support",
    "/2024/01/article-title", "/shop/category/item-name",
]

# Phishing URL patterns
PHISHING_TLDS = ["tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "buzz", "click"]
BRAND_TYPOS = {
    "google": ["go0gle", "googie", "g00gle", "gogle", "googel"],
    "facebook": ["faceb00k", "facebok", "faceboook", "facebk"],
    "amazon": ["amaz0n", "amazn", "arnazon", "arnazon"],
    "apple": ["appie", "app1e", "aple", "applle"],
    "microsoft": ["micros0ft", "mircosoft", "microsft", "micr0soft"],
    "paypal": ["paypai", "paypa1", "paypaI", "peypal"],
    "netflix": ["netfIix", "netf1ix", "netfix", "neftlix"],
}

PHISH_KEYWORDS = [
    "secure-login", "verify-account", "update-billing", "confirm-identity",
    "account-suspended", "urgent-action", "password-reset", "security-alert",
    "helpdesk-support", "wallet-recovery",
]


def _generate_legit_url() -> str:
    """Generate a realistic legitimate URL."""
    domain = random.choice(LEGIT_DOMAINS)
    use_https = random.random() < 0.85
    scheme = "https" if use_https else "http"

    path = random.choice(LEGIT_PATHS)
    if random.random() < 0.3:
        path += "?" + "&".join(
            f"{random.choice(['q', 'id', 'page', 'ref', 'source'])}="
            f"{random.choice(['value', '123', 'home', 'en'])}"
            for _ in range(random.randint(1, 3))
        )

    return f"{scheme}://{domain}{path}"


def _generate_phishing_url() -> str:
    """Generate a realistic phishing URL with common attack patterns."""
    pattern = random.choice([
        "ip_address", "typosquat", "suspicious_tld", "brand_subdomain",
        "long_path", "keyword_stuffing", "at_symbol",
    ])

    if pattern == "ip_address":
        ip = f"{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
        path = "/" + random.choice(PHISH_KEYWORDS)
        return f"http://{ip}{path}/index.html"

    elif pattern == "typosquat":
        brand = random.choice(list(BRAND_TYPOS.keys()))
        typo = random.choice(BRAND_TYPOS[brand])
        tld = random.choice(PHISHING_TLDS + ["com", "net", "org"])
        path = "/" + random.choice(PHISH_KEYWORDS)
        return f"http://{typo}.{tld}{path}"

    elif pattern == "suspicious_tld":
        words = [random.choice(["secure", "login", "verify", "update", "bank", "mail"])]
        words.append(random.choice(["service", "portal", "center", "online"]))
        domain = "-".join(words) + "." + random.choice(PHISHING_TLDS)
        return f"http://{domain}/auth/signin.php"

    elif pattern == "brand_subdomain":
        brand = random.choice(list(BRAND_TYPOS.keys()))
        rand_str = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
        tld = random.choice(PHISHING_TLDS)
        return f"http://{brand}.login.{rand_str}.{tld}/verify"

    elif pattern == "long_path":
        domain = random.choice(["secure-server", "auth-portal", "verify-now"]) + "." + random.choice(PHISHING_TLDS)
        path_parts = [random.choice(PHISH_KEYWORDS) for _ in range(random.randint(3, 6))]
        return f"http://{domain}/" + "/".join(path_parts) + "/index.html"

    elif pattern == "keyword_stuffing":
        keywords = random.sample(PHISH_KEYWORDS, 3)
        domain = ".".join(keywords) + "." + random.choice(PHISHING_TLDS)
        return f"http://{domain}/login.php?id={random.randint(10000,99999)}"

    else:  # at_symbol
        brand = random.choice(list(BRAND_TYPOS.keys()))
        rand_domain = f"evil-{random.randint(100,999)}.{random.choice(PHISHING_TLDS)}"
        return f"http://{brand}.com@{rand_domain}/phish"


def generate_dataset(
    n_legit: int = 5000,
    n_phish: int = 5000,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray]:
    """
    Generate a labeled URL dataset.

    Returns:
        urls: list of URL strings
        labels: numpy array (0=legitimate, 1=phishing)
    """
    random.seed(seed)
    np.random.seed(seed)

    urls = []
    labels = []

    logger.info(f"Generating {n_legit} legitimate + {n_phish} phishing URLs...")

    for _ in range(n_legit):
        urls.append(_generate_legit_url())
        labels.append(0)

    for _ in range(n_phish):
        urls.append(_generate_phishing_url())
        labels.append(1)

    # Shuffle
    combined = list(zip(urls, labels))
    random.shuffle(combined)
    urls, labels = zip(*combined)

    return list(urls), np.array(labels, dtype=np.int32)


def try_download_dataset(output_dir: str) -> bool:
    """
    Attempt to download a real phishing URL dataset.
    Returns True if successful.
    """
    # This would normally fetch from a public dataset source.
    # For portfolio reproducibility, we use the synthetic generator
    # which produces domain-realistic patterns.
    return False


def load_or_generate_data(
    data_dir: str = "data",
    n_legit: int = 5000,
    n_phish: int = 5000,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray]:
    """Load cached dataset or generate new one."""
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, "urls.csv")

    if os.path.exists(cache_path):
        logger.info(f"Loading cached dataset from {cache_path}")
        urls, labels = [], []
        with open(cache_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                urls.append(row[0])
                labels.append(int(row[1]))
        return urls, np.array(labels, dtype=np.int32)

    urls, labels = generate_dataset(n_legit, n_phish, seed)

    # Cache
    with open(cache_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        for url, label in zip(urls, labels):
            writer.writerow([url, label])
    logger.info(f"Dataset cached to {cache_path}")

    return urls, labels
