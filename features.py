"""
features.py  — 52-feature extractor for phishing URL detection
---------------------------------------------------------------
Key design principles:
  • Continuous values wherever possible (models learn thresholds better)
  • No padding / placeholder constants
  • No magnitude-destroying binary ±1 encoding for numeric signals
  • Binary 0/1 flags only for genuine yes/no signals
  • Nigerian / West-African fintech brand list included
"""

import re
import math
import tldextract
import difflib
from urllib.parse import urlparse, parse_qs, unquote
from collections import Counter


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_entropy(s: str) -> float:
    """Shannon entropy of a string."""
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


# Levenshtein distance (pure Python, no extra deps)
def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j] + (ca != cb), prev[j + 1] + 1, curr[j] + 1))
        prev = curr
    return prev[-1]


# ── Brand lists ───────────────────────────────────────────────────────────────

NIGERIAN_BRANDS = [
    'opay', 'palmpay', 'kuda', 'moniepoint', 'piggyvest', 'cowrywise',
    'zenithbank', 'gtbank', 'accessbank', 'firstbank', 'uba', 'stanbic',
    'fidelitybank', 'sterlingbank', 'fcmb', 'unionbank', 'providusbank',
    'vfd', 'rubies', 'carbon', 'fairmoney', 'monie', 'flutterwave',
    'paystack', 'interswitch', 'nibss', 'remita'
]

GLOBAL_BRANDS = [
    'paypal', 'apple', 'google', 'microsoft', 'amazon', 'netflix',
    'facebook', 'instagram', 'twitter', 'whatsapp', 'binance', 'coinbase'
]

ALL_BRANDS = NIGERIAN_BRANDS + GLOBAL_BRANDS

SHORTENERS = re.compile(
    r'bit\.ly|goo\.gl|tinyurl|t\.co|rb\.gy|ow\.ly|is\.gd|'
    r'buff\.ly|cutt\.ly|shorturl|shorte\.st|adf\.ly'
)

SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'sign-in', 'verify', 'verification',
    'update', 'confirm', 'secure', 'security', 'account',
    'banking', 'portal', 'bvn', 'nin', 'otp', 'password',
    'credential', 'wallet', 'transfer', 'payment', 'refund',
    'suspend', 'unusual', 'alert', 'locked', 'unlock',
    'customer-care', 'support', 'helpdesk', 'winner',
    'claim', 'prize', 'reward', 'bonus', 'kyc'
]

SUSPICIOUS_TLDS = {
    'tk', 'ml', 'ga', 'cf', 'gq',           # Freenom free TLDs
    'xyz', 'top', 'live', 'click', 'link',
    'buzz', 'icu', 'vip', 'work', 'pw',
    'loan', 'date', 'review', 'faith',
    'stream', 'download', 'racing'
}

HOMOGLYPHS = re.compile(
    r'[0o]{1}(?=[a-z])|[1il]{1}(?=[a-z])|rn(?=[a-z])|vv(?=[a-z])'
)


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_features(url: str) -> list:
    """
    Returns a list of 52 numeric features (mix of continuous + binary 0/1).
    Feature names are listed in FEATURE_NAMES at the bottom of this file.
    """
    features = []
    url_raw  = url                       # preserve original for some checks
    url      = url.lower().strip()

    parsed  = urlparse(url)
    ext     = tldextract.extract(url)
    domain  = ext.domain or ''
    suffix  = ext.suffix  or ''
    sub     = ext.subdomain or ''
    netloc  = parsed.netloc or ''
    path    = parsed.path   or ''
    query   = parsed.query  or ''

    full_domain = f"{sub}.{domain}.{suffix}".strip('.')


    # ── SECTION A: URL-LEVEL SIGNALS (continuous) ────────────────────────────

    # 1. Raw URL length
    features.append(len(url))

    # 2. Shannon entropy of the full URL
    features.append(round(get_entropy(url), 4))

    # 3. Digit ratio in URL
    features.append(round(sum(c.isdigit() for c in url) / max(len(url), 1), 4))

    # 4. Special character count (non-word, non-slash, non-dot)
    features.append(len(re.findall(r'[^\w./:?=&%-]', url)))

    # 5. Count of dots in full URL
    features.append(url.count('.'))

    # 6. Count of hyphens in full URL
    features.append(url.count('-'))

    # 7. Count of underscores
    features.append(url.count('_'))

    # 8. Count of percent-encoded characters (%XX) → obfuscation signal
    features.append(len(re.findall(r'%[0-9a-f]{2}', url)))

    # 9. Count of query parameters
    features.append(len(parse_qs(query)))

    # 10. Query string length
    features.append(len(query))

    # 11. Path depth (number of '/' after host)
    features.append(path.count('/'))

    # 12. Path length
    features.append(len(path))


    # ── SECTION B: DOMAIN-LEVEL SIGNALS (continuous + binary) ────────────────

    # 13. Domain character length
    features.append(len(domain))

    # 14. Full domain length (including subdomains + TLD)
    features.append(len(full_domain))

    # 15. Subdomain depth (number of labels in subdomain)
    features.append(sub.count('.') + (1 if sub else 0))

    # 16. Digit count in domain only
    features.append(sum(c.isdigit() for c in domain))

    # 17. Hyphen count in domain
    features.append(domain.count('-'))

    # 18. Consonant-to-vowel ratio in domain (high ratio → random/generated)
    vowels     = sum(c in 'aeiou' for c in domain)
    consonants = sum(c.isalpha() and c not in 'aeiou' for c in domain)
    features.append(round(consonants / max(vowels, 1), 4))

    # 19. Domain entropy
    features.append(round(get_entropy(domain), 4))


    # ── SECTION C: BINARY STRUCTURAL FLAGS (0 / 1) ───────────────────────────

    # 20. Has IP address instead of domain name
    features.append(1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', netloc) else 0)

    # 21. Uses URL shortener
    features.append(1 if SHORTENERS.search(url) else 0)

    # 22. Contains @ symbol (redirects browser to part after @)
    features.append(1 if '@' in url else 0)

    # 23. Double-slash redirect outside of scheme (e.g. http://site.com//evil)
    features.append(1 if url.rfind('//') > 7 else 0)

    # 24. Non-HTTPS scheme
    features.append(0 if parsed.scheme == 'https' else 1)

    # 25. Non-standard port
    try:
        port = parsed.port
        features.append(1 if port and port not in (80, 443) else 0)
    except ValueError:
        features.append(1)

    # 26. Suspicious TLD
    features.append(1 if suffix in SUSPICIOUS_TLDS else 0)

    # 27. Domain starts with common spoof prefix
    features.append(1 if re.match(r'^(login|signin|secure|verify|account|update)-', domain) else 0)

    # 28. Domain ends with common spoof suffix
    features.append(1 if re.search(r'-(login|update|verify|secure|alert|support)$', domain) else 0)

    # 29. Homoglyph / typosquatting characters in domain (rn→m, vv→w, 0→o, 1→l)
    features.append(1 if HOMOGLYPHS.search(domain) else 0)

    # 30. Non-ASCII / punycode characters (IDN homograph attack)
    features.append(1 if re.search(r'xn--', netloc) or any(ord(c) > 127 for c in url_raw) else 0)

    # 31. Hex-encoded domain / path segments (obfuscation)
    features.append(1 if re.search(r'%[0-9a-f]{2}', netloc + path) else 0)


    # ── SECTION D: KEYWORD SIGNALS ────────────────────────────────────────────

    # 32. Count of suspicious keywords found in URL
    kw_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url)
    features.append(kw_hits)

    # 33. Binary: at least one suspicious keyword present
    features.append(1 if kw_hits > 0 else 0)

    # 34. Binary: brand keyword appears in PATH/QUERY but not as the domain
    #     (e.g. evil.com/paypal-login → spoofing brand in path)
    path_brand_spoof = 0
    for brand in ALL_BRANDS:
        if brand in (path + query) and brand not in domain:
            path_brand_spoof = 1
            break
    features.append(path_brand_spoof)


    # ── SECTION E: BRAND SPOOFING (continuous similarity scores) ─────────────

    # 35. Minimum Levenshtein distance to any known brand
    #     Low distance + not exact match = likely spoof
    lev_distances = [_levenshtein(domain, b) for b in ALL_BRANDS]
    min_lev = min(lev_distances)
    features.append(min_lev)

    # 36. Closest fuzzy similarity ratio (0–1) to any brand
    sim_scores = [difflib.SequenceMatcher(None, domain, b).ratio() for b in ALL_BRANDS]
    max_sim = round(max(sim_scores), 4)
    features.append(max_sim)

    # 37. Binary: domain is a close-but-not-exact brand spoof
    #     (similarity 0.55–0.99 OR lev distance 1–3 on short domains)
    exact_match = domain in ALL_BRANDS
    is_spoof = (
        not exact_match and (
            (0.55 < max_sim < 1.0) or
            (min_lev in (1, 2, 3) and len(domain) <= 12)
        )
    )
    features.append(1 if is_spoof else 0)

    # 38. Count of brand names appearing anywhere in the full URL
    brand_count_in_url = sum(1 for b in ALL_BRANDS if b in url)
    features.append(brand_count_in_url)


    # ── SECTION F: REPUTATION / STRUCTURAL HEURISTICS ────────────────────────

    # 39. Ratio of domain length to full URL length (short domain in long URL)
    features.append(round(len(domain) / max(len(url), 1), 4))

    # 40. Binary: URL has fragment (#) — sometimes used to hide redirect targets
    features.append(1 if parsed.fragment else 0)

    # 41. Binary: multiple subdomains stacked (e.g. pay.login.evil.com)
    features.append(1 if sub.count('.') >= 2 else 0)

    # 42. Binary: free subdomain service abuse (often used in phishing)
    FREE_HOSTS = re.compile(
        r'000webhostapp|weebly|wixsite|blogspot|wordpress|firebaseapp|'
        r'netlify|pages\.dev|github\.io|glitch\.me|vercel\.app'
    )
    features.append(1 if FREE_HOSTS.search(url) else 0)

    # 43. Repeated character sequences in domain (e.g. aaaa, 1111) → DGA signal
    features.append(1 if re.search(r'(.)\1{3,}', domain) else 0)

    # 44. Binary: URL contains common redirect patterns
    features.append(1 if re.search(r'redirect|return|next|url=|continue=|goto=', url) else 0)

    # 45. Binary: query string contains another URL (open redirect / nested URL)
    features.append(1 if re.search(r'https?%3a|https?://', query) else 0)

    # 46. Count of semicolons in URL (unusual separator, obfuscation signal)
    features.append(url.count(';'))

    # 47. Binary: domain is purely numeric (except dots) — likely IP or DGA
    features.append(1 if re.fullmatch(r'[\d.]+', netloc) else 0)

    # 48. Vowel ratio in full URL (DGA domains tend to be very low)
    features.append(round(sum(c in 'aeiou' for c in url) / max(len(url), 1), 4))

    # 49. Token count when splitting URL on non-alphanumeric chars
    tokens = re.split(r'[^a-z0-9]', url)
    features.append(len([t for t in tokens if t]))

    # 50. Max token length (very long tokens signal obfuscation)
    token_lengths = [len(t) for t in tokens if t]
    features.append(max(token_lengths) if token_lengths else 0)

    # 51. Binary: URL contains base64-like segment (long alphanumeric run ≥ 20 chars)
    features.append(1 if re.search(r'[a-z0-9]{20,}', url) else 0)

    # 52. Entropy of path only (high path entropy → encoded / obfuscated path)
    features.append(round(get_entropy(path), 4))

    assert len(features) == 52, f"Feature count mismatch: {len(features)}"
    return features


# ── Feature name reference (for importance plots) ────────────────────────────

FEATURE_NAMES = [
    # A – URL-level
    "url_length",           "url_entropy",         "digit_ratio",
    "special_char_count",   "dot_count",            "hyphen_count",
    "underscore_count",     "pct_encoded_count",    "query_param_count",
    "query_length",         "path_depth",           "path_length",
    # B – Domain-level
    "domain_length",        "full_domain_length",   "subdomain_depth",
    "digit_count_domain",   "hyphen_count_domain",  "consonant_vowel_ratio",
    "domain_entropy",
    # C – Binary structural
    "has_ip",               "is_shortener",         "has_at_symbol",
    "double_slash",         "non_https",            "non_std_port",
    "suspicious_tld",       "spoof_prefix",         "spoof_suffix",
    "homoglyph",            "non_ascii",            "hex_encoded",
    # D – Keywords
    "kw_count",             "has_keyword",          "path_brand_spoof",
    # E – Brand spoofing
    "min_lev_dist",         "max_brand_sim",        "is_brand_spoof",
    "brand_count_in_url",
    # F – Heuristics
    "domain_url_ratio",     "has_fragment",         "stacked_subdomains",
    "free_host",            "repeated_chars",        "redirect_pattern",
    "nested_url",           "semicolon_count",      "numeric_domain",
    "vowel_ratio",          "token_count",          "max_token_length",
    "has_b64_segment",      "path_entropy",
]