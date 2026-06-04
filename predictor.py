"""
predictor.py
------------
All business logic for PhishGuard.
app.py should only call functions from here — no logic lives in the UI layer.
"""

import joblib
import tldextract
from features import extract_features, FEATURE_NAMES


# ─────────────────────────────────────────────
# TRUSTED DOMAIN WHITELIST
# ─────────────────────────────────────────────
TRUSTED_DOMAINS = {
    # Global tech
    'google.com', 'youtube.com', 'gmail.com', 'googleapis.com',
    'github.com', 'microsoft.com', 'apple.com', 'amazon.com',
    'wikipedia.org', 'linkedin.com', 'twitter.com', 'x.com',
    'facebook.com', 'instagram.com', 'whatsapp.com', 'netflix.com',
    # Nigerian banks & fintechs
    'gtbank.com', 'zenithbank.com', 'accessbank.com',
    'firstbanknigeria.com', 'uba.com', 'fcmb.com',
    'opay.com', 'palmpay.com', 'kuda.com', 'moniepoint.com',
    'piggyvest.com', 'cowrywise.com', 'flutterwave.com',
    'paystack.com', 'interswitch.com',
}

# ─────────────────────────────────────────────
# SIGNAL MAP  (feature name → plain English)
# ─────────────────────────────────────────────
SIGNAL_MAP = {
    "has_ip":            "IP address used instead of domain",
    "is_shortener":      "URL has been shortened",
    "has_at_symbol":     "Contains @ symbol",
    "double_slash":      "Redirect pattern detected",
    "non_https":         "Not encrypted (no HTTPS)",
    "non_std_port":      "Uses a non-standard port",
    "suspicious_tld":    "Suspicious domain extension",
    "spoof_prefix":      "Domain mimics a login page",
    "spoof_suffix":      "Domain mimics an update page",
    "homoglyph":         "Look-alike characters in domain",
    "non_ascii":         "Contains hidden characters",
    "hex_encoded":       "Contains encoded/obfuscated parts",
    "has_keyword":       "Contains phishing keywords",
    "path_brand_spoof":  "Brand name hidden in URL path",
    "is_brand_spoof":    "Impersonating a known brand",
    "stacked_subdomains":"Multiple suspicious subdomains",
    "free_host":         "Hosted on a free platform",
    "repeated_chars":    "Unusual repeated characters",
    "redirect_pattern":  "Contains redirect parameter",
    "nested_url":        "URL hidden inside URL",
    "numeric_domain":    "Numeric-only domain",
    "has_b64_segment":   "Contains obfuscated data",
}


# ─────────────────────────────────────────────
# PUBLIC FUNCTIONS
# ─────────────────────────────────────────────

def load_model(model_path: str = 'phish_model.pkl',
               scaler_path: str = 'scaler.pkl') -> tuple:
    """
    Load the trained model and scaler from disk.
    Returns (model, scaler, success: bool).
    """
    try:
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, True
    except Exception:
        return None, None, False


def is_trusted(url: str) -> bool:
    """
    Return True if the URL's apex domain is on the trusted whitelist.
    Trusted domains bypass the model entirely.
    """
    try:
        ext  = tldextract.extract(url.lower())
        apex = f"{ext.domain}.{ext.suffix}"
        return apex in TRUSTED_DOMAINS
    except Exception:
        return False


def get_signals(feature_vector: list, max_signals: int = 5) -> list[str]:
    """
    Given a raw feature vector, return a list of plain-English
    signal descriptions for every triggered binary flag.
    Capped at max_signals to keep the UI clean.
    """
    return [
        SIGNAL_MAP[name]
        for i, name in enumerate(FEATURE_NAMES)
        if name in SIGNAL_MAP and feature_vector[i] == 1
    ][:max_signals]


def analyse(url: str, model, scaler) -> dict:
    """
    Run the full phishing analysis pipeline on a URL.

    Returns a dict:
    {
        "status":     "trusted" | "phishing" | "safe",
        "confidence": float (0–100),
        "signals":    list[str],   # plain-English triggered flags
    }
    """
    # 1. Whitelist check — no model needed
    if is_trusted(url):
        return {
            "status":     "trusted",
            "confidence": 100.0,
            "signals":    [],
        }

    # 2. Extract features
    vector = extract_features(url)

    # 3. Scale + predict
    vector_scaled = scaler.transform([vector])
    prediction    = model.predict(vector_scaled)[0]
    proba         = model.predict_proba(vector_scaled)[0]

    is_phish   = int(prediction) == 1
    confidence = (proba[1] if is_phish else proba[0]) * 100

    return {
        "status":     "phishing" if is_phish else "safe",
        "confidence": round(confidence, 1),
        "signals":    get_signals(vector),
    }