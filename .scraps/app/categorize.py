"""
Transaction categorisation & payee name resolution.

Two-layer approach:
  1. derive_payee(description) -> clean short vendor name
     - Handles DBS-specific prefixes, location stripping, alias merging
  2. predict_category(payee) -> category string
     - Looks up the payee in a learned dictionary (lookup.json)
     - Falls back to fuzzy matching against known payees
     - Returns "Other" if nothing matches

The lookup is bootstrapped from app/lookup.json and persisted to
/data/models/lookup.json at runtime. Editable via /api/lookup endpoints.
"""

import json
import re
import shutil
from typing import Dict, Tuple
from pathlib import Path

from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = Path("/data/models")
LOOKUP_PATH = MODEL_DIR / "lookup.json"
DEFAULT_LOOKUP_PATH = Path(__file__).parent / "lookup.json"

# ---------------------------------------------------------------------------
# Location / noise patterns to strip from bank descriptions
# ---------------------------------------------------------------------------
_LOCATION_PATTERNS = [
    r"\s+SINGAPORE\s+\S*$",
    r"\s+JOHOR\s+BAHRU\s+MYS$",
    r"\s+JOHOR\s+MYS$",
    r"\s+KUALA\s+LUMPUR\s+MYS.*$",
    r"\s+BATU\s+PAHAT\s+MYS.*$",
    r"\s+INTERNET\s+HKG$",
    r"\s+PETALING\s+JAYA\s+MYS$",
    r"\s+SAN\s+FRANCISCO\s+CA$",
    r"\s+NEW\s+YORK\s+NY$",
    r"\s+TORONTO\s+ON$",
    r"\s+FORT\s+COLLINS\s+CO$",
    r"\s+IRVINE\s+CA$",
    r"\s+HAMBURG\s+DEU$",
    r"\s+HONGKONG\s+HKG$",
    r"\s+MIAMI\s+FL$",
    r"\s+CAMBRIDGE\s+LND$",
    r"\s+TALLINN\s+DUB.*$",
    r"\s+LUGANO\s+DUB$",
    r"\s+SAN\s+JOSE\s+CA$",
    r"\s+MALAYSIAN\s+RINGGIT\s+[\d.]+$",
    r"\s+EUROPEAN\s+MONETARY.*$",
    r"\s+\d{3,}$",
]

_LOCATION_RE = [re.compile(p) for p in _LOCATION_PATTERNS]


def _clean_desc(d: str) -> str:
    """Strip trailing location codes, country names, and DBS noise."""
    d = d.strip()
    for pat in _LOCATION_RE:
        d = pat.sub("", d)
    return d.strip()


# ---------------------------------------------------------------------------
# Payee derivation (description -> short vendor name)
# ---------------------------------------------------------------------------
def _derive_payee_raw(desc: str) -> str:
    """Convert a raw bank description into a short vendor name (before alias)."""
    d = _clean_desc(desc)

    # DBS loan / installment lines: 008MY PREFERRED PAYMENT PLAN03 (01)
    if re.match(r"^\d{2,3}(CARDS|MY PREFERRED|LAZADA)", d):
        m = re.match(r"^\d{2,3}(.+?)(\s*\(.*)?$", d)
        if m:
            name = m.group(1).strip()
            if "LAZADA" in name:
                return "Lazada"
            if "CARDS IL" in name:
                return "DBS Cards IL"
            if "MY PREFERRED" in name:
                return "DBS Payment Plan"
            return name

    if d.startswith("BUS/MRT"):
        return "TransitLink"
    if d.startswith("GRAB"):
        return "Grab"
    if "TADA.G" in d or d.startswith("TADA"):
        return "TADA"

    # TS/ prefix (tenant/shop code in malls)
    if d.startswith("TS/"):
        name = d[3:].split(" - ")[0].split(" COMP")[0].strip()
        return name.title()

    # SP prefix (Shopify / Stripe merchants)
    if d.startswith("SP "):
        return d[3:].strip().title()

    # PayPal
    if d.startswith("PAYPAL *") or d.startswith("PAYPAL*"):
        name = re.sub(r"^PAYPAL\s*\*\s*", "", d)
        name = re.sub(r"\s+\d{10,}.*$", "", name)
        return name.title()

    if d.startswith("MICROSOFT"):
        return "Microsoft"

    # WWW. domains
    if d.startswith("WWW."):
        m = re.match(r"WWW\.(\S+)", d)
        if m:
            domain = m.group(1).rstrip("*").rstrip(".")
            if "TADA" in domain.upper():
                return "TADA"
            if "COMMON" in domain.upper():
                parts = d.split("*", 1)
                rest = parts[1].strip() if len(parts) > 1 else ""
                if rest:
                    return rest.split()[0].title()
                return domain.title()

    # Amazon Prime vs Amazon Marketplace
    if d.startswith("AMZNPRIME"):
        return "Amazon Prime"
    if d.startswith("AMZN") or d.startswith("AMAZON"):
        return "Amazon"

    if "SHOPEE" in d:
        return "Shopee"
    if d.startswith("Lazada") or d.startswith("LAZADA"):
        return "Lazada"
    if "STARBUCKS" in d.upper():
        return "Starbucks"
    if d.upper().startswith("MCDONALD"):
        return "McDonald's"
    if d.upper().startswith("BURGER KING"):
        return "Burger King"
    if "TNG" in d and "EWALLET" in d:
        return "TNG eWallet"
    if d.startswith("GOOGLE "):
        rest = d[7:].strip()
        return ("Google " + rest.title()) if rest else "Google"
    if d.startswith("APPLE"):
        return "Apple"
    if "OPENAI" in d or "CHATGPT" in d:
        return "ChatGPT"
    if d.startswith("PRUDENTIAL"):
        return "Prudential"
    if "STARHUB" in d:
        return "StarHub"
    if "ANNUAL FEE" in d:
        return "Annual Fee"
    if d.startswith("ACRA"):
        return "ACRA"
    if d.startswith("AGODA"):
        return "Agoda"

    # Generic fallback: take the part before branch/location separators
    name = d.split(" - ")[0]
    name = re.sub(r"\s*@\s*.*$", "", name)
    name = re.sub(r"\s*_\S+$", "", name)
    name = re.sub(r"\s*\*\s*.*$", "", name)
    name = re.sub(r"\s+\(.*$", "", name)
    name = name.strip()
    return name.title() if name else desc[:20].title()


# ---------------------------------------------------------------------------
# Lookup loading
# ---------------------------------------------------------------------------
_cache: Dict = {}


def _load_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load payee_categories and payee_aliases; bootstrap on first run."""
    if "lookup" in _cache:
        return _cache["lookup"]

    if not LOOKUP_PATH.exists() and DEFAULT_LOOKUP_PATH.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DEFAULT_LOOKUP_PATH, LOOKUP_PATH)

    if not LOOKUP_PATH.exists():
        _cache["lookup"] = ({}, {})
        return _cache["lookup"]

    try:
        raw = json.loads(LOOKUP_PATH.read_text(encoding="utf-8"))
        payee_cats = raw.get("payee_categories", {})
        aliases = raw.get("payee_aliases", {})
        _cache["lookup"] = (payee_cats, aliases)
    except Exception:
        _cache["lookup"] = ({}, {})

    return _cache["lookup"]


def reload_lookup() -> None:
    """Force-reload from disk (call after external edits)."""
    _cache.pop("lookup", None)
    _load_lookup()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def derive_payee(description: str) -> str:
    """Convert a raw bank description into a clean, short payee name."""
    _, aliases = _load_lookup()
    raw = _derive_payee_raw(description)
    return aliases.get(raw.lower(), raw)


def predict_category(
    description: str = "",
    payee: str = "",
    memo: str = "",
    refund_hint: bool = False,
) -> str:
    """
    Predict the spending category for a transaction.

    Uses an exact lookup first, then falls back to fuzzy matching.
    """
    payee_cats, _ = _load_lookup()

    if not payee:
        payee = derive_payee(description)

    # 1. Exact match
    if payee in payee_cats:
        return payee_cats[payee]

    # 2. Case-insensitive match
    payee_lower = payee.lower()
    for known, cat in payee_cats.items():
        if known.lower() == payee_lower:
            return cat

    # 3. Fuzzy match against known payees (handles slight variations)
    best_cat, best_score = None, -1
    for known, cat in payee_cats.items():
        score = fuzz.ratio(payee_lower, known.lower())
        if score > best_score:
            best_cat, best_score = cat, score

    if best_score >= 80:
        return best_cat

    return "Other"


# ---------------------------------------------------------------------------
# Lookup management (for /api/lookup endpoints)
# ---------------------------------------------------------------------------
def load_lookup() -> Dict:
    """Return the full lookup dict for the API."""
    payee_cats, aliases = _load_lookup()
    return {"payee_categories": payee_cats, "payee_aliases": aliases}


def save_lookup(data: Dict) -> None:
    """Save updated lookup from the API."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOOKUP_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _cache.pop("lookup", None)  # bust cache
