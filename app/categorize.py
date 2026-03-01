import json, re, shutil
from typing import Dict, List, Optional
from pathlib import Path
from rapidfuzz import fuzz

# Runtime rules file (mounted volume)
MODEL_DIR  = Path("/data/models")
RULES_PATH = MODEL_DIR / "rules.json"

# Ship a default in-repo file; copied on first run
DEFAULT_RULES_PATH = Path(__file__).parent / "rules.json"

def _norm(s: str) -> str:
    s = (s or "").upper()
    s = re.sub(r"[^A-Z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _coerce_to_category_dict(data) -> Dict[str, List[str]]:
    """Accept dict {Category: [patterns...]} OR legacy list[{pattern,category}]."""
    out: Dict[str, List[str]] = {}
    if isinstance(data, dict):
        for cat, pats in data.items():
            arr = [str(p).strip() for p in (pats or []) if str(p).strip()]
            if arr: out[cat] = sorted(set(arr))
        return out
    if isinstance(data, list):
        for item in data:
            cat = str(item.get("category","")).strip()
            pat = str(item.get("pattern","")).strip()
            if cat and pat:
                out.setdefault(cat, []).append(pat)
        for cat in list(out.keys()):
            out[cat] = sorted(set(out[cat]))
        return out
    return {}

def load_rules() -> Dict[str, List[str]]:
    """Load category rules; bootstrap from in-repo rules on first run."""
    if not RULES_PATH.exists() and DEFAULT_RULES_PATH.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DEFAULT_RULES_PATH, RULES_PATH)
    if not RULES_PATH.exists():
        return {}
    try:
        raw = json.loads(RULES_PATH.read_text(encoding="utf-8"))
        return _coerce_to_category_dict(raw)
    except Exception:
        return {}

def save_rules(rules: Dict[str, List[str]]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cleaned = _coerce_to_category_dict(rules)
    RULES_PATH.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")

def apply_rules(payee: str, memo: str, threshold: int = 80) -> Optional[str]:
    """Fuzzy (partial) match payee/memo to user patterns, category-first."""
    rules = load_rules()
    if not rules:
        return None
    s = _norm(f"{payee} {memo}")
    best_cat, best_score = None, -1
    for cat, pats in rules.items():
        for pat in pats:
            score = fuzz.partial_ratio(_norm(pat), s)
            if score > best_score:
                best_cat, best_score = cat, score
    return best_cat if best_score >= threshold else None

def predict_category(payee: str, memo: str, refund_hint: bool = False) -> str:
    """Rules-only prediction (kept name for compatibility)."""
    cat = apply_rules(payee, memo)
    if cat:
        return cat
    if refund_hint:
        cat = apply_rules("REFUND", memo) or None
        if cat:
            return cat
    return "Other"
