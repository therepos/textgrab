"""Scheme registry with auto-discovery.

Any .py file in this folder (except __init__) that exposes:
  - LABEL: str                      — display name for the UI dropdown
  - ACCEPTS: list[str]              — list of file extensions this scheme handles
  - MULTI_FILE: bool                — whether multi-file input is supported
  - OUTPUT_OPTIONS: list[str]       — output mode options (e.g. ["consolidated", "individual"])
  - transform(texts: dict, output_mode: str) -> dict
      texts: {filename: extracted_text_str, ...}
      output_mode: one of OUTPUT_OPTIONS
      returns: {"text": str, "filename": str, "files": list[dict] (optional)}

…will be automatically registered as a scheme.
"""

import importlib
import pkgutil
import logging
from typing import Dict, Any, List, Optional

log = logging.getLogger(__name__)

_SKIP = {"__init__"}

_registry: Dict[str, Any] = {}

for _finder, _name, _ispkg in pkgutil.iter_modules(__path__):
    if _name in _SKIP or _ispkg:
        continue
    try:
        _mod = importlib.import_module(f".{_name}", __package__)
        required = ("LABEL", "ACCEPTS", "MULTI_FILE", "OUTPUT_OPTIONS", "transform")
        if all(hasattr(_mod, attr) for attr in required):
            _registry[_name] = _mod
            log.info(f"Registered scheme: {_name} ({_mod.LABEL})")
        else:
            missing = [a for a in required if not hasattr(_mod, a)]
            log.warning(f"Skipping scheme {_name}: missing {missing}")
    except Exception as e:
        log.warning(f"Failed to load scheme {_name}: {e}")


def get_schemes() -> Dict[str, dict]:
    """Return {slug: {label, accepts, multi_file, output_options}} for all schemes."""
    schemes = {"raw": {"label": "Raw Text", "accepts": ["*"], "multi_file": True, "output_options": []}}
    for slug, mod in sorted(_registry.items()):
        schemes[slug] = {
            "label": mod.LABEL,
            "accepts": mod.ACCEPTS,
            "multi_file": mod.MULTI_FILE,
            "output_options": mod.OUTPUT_OPTIONS,
        }
    return schemes


def get_scheme(slug: str) -> Optional[Any]:
    return _registry.get(slug)


def list_schemes() -> List[str]:
    return ["raw"] + list(_registry.keys())
