"""Parser registry with auto-discovery.

Any .py file in this folder (except __init__, generic, helpers) that exposes:
  - LABEL: str          — display name for the UI dropdown
  - detect(raw_text: str) -> bool  — returns True if this parser matches
  - parse(content: bytes) -> list  — returns list of transaction dicts

…will be automatically registered as a template.

Transaction dict format:
  {"date": "YYYY-MM-DD", "payee": str, "memo": str, "amount": float, "credit": bool}
"""

import importlib
import pkgutil
import logging
from typing import Dict, Any, Callable, List, Optional

log = logging.getLogger(__name__)

# Skip these modules during auto-discovery
_SKIP = {"__init__", "generic", "helpers"}

# Registry populated at import time
# key = template slug (filename), value = module
_registry: Dict[str, Any] = {}

# Discover and register parsers
for _finder, _name, _ispkg in pkgutil.iter_modules(__path__):
    if _name in _SKIP or _ispkg:
        continue
    try:
        _mod = importlib.import_module(f".{_name}", __package__)
        # Validate required interface
        if all(hasattr(_mod, attr) for attr in ("LABEL", "detect", "parse")):
            _registry[_name] = _mod
            log.info(f"Registered parser: {_name} ({_mod.LABEL})")
        else:
            log.warning(
                f"Skipping {_name}: missing LABEL, detect(), or parse()"
            )
    except Exception as e:
        log.warning(f"Failed to load parser {_name}: {e}")


def get_templates() -> Dict[str, str]:
    """Return {slug: label} for all registered parsers, plus 'auto'."""
    templates = {"auto": "Auto-detect (generic table parser)"}
    for slug, mod in sorted(_registry.items()):
        templates[slug] = mod.LABEL
    return templates


def get_parser(slug: str) -> Optional[Any]:
    """Return the parser module for a given slug, or None."""
    return _registry.get(slug)


def auto_detect(raw_text: str) -> Optional[str]:
    """Run detect() on all registered parsers. Return first matching slug."""
    for slug, mod in _registry.items():
        try:
            if mod.detect(raw_text):
                return slug
        except Exception as e:
            log.warning(f"detect() failed for {slug}: {e}")
    return None


def list_parsers() -> List[str]:
    """Return list of registered parser slugs."""
    return list(_registry.keys())
