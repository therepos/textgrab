# Simple registry for pluggable parsers

from typing import Callable, List, Dict, Any, Tuple

Parser = Tuple[str, Callable[[str], bool], Callable[[str, int], List[Dict[str, Any]]]]
_REGISTRY: List[Parser] = []

def register(name: str):
    """Decorator to register a parser with (name, detect, parse)."""
    def deco(mod):
        detect = getattr(mod, "detect", None)
        parse  = getattr(mod, "parse", None)
        if not callable(detect) or not callable(parse):
            raise ValueError(f"{name}: module must expose detect(text) and parse(text, year)")
        _REGISTRY.append((name, detect, parse))
        return mod
    return deco

def available() -> List[str]:
    return [n for n,_,_ in _REGISTRY]

def dispatch(text: str, year: int):
    """Find the first parser whose detect() returns True, then parse()."""
    for name, detect, parse in _REGISTRY:
        if detect(text):
            return parse(text, year)
    raise RuntimeError("Unknown statement format.")
