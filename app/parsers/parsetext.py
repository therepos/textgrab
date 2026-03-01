from typing import List, Dict, Any
# Importing registers the parsers
from .parsers import dispatch, available  # noqa: F401
from .parsers import creditcard as _credit  # instead of credit_card
from .parsers import deposit as _deposit     # noqa: F401

def extract_transactions_from_text(raw_text: str, year: int = 2025) -> List[Dict[str, Any]]:
    return dispatch(raw_text, year)
