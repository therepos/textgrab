import io
import re
from typing import List, Dict
from collections import defaultdict

import pdfplumber


def extract_deposit_table(content: bytes) -> List[Dict[str, str]]:
    """
    Extract the 'Details of Transactions' table for deposit statements by using
    the column x-positions from the header row.

    Returns rows with keys: date, desc, withdrawal, deposit, balance (all strings).
    """
    rows: List[Dict[str, str]] = []

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False) or []
            if not words:
                continue

            x_date = _find_header_x(words, "DATE")
            x_desc = _find_header_x(words, "DETAILS OF TRANSACTIONS")
            x_wdr  = _find_header_x(words, "WITHDRAWAL(")
            x_dep  = _find_header_x(words, "DEPOSIT(")
            x_bal  = _find_header_x(words, "BALANCE(")
            if not all([x_date, x_desc, x_wdr, x_dep, x_bal]):
                continue

            edges = [x_date, x_desc, x_wdr, x_dep, x_bal, page.width]

            lines: Dict[int, List[dict]] = defaultdict(list)
            for w in words:
                key = int(round(float(w["top"]) / 3.0))
                lines[key].append(w)

            for key in sorted(lines):
                ws = sorted(lines[key], key=lambda z: float(z["x0"]))
                cols = ["", "", "", "", ""]
                for w in ws:
                    x = float(w["x0"])
                    idx = sum(1 for e in edges if x >= e) - 1
                    if 0 <= idx < 5:
                        cols[idx] = (cols[idx] + " " + w["text"]).strip()

                if re.match(r"^\d{2}\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b", cols[0], re.I):
                    rows.append({
                        "date": cols[0],
                        "desc": cols[1],
                        "withdrawal": cols[2],
                        "deposit": cols[3],
                        "balance": cols[4],
                    })

    return rows


def _find_header_x(words: List[dict], token_starts: str) -> float | None:
    token = token_starts.upper()
    for w in words:
        if w.get("text", "").upper().startswith(token):
            return float(w["x0"])
    return None
