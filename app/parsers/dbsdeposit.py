"""DBS/POSB Deposit Statement parser.

Handles the multi-line row format with columns:
  DATE | DETAILS OF TRANSACTIONS | WITHDRAWAL($) | DEPOSIT($) | BALANCE($)
"""

import io
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict

import pdfplumber

from .helpers import MONTH, MONEY_RE, extract_year_from_pdf

# --- Required interface ---
LABEL = "DBS/POSB Deposit Statement"


def detect(raw_text: str) -> bool:
    """Return True if the PDF text looks like a DBS/POSB deposit statement."""
    return bool(re.search(
        r"DETAILS OF TRANSACTIONS\s+WITHDRAWAL\(\$\)\s+DEPOSIT\(\$\)",
        raw_text, re.I,
    ))


def parse(content: bytes) -> List[Dict[str, Any]]:
    """Parse a DBS/POSB deposit statement PDF into transaction dicts."""
    year = extract_year_from_pdf(content)
    if year is None:
        from datetime import date as _date
        year = _date.today().year

    raw_rows = _extract_table_rows(content)
    return _rows_to_transactions(raw_rows, year)


# --- Internal helpers ---

def _extract_table_rows(content: bytes) -> List[Dict[str, str]]:
    """Extract table rows using column x-positions from the header row."""
    rows: List[Dict[str, str]] = []

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(
                x_tolerance=2, y_tolerance=3, keep_blank_chars=False
            ) or []
            if not words:
                continue

            # Find header row by locating WITHDRAWAL($) — unique anchor
            header_y = None
            for w in words:
                if w.get("text", "").upper().startswith("WITHDRAWAL"):
                    header_y = float(w["top"])
                    break
            if header_y is None:
                continue

            # Get all words on the header row (within 3pt tolerance)
            header_words = [
                w for w in words if abs(float(w["top"]) - header_y) < 3
            ]

            def _find_x(token: str) -> Optional[float]:
                t = token.upper()
                for w in header_words:
                    if w.get("text", "").upper().startswith(t):
                        return float(w["x0"])
                return None

            x_date = _find_x("DATE")
            x_desc = _find_x("DETAILS")
            x_wdr = _find_x("WITHDRAWAL")
            x_dep = _find_x("DEPOSIT")
            x_bal = _find_x("BALANCE")
            if not all([x_date, x_desc, x_wdr, x_dep, x_bal]):
                continue

            edges = [x_date, x_desc, x_wdr, x_dep, x_bal, page.width]

            # Group words into lines by y-position
            lines: Dict[int, list] = defaultdict(list)
            for w in words:
                key = int(round(float(w["top"]) / 3.0))
                lines[key].append(w)

            for key in sorted(lines):
                ws = sorted(lines[key], key=lambda z: float(z["x0"]))
                cols = ["", "", "", "", ""]
                for w in ws:
                    x = float(w["x0"])
                    # 2pt tolerance for sub-pixel alignment differences
                    idx = sum(1 for e in edges if x >= e - 2.0) - 1
                    if 0 <= idx < 5:
                        cols[idx] = (cols[idx] + " " + w["text"]).strip()

                if re.match(
                    r"^\d{2}\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b",
                    cols[0], re.I,
                ):
                    rows.append({
                        "date": cols[0],
                        "desc": cols[1],
                        "withdrawal": cols[2],
                        "deposit": cols[3],
                        "balance": cols[4],
                    })

    return rows


def _rows_to_transactions(
    rows: List[Dict[str, str]], year: int
) -> List[Dict[str, Any]]:
    """Convert raw table rows into standardised transaction dicts."""
    txns: List[Dict[str, Any]] = []

    for r in rows:
        m = re.match(
            r"^(\d{2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b",
            r["date"], re.I,
        )
        if not m:
            continue

        day, mon = m.group(1), m.group(2)
        wm = MONEY_RE.search(r.get("withdrawal", "") or "")
        dm = MONEY_RE.search(r.get("deposit", "") or "")
        if not (wm or dm):
            continue

        amt = float((dm or wm).group(1).replace(",", ""))
        credit = bool(dm)
        date_str = f"{year:04d}-{MONTH[mon.upper()]:02d}-{int(day):02d}"

        txns.append({
            "date": date_str,
            "payee": r.get("desc", "").strip(),
            "memo": "",
            "amount": amt,
            "credit": credit,
        })

    return txns
