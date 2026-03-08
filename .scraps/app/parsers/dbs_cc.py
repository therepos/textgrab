"""DBS/POSB Credit Card Statement parser.

Handles the text-based format with columns:
  DATE | DESCRIPTION | AMOUNT (S$)
Amounts ending with CR are credits/refunds.
"""

import io
import re
from typing import List, Dict, Any, Optional

import pdfplumber

from .helpers import MONTH, extract_statement_month

# --- Required interface ---
LABEL = "DBS/POSB Credit Card Statement"


def detect(raw_text: str) -> bool:
    """Return True if the PDF text looks like a DBS/POSB credit card statement."""
    return bool(re.search(r"\bNEW TRANSACTIONS\b", raw_text, re.I))


def parse(content: bytes) -> List[Dict[str, Any]]:
    """Parse a DBS/POSB credit card statement PDF into transaction dicts."""
    # Extract raw text from all pages
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        raw_text = ""
        for p in pdf.pages:
            raw_text += (p.extract_text(x_tolerance=2, y_tolerance=3) or "") + "\n"

    # Statement date for year inference
    stmt_year, stmt_mon = extract_statement_month(content)

    def _infer_year(txn_mon: int) -> int:
        """If transaction month > statement month, it's from the previous year."""
        if txn_mon > stmt_mon:
            return stmt_year - 1
        return stmt_year

    # Find transaction section boundaries
    lines = [
        re.sub(r"\s+", " ", ln).strip()
        for ln in raw_text.splitlines()
        if ln.strip()
    ]

    s = next(
        (i for i, l in enumerate(lines) if re.search(r"\bNEW TRANSACTIONS\b", l)),
        None,
    )
    if s is None:
        s = next(
            (i for i, l in enumerate(lines)
             if re.search(r"\bPREVIOUS BALANCE\b", l)),
            None,
        )
    if s is None:
        return []

    e = next(
        (j for j in range(s + 1, len(lines))
         if re.match(r"^(SUB-TOTAL:|TOTAL:|INSTALMENT PLANS SUMMARY)", lines[j])),
        len(lines),
    )
    txn_lines = lines[s + 1 : e]

    # Regex patterns
    date_re = re.compile(
        r"^(?P<d>\d{2}) (?P<m>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b",
        re.I,
    )
    amt_line = re.compile(
        r"^(?P<a>[0-9]{1,3}(?:,[0-9]{3})*\.\d{2})(?:\s*(?P<cr>CR))?$", re.I,
    )
    amt_inline = re.compile(
        r"(?P<a>[0-9]{1,3}(?:,[0-9]{3})*\.\d{2})(?:\s*(?P<cr>CR))?$",
    )
    fx_info = re.compile(
        r"(U\. S\. DOLLAR|MALAYSIAN RINGGIT)\s+"
        r"([0-9]{1,3}(?:,[0-9]{3})*\.\d{2})",
        re.I,
    )
    skip_re = re.compile(
        r"^(Credit Cards|DBS Cards|Hotline:|Statement of Account|PDS_|\d+ of \d+)",
        re.I,
    )

    txns: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    for ln in txn_lines:
        m = date_re.match(ln)
        if m:
            # Flush previous transaction
            if cur and ("amount" in cur) and cur.get("payee"):
                txns.append(cur)

            d, mon = m.group("d"), m.group("m")
            rest = ln[m.end() :].strip()

            # Check for inline amount at end of line
            mi = amt_inline.search(rest)
            cr = False
            amt = None
            if mi:
                amt = mi.group("a")
                cr = bool(mi.group("cr"))
                rest = rest[: mi.start()].strip()

            txn_mon = MONTH[mon.upper()]
            year = _infer_year(txn_mon)
            date_str = f"{year:04d}-{txn_mon:02d}-{int(d):02d}"

            cur = {"date": date_str, "payee": rest, "memo": ""}
            if amt:
                cur["amount"] = float(amt.replace(",", ""))
                cur["credit"] = cr
            continue

        if cur is None:
            continue

        # Foreign currency info → memo
        if fx_info.search(ln):
            cur["memo"] = (
                cur.get("memo", "")
                + ("; " if cur.get("memo") else "")
                + ln
            ).strip()
            continue

        # Standalone amount line
        ma = amt_line.match(ln)
        if ma and "amount" not in cur:
            cur["amount"] = float(ma.group("a").replace(",", ""))
            cur["credit"] = bool(ma.group("cr"))
            continue

        # Skip page headers/footers, append rest to payee
        if not skip_re.match(ln):
            cur["payee"] = (cur["payee"] + " " + ln).strip()

    # Flush last transaction
    if cur and ("amount" in cur) and cur.get("payee"):
        txns.append(cur)

    # Clean up payee whitespace
    for t in txns:
        t["payee"] = re.sub(r"\s{2,}", " ", t["payee"]).strip()

    return txns
