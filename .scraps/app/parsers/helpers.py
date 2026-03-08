"""Shared helpers for bank statement parsers."""

import io
import re
from typing import Optional, Tuple

import pdfplumber

MONTH = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

MONEY_RE = re.compile(r"([0-9]{1,3}(?:,[0-9]{3})*\.\d{2})")


def extract_year_from_pdf(content: bytes) -> Optional[int]:
    """Try to extract statement year from the first page of a PDF."""
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        text = pdf.pages[0].extract_text(x_tolerance=2, y_tolerance=3) or ""

    # "As at 31 Jan 2026" (DBS deposit)
    m = re.search(r"As at \d{1,2}\s+\w+\s+(\d{4})", text)
    if m:
        return int(m.group(1))

    # "23 Feb 2026" (DBS credit card / general)
    m = re.search(
        r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})",
        text,
    )
    if m:
        return int(m.group(3))

    # "2026-02-23" ISO format
    m = re.search(r"(\d{4})-\d{2}-\d{2}", text)
    if m:
        return int(m.group(1))

    return None


def extract_statement_month(content: bytes) -> Tuple[int, int]:
    """Extract statement year and month from the first page.
    Returns (year, month). Falls back to (current_year, 1)."""
    from datetime import date as _date

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        text = pdf.pages[0].extract_text(x_tolerance=2, y_tolerance=3) or ""

    # "As at 31 Jan 2026"
    m = re.search(r"As at \d{1,2}\s+(\w+)\s+(\d{4})", text)
    if m:
        mon = MONTH.get(m.group(1).upper()[:3], 1)
        return int(m.group(2)), mon

    # "23 Feb 2026"
    m = re.search(
        r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})",
        text,
    )
    if m:
        mon = MONTH.get(m.group(2).upper()[:3], 1)
        return int(m.group(3)), mon

    return _date.today().year, 1
