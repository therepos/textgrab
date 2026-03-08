"""
Generic bank statement parser.

Uses structured table extraction (img2table) to find tables with financial
column headers (Date, Description, Amount, Withdrawal, Deposit, etc.)
and converts them to transactions — works across bank formats.
"""
import re
from typing import List, Dict, Any, Optional
from app.extract import ExtractedTable

# Common column header patterns (case-insensitive)
_DATE_PATTERNS = [
    r"date", r"trans.*date", r"posting.*date", r"value.*date",
]
_DESC_PATTERNS = [
    r"desc", r"detail", r"particular", r"narration", r"transaction",
    r"payee", r"merchant", r"reference",
]
_WITHDRAWAL_PATTERNS = [
    r"withdraw", r"debit", r"dr", r"outflow", r"payment",
    r"charge", r"withdrawal\s*\(\$?\)",
]
_DEPOSIT_PATTERNS = [
    r"deposit", r"credit", r"cr", r"inflow",
    r"deposit\s*\(\$?\)",
]
_AMOUNT_PATTERNS = [
    r"^amount$", r"^sum$", r"^total$",
]
_BALANCE_PATTERNS = [
    r"balance", r"bal",
]

MONTH_MAP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
}

_money_re = re.compile(r"[0-9]{1,3}(?:,[0-9]{3})*\.\d{2}")


def _match_column(header: str, patterns: List[str]) -> bool:
    """Check if a header matches any pattern."""
    h = header.strip().lower()
    return any(re.search(p, h) for p in patterns)


def _find_column(headers: List[str], patterns: List[str]) -> Optional[int]:
    """Find the index of a column matching patterns."""
    for i, h in enumerate(headers):
        if _match_column(h, patterns):
            return i
    return None


def _parse_amount(s: str) -> Optional[float]:
    """Parse a money string like '1,234.56' or '(1,234.56)' into a float."""
    if not s or not s.strip():
        return None
    s = s.strip()
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    if s.startswith("-"):
        negative = True
        s = s[1:]
    m = _money_re.search(s)
    if not m:
        return None
    val = float(m.group().replace(",", ""))
    return -val if negative else val


def _parse_date(s: str, year: int) -> Optional[str]:
    """Try to parse various date formats into YYYY-MM-DD."""
    s = s.strip()
    if not s:
        return None

    # DD Mon (e.g. "01 Sep", "15 JAN")
    m = re.match(r"(\d{1,2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)", s, re.I)
    if m:
        day = int(m.group(1))
        mon = MONTH_MAP.get(m.group(2).upper())
        if mon:
            return f"{year:04d}-{mon:02d}-{day:02d}"

    # DD/MM/YYYY or DD-MM-YYYY
    m = re.match(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", s)
    if m:
        day, mon, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mon <= 12 and 1 <= day <= 31:
            return f"{y:04d}-{mon:02d}-{day:02d}"

    # DD/MM/YY
    m = re.match(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2})", s)
    if m:
        day, mon, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y += 2000 if y < 50 else 1900
        if 1 <= mon <= 12 and 1 <= day <= 31:
            return f"{y:04d}-{mon:02d}-{day:02d}"

    # YYYY-MM-DD
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return m.group(0)

    # Mon DD (e.g. "Sep 01")
    m = re.match(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{1,2})", s, re.I)
    if m:
        mon = MONTH_MAP.get(m.group(1).upper())
        day = int(m.group(2))
        if mon:
            return f"{year:04d}-{mon:02d}-{day:02d}"

    return None


def detect_financial_table(table: ExtractedTable) -> Optional[Dict[str, int]]:
    """
    Check if a table looks like a financial statement.
    Returns column mapping dict or None.
    """
    headers = table.headers
    if not headers or len(headers) < 2:
        return None

    date_col = _find_column(headers, _DATE_PATTERNS)
    desc_col = _find_column(headers, _DESC_PATTERNS)
    withdraw_col = _find_column(headers, _WITHDRAWAL_PATTERNS)
    deposit_col = _find_column(headers, _DEPOSIT_PATTERNS)
    amount_col = _find_column(headers, _AMOUNT_PATTERNS)
    balance_col = _find_column(headers, _BALANCE_PATTERNS)

    # Must have at least a date column and one money column
    if date_col is None:
        return None
    if withdraw_col is None and deposit_col is None and amount_col is None:
        return None

    return {
        "date": date_col,
        "desc": desc_col,
        "withdrawal": withdraw_col,
        "deposit": deposit_col,
        "amount": amount_col,
        "balance": balance_col,
    }


def parse_financial_table(
    table: ExtractedTable,
    col_map: Dict[str, int],
    year: int,
) -> List[Dict[str, Any]]:
    """Parse a financial table into transaction dicts."""
    txns = []

    for row in table.rows:
        # Parse date
        date_val = row[col_map["date"]] if col_map["date"] < len(row) else ""
        date = _parse_date(date_val, year)
        if not date:
            continue  # Skip non-transaction rows

        # Parse description
        desc = ""
        if col_map.get("desc") is not None and col_map["desc"] < len(row):
            desc = row[col_map["desc"]].strip()

        # Parse amounts — three possible layouts:
        # 1. Separate withdrawal/deposit columns
        # 2. Single amount column (positive = credit, negative = debit)
        # 3. Single amount column with separate CR indicator

        withdrawal = None
        deposit = None

        if col_map.get("withdrawal") is not None and col_map.get("deposit") is not None:
            # Layout 1: separate columns
            w_val = row[col_map["withdrawal"]] if col_map["withdrawal"] < len(row) else ""
            d_val = row[col_map["deposit"]] if col_map["deposit"] < len(row) else ""
            withdrawal = _parse_amount(w_val)
            deposit = _parse_amount(d_val)
        elif col_map.get("amount") is not None:
            # Layout 2: single amount column
            a_val = row[col_map["amount"]] if col_map["amount"] < len(row) else ""
            amt = _parse_amount(a_val)
            if amt is not None:
                if amt < 0:
                    withdrawal = abs(amt)
                else:
                    deposit = amt
        elif col_map.get("withdrawal") is not None:
            # Only withdrawal column
            w_val = row[col_map["withdrawal"]] if col_map["withdrawal"] < len(row) else ""
            withdrawal = _parse_amount(w_val)
        elif col_map.get("deposit") is not None:
            # Only deposit column
            d_val = row[col_map["deposit"]] if col_map["deposit"] < len(row) else ""
            deposit = _parse_amount(d_val)

        if withdrawal is None and deposit is None:
            continue  # No amount found

        is_credit = deposit is not None and deposit > 0
        amount = deposit if is_credit else (withdrawal or 0)

        txns.append({
            "date": date,
            "payee": desc,
            "memo": "",
            "amount": abs(amount),
            "credit": is_credit,
        })

    return txns
