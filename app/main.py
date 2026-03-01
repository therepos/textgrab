import io
import re
import csv
import os
import logging
from typing import List, Dict, Any
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import pdfplumber

from .extract import extract_structured, extract_structured_from_pdf, extract_text_from_pdf
from .categorize import predict_category, load_rules, save_rules
from .parsers.generic import detect_financial_table, parse_financial_table

log = logging.getLogger(__name__)

app = FastAPI(
    title="textgrab",
    version="3.0",
    description="PDF/image text extraction + bank statement CSV converter",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

UPLOAD_DIR = "/tmp/textgrab_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MONTH = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
}
_money_re = re.compile(r"([0-9]{1,3}(?:,[0-9]{3})*\.\d{2})")


# -------------------------------------------------------------------
# TEXT EXTRACTION
# -------------------------------------------------------------------
@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    """Extract text from a PDF or image. Returns markdown with tables formatted."""
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(input_path, "wb") as f:
        f.write(await file.read())

    try:
        doc = extract_structured(input_path)
        markdown = doc.to_markdown()
        plain = doc.to_plain_text()
        table_count = len(doc.all_tables)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    return {
        "text": markdown,
        "plain_text": plain,
        "table_count": table_count,
        "filename": file.filename,
    }


@app.post("/api/extract-structured")
async def extract_structured_endpoint(file: UploadFile = File(...)):
    """Extract structured content (text + tables as JSON) from a PDF or image."""
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(input_path, "wb") as f:
        f.write(await file.read())

    try:
        doc = extract_structured(input_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    return doc.to_dict()


# -------------------------------------------------------------------
# BANK STATEMENT → CSV (helpers)
# -------------------------------------------------------------------
def _txns_to_csv(txns, single_amount_col=True) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    if single_amount_col:
        w.writerow(["Date", "Payee", "Category", "Memo", "Amount"])
    else:
        w.writerow(["Date", "Payee", "Category", "Memo", "Outflow", "Inflow"])

    for t in txns:
        date = t["date"]
        payee = t["payee"]
        memo = t.get("memo", "")
        amount = float(t.get("amount", 0.0))
        is_credit = bool(t.get("credit", False))
        cat = predict_category(payee, memo, refund_hint=is_credit)

        if single_amount_col:
            amt = amount if is_credit else -amount
            w.writerow([date, payee, cat, memo, f"{amt:.2f}"])
        else:
            outflow = "" if is_credit else f"{amount:.2f}"
            inflow = f"{amount:.2f}" if is_credit else ""
            w.writerow([date, payee, cat, memo, outflow, inflow])

    return buf.getvalue().encode("utf-8")


# -------------------------------------------------------------------
# DBS fallback: deposit statement (inlined, no external files needed)
# -------------------------------------------------------------------
def _dbs_detect(raw_text: str) -> bool:
    return bool(re.search(
        r"DETAILS OF TRANSACTIONS\s+WITHDRAWAL\(\$\)\s+DEPOSIT\(\$\)", raw_text, re.I
    ))


def _dbs_extract_deposit_table(content: bytes) -> List[Dict[str, str]]:
    """Extract DBS deposit table using column x-positions from header row."""
    rows = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False) or []
            if not words:
                continue

            # Find the header row by locating WITHDRAWAL($) — unique enough
            header_y = None
            for w in words:
                if w.get("text", "").upper().startswith("WITHDRAWAL"):
                    header_y = float(w["top"])
                    break
            if header_y is None:
                continue

            # Get all words on the header row (within 3pt tolerance)
            header_words = [w for w in words if abs(float(w["top"]) - header_y) < 3]

            def _find_x_in_row(token):
                t = token.upper()
                for w in header_words:
                    if w.get("text", "").upper().startswith(t):
                        return float(w["x0"])
                return None

            x_date = _find_x_in_row("DATE")
            x_desc = _find_x_in_row("DETAILS")
            x_wdr = _find_x_in_row("WITHDRAWAL")
            x_dep = _find_x_in_row("DEPOSIT")
            x_bal = _find_x_in_row("BALANCE")
            if not all([x_date, x_desc, x_wdr, x_dep, x_bal]):
                continue

            edges = [x_date, x_desc, x_wdr, x_dep, x_bal, page.width]
            lines: Dict[int, list] = defaultdict(list)
            for w in words:
                key = int(round(float(w["top"]) / 3.0))
                lines[key].append(w)

            for key in sorted(lines):
                ws = sorted(lines[key], key=lambda z: float(z["x0"]))
                cols = ["", "", "", "", ""]
                for w in ws:
                    x = float(w["x0"])
                    # Use 2pt tolerance to handle sub-pixel alignment differences
                    idx = sum(1 for e in edges if x >= e - 2.0) - 1
                    if 0 <= idx < 5:
                        cols[idx] = (cols[idx] + " " + w["text"]).strip()

                if re.match(r"^\d{2}\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b", cols[0], re.I):
                    rows.append({
                        "date": cols[0], "desc": cols[1],
                        "withdrawal": cols[2], "deposit": cols[3], "balance": cols[4],
                    })
    return rows


def _dbs_parse_deposit(rows: List[Dict[str, str]], year: int) -> List[Dict[str, Any]]:
    """Parse DBS deposit table rows into transaction dicts."""
    txns = []
    for r in rows:
        m = re.match(r"^(\d{2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b", r["date"], re.I)
        if not m:
            continue
        day, mon = m.group(1), m.group(2)
        w = _money_re.search(r.get("withdrawal", "") or "")
        d = _money_re.search(r.get("deposit", "") or "")
        if not (w or d):
            continue
        amt = float((d or w).group(1).replace(",", ""))
        credit = bool(d)
        date_str = f"{year:04d}-{MONTH[mon.upper()]:02d}-{int(day):02d}"
        txns.append({
            "date": date_str,
            "payee": r.get("desc", "").strip(),
            "memo": "",
            "amount": amt,
            "credit": credit,
        })
    return txns


# -------------------------------------------------------------------
# DBS fallback: credit card statement (inlined)
# -------------------------------------------------------------------
def _dbs_detect_cc(raw_text: str) -> bool:
    return bool(re.search(r"\bNEW TRANSACTIONS\b", raw_text, re.I))


def _dbs_parse_cc(raw_text: str, year: int) -> List[Dict[str, Any]]:
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw_text.splitlines() if ln.strip()]
    s = next((i for i, l in enumerate(lines) if re.search(r"\bNEW TRANSACTIONS\b", l)), None)
    if s is None:
        s = next((i for i, l in enumerate(lines) if re.search(r"\bPREVIOUS BALANCE\b", l)), None)
    if s is None:
        return []

    e = next((j for j in range(s + 1, len(lines))
              if re.match(r"^(SUB-TOTAL:|TOTAL:|INSTALMENT PLANS SUMMARY)", lines[j])), len(lines))
    txn_lines = lines[s + 1:e]

    date_re = re.compile(r'^(?P<d>\d{2}) (?P<m>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b', re.I)
    amt_line = re.compile(r'^(?P<a>[0-9]{1,3}(?:,[0-9]{3})*\.\d{2})(?:\s*(?P<cr>CR))?$', re.I)
    amt_inline = re.compile(r'(?P<a>[0-9]{1,3}(?:,[0-9]{3})*\.\d{2})(?:\s*(?P<cr>CR))?$')
    fx_info = re.compile(r'(U\. S\. DOLLAR|MALAYSIAN RINGGIT)\s+([0-9]{1,3}(?:,[0-9]{3})*\.\d{2})', re.I)

    txns, cur = [], None
    for ln in txn_lines:
        m = date_re.match(ln)
        if m:
            if cur and ('amount' in cur) and cur.get('payee'):
                txns.append(cur)
            d, mon = m.group('d'), m.group('m')
            rest = ln[m.end():].strip()
            mi = amt_inline.search(rest)
            cr = False; amt = None
            if mi:
                amt = mi.group('a'); cr = bool(mi.group('cr')); rest = rest[:mi.start()].strip()
            date_str = f"{year:04d}-{MONTH[mon.upper()]:02d}-{int(d):02d}"
            cur = {'date': date_str, 'payee': rest, 'memo': ''}
            if amt:
                cur['amount'] = float(amt.replace(',', ''))
                cur['credit'] = cr
            continue

        if cur is None:
            continue
        if fx_info.search(ln):
            cur['memo'] = (cur.get('memo', '') + ('; ' if cur.get('memo') else '') + ln).strip()
            continue
        ma = amt_line.match(ln)
        if ma and 'amount' not in cur:
            cur['amount'] = float(ma.group('a').replace(',', ''))
            cur['credit'] = bool(ma.group('cr'))
            continue
        if not re.match(r'^(Credit Cards|DBS Cards|Hotline:|Statement of Account|PDS_)', ln, re.I):
            cur['payee'] = (cur['payee'] + ' ' + ln).strip()

    if cur and ('amount' in cur) and cur.get('payee'):
        txns.append(cur)
    for t in txns:
        t['payee'] = re.sub(r'\s{2,}', ' ', t['payee']).strip()
    return txns


# -------------------------------------------------------------------
# CONVERT ENDPOINT
# -------------------------------------------------------------------
@app.post("/api/convert", response_class=StreamingResponse)
async def convert_pdf(
    pdf: UploadFile = File(...),
    year: int = Form(2025),
    single_amount_col: bool = Form(True),
):
    """Upload a bank statement PDF and receive a categorized CSV."""
    content = await pdf.read()

    # --- Strategy 1: Generic table-based extraction (works across banks) ---
    try:
        doc = extract_structured_from_pdf(content)
        all_txns = []
        for table in doc.all_tables:
            col_map = detect_financial_table(table)
            if col_map:
                txns = parse_financial_table(table, col_map, year)
                all_txns.extend(txns)

        if all_txns:
            log.info(f"Generic parser: {len(all_txns)} transactions")
            data = _txns_to_csv(all_txns, single_amount_col)
            return StreamingResponse(
                io.BytesIO(data),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=statement.csv"},
            )
    except Exception as e:
        log.warning(f"Generic table parser failed: {e}")

    # --- Strategy 2: DBS-specific fallback (inlined) ---
    try:
        raw_text = extract_text_from_pdf(content)

        # DBS deposit
        if _dbs_detect(raw_text):
            log.info("Fallback: DBS deposit detected")
            rows = _dbs_extract_deposit_table(content)
            txns = _dbs_parse_deposit(rows, year)
            if txns:
                data = _txns_to_csv(txns, single_amount_col)
                return StreamingResponse(
                    io.BytesIO(data),
                    media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=statement.csv"},
                )

        # DBS credit card
        if _dbs_detect_cc(raw_text):
            log.info("Fallback: DBS credit card detected")
            txns = _dbs_parse_cc(raw_text, year)
            if txns:
                data = _txns_to_csv(txns, single_amount_col)
                return StreamingResponse(
                    io.BytesIO(data),
                    media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=statement.csv"},
                )
    except Exception as e:
        log.warning(f"DBS fallback failed: {e}")

    # --- Nothing worked ---
    log.warning("No transactions found")
    data = _txns_to_csv([], single_amount_col)
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=statement.csv"},
    )


# -------------------------------------------------------------------
# RULES MANAGEMENT
# -------------------------------------------------------------------
@app.get("/api/rules")
def get_rules():
    return load_rules()


@app.post("/api/rules")
def upsert_rules(
    rules: dict = Body(
        ...,
        example={"Food": ["ROYAL CABRI", "STARBUCKS"], "Transport": ["GRAB"]},
    )
):
    save_rules(rules or {})
    return {"count": sum(len(v) for v in (rules or {}).values())}


# -------------------------------------------------------------------
# HEALTH + VERSION
# -------------------------------------------------------------------
@app.get("/api/health")
def health():
    rules_path = Path("/data/models/rules.json")
    return {"ok": True, "rules_exists": rules_path.exists(), "rules_path": str(rules_path)}


@app.get("/api/version")
def version():
    return {"service": "textgrab", "version": "3.0"}


# -------------------------------------------------------------------
# STATIC FILES (frontend) — must be last
# -------------------------------------------------------------------
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
