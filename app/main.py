import io
import csv
import os
import logging
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .extract import extract_structured, extract_structured_from_pdf, extract_text_from_pdf
from .categorize import predict_category, load_rules, save_rules
from .parsers import get_templates, get_parser, auto_detect
from .parsers.generic import detect_financial_table, parse_financial_table
from .parsers.helpers import extract_year_from_pdf

log = logging.getLogger(__name__)

app = FastAPI(
    title="textgrab",
    version="3.1",
    description="PDF/image text extraction + bank statement CSV converter",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

UPLOAD_DIR = "/tmp/textgrab_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ===================================================================
# TEXT EXTRACTION ENDPOINTS
# ===================================================================
@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    """Extract text from a PDF or image. Returns markdown with tables."""
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


# ===================================================================
# CSV HELPER
# ===================================================================
def _txns_to_csv(txns: List[Dict[str, Any]], single_amount_col: bool = True) -> bytes:
    """Convert transaction list to CSV bytes."""
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


# ===================================================================
# CONVERT ENDPOINT
# ===================================================================
@app.post("/api/convert", response_class=StreamingResponse)
async def convert_pdf(
    pdf: UploadFile = File(...),
    template: str = Form("auto"),
    single_amount_col: bool = Form(True),
):
    """Upload a bank statement PDF and receive a categorized CSV.

    Use /api/templates to list available templates.
    """
    content = await pdf.read()
    txns: List[Dict[str, Any]] = []

    if template != "auto":
        # --- Named template: dispatch directly ---
        parser = get_parser(template)
        if parser is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown template: {template}. "
                       f"Available: {list(get_templates().keys())}",
            )
        txns = parser.parse(content)

    else:
        # --- Auto mode ---
        # Strategy 1: Generic table parser (img2table)
        try:
            doc = extract_structured_from_pdf(content)
            for table in doc.all_tables:
                col_map = detect_financial_table(table)
                if col_map:
                    year = extract_year_from_pdf(content) or 2025
                    parsed = parse_financial_table(table, col_map, year)
                    txns.extend(parsed)
        except Exception as e:
            log.warning(f"Generic table parser failed: {e}")

        # Strategy 2: Auto-detect registered parsers
        if not txns:
            try:
                raw_text = extract_text_from_pdf(content)
                slug = auto_detect(raw_text)
                if slug:
                    parser = get_parser(slug)
                    log.info(f"Auto-detected: {slug} ({parser.LABEL})")
                    txns = parser.parse(content)
            except Exception as e:
                log.warning(f"Auto-detect failed: {e}")

    if txns:
        log.info(f"Parsed {len(txns)} transactions (template={template})")

    data = _txns_to_csv(txns, single_amount_col)
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=statement.csv"},
    )


# ===================================================================
# TEMPLATE LIST
# ===================================================================
@app.get("/api/templates")
def list_templates():
    """Return available statement templates (auto-discovered)."""
    return get_templates()


# ===================================================================
# RULES MANAGEMENT
# ===================================================================
@app.get("/api/rules")
def get_rules():
    return load_rules()


@app.post("/api/rules")
def upsert_rules(
    rules: dict = Body(
        ...,
        example={"Food": ["ROYAL CABRI", "STARBUCKS"], "Transport": ["GRAB"]},
    ),
):
    save_rules(rules or {})
    return {"count": sum(len(v) for v in (rules or {}).values())}


# ===================================================================
# HEALTH + VERSION
# ===================================================================
@app.get("/api/health")
def health():
    rules_path = Path("/data/models/rules.json")
    return {
        "ok": True,
        "rules_exists": rules_path.exists(),
        "rules_path": str(rules_path),
    }


@app.get("/api/version")
def version():
    return {"service": "textgrab", "version": "3.1"}


# ===================================================================
# STATIC FILES (frontend) — must be last
# ===================================================================
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
