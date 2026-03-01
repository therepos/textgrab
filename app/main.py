import io
import csv
import os
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .extract import extract_structured, extract_structured_from_pdf, extract_text_from_pdf
from .categorize import predict_category, load_rules, save_rules
from .parsers.generic import detect_financial_table, parse_financial_table
from .parsepdf import extract_deposit_table
from .parsers.deposit import _ as deposit_mod
from .parsers import dispatch

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
# BANK STATEMENT → CSV
# -------------------------------------------------------------------
def _txns_to_csv(txns, single_amount_col=True) -> bytes:
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
            log.info(f"Generic parser: {len(all_txns)} transactions from {len(doc.all_tables)} tables")
            data = _txns_to_csv(all_txns, single_amount_col)
            return StreamingResponse(
                io.BytesIO(data),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=statement.csv"},
            )
    except Exception as e:
        log.warning(f"Generic table parser failed: {e}")

    # --- Strategy 2: DBS-specific parsers (fallback) ---
    try:
        raw_text = extract_text_from_pdf(content)

        if deposit_mod.detect(raw_text):
            log.info("Fallback: DBS deposit format detected")
            rows = extract_deposit_table(content)
            txns = deposit_mod.parse_from_table(rows, year)
            if txns:
                data = _txns_to_csv(txns, single_amount_col)
                return StreamingResponse(
                    io.BytesIO(data),
                    media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=statement.csv"},
                )
    except Exception as e:
        log.warning(f"DBS deposit parser failed: {e}")

    # --- Strategy 3: Text-based regex parsers (last resort) ---
    try:
        if not raw_text:
            raw_text = extract_text_from_pdf(content)
        txns = dispatch(raw_text, year)
        if txns:
            log.info(f"Fallback: text parser matched, {len(txns)} transactions")
            data = _txns_to_csv(txns, single_amount_col)
            return StreamingResponse(
                io.BytesIO(data),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=statement.csv"},
            )
    except RuntimeError:
        pass
    except Exception as e:
        log.warning(f"Text parser failed: {e}")

    # --- Nothing worked ---
    log.warning("No transactions found in any strategy")
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
