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
from .categorize import predict_category, derive_payee, load_lookup, save_lookup
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


@app.post("/api/extract-bulk")
async def extract_bulk(files: List[UploadFile] = File(...)):
    """Extract text from multiple PDFs/images and return one merged result."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    results = []
    total_tables = 0

    for file in files:
        input_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(input_path, "wb") as f:
            f.write(await file.read())

        try:
            doc = extract_structured(input_path)
            markdown = doc.to_markdown()
            plain = doc.to_plain_text()
            table_count = len(doc.all_tables)
            total_tables += table_count
            results.append({
                "filename": file.filename,
                "text": markdown,
                "plain_text": plain,
                "table_count": table_count,
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "text": "",
                "plain_text": "",
                "table_count": 0,
                "error": str(e),
            })
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    # Build merged output
    merged_md_parts = []
    merged_plain_parts = []
    for r in results:
        header = f"## 📄 {r['filename']}"
        if r.get("error"):
            merged_md_parts.append(f"{header}\n\n⚠️ Extraction failed: {r['error']}")
            merged_plain_parts.append(f"=== {r['filename']} ===\n\nExtraction failed: {r['error']}")
        else:
            merged_md_parts.append(f"{header}\n\n{r['text']}")
            merged_plain_parts.append(f"=== {r['filename']} ===\n\n{r['plain_text']}")

    merged_md = "\n\n---\n\n".join(merged_md_parts)
    merged_plain = "\n\n---\n\n".join(merged_plain_parts)

    return {
        "text": merged_md,
        "plain_text": merged_plain,
        "table_count": total_tables,
        "file_count": len(files),
        "files": results,
    }


# ===================================================================
# CONVERSION HELPERS
# ===================================================================
def _parse_transactions_from_pdf(content: bytes, template: str = "auto") -> List[Dict[str, Any]]:
    """Parse one statement PDF into standardised transaction dicts."""
    txns: List[Dict[str, Any]] = []

    if template != "auto":
        parser = get_parser(template)
        if parser is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown template: {template}. "
                       f"Available: {list(get_templates().keys())}",
            )
        return parser.parse(content)

    try:
        doc = extract_structured_from_pdf(content)
        for table in doc.all_tables:
            col_map = detect_financial_table(table)
            if col_map:
                year = extract_year_from_pdf(content) or 2025
                txns.extend(parse_financial_table(table, col_map, year))
    except Exception as e:
        log.warning(f"Generic table parser failed: {e}")

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

    return txns


def _dedupe_transactions(txns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop exact duplicate transactions that can happen across overlapping statements."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for t in txns:
        key = (
            t.get("date", ""),
            t.get("payee", ""),
            t.get("memo", ""),
            round(float(t.get("amount", 0.0)), 2),
            bool(t.get("credit", False)),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _sort_transactions(txns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        txns,
        key=lambda t: (
            t.get("date", ""),
            t.get("payee", ""),
            t.get("memo", ""),
            float(t.get("amount", 0.0)),
            bool(t.get("credit", False)),
        ),
    )


def _txns_to_csv(
    txns: List[Dict[str, Any]],
    single_amount_col: bool = True,
    include_source_file: bool = False,
) -> bytes:
    """Convert transaction list to CSV bytes."""
    buf = io.StringIO()
    w = csv.writer(buf)

    headers = ["Date", "Description", "Payee", "Category", "Memo"]
    if include_source_file:
        headers.append("SourceFile")
    if single_amount_col:
        headers.append("Amount")
    else:
        headers.extend(["Outflow", "Inflow"])
    w.writerow(headers)

    for t in txns:
        date = t["date"]
        description = t["payee"]  # raw bank description
        memo = t.get("memo", "")
        amount = float(t.get("amount", 0.0))
        is_credit = bool(t.get("credit", False))

        payee = derive_payee(description)
        cat = predict_category(description=description, payee=payee, memo=memo, refund_hint=is_credit)

        row = [date, description, payee, cat, memo]
        if include_source_file:
            row.append(t.get("source_file", ""))

        if single_amount_col:
            amt = amount if is_credit else -amount
            row.append(f"{amt:.2f}")
        else:
            outflow = "" if is_credit else f"{amount:.2f}"
            inflow = f"{amount:.2f}" if is_credit else ""
            row.extend([outflow, inflow])

        w.writerow(row)

    return buf.getvalue().encode("utf-8")


# ===================================================================
# CONVERT ENDPOINTS
# ===================================================================
@app.post("/api/convert", response_class=StreamingResponse)
async def convert_pdf(
    pdf: UploadFile = File(...),
    template: str = Form("auto"),
    single_amount_col: bool = Form(True),
):
    """Upload a bank statement PDF and receive a categorized CSV."""
    content = await pdf.read()
    txns = _parse_transactions_from_pdf(content, template)

    if txns:
        log.info(f"Parsed {len(txns)} transactions (template={template})")

    data = _txns_to_csv(txns, single_amount_col)
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=statement.csv"},
    )


@app.post("/api/convert-bulk", response_class=StreamingResponse)
async def convert_pdf_bulk(
    pdfs: List[UploadFile] = File(...),
    template: str = Form("auto"),
    single_amount_col: bool = Form(True),
    merge: bool = Form(True),
    dedupe: bool = Form(True),
    include_source_file: bool = Form(True),
):
    """Upload multiple bank statement PDFs and receive one merged categorized CSV."""
    if not pdfs:
        raise HTTPException(status_code=400, detail="No PDF files uploaded")

    if not merge:
        raise HTTPException(status_code=400, detail="Only merged output is currently supported")

    all_txns: List[Dict[str, Any]] = []
    file_stats = []

    for pdf in pdfs:
        content = await pdf.read()
        txns = _parse_transactions_from_pdf(content, template)
        for t in txns:
            t["source_file"] = pdf.filename or "statement.pdf"
        all_txns.extend(txns)
        file_stats.append((pdf.filename or "statement.pdf", len(txns)))

    if dedupe:
        all_txns = _dedupe_transactions(all_txns)
    all_txns = _sort_transactions(all_txns)

    for name, count in file_stats:
        log.info(f"Parsed {count} transactions from {name} (template={template})")
    log.info(f"Bulk parsed {len(all_txns)} merged transactions from {len(file_stats)} files")

    data = _txns_to_csv(
        all_txns,
        single_amount_col=single_amount_col,
        include_source_file=include_source_file,
    )
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=statements_merged.csv"},
    )


# ===================================================================
# TEMPLATE LIST
# ===================================================================
@app.get("/api/templates")
def list_templates():
    """Return available statement templates (auto-discovered)."""
    return get_templates()


# ===================================================================
# LOOKUP MANAGEMENT
# ===================================================================
@app.get("/api/lookup")
def get_lookup():
    """Return the payee → category lookup and alias table."""
    return load_lookup()


@app.post("/api/lookup")
def upsert_lookup(
    data: dict = Body(
        ...,
        example={
            "payee_categories": {"Starbucks": "Food", "Grab": "Transport"},
            "payee_aliases": {"starbucks": "Starbucks"},
        },
    ),
):
    """Replace the full lookup table."""
    save_lookup(data or {})
    cats = (data or {}).get("payee_categories", {})
    aliases = (data or {}).get("payee_aliases", {})
    return {"payees": len(cats), "aliases": len(aliases)}


# ===================================================================
# HEALTH + VERSION
# ===================================================================
@app.get("/api/health")
def health():
    lookup_path = Path("/data/models/lookup.json")
    return {
        "ok": True,
        "lookup_exists": lookup_path.exists(),
        "lookup_path": str(lookup_path),
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
