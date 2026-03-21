import io
import csv
import os
import logging
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .extracttext import extract_text_from_bytes, get_file_type, SUPPORTED_EXTENSIONS
from .extracttabular import (
    extract_structured,
    extract_structured_from_pdf,
    extract_text_from_pdf,
)
from .categorize import predict_category, derive_payee, load_lookup, save_lookup
from .parsers import get_templates, get_parser, auto_detect
from .parsers.generic import detect_financial_table, parse_financial_table
from .parsers.helpers import extract_year_from_pdf
from .schemes import get_schemes, get_scheme

log = logging.getLogger(__name__)

app = FastAPI(
    title="textgrab",
    version="4.0",
    description="Text extraction + tabular data conversion",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

UPLOAD_DIR = "/tmp/textgrab_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ===================================================================
# TAB 1: TEXT — extract text with optional scheme transform
# ===================================================================
@app.post("/api/text")
async def text_extract(
    files: List[UploadFile] = File(...),
    scheme: str = Form("raw"),
    output_mode: str = Form("consolidated"),
):
    """Extract text from file(s) with optional scheme transformation."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Validate scheme
    schemes = get_schemes()
    if scheme != "raw" and scheme not in schemes:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scheme: {scheme}. Available: {list(schemes.keys())}",
        )

    # Check if scheme wants raw bytes instead of extracted text
    raw_input = False
    if scheme != "raw":
        scheme_mod_check = get_scheme(scheme)
        if scheme_mod_check and getattr(scheme_mod_check, "RAW_INPUT", False):
            raw_input = True

    # Extract text (or collect raw bytes) from each file
    extracted = {}
    errors = []
    for f in files:
        try:
            content = await f.read()
            if raw_input:
                extracted[f.filename] = content
            else:
                text = extract_text_from_bytes(content, f.filename)
                extracted[f.filename] = text
        except Exception as e:
            errors.append({"filename": f.filename, "error": str(e)})

    if not extracted and errors:
        raise HTTPException(
            status_code=500,
            detail=f"All extractions failed: {errors}",
        )

    # Apply scheme transform
    if scheme == "raw":
        if len(extracted) == 1:
            fname, text = next(iter(extracted.items()))
            return {
                "scheme": "raw",
                "text": text,
                "filename": fname,
                "errors": errors,
            }
        else:
            # Multi-file raw: merge or separate
            if output_mode == "individual":
                file_results = [
                    {"filename": fn, "text": txt}
                    for fn, txt in sorted(extracted.items())
                ]
                return {
                    "scheme": "raw",
                    "merged": False,
                    "files": file_results,
                    "errors": errors,
                }
            else:
                merged_parts = []
                for fn, txt in sorted(extracted.items()):
                    merged_parts.append(f"=== {fn} ===\n\n{txt}")
                return {
                    "scheme": "raw",
                    "merged": True,
                    "text": "\n\n---\n\n".join(merged_parts),
                    "filename": "merged_extraction.txt",
                    "file_count": len(extracted),
                    "errors": errors,
                }

    # Non-raw scheme: run transform
    scheme_mod = get_scheme(scheme)
    if scheme_mod is None:
        raise HTTPException(status_code=400, detail=f"Scheme '{scheme}' not found")

    # Validate file types against scheme
    for fname in extracted:
        ext = os.path.splitext(fname)[-1].lower()
        if ext not in scheme_mod.ACCEPTS:
            raise HTTPException(
                status_code=400,
                detail=f"File '{fname}' ({ext}) not accepted by scheme '{scheme}'. "
                       f"Accepted: {scheme_mod.ACCEPTS}",
            )

    try:
        result = scheme_mod.transform(extracted, output_mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scheme transform failed: {str(e)}")

    return {
        "scheme": scheme,
        "errors": errors,
        **result,
    }


@app.get("/api/schemes")
def list_schemes():
    """Return available schemes with their accepted file types."""
    return get_schemes()


# ===================================================================
# TAB 2: TABULAR — structured extraction + bank statement conversion
# ===================================================================
@app.post("/api/tabular/extract")
async def tabular_extract(file: UploadFile = File(...)):
    """Extract structured content (text + tables) from a PDF or image."""
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


@app.post("/api/tabular/convert", response_class=StreamingResponse)
async def tabular_convert(
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


@app.post("/api/tabular/convert-bulk", response_class=StreamingResponse)
async def tabular_convert_bulk(
    pdfs: List[UploadFile] = File(...),
    template: str = Form("auto"),
    single_amount_col: bool = Form(True),
    merge: bool = Form(True),
    dedupe: bool = Form(True),
    include_source_file: bool = Form(True),
):
    """Upload multiple bank statement PDFs and receive one merged CSV."""
    if not pdfs:
        raise HTTPException(status_code=400, detail="No PDF files uploaded")

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
# BACKWARD COMPAT — keep old endpoints working
# ===================================================================
@app.post("/api/extract")
async def extract_compat(file: UploadFile = File(...)):
    """Legacy endpoint — redirects to /api/text."""
    return await text_extract(files=[file], scheme="raw", output_mode="consolidated")


@app.post("/api/convert", response_class=StreamingResponse)
async def convert_compat(
    pdf: UploadFile = File(...),
    template: str = Form("auto"),
    single_amount_col: bool = Form(True),
):
    """Legacy endpoint — redirects to /api/tabular/convert."""
    return await tabular_convert(pdf=pdf, template=template, single_amount_col=single_amount_col)


# ===================================================================
# CONVERSION HELPERS (unchanged)
# ===================================================================
def _parse_transactions_from_pdf(content: bytes, template: str = "auto") -> List[Dict[str, Any]]:
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
        description = t["payee"]
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
# TEMPLATE + LOOKUP ENDPOINTS
# ===================================================================
@app.get("/api/templates")
def list_templates():
    return get_templates()


@app.get("/api/lookup")
def get_lookup():
    return load_lookup()


@app.post("/api/lookup")
def upsert_lookup(
    data: dict = Body(
        ...,
        example={
            "payee_categories": {"Starbucks": "Food"},
            "payee_aliases": {"starbucks": "Starbucks"},
        },
    ),
):
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
    return {
        "service": "textgrab",
        "version": "4.1",
    }


# ===================================================================
# STATIC FILES — must be last
# ===================================================================
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
