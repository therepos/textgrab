import io
import os
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import pdfplumber
from PIL import Image
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

_doctr_model = None


def _get_doctr_model():
    global _doctr_model
    if _doctr_model is None:
        logger.info("Loading doctr OCR model...")
        from doctr.models import ocr_predictor
        _doctr_model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
        )
        logger.info("doctr OCR model loaded.")
    return _doctr_model


def _preprocess_image(img: Image.Image) -> Image.Image:
    import cv2
    import numpy as np
    arr = np.array(img)
    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    denoised = cv2.fastNlMeansDenoising(binary, h=10)
    return Image.fromarray(denoised)


def _ocr_with_doctr(images: list[Image.Image]) -> str:
    import numpy as np
    model = _get_doctr_model()
    pages = [np.array(img.convert("RGB")) for img in images]
    result = model(pages)
    text_parts = []
    for page in result.pages:
        page_text = []
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join(word.value for word in line.words)
                page_text.append(line_text)
        text_parts.append("\n".join(page_text))
    return "\n\n".join(text_parts)


# ---------------------------------------------------------------------------
# Structured extraction types
# ---------------------------------------------------------------------------
@dataclass
class ExtractedTable:
    headers: List[str]
    rows: List[List[str]]
    page: int = 0

    def to_markdown(self) -> str:
        if not self.headers and not self.rows:
            return ""
        cols = self.headers if self.headers else [f"Col{i+1}" for i in range(len(self.rows[0]))]
        def _clean(c):
            return str(c or "").replace("|", "\\|").replace("\n", " ").strip()
        lines = []
        lines.append("| " + " | ".join(_clean(c) for c in cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for row in self.rows:
            padded = row + [""] * (len(cols) - len(row))
            lines.append("| " + " | ".join(_clean(c) for c in padded[:len(cols)]) + " |")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"headers": self.headers, "rows": self.rows, "page": self.page}


@dataclass
class StructuredPage:
    page_num: int
    text: str = ""
    tables: List[ExtractedTable] = field(default_factory=list)


@dataclass
class StructuredDocument:
    pages: List[StructuredPage] = field(default_factory=list)

    @property
    def all_tables(self) -> List[ExtractedTable]:
        return [t for p in self.pages for t in p.tables]

    def to_plain_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    def to_markdown(self) -> str:
        parts = []
        for page in self.pages:
            page_parts = []
            if page.text.strip():
                page_parts.append(page.text.strip())
            for table in page.tables:
                md = table.to_markdown()
                if md:
                    page_parts.append(md)
            if page_parts:
                parts.append("\n\n".join(page_parts))
        return "\n\n---\n\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pages": [
                {
                    "page": p.page_num,
                    "text": p.text,
                    "tables": [t.to_dict() for t in p.tables],
                }
                for p in self.pages
            ],
            "table_count": len(self.all_tables),
        }


# ---------------------------------------------------------------------------
# pdfplumber structured extraction (native/digital PDFs)
# ---------------------------------------------------------------------------
def _clean_table(raw_table: List[List]) -> Optional[ExtractedTable]:
    if not raw_table or len(raw_table) < 2:
        return None
    headers = [str(c or "").strip() for c in raw_table[0]]
    if not any(headers):
        if len(raw_table) > 2:
            headers = [str(c or "").strip() for c in raw_table[1]]
            raw_table = raw_table[1:]
        else:
            return None
    rows = []
    for raw_row in raw_table[1:]:
        row = [str(c or "").strip() for c in raw_row]
        if any(row):
            rows.append(row)
    if not rows:
        return None
    return ExtractedTable(headers=headers, rows=rows)


def _extract_text_without_tables(page, tables) -> str:
    if not tables:
        return page.extract_text(x_tolerance=2, y_tolerance=3) or ""

    bboxes = []
    for t in tables:
        if hasattr(t, "bbox") and t.bbox:
            bboxes.append(t.bbox)

    if not bboxes:
        return page.extract_text(x_tolerance=2, y_tolerance=3) or ""

    words = page.extract_words(x_tolerance=2, y_tolerance=3) or []
    filtered = []
    for w in words:
        wx0, wtop, wx1, wbottom = float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"])
        in_table = False
        for (tx0, ttop, tx1, tbottom) in bboxes:
            if wx0 >= tx0 - 2 and wtop >= ttop - 2 and wx1 <= tx1 + 2 and wbottom <= tbottom + 2:
                in_table = True
                break
        if not in_table:
            filtered.append(w)

    if not filtered:
        return ""

    lines: Dict[int, List[dict]] = defaultdict(list)
    for w in filtered:
        key = int(round(float(w["top"]) / 3.0))
        lines[key].append(w)

    text_lines = []
    for key in sorted(lines):
        ws = sorted(lines[key], key=lambda z: float(z["x0"]))
        text_lines.append(" ".join(w["text"] for w in ws))

    return "\n".join(text_lines)


def _structured_from_pdfplumber(content: bytes) -> Optional[StructuredDocument]:
    doc = StructuredDocument()
    has_content = False

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for i, page in enumerate(pdf.pages):
            sp = StructuredPage(page_num=i + 1)

            # Find tables — try line-based first, then text-based
            raw_tables = page.find_tables({
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 5,
            }) or []

            if not raw_tables:
                raw_tables = page.find_tables({
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 5,
                    "min_words_vertical": 3,
                    "min_words_horizontal": 2,
                }) or []

            for rt in raw_tables:
                extracted = rt.extract()
                table = _clean_table(extracted)
                if table:
                    table.page = i + 1
                    sp.tables.append(table)

            sp.text = _extract_text_without_tables(page, raw_tables)
            if sp.text.strip() or sp.tables:
                has_content = True

            doc.pages.append(sp)

    return doc if has_content else None


# ---------------------------------------------------------------------------
# doctr structured extraction (scanned/image PDFs)
# ---------------------------------------------------------------------------
def _structured_from_doctr_images(images: list[Image.Image]) -> StructuredDocument:
    import numpy as np

    model = _get_doctr_model()
    pages_arr = [np.array(img.convert("RGB")) for img in images]
    result = model(pages_arr)

    doc = StructuredDocument()

    for i, page in enumerate(result.pages):
        sp = StructuredPage(page_num=i + 1)

        page_lines = []
        for block in page.blocks:
            for line in block.lines:
                words = []
                for word in line.words:
                    words.append({
                        "text": word.value,
                        "geometry": word.geometry,
                    })
                if words:
                    line_text = " ".join(w["text"] for w in words)
                    y_pos = words[0]["geometry"][0][1]
                    x_pos = words[0]["geometry"][0][0]
                    page_lines.append({
                        "text": line_text,
                        "words": words,
                        "y": y_pos,
                        "x": x_pos,
                    })

        page_lines.sort(key=lambda l: (l["y"], l["x"]))

        tables = _detect_tables_from_lines(page_lines, page_num=i + 1)

        table_line_indices = set()
        for table in tables:
            table_line_indices.update(table.get("_line_indices", set()))

        text_lines = []
        for idx, line in enumerate(page_lines):
            if idx not in table_line_indices:
                text_lines.append(line["text"])

        sp.text = "\n".join(text_lines)
        for table in tables:
            et = ExtractedTable(
                headers=table["headers"],
                rows=table["rows"],
                page=i + 1,
            )
            sp.tables.append(et)

        doc.pages.append(sp)

    return doc


def _detect_tables_from_lines(lines: List[Dict], page_num: int) -> List[Dict]:
    """Heuristic table detection from OCR'd lines by column alignment."""
    if len(lines) < 3:
        return []

    tables = []

    def _get_word_columns(line_data: Dict) -> List[float]:
        return [w["geometry"][0][0] for w in line_data["words"]]

    def _columns_match(cols_a: List[float], cols_b: List[float], tolerance: float = 0.03) -> bool:
        if abs(len(cols_a) - len(cols_b)) > 1:
            return False
        if len(cols_a) < 3 or len(cols_b) < 3:
            return False
        matches = 0
        for ca in cols_a:
            for cb in cols_b:
                if abs(ca - cb) < tolerance:
                    matches += 1
                    break
        return matches >= min(len(cols_a), len(cols_b)) * 0.5

    i = 0
    while i < len(lines):
        cols_i = _get_word_columns(lines[i])
        if len(cols_i) < 3:
            i += 1
            continue

        run = [i]
        for j in range(i + 1, len(lines)):
            cols_j = _get_word_columns(lines[j])
            if _columns_match(cols_i, cols_j):
                run.append(j)
            elif len(run) >= 3:
                break
            else:
                run = [j]
                cols_i = cols_j

        if len(run) >= 3:
            header_words = [w["text"] for w in lines[run[0]]["words"]]
            data_rows = []
            for idx in run[1:]:
                row = [w["text"] for w in lines[idx]["words"]]
                data_rows.append(row)

            tables.append({
                "headers": header_words,
                "rows": data_rows,
                "_line_indices": set(run),
            })
            i = run[-1] + 1
        else:
            i += 1

    return tables


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_text_from_pdf(content: bytes) -> str:
    """Extract plain text from PDF. pdfplumber first, doctr fallback."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for p in pdf.pages:
            txt = p.extract_text(x_tolerance=2, y_tolerance=3) or ""
            text_parts.append(txt.rstrip())

    text = "\n".join(text_parts).strip()
    if text and len(text) > 50:
        return text

    logger.info("pdfplumber sparse, falling back to doctr OCR")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        images = convert_from_path(tmp_path, dpi=300)
        preprocessed = [_preprocess_image(img) for img in images]
        return _ocr_with_doctr(preprocessed)
    finally:
        os.unlink(tmp_path)


def extract_structured_from_pdf(content: bytes) -> StructuredDocument:
    """Extract structured content (text + tables) from PDF."""
    doc = _structured_from_pdfplumber(content)
    if doc and (doc.all_tables or any(p.text.strip() for p in doc.pages)):
        total_text = sum(len(p.text) for p in doc.pages)
        if total_text > 50 or doc.all_tables:
            return doc

    logger.info("Falling back to doctr for structured extraction")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        images = convert_from_path(tmp_path, dpi=300)
        preprocessed = [_preprocess_image(img) for img in images]
        return _structured_from_doctr_images(preprocessed)
    finally:
        os.unlink(tmp_path)


def extract_structured_from_image(path: str) -> StructuredDocument:
    img = Image.open(path)
    preprocessed = _preprocess_image(img)
    return _structured_from_doctr_images([preprocessed])


def extract_text_from_image(path: str) -> str:
    img = Image.open(path)
    preprocessed = _preprocess_image(img)
    return _ocr_with_doctr([preprocessed])


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        with open(path, "rb") as f:
            return extract_text_from_pdf(f.read())
    elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"]:
        return extract_text_from_image(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def extract_structured(path: str) -> StructuredDocument:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        with open(path, "rb") as f:
            return extract_structured_from_pdf(f.read())
    elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"]:
        return extract_structured_from_image(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
