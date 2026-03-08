import io
import os
import tempfile
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
    bbox: tuple = None  # (x0, y0, x1, y1) for excluding from text

    def to_markdown(self) -> str:
        if not self.headers and not self.rows:
            return ""
        cols = (
            self.headers
            if self.headers
            else [f"Col{i+1}" for i in range(len(self.rows[0]))]
        )

        def _clean(c):
            return str(c or "").replace("|", "\\|").replace("\n", " ").strip()

        lines = []
        lines.append("| " + " | ".join(_clean(c) for c in cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for row in self.rows:
            padded = row + [""] * (len(cols) - len(row))
            lines.append("| " + " | ".join(_clean(c) for c in padded[: len(cols)]) + " |")
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
# img2table extraction
# ---------------------------------------------------------------------------
def _tables_from_img2table_pdf(content: bytes) -> Dict[int, List[ExtractedTable]]:
    """Use img2table to detect and extract tables from a PDF."""
    from img2table.document import PDF
    from img2table.ocr import DocTR

    tables_by_page: Dict[int, List[ExtractedTable]] = {}

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        ocr = DocTR()
        doc = PDF(tmp_path)
        extracted = doc.extract_tables(
            ocr=ocr,
            implicit_rows=True,
            implicit_columns=True,
            borderless_tables=True,
            min_confidence=50,
        )

        for page_idx, page_tables in extracted.items():
            page_num = page_idx + 1
            tables_by_page[page_num] = []
            for table in page_tables:
                df = table.df
                if df is None or df.empty:
                    continue
                headers = [str(c).strip() for c in df.columns.tolist()]
                rows = []
                for _, row in df.iterrows():
                    rows.append([str(v).strip() for v in row.tolist()])
                if rows:
                    bbox = None
                    if hasattr(table, "bbox"):
                        bbox = table.bbox
                    tables_by_page[page_num].append(
                        ExtractedTable(headers=headers, rows=rows, page=page_num, bbox=bbox)
                    )
    finally:
        os.unlink(tmp_path)

    return tables_by_page


def _tables_from_img2table_image(path: str) -> List[ExtractedTable]:
    """Use img2table to detect and extract tables from an image."""
    from img2table.document import Image as Img2TableImage
    from img2table.ocr import DocTR

    ocr = DocTR()
    doc = Img2TableImage(path)
    extracted = doc.extract_tables(
        ocr=ocr,
        implicit_rows=True,
        implicit_columns=True,
        borderless_tables=True,
        min_confidence=50,
    )

    tables = []
    for table in extracted:
        df = table.df
        if df is None or df.empty:
            continue
        headers = [str(c).strip() for c in df.columns.tolist()]
        rows = []
        for _, row in df.iterrows():
            rows.append([str(v).strip() for v in row.tolist()])
        if rows:
            tables.append(ExtractedTable(headers=headers, rows=rows, page=1))

    return tables


# ---------------------------------------------------------------------------
# Text extraction excluding table regions
# ---------------------------------------------------------------------------
def _bbox_to_tuple(b) -> Optional[tuple]:
    """Normalize bbox-ish objects into (x0, y0, x1, y1) tuple.

    img2table may return a custom BBox object depending on version.
    """
    if b is None:
        return None

    # Already a 4-tuple/list
    if isinstance(b, (list, tuple)) and len(b) == 4:
        try:
            return tuple(float(x) for x in b)
        except Exception:
            return None

    # Dict-like
    if isinstance(b, dict):
        for keys in (
            ("x0", "y0", "x1", "y1"),
            ("x1", "y1", "x2", "y2"),
            ("left", "top", "right", "bottom"),
        ):
            if all(k in b for k in keys):
                try:
                    return tuple(float(b[k]) for k in keys)
                except Exception:
                    return None

    # Object with common coordinate attrs
    for attrs in (
        ("x0", "y0", "x1", "y1"),
        ("x1", "y1", "x2", "y2"),
        ("left", "top", "right", "bottom"),
    ):
        if all(hasattr(b, a) for a in attrs):
            try:
                return tuple(float(getattr(b, a)) for a in attrs)
            except Exception:
                return None

    return None


def _extract_text_without_tables(page, table_bboxes: List[tuple]) -> str:
    """Extract text from a pdfplumber page excluding table bounding boxes."""
    if not table_bboxes:
        return page.extract_text(x_tolerance=2, y_tolerance=3) or ""

    words = page.extract_words(x_tolerance=2, y_tolerance=3) or []
    filtered = []
    for w in words:
        wx0, wtop, wx1, wbottom = (
            float(w["x0"]),
            float(w["top"]),
            float(w["x1"]),
            float(w["bottom"]),
        )
        in_table = False
        for bbox in table_bboxes:
            if bbox is None:
                continue
            tx0, ty0, tx1, ty1 = bbox
            if wx0 >= tx0 - 5 and wtop >= ty0 - 5 and wx1 <= tx1 + 5 and wbottom <= ty1 + 5:
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


# ---------------------------------------------------------------------------
# Structured extraction — combines pdfplumber text + img2table tables
# ---------------------------------------------------------------------------
def _structured_from_pdf(content: bytes) -> StructuredDocument:
    """Extract structured content from PDF: pdfplumber for text, img2table for tables."""
    doc = StructuredDocument()

    # Get tables via img2table
    try:
        tables_by_page = _tables_from_img2table_pdf(content)
    except Exception as e:
        logger.warning(f"img2table failed: {e}, proceeding without table detection")
        tables_by_page = {}

    # Get text via pdfplumber, excluding table regions
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            sp = StructuredPage(page_num=page_num)

            # Get table bboxes for this page to exclude from text
            page_tables = tables_by_page.get(page_num, [])
            table_bboxes = [_bbox_to_tuple(t.bbox) for t in page_tables if t.bbox]
            table_bboxes = [b for b in table_bboxes if b is not None]

            sp.text = _extract_text_without_tables(page, table_bboxes)
            sp.tables = page_tables
            doc.pages.append(sp)

    # If pdfplumber got no text at all, fallback to doctr for text portions
    total_text = sum(len(p.text) for p in doc.pages)
    if total_text < 50 and not doc.all_tables:
        logger.info("Sparse content, falling back to doctr OCR for text")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            images = convert_from_path(tmp_path, dpi=300)
            preprocessed = [_preprocess_image(img) for img in images]
            ocr_text = _ocr_with_doctr(preprocessed)
            # Distribute OCR text across pages
            ocr_pages = ocr_text.split("\n\n")
            for j, sp in enumerate(doc.pages):
                if j < len(ocr_pages):
                    sp.text = ocr_pages[j]
        finally:
            os.unlink(tmp_path)

    return doc


def _structured_from_image(path: str) -> StructuredDocument:
    """Extract structured content from an image: doctr for text, img2table for tables."""
    doc = StructuredDocument()
    sp = StructuredPage(page_num=1)

    # Get tables
    try:
        sp.tables = _tables_from_img2table_image(path)
    except Exception as e:
        logger.warning(f"img2table failed on image: {e}")
        sp.tables = []

    # Get text via doctr
    img = Image.open(path)
    preprocessed = _preprocess_image(img)
    sp.text = _ocr_with_doctr([preprocessed])

    doc.pages.append(sp)
    return doc


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
    return _structured_from_pdf(content)


def extract_structured_from_image(path: str) -> StructuredDocument:
    return _structured_from_image(path)


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