"""Text extraction engine for the Text tab.

Extracts clean text from various file formats without table detection.
Supported: PDF, images, MHTML, HTML, DOCX.
"""

import io
import os
import email
import tempfile
import logging
from typing import Optional
from html.parser import HTMLParser
from zipfile import ZipFile
import xml.etree.ElementTree as ET

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
    arr = numpy_array = __import__("numpy").array(img)
    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    denoised = cv2.fastNlMeansDenoising(binary, h=10)
    return Image.fromarray(denoised)


def _ocr_with_doctr(images: list) -> str:
    import numpy as np
    model = _get_doctr_model()
    pages = [np.array(img.convert("RGB")) for img in images]
    result = model(pages)
    text_parts = []
    for page_idx, page in enumerate(result.pages):
        h, w = pages[page_idx].shape[:2]
        words = []
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (x0, y0), (x1, y1) = word.geometry
                    words.append({
                        'text': word.value,
                        'x0': x0 * w,
                        'y0': y0 * h,
                        'x1': x1 * w,
                        'y1': y1 * h,
                        'cy': (y0 + y1) * h / 2,
                    })

        if not words:
            text_parts.append("")
            continue

        page_text = _render_as_markdown(words, w, h)
        text_parts.append(page_text)
    return "\n\n".join(text_parts)


def _render_as_markdown(words: list, page_w: float, page_h: float) -> str:
    """Render OCR words as markdown with tables for multi-column regions."""
    from typing import List

    # --- Step 1: Group words into visual rows ---
    words.sort(key=lambda wd: wd['cy'])
    avg_h = sum(wd['y1'] - wd['y0'] for wd in words) / len(words)
    row_threshold = avg_h * 0.6

    rows: List[list] = []
    current_row = [words[0]]
    current_cy = words[0]['cy']
    for wd in words[1:]:
        if abs(wd['cy'] - current_cy) < row_threshold:
            current_row.append(wd)
        else:
            rows.append(current_row)
            current_row = [wd]
            current_cy = wd['cy']
    rows.append(current_row)

    for row in rows:
        row.sort(key=lambda wd: wd['x0'])

    # --- Step 2: For each row, find its column splits ---
    # Compute a global "normal word spacing" from all rows — the typical gap
    # between words within a cell. This anchors our threshold.
    all_small_gaps = []
    for row in rows:
        for i in range(1, len(row)):
            g = row[i]['x0'] - row[i - 1]['x1']
            if 0 < g < avg_h * 2:
                all_small_gaps.append(g)
    typical_word_gap = sorted(all_small_gaps)[len(all_small_gaps) // 2] if all_small_gaps else avg_h * 0.5

    def _split_row(row_words: list) -> List[str]:
        """Split a row into cells based on large gaps. Returns list of cell texts."""
        if len(row_words) < 2:
            return [" ".join(wd['text'] for wd in row_words)]

        # Calculate gaps between consecutive words
        gaps = []
        for i in range(1, len(row_words)):
            gap_size = row_words[i]['x0'] - row_words[i - 1]['x1']
            gaps.append(gap_size)

        # A gap is a "column break" if it's much larger than typical word spacing.
        # Use absolute threshold based on global typical word gap, not per-row median.
        threshold = max(typical_word_gap * 4.0, avg_h * 1.0)

        # Build cells by splitting at large gaps
        cells = []
        current_cell_words = [row_words[0]]
        for i in range(1, len(row_words)):
            if gaps[i - 1] > threshold:
                cells.append(" ".join(wd['text'] for wd in current_cell_words))
                current_cell_words = [row_words[i]]
            else:
                current_cell_words.append(row_words[i])
        cells.append(" ".join(wd['text'] for wd in current_cell_words))

        return cells

    # Compute cells for every row
    row_cells = [_split_row(row) for row in rows]

    # --- Step 3: Group consecutive rows with same column count into tables ---
    output_lines = []
    i = 0
    while i < len(row_cells):
        cells = row_cells[i]
        n_cols = len(cells)

        if n_cols <= 1:
            # Single-column row = plain text
            text = cells[0] if cells else ""
            if text.strip():
                output_lines.append(text)
            i += 1
            continue

        # Multi-column row: look ahead for consecutive rows with same column count
        group_start = i
        group = [cells]
        j = i + 1
        while j < len(row_cells) and len(row_cells[j]) == n_cols:
            group.append(row_cells[j])
            j += 1

        # Render as markdown table
        # If group has 2+ rows, first row = header
        if len(group) >= 2:
            header = group[0]
            output_lines.append("")
            output_lines.append("| " + " | ".join(c.strip() for c in header) + " |")
            output_lines.append("| " + " | ".join("---" for _ in header) + " |")
            for body_row in group[1:]:
                output_lines.append("| " + " | ".join(c.strip() for c in body_row) + " |")
            output_lines.append("")
        else:
            # Single multi-column row — still a table (no header/body distinction)
            output_lines.append("")
            output_lines.append("| " + " | ".join(c.strip() for c in cells) + " |")
            output_lines.append("| " + " | ".join("---" for _ in cells) + " |")
            output_lines.append("")

        i = j

    # Clean up excessive blank lines
    cleaned = []
    prev_blank = False
    for line in output_lines:
        if line.strip() == "":
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    return "\n".join(cleaned).strip()


def _cluster_positions(values: list, tolerance: float) -> list:
    """Cluster numeric values and return the mean of each cluster."""
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] < tolerance:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


# ---------------------------------------------------------------------------
# HTML text extractor (shared by MHTML and HTML)
# ---------------------------------------------------------------------------
class _HTMLTextExtractor(HTMLParser):
    """Extract clean text from HTML, preserving structural breaks."""

    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = False
        if tag in ("p", "div", "tr", "br", "h1", "h2", "h3", "h4", "h5",
                    "h6", "li", "td", "th", "dt", "dd", "blockquote"):
            self.text.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.text.append(data)


def _clean_lines(text: str) -> str:
    """Clean extracted text: strip lines, remove blanks, join."""
    lines = [l.strip() for l in text.split("\n")]
    lines = [l for l in lines if l]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Format-specific extractors
# ---------------------------------------------------------------------------
def _extract_from_pdf(content: bytes) -> str:
    """Extract text from PDF. pdfplumber (layout-aware) first, doctr OCR fallback."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for p in pdf.pages:
            # Use layout mode to preserve table column alignment
            try:
                txt = p.extract_text(layout=True, x_tolerance=2, y_tolerance=3) or ""
            except TypeError:
                # Fallback if layout param not supported in this version
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


def _extract_from_image(path: str) -> str:
    """Extract text from an image using doctr OCR."""
    img = Image.open(path)
    preprocessed = _preprocess_image(img)
    return _ocr_with_doctr([preprocessed])


def _extract_from_html(html_content: str) -> str:
    """Extract clean text from HTML string."""
    extractor = _HTMLTextExtractor()
    extractor.feed(html_content)
    return _clean_lines("".join(extractor.text))


def _extract_from_mhtml(content: bytes) -> str:
    """Extract clean text from MHTML (web archive) file."""
    text_content = content.decode("utf-8", errors="ignore")
    msg = email.message_from_string(text_content)

    html_part = None
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            payload = part.get_payload(decode=True)
            if payload:
                html_part = payload.decode("utf-8", errors="ignore")
                break

    if not html_part:
        # Fallback: try treating the whole thing as HTML
        return _extract_from_html(text_content)

    return _extract_from_html(html_part)


def _extract_from_docx(content: bytes) -> str:
    """Extract text from DOCX file."""
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []

    with ZipFile(io.BytesIO(content)) as z:
        if "word/document.xml" not in z.namelist():
            raise ValueError("Invalid DOCX: no word/document.xml found")
        with z.open("word/document.xml") as f:
            tree = ET.parse(f)

    for p in tree.findall(".//w:p", ns):
        texts = [t.text for t in p.findall(".//w:t", ns) if t.text]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)

    return "\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp",
    ".mhtml", ".mht", ".html", ".htm", ".docx",
}


def extract_text(path: str) -> str:
    """Extract clean text from any supported file type."""
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".pdf":
        with open(path, "rb") as f:
            return _extract_from_pdf(f.read())

    elif ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"):
        return _extract_from_image(path)

    elif ext in (".mhtml", ".mht"):
        with open(path, "rb") as f:
            return _extract_from_mhtml(f.read())

    elif ext in (".html", ".htm"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return _extract_from_html(f.read())

    elif ext == ".docx":
        with open(path, "rb") as f:
            return _extract_from_docx(f.read())

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def extract_text_from_bytes(content: bytes, filename: str) -> str:
    """Extract text from bytes content given the original filename."""
    ext = os.path.splitext(filename)[-1].lower()

    if ext == ".pdf":
        return _extract_from_pdf(content)
    elif ext in (".mhtml", ".mht"):
        return _extract_from_mhtml(content)
    elif ext in (".html", ".htm"):
        return _extract_from_html(content.decode("utf-8", errors="ignore"))
    elif ext == ".docx":
        return _extract_from_docx(content)
    elif ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"):
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            return _extract_from_image(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def get_file_type(filename: str) -> Optional[str]:
    """Return the file extension (lowercase, with dot) or None."""
    ext = os.path.splitext(filename)[-1].lower()
    return ext if ext in SUPPORTED_EXTENSIONS else None
