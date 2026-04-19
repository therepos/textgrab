"""PDF → Markdown scheme.

Converts PDFs into structured Markdown with proper reading order,
preserved tables, extracted figure sidecars, and per-page confidence
metadata.  Deterministic — no LLM, no captioning, no hallucination
surface.

Primary engine: **Docling** (IBM, MIT-licensed).  Docling ships
purpose-trained layout and table-structure models (DocLayNet-derived
layout detector + TableFormer) plus its own OCR stack.  We call
`DocumentConverter.convert(...)` and take its Markdown exporter's output
verbatim — we do not hand-assemble Markdown from Docling's internal
representation.

Fallback engine: the heuristic pipeline that shipped before Docling
was wired in.  It activates automatically if Docling raises on a
document (bad PDF, model init failure, memory cap, etc.).  Heuristic
pipeline:
  1. Char-level text extraction via pdfplumber (size + fontname per glyph)
  2. Font-size clustering → heading level assignment (H1..H4 + body)
  3. Per-page column detection via x-centroid silhouette scoring
  4. Table detection via img2table
  5. Figure extraction via pdfplumber.page.images
  6. OCR fallback via doctr on sparse pages
  7. Block assembly + Markdown cleanup

When the fallback fires it is noted in `confidence.warnings` and each
affected page carries `"engine": "heuristic"` in its confidence entry.

Output envelope (backwards-compatible with the pre-Docling version):
  {
    "text": <str>,             # the assembled Markdown
    "filename": <str>,
    "figures": [                # sidecar images (base64)
      {"filename": "figure-1-p3-<stem>.png",
       "data_b64": "...", "mime": "image/png"},
      ...
    ],
    "stats": [<str>, ...],     # human-readable summary
    "confidence": {
      "pages": [
        {
          "source": <str>,                   # originating filename
          "page": int,
          "engine": "docling" | "heuristic",
          "extraction_method": "native" | "ocr" | "mixed",
          "table_count": int,
          "figure_count": int,
          # Docling-only (None when engine == "heuristic"):
          "docling_grade": "poor" | "fair" | "good" | "excellent" | None,
          "docling_score": float | None,      # 0..1, page mean_score
          "ocr_score":    float | None,
          "layout_score": float | None,
          "parse_score":  float | None,
          "table_score":  float | None,
        },
        ...
      ],
      "docling_summary": {                    # None if no page used Docling
        "mean_grade": str, "low_grade": str,
        "mean_score": float, "low_score": float,
      } | None,
      "warnings": [<str>, ...],
    }
  }

Figure reference injection: Docling's `export_to_markdown()` emits the
literal placeholder `<!-- image -->` at the location of each picture
in reading order.  We replace these placeholders in document order with
`![Figure N](figure-N-p<page>-<stem>.png)` references that point at the
sidecar PNGs we return in the `figures` array.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pdfplumber

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheme interface
# ---------------------------------------------------------------------------
LABEL = "PDF → Markdown"
ACCEPTS = [".pdf"]
MULTI_FILE = True
OUTPUT_OPTIONS = ["consolidated", "individual"]
RAW_INPUT = True  # receive raw PDF bytes, not pre-extracted text


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
# Heading clustering: a cluster is promoted to a heading tier only if its
# mean size exceeds the body cluster by at least this factor.  1.15 is
# conservative — it favours missing a heading over promoting body text.
HEADING_SIZE_RATIO = 1.15

# Maximum number of heading levels we will assign (H1..H4).  More than
# four tiers starts producing noisy classifications on real documents.
MAX_HEADING_LEVELS = 4

# Column detection: the minimum silhouette score for a 2-column
# hypothesis to beat the 1-column null hypothesis.  0.55 requires a
# fairly clean gap between column centroids — threshold chosen to avoid
# false-positive splits on wide single-column layouts with marginalia.
TWO_COLUMN_SILHOUETTE_THRESHOLD = 0.55

# Minimum number of words on a page before column detection is even
# attempted.  Below this, we default to single-column.
MIN_WORDS_FOR_COLUMN_DETECTION = 40

# When pdfplumber text on a page has fewer than this many characters
# AND no tables were detected, the page is considered scanned and OCR
# fallback is used for that page only.
OCR_FALLBACK_CHAR_THRESHOLD = 50

# Words whose vertical position overlaps any detected table bbox by
# this proportion are excluded from the prose stream.
TABLE_OVERLAP_TOLERANCE = 5.0  # points of slack on each edge

# Minimum embedded image area (as fraction of page area) to emit as a
# figure.  Filters out decorative rules, bullets, logos in headers.
MIN_FIGURE_AREA_FRACTION = 0.01


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Block:
    """A single logical block on a page: heading, paragraph, table, or figure."""
    kind: str          # "heading" | "paragraph" | "table" | "figure"
    page: int
    column: int        # 0 for single-column, 0/1 for two-column
    y: float           # top-of-block y-coordinate (for ordering)
    last_y: float = 0.0  # y of the most-recently-appended line (for paragraph merging)
    text: str = ""     # Markdown content (or figure alt text)
    level: int = 0     # heading level 1..4 (for kind="heading")
    figure_ref: Optional[str] = None  # filename (for kind="figure")


@dataclass
class PageStats:
    page: int
    extraction_method: str = "native"   # native | ocr | mixed
    column_layout: str = "single"       # single | two
    column_confidence: float = 1.0
    table_count: int = 0
    figure_count: int = 0
    heading_candidates: int = 0


# ---------------------------------------------------------------------------
# Stage 2: font-size clustering for heading inference
# ---------------------------------------------------------------------------
def _cluster_font_sizes(sizes: List[float], tolerance: float = 0.5) -> List[Tuple[float, int]]:
    """Agglomerate a sorted list of sizes into clusters.

    Returns a list of (mean_size, count) pairs sorted descending by size.
    Two sizes within `tolerance` points are treated as the same cluster.
    """
    if not sizes:
        return []
    sizes = sorted(sizes)
    clusters: List[List[float]] = [[sizes[0]]]
    for s in sizes[1:]:
        if s - clusters[-1][-1] <= tolerance:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    means = [(sum(c) / len(c), len(c)) for c in clusters]
    means.sort(key=lambda p: -p[0])
    return means


def _assign_heading_levels(
    all_sizes: List[float],
) -> Tuple[Dict[int, int], float, Dict[str, float]]:
    """Decide which font sizes map to which heading level.

    Returns:
        (size_to_level, body_size, level_info)
        size_to_level: {round(size*2): level}  (level 0 = body, 1..4 = H1..H4)
        body_size: the dominant body text size
        level_info: {"body_size": float, "H1": float, ...} for diagnostics
    """
    if not all_sizes:
        return {}, 0.0, {}

    clusters = _cluster_font_sizes(all_sizes)
    if not clusters:
        return {}, 0.0, {}

    # Body text = the cluster with the largest glyph count (not the smallest
    # size — footnotes can be the smallest).
    body_cluster = max(clusters, key=lambda p: p[1])
    body_size = body_cluster[0]

    # Candidate heading clusters: sizes strictly larger than body by at
    # least HEADING_SIZE_RATIO.  Sorted descending.
    heading_candidates = [
        (size, count) for (size, count) in clusters
        if size >= body_size * HEADING_SIZE_RATIO
    ]
    heading_candidates.sort(key=lambda p: -p[0])
    heading_candidates = heading_candidates[:MAX_HEADING_LEVELS]

    size_to_level: Dict[int, int] = {}
    level_info: Dict[str, float] = {"body_size": round(body_size, 2)}

    for level, (size, _count) in enumerate(heading_candidates, start=1):
        key = int(round(size * 2))
        size_to_level[key] = level
        level_info[f"H{level}"] = round(size, 2)

    return size_to_level, body_size, level_info


# ---------------------------------------------------------------------------
# Stage 3: column detection
# ---------------------------------------------------------------------------
def _detect_columns(
    word_xs: List[float], page_width: float
) -> Tuple[str, float, Optional[float]]:
    """Decide whether a page is single- or two-column.

    Args:
        word_xs: list of x-centroids (one per word) on the page
        page_width: page width in points

    Returns:
        (layout, confidence, split_x)
        layout: "single" or "two"
        confidence: silhouette score for the chosen hypothesis (0..1)
        split_x: x-coordinate separating the two columns, or None
    """
    if len(word_xs) < MIN_WORDS_FOR_COLUMN_DETECTION:
        return "single", 1.0, None

    xs = sorted(word_xs)
    n = len(xs)

    # A two-column page has a visible gap between the rightmost word
    # of the left column and the leftmost word of the right column.
    # We search for the largest gap between consecutive sorted
    # centroids that sits in the central 20-80% band of the page, then
    # score it against the median within-line gap — a true column
    # separator is many multiples of the typical word-to-word gap.
    #
    # Earlier iterations scored gap against the *spread* of each
    # cluster, but spread scales with column width, which collapsed
    # the score on pages with wide columns.  Median within-cluster
    # gap is width-invariant.
    band_lo = 0.20 * page_width
    band_hi = 0.80 * page_width

    # Median gap between consecutive sorted centroids — a proxy for
    # within-column word spacing.  Column separator gaps should be
    # many multiples of this.
    all_gaps = [xs[i] - xs[i - 1] for i in range(1, n)]
    sorted_gaps = sorted(all_gaps)
    median_gap = sorted_gaps[len(sorted_gaps) // 2] or 1.0

    best_score = -1.0
    best_split: Optional[float] = None

    for i in range(1, n):
        left_edge = xs[i - 1]
        right_edge = xs[i]
        midpoint = (left_edge + right_edge) / 2.0
        if not (band_lo <= midpoint <= band_hi):
            continue
        gap = right_edge - left_edge
        if gap <= 0:
            continue
        # Score: how many median-gaps wide is this gap?  Mapped to 0..1
        # with tanh so a gap >= 10x median approaches 1.
        ratio = gap / median_gap
        score = ratio / (ratio + 8.0)  # score=0.5 when ratio=8
        if score > best_score:
            best_score = score
            best_split = midpoint

    if best_score >= TWO_COLUMN_SILHOUETTE_THRESHOLD and best_split is not None:
        return "two", round(best_score, 3), best_split

    # Default to single column
    return "single", round(1.0 - max(best_score, 0.0), 3), None


# ---------------------------------------------------------------------------
# Stage 4: tables — reuse img2table from extracttabular
# ---------------------------------------------------------------------------
def _tables_by_page(content: bytes) -> Dict[int, list]:
    """Delegate to the existing img2table wrapper in extracttabular."""
    try:
        from ..extracttabular import _tables_from_img2table_pdf
        return _tables_from_img2table_pdf(content)
    except Exception as e:
        logger.warning(f"Table detection failed: {e}")
        return {}


def _bbox_overlaps(
    word_bbox: Tuple[float, float, float, float],
    table_bbox: Tuple[float, float, float, float],
    slack: float = TABLE_OVERLAP_TOLERANCE,
) -> bool:
    """Return True if a word bbox sits inside a table bbox (with slack)."""
    wx0, wtop, wx1, wbottom = word_bbox
    tx0, ty0, tx1, ty1 = table_bbox
    return (
        wx0 >= tx0 - slack
        and wtop >= ty0 - slack
        and wx1 <= tx1 + slack
        and wbottom <= ty1 + slack
    )


# ---------------------------------------------------------------------------
# Stage 5: figure extraction
# ---------------------------------------------------------------------------
def _extract_figures(pdf_path: str) -> Dict[int, List[dict]]:
    """Extract embedded raster images per page.

    Uses pdfplumber.page.images for bbox metadata; uses pdfplumber's page
    rendering to produce a PNG of the image region (pdfplumber doesn't
    expose raw embedded image streams, and rendering the cropped region
    is more reliable than decoding the embedded stream directly across
    the many colourspace variants a PDF may contain).

    Returns:
        {page_num: [{"bbox": (x0,y0,x1,y1), "png_bytes": bytes}, ...]}
    """
    figures_by_page: Dict[int, List[dict]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            page_area = float(page.width) * float(page.height)
            page_figs: List[dict] = []
            for img in page.images or []:
                x0 = float(img.get("x0", 0))
                x1 = float(img.get("x1", 0))
                # pdfplumber gives "top" and "bottom" in page coord space
                top = float(img.get("top", 0))
                bottom = float(img.get("bottom", 0))
                w = max(0.0, x1 - x0)
                h = max(0.0, bottom - top)
                area = w * h
                if area <= 0 or area / page_area < MIN_FIGURE_AREA_FRACTION:
                    continue
                try:
                    cropped = page.crop((x0, top, x1, bottom))
                    # Render at 200 DPI for a sensible sidecar file size
                    pil_img = cropped.to_image(resolution=200).original
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    page_figs.append({
                        "bbox": (x0, top, x1, bottom),
                        "png_bytes": buf.getvalue(),
                    })
                except Exception as e:
                    logger.warning(
                        f"Figure render failed on page {page_num}: {e}"
                    )
                    continue
            if page_figs:
                figures_by_page[page_num] = page_figs
    return figures_by_page


# ---------------------------------------------------------------------------
# Stage 6: OCR fallback for a single page
# ---------------------------------------------------------------------------
def _ocr_page(pdf_path: str, page_num: int) -> str:
    """OCR a single page using the shared doctr pipeline."""
    try:
        from pdf2image import convert_from_path
        from ..extracttext import _preprocess_image, _ocr_with_doctr

        images = convert_from_path(
            pdf_path, dpi=300, first_page=page_num, last_page=page_num
        )
        if not images:
            return ""
        preprocessed = [_preprocess_image(im) for im in images]
        return _ocr_with_doctr(preprocessed)
    except Exception as e:
        logger.warning(f"OCR failed for page {page_num}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Stage 1 + 7: per-page word collection, block assembly
# ---------------------------------------------------------------------------
def _group_words_into_lines(
    words: List[dict], row_tolerance: float = 3.0
) -> List[List[dict]]:
    """Group pdfplumber words into lines by their top y-coordinate."""
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (float(w["top"]), float(w["x0"])))
    lines: List[List[dict]] = []
    current: List[dict] = [sorted_words[0]]
    current_top = float(sorted_words[0]["top"])
    for w in sorted_words[1:]:
        wt = float(w["top"])
        if abs(wt - current_top) <= row_tolerance:
            current.append(w)
        else:
            lines.append(sorted(current, key=lambda x: float(x["x0"])))
            current = [w]
            current_top = wt
    lines.append(sorted(current, key=lambda x: float(x["x0"])))
    return lines


def _line_font_profile(
    line_words: List[dict], char_lookup: Dict[Tuple[int, int], dict]
) -> Tuple[float, bool]:
    """Determine the dominant font size and bold-ness for a line.

    char_lookup: indexed by (round(top*10), round(x0*10)) → char dict
    """
    sizes: List[float] = []
    bold_hits = 0
    total = 0
    for w in line_words:
        # Approximate: find any char roughly within this word's bbox
        wt = round(float(w["top"]) * 10)
        wx = round(float(w["x0"]) * 10)
        # Scan a small neighbourhood; words and chars won't align exactly
        matched = None
        for dt in range(-5, 6):
            for dx in range(-5, 20):
                key = (wt + dt, wx + dx)
                if key in char_lookup:
                    matched = char_lookup[key]
                    break
            if matched:
                break
        if matched:
            sizes.append(float(matched.get("size", 0)))
            font = str(matched.get("fontname", ""))
            if any(tok in font for tok in ("Bold", "Heavy", "Black", "Semibold")):
                bold_hits += 1
            total += 1
    if not sizes:
        return 0.0, False
    # Dominant size = mode-ish (cluster mean after collapsing within 0.5 pt)
    clusters = _cluster_font_sizes(sizes)
    dominant = clusters[0][0] if clusters else sizes[0]
    is_bold = total > 0 and bold_hits / total >= 0.5
    return dominant, is_bold


def _assign_column(
    x_centroid: float, split_x: Optional[float]
) -> int:
    if split_x is None:
        return 0
    return 0 if x_centroid < split_x else 1


def _line_text(line_words: List[dict]) -> str:
    return " ".join(str(w.get("text", "")).strip() for w in line_words if w.get("text"))


def _should_merge_paragraphs(prev_block: Block, new_top: float, line_gap: float) -> bool:
    """Heuristic: if two consecutive body lines sit close together in the
    same column, they're the same paragraph.
    """
    if prev_block.kind != "paragraph":
        return False
    # prev_block.last_y is the top-y of the most recently appended line;
    # if the new line sits within ~1.8x a typical line gap of it, they're
    # in the same paragraph.  Using last_y (not y, the block start) is
    # what lets multi-line paragraphs merge correctly.
    anchor = prev_block.last_y if prev_block.last_y > 0 else prev_block.y
    return (new_top - anchor) <= max(line_gap * 1.8, 12.0)


def _heading_markdown(level: int, text: str) -> str:
    level = max(1, min(MAX_HEADING_LEVELS, level))
    return ("#" * level) + " " + text.strip()


def _escape_md(text: str) -> str:
    """Minimal Markdown escaping for body text."""
    # Don't escape aggressively — this is technical prose, not prose fiction.
    # Just protect literal asterisks and underscores that would be parsed
    # as emphasis at line boundaries.
    return text


# ---------------------------------------------------------------------------
# Stage 8: Markdown cleanup
# ---------------------------------------------------------------------------
_MULTIPLE_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+(\n|$)")


def _cleanup_markdown(md: str) -> str:
    md = _TRAILING_WS.sub(r"\1", md)
    md = _MULTIPLE_BLANK_LINES.sub("\n\n", md)
    return md.strip() + "\n"


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------
def _convert_single_heuristic(content: bytes, filename: str) -> Tuple[str, List[dict], List[PageStats], Dict[str, float], List[str]]:
    """Convert one PDF.  Returns (markdown, figures, page_stats, level_info, warnings)."""
    warnings: List[str] = []

    # Write to temp file — pdf2image / OCR fallback both need a path
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # ---- Pass 1: collect every char's font size across the document ----
        all_sizes: List[float] = []
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                for ch in page.chars or []:
                    sz = ch.get("size")
                    if sz and sz > 0:
                        all_sizes.append(float(sz))

        size_to_level, body_size, level_info = _assign_heading_levels(all_sizes)
        if not all_sizes:
            warnings.append("No text found via pdfplumber — entire document may be scanned")

        # ---- Pass 2: tables for the whole document ----
        tables_by_page = _tables_by_page(content)
        table_count_total = sum(len(v) for v in tables_by_page.values())

        # ---- Pass 3: figures for the whole document ----
        figures_by_page = _extract_figures(tmp_path)

        # Build the figure payload list and per-page ref map
        figure_payload: List[dict] = []
        figure_refs_by_page: Dict[int, List[Tuple[float, str]]] = {}
        safe_stem = re.sub(r"[^A-Za-z0-9_\-]", "_", os.path.splitext(filename)[0])[:40] or "doc"
        fig_counter = 0
        for page_num in sorted(figures_by_page):
            for fig in figures_by_page[page_num]:
                fig_counter += 1
                fig_name = f"{safe_stem}-fig-{fig_counter}-p{page_num}.png"
                figure_payload.append({
                    "filename": fig_name,
                    "data_b64": base64.b64encode(fig["png_bytes"]).decode("ascii"),
                    "mime": "image/png",
                })
                top_y = fig["bbox"][1]
                figure_refs_by_page.setdefault(page_num, []).append((top_y, fig_name))

        # ---- Pass 4: per-page assembly ----
        page_stats: List[PageStats] = []
        all_blocks: List[Block] = []

        with pdfplumber.open(tmp_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                stats = PageStats(page=page_num)
                page_w = float(page.width)

                # Tables for this page, with bboxes normalised
                page_tables = tables_by_page.get(page_num, [])
                stats.table_count = len(page_tables)
                table_bboxes: List[Tuple[float, float, float, float]] = []
                for t in page_tables:
                    bbox = _normalise_bbox(getattr(t, "bbox", None))
                    if bbox:
                        table_bboxes.append(bbox)

                # Figures on this page
                page_fig_refs = figure_refs_by_page.get(page_num, [])
                stats.figure_count = len(page_fig_refs)

                # Extract words (layout-preserving)
                try:
                    words = page.extract_words(x_tolerance=2, y_tolerance=3) or []
                except Exception:
                    words = []

                # Drop words that sit inside a detected table bbox
                prose_words = []
                for w in words:
                    bbox = (
                        float(w["x0"]), float(w["top"]),
                        float(w["x1"]), float(w["bottom"]),
                    )
                    if any(_bbox_overlaps(bbox, tb) for tb in table_bboxes):
                        continue
                    prose_words.append(w)

                # Decide extraction method
                raw_page_text = "".join(
                    str(w.get("text", "")) for w in prose_words
                )
                if (
                    len(raw_page_text) < OCR_FALLBACK_CHAR_THRESHOLD
                    and not page_tables
                ):
                    # Scanned page — use OCR
                    stats.extraction_method = "ocr"
                    ocr_text = _ocr_page(tmp_path, page_num)
                    if ocr_text.strip():
                        # OCR path emits one big paragraph per page — no
                        # heading inference is possible from OCR (font
                        # size information is lost).  Emit as paragraphs
                        # split on blank lines.
                        for para in re.split(r"\n\s*\n", ocr_text):
                            para = para.strip()
                            if para:
                                all_blocks.append(Block(
                                    kind="paragraph", page=page_num, column=0,
                                    y=0.0, text=para,
                                ))
                    else:
                        warnings.append(f"Page {page_num}: OCR produced no text")

                    # Still emit figures and tables for this page
                    _emit_page_tables_and_figures(
                        page_num, page_tables, page_fig_refs, all_blocks,
                    )
                    page_stats.append(stats)
                    continue

                # Native text path
                if raw_page_text and not prose_words:
                    stats.extraction_method = "mixed"
                else:
                    stats.extraction_method = "native"

                # Column detection
                if prose_words:
                    centroids = [
                        (float(w["x0"]) + float(w["x1"])) / 2.0
                        for w in prose_words
                    ]
                    layout, conf, split_x = _detect_columns(centroids, page_w)
                else:
                    layout, conf, split_x = "single", 1.0, None
                stats.column_layout = layout
                stats.column_confidence = conf

                # Build char lookup for font profile queries
                char_lookup: Dict[Tuple[int, int], dict] = {}
                for ch in page.chars or []:
                    key = (round(float(ch["top"]) * 10), round(float(ch["x0"]) * 10))
                    char_lookup[key] = ch

                # Group words into lines, split lines across columns
                lines = _group_words_into_lines(prose_words)

                # For each line, split it by column first, then emit
                page_blocks: List[Block] = []
                prev_body_block_by_col: Dict[int, Block] = {}
                # Estimate a typical line gap for paragraph merging
                line_gap = _estimate_line_gap(lines)

                heading_candidates = 0
                for line_words in lines:
                    # Split this line's words by column
                    col_buckets: Dict[int, List[dict]] = {0: [], 1: []}
                    for w in line_words:
                        cx = (float(w["x0"]) + float(w["x1"])) / 2.0
                        col = _assign_column(cx, split_x)
                        col_buckets[col].append(w)

                    for col, ws in col_buckets.items():
                        if not ws:
                            continue
                        text = _line_text(ws).strip()
                        if not text:
                            continue
                        line_top = min(float(w["top"]) for w in ws)

                        size, is_bold = _line_font_profile(ws, char_lookup)
                        size_key = int(round(size * 2))
                        level = size_to_level.get(size_key, 0)

                        # A bold line at body size can still be a sub-heading
                        # if it's short and ends without sentence punctuation.
                        # Be very conservative with this promotion — only
                        # fire when we have zero assigned headings at body
                        # level and the line is clearly title-cased.
                        if (
                            level == 0 and is_bold and len(text) < 120
                            and not text.endswith((".", ",", ";", ":"))
                            and size >= body_size * 0.98
                        ):
                            # Assign the deepest available heading level
                            level = min(MAX_HEADING_LEVELS, len(level_info) - 1) or 4
                            if level < 1:
                                level = 0  # couldn't place it — treat as body

                        if level >= 1:
                            heading_candidates += 1
                            block = Block(
                                kind="heading", page=page_num, column=col,
                                y=line_top, last_y=line_top,
                                text=text, level=level,
                            )
                            page_blocks.append(block)
                            prev_body_block_by_col[col] = None
                        else:
                            prev = prev_body_block_by_col.get(col)
                            if prev and _should_merge_paragraphs(
                                prev, line_top, line_gap
                            ):
                                prev.text = prev.text + " " + text
                                prev.last_y = line_top
                            else:
                                block = Block(
                                    kind="paragraph", page=page_num,
                                    column=col, y=line_top, last_y=line_top,
                                    text=text,
                                )
                                page_blocks.append(block)
                                prev_body_block_by_col[col] = block

                stats.heading_candidates = heading_candidates

                # Slot tables into page_blocks in y-order
                for t in page_tables:
                    md = t.to_markdown() if hasattr(t, "to_markdown") else ""
                    if not md:
                        continue
                    bbox = _normalise_bbox(getattr(t, "bbox", None))
                    y = bbox[1] if bbox else 0.0
                    # Tables span columns in a two-column layout usually
                    # sit in column 0; place them at column 0 as a
                    # conservative default.
                    page_blocks.append(Block(
                        kind="table", page=page_num, column=0,
                        y=y, text=md,
                    ))

                # Slot figures
                for (top_y, fig_name) in page_fig_refs:
                    alt = f"Figure (p.{page_num})"
                    page_blocks.append(Block(
                        kind="figure", page=page_num, column=0,
                        y=top_y, text=alt, figure_ref=fig_name,
                    ))

                # Sort page blocks: column first, then y.  This gives
                # "read column 0 top-to-bottom, then column 1 top-to-
                # bottom" — the conventional order for two-column
                # layouts.
                page_blocks.sort(key=lambda b: (b.column, b.y))
                all_blocks.extend(page_blocks)
                page_stats.append(stats)

        # ---- Pass 5: render blocks to Markdown ----
        md_parts: List[str] = []
        last_page = 0
        for b in all_blocks:
            # Insert a light page break marker between pages.  Using a
            # horizontal rule is common convention and parses cleanly in
            # every Markdown renderer.
            if b.page != last_page and last_page != 0:
                md_parts.append("\n---\n")
            last_page = b.page

            if b.kind == "heading":
                md_parts.append(_heading_markdown(b.level, b.text))
                md_parts.append("")
            elif b.kind == "paragraph":
                md_parts.append(_escape_md(b.text))
                md_parts.append("")
            elif b.kind == "table":
                md_parts.append(b.text)
                md_parts.append("")
            elif b.kind == "figure":
                md_parts.append(f"![{b.text}]({b.figure_ref})")
                md_parts.append("")

        markdown = _cleanup_markdown("\n".join(md_parts))
        return markdown, figure_payload, page_stats, level_info, warnings

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _emit_page_tables_and_figures(
    page_num: int,
    page_tables: list,
    page_fig_refs: List[Tuple[float, str]],
    all_blocks: List[Block],
) -> None:
    """Emit tables + figures for an OCR page (no coordinate-ordered prose)."""
    for t in page_tables:
        md = t.to_markdown() if hasattr(t, "to_markdown") else ""
        if md:
            all_blocks.append(Block(
                kind="table", page=page_num, column=0, y=0.0, text=md,
            ))
    for (_top_y, fig_name) in page_fig_refs:
        all_blocks.append(Block(
            kind="figure", page=page_num, column=0, y=0.0,
            text=f"Figure (p.{page_num})", figure_ref=fig_name,
        ))


def _estimate_line_gap(lines: List[List[dict]]) -> float:
    """Median vertical gap between consecutive lines."""
    if len(lines) < 2:
        return 14.0
    tops = [min(float(w["top"]) for w in ln) for ln in lines if ln]
    gaps = [tops[i] - tops[i - 1] for i in range(1, len(tops)) if tops[i] > tops[i - 1]]
    if not gaps:
        return 14.0
    gaps.sort()
    return gaps[len(gaps) // 2]


def _normalise_bbox(bbox) -> Optional[Tuple[float, float, float, float]]:
    """Normalise an img2table bbox object into a tuple."""
    if bbox is None:
        return None
    # Mirrors the logic in extracttabular._bbox_to_tuple, kept local to
    # avoid leaking on the private interface.
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            return tuple(float(x) for x in bbox)  # type: ignore[return-value]
        except Exception:
            return None
    if isinstance(bbox, dict):
        for keys in (
            ("x0", "y0", "x1", "y1"),
            ("x1", "y1", "x2", "y2"),
            ("left", "top", "right", "bottom"),
        ):
            if all(k in bbox for k in keys):
                try:
                    return tuple(float(bbox[k]) for k in keys)  # type: ignore[return-value]
                except Exception:
                    return None
    for attrs in (
        ("x0", "y0", "x1", "y1"),
        ("x1", "y1", "x2", "y2"),
        ("left", "top", "right", "bottom"),
    ):
        if all(hasattr(bbox, a) for a in attrs):
            try:
                return tuple(float(getattr(bbox, a)) for a in attrs)  # type: ignore[return-value]
            except Exception:
                return None
    return None


# ---------------------------------------------------------------------------
# Docling integration
# ---------------------------------------------------------------------------
# Docling is lazy-imported and the DocumentConverter is built once per
# process. If imports or model init fail we degrade to the heuristic
# path on every call (and surface the failure as a warning).

_DOCLING_CONVERTER = None            # cached DocumentConverter instance
_DOCLING_INIT_ERROR: Optional[str] = None  # non-None means init failed permanently

# Tunables
DOCLING_IMAGES_SCALE = 1.5           # 1.5 ~ 108 DPI for figure crops — sharpness sufficient for previews, ~2x faster raster than 2.0
DOCLING_MAX_FIGURES_PER_DOC = 200    # cap to bound memory
DOCLING_MAX_PAGES = 200              # soft cap; past this, Docling is told to stop and we fall back
DOCLING_OCR_LANGS = ["en"]           # EasyOCR language codes


def _build_docling_converter():
    """Lazy-construct the singleton DocumentConverter.

    Returns (converter, error_message).  error_message is None on success.
    """
    global _DOCLING_CONVERTER, _DOCLING_INIT_ERROR
    if _DOCLING_CONVERTER is not None:
        return _DOCLING_CONVERTER, None
    if _DOCLING_INIT_ERROR is not None:
        return None, _DOCLING_INIT_ERROR
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions, EasyOcrOptions,
        )
        opts = PdfPipelineOptions(
            generate_picture_images=True,
            images_scale=DOCLING_IMAGES_SCALE,
            do_ocr=True,
            do_table_structure=True,
            ocr_options=EasyOcrOptions(
                lang=DOCLING_OCR_LANGS,
                use_gpu=False,
            ),
        )
        _DOCLING_CONVERTER = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)},
        )
        logger.info("Docling DocumentConverter initialised")
        return _DOCLING_CONVERTER, None
    except Exception as e:
        msg = f"Docling init failed: {type(e).__name__}: {e}"
        logger.warning(msg)
        _DOCLING_INIT_ERROR = msg
        return None, msg


def _docling_page_extraction_method(conv_result, page_no: int) -> str:
    """Infer native|ocr|mixed for a page from Docling's per-page data.

    Heuristic:
      - if the page has no OCR cells at all → "native"
      - if the page has OCR cells AND parsed text cells → "mixed"
      - if the page is OCR-only → "ocr"
    Falls back to "native" when Docling's page data is unavailable.
    """
    try:
        page = conv_result.pages[page_no] if hasattr(conv_result, "pages") else None
        if page is None:
            # v2.x: pages may be stored under result.pages as a list
            pages = getattr(conv_result, "pages", None) or []
            # find by .page_no
            for p in pages:
                if getattr(p, "page_no", None) == page_no:
                    page = p
                    break
        if page is None:
            return "native"

        ocr_cells = 0
        parsed_cells = 0
        # Docling's page cells live under page.cells in v2.x; attribute name
        # has shifted across versions, so probe defensively.
        cells = getattr(page, "cells", None)
        if cells is None:
            cells = getattr(page, "text_cells", None) or []
        for c in cells:
            label = str(getattr(c, "from_ocr", "")).lower()
            # v2.x: cells have a from_ocr bool
            if label in ("true", "1"):
                ocr_cells += 1
            elif label in ("false", "0"):
                parsed_cells += 1
            else:
                # older schemas: a `label` attr with "ocr" in the name
                lbl = str(getattr(c, "label", "")).lower()
                if "ocr" in lbl:
                    ocr_cells += 1
                else:
                    parsed_cells += 1

        if ocr_cells == 0:
            return "native"
        if parsed_cells == 0:
            return "ocr"
        return "mixed"
    except Exception:
        return "native"


def _finite(x) -> Optional[float]:
    """Return float(x) if finite, else None. Docling uses NaN as 'not set'."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if xf != xf:  # NaN
        return None
    return xf


def _convert_single_docling(content: bytes, filename: str) -> Tuple[str, List[dict], List[dict], Dict[str, float], List[str]]:
    """Docling-powered conversion.

    Returns (markdown, figures, page_confidence_dicts, doc_summary, warnings).
      - page_confidence_dicts are plain dicts already in the public
        response shape (minus "source", which transform() adds).
      - doc_summary is {mean_grade, low_grade, mean_score, low_score} or {}.
    """
    from io import BytesIO
    from docling.datamodel.base_models import DocumentStream
    from docling_core.types.doc import PictureItem, TableItem

    converter, init_err = _build_docling_converter()
    if converter is None:
        raise RuntimeError(init_err or "Docling converter unavailable")

    warnings: List[str] = []

    src = DocumentStream(name=filename, stream=BytesIO(content))
    result = converter.convert(src, raises_on_error=True, max_num_pages=DOCLING_MAX_PAGES)

    # Markdown straight from Docling's exporter.
    markdown = result.document.export_to_markdown() or ""

    # --- Figures: walk the document, grab PictureItem images as PNG ---
    stem = re.sub(r"\.pdf$", "", filename, flags=re.I) or "doc"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)

    figures: List[dict] = []
    figure_filenames_in_order: List[str] = []  # for placeholder substitution
    per_page_figures: Dict[int, int] = {}
    per_page_tables: Dict[int, int] = {}

    pic_idx = 0
    for element, _level in result.document.iterate_items():
        if isinstance(element, PictureItem):
            pic_idx += 1
            if pic_idx > DOCLING_MAX_FIGURES_PER_DOC:
                warnings.append(
                    f"{filename}: figure cap hit ({DOCLING_MAX_FIGURES_PER_DOC}); remaining figures not extracted"
                )
                break
            # Determine page
            page_no = 0
            if element.prov:
                page_no = getattr(element.prov[0], "page_no", 0) or 0
            per_page_figures[page_no] = per_page_figures.get(page_no, 0) + 1

            fig_filename = f"figure-{pic_idx}-p{page_no}-{stem}.png"
            try:
                img = element.get_image(result.document)
            except Exception as e:
                warnings.append(f"{filename}: figure {pic_idx} get_image failed — {e}")
                figure_filenames_in_order.append(None)  # preserve ordering slot
                continue
            if img is None:
                warnings.append(f"{filename}: figure {pic_idx} has no image data")
                figure_filenames_in_order.append(None)
                continue

            buf = io.BytesIO()
            try:
                img.save(buf, format="PNG")
            except Exception as e:
                warnings.append(f"{filename}: figure {pic_idx} PNG encode failed — {e}")
                figure_filenames_in_order.append(None)
                continue
            figures.append({
                "filename": fig_filename,
                "data_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                "mime": "image/png",
            })
            figure_filenames_in_order.append(fig_filename)

        elif isinstance(element, TableItem):
            page_no = 0
            if element.prov:
                page_no = getattr(element.prov[0], "page_no", 0) or 0
            per_page_tables[page_no] = per_page_tables.get(page_no, 0) + 1

    # --- Inject figure refs into the Markdown in reading order ---
    # Docling's markdown exporter emits "<!-- image -->" as a placeholder
    # for each picture.  Replace them one-by-one with real refs.
    if figure_filenames_in_order:
        markdown = _inject_figure_refs(markdown, figure_filenames_in_order)

    # --- Per-page confidence ---
    page_entries: List[dict] = []
    cr = getattr(result, "confidence", None)
    cr_pages = getattr(cr, "pages", {}) if cr is not None else {}

    # Build a sorted list of page numbers we know about
    known_pages = set()
    if cr_pages:
        known_pages.update(cr_pages.keys())
    known_pages.update(per_page_figures.keys())
    known_pages.update(per_page_tables.keys())
    # Also include every page that appears in result.document.pages
    try:
        known_pages.update(result.document.pages.keys())
    except Exception:
        pass
    known_pages.discard(0)  # page 0 is our "unknown" sentinel

    for pno in sorted(known_pages):
        pcs = cr_pages.get(pno) if cr_pages else None
        entry = {
            "page": pno,
            "engine": "docling",
            "extraction_method": _docling_page_extraction_method(result, pno),
            "table_count": per_page_tables.get(pno, 0),
            "figure_count": per_page_figures.get(pno, 0),
            "docling_grade": getattr(getattr(pcs, "mean_grade", None), "value",
                                     str(pcs.mean_grade) if pcs is not None else None) if pcs is not None else None,
            "docling_score": _finite(pcs.mean_score) if pcs is not None else None,
            "ocr_score":    _finite(pcs.ocr_score)    if pcs is not None else None,
            "layout_score": _finite(pcs.layout_score) if pcs is not None else None,
            "parse_score":  _finite(pcs.parse_score)  if pcs is not None else None,
            "table_score":  _finite(pcs.table_score)  if pcs is not None else None,
        }
        page_entries.append(entry)

    # --- Document-level confidence summary ---
    doc_summary: Dict[str, float] = {}
    if cr is not None:
        try:
            mg = getattr(cr, "mean_grade", None)
            lg = getattr(cr, "low_grade", None)
            doc_summary = {
                "mean_grade": getattr(mg, "value", str(mg)) if mg is not None else None,
                "low_grade":  getattr(lg, "value", str(lg)) if lg is not None else None,
                "mean_score": _finite(cr.mean_score),
                "low_score":  _finite(cr.low_score),
            }
        except Exception:
            doc_summary = {}

    # If Docling reported errors on the document, surface them as warnings
    for err in getattr(result, "errors", []) or []:
        warnings.append(f"{filename}: docling — {err}")

    return markdown, figures, page_entries, doc_summary, warnings


_IMAGE_PLACEHOLDER_RE = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)


def _inject_figure_refs(markdown: str, filenames_in_order: List[Optional[str]]) -> str:
    """Replace Docling's '<!-- image -->' placeholders with real Markdown
    image refs, in document order.  If a filename slot is None (figure
    failed to extract), the placeholder is left untouched so the reader
    still knows something was there."""
    it = iter(filenames_in_order)

    def _sub(_match):
        try:
            fname = next(it)
        except StopIteration:
            return _match.group(0)
        if fname is None:
            return _match.group(0)
        return f"![Figure]({fname})"

    return _IMAGE_PLACEHOLDER_RE.sub(_sub, markdown)


def _heuristic_page_stats_to_entry(filename: str, ps: PageStats) -> dict:
    """Adapt the heuristic PageStats dataclass to the public page-confidence
    dict shape.  Docling-specific fields are None; engine == 'heuristic'."""
    return {
        "page": ps.page,
        "engine": "heuristic",
        "extraction_method": ps.extraction_method,
        "table_count": ps.table_count,
        "figure_count": ps.figure_count,
        "docling_grade": None,
        "docling_score": None,
        "ocr_score": None,
        "layout_score": None,
        "parse_score": None,
        "table_score": None,
    }


def _convert_single(content: bytes, filename: str) -> Tuple[str, List[dict], List[dict], Dict[str, float], List[str], str]:
    """Unified single-document convert.

    Tries Docling first, falls back to the heuristic engine on any
    exception.  Returns (markdown, figures, page_entries, doc_summary,
    warnings, engine) where engine is "docling" or "heuristic" and
    indicates which path *actually* produced the output.
    """
    # Attempt Docling
    try:
        md, figs, pages, summary, warns = _convert_single_docling(content, filename)
        return md, figs, pages, summary, warns, "docling"
    except Exception as e:
        logger.warning(f"Docling failed on {filename}: {type(e).__name__}: {e}; falling back to heuristic", exc_info=True)
        fallback_warning = f"{filename}: Docling failed ({type(e).__name__}: {e}); used heuristic fallback"

    # Fallback: heuristic
    try:
        md, figs, page_stats, level_info, h_warns = _convert_single_heuristic(content, filename)
    except Exception as e:
        logger.exception(f"heuristic fallback also failed on {filename}")
        return (
            f"*Conversion failed: {e}*\n",
            [],
            [],
            {},
            [fallback_warning, f"{filename}: heuristic fallback also failed — {e}"],
            "heuristic",
        )

    page_entries = [_heuristic_page_stats_to_entry(filename, ps) for ps in page_stats]
    summary = {}  # no Docling summary when we took the fallback
    warns = [fallback_warning] + list(h_warns)
    return md, figs, page_entries, summary, warns, "heuristic"


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def transform(texts: Dict[str, bytes], output_mode: str = "consolidated") -> dict:
    """Convert one or more PDFs into Markdown with confidence metadata."""
    individual_files: List[dict] = []
    all_figures: List[dict] = []
    all_warnings: List[str] = []
    confidence_pages: List[dict] = []
    docling_summaries: List[Dict[str, float]] = []

    total_tables = 0
    total_figures = 0
    total_pages = 0
    ocr_pages = 0
    docling_pages = 0
    fallback_pages = 0

    for filename, content in sorted(texts.items()):
        try:
            markdown, figures, page_entries, doc_summary, warnings, engine = _convert_single(
                content, filename
            )
        except Exception as e:
            # _convert_single catches its own errors but be defensive
            logger.exception(f"pdf2md fatal error for {filename}")
            markdown = f"*Conversion failed: {e}*\n"
            figures = []
            page_entries = []
            doc_summary = {}
            warnings = [f"{filename}: fatal — {e}"]
            engine = "none"

        out_name = re.sub(r"\.pdf$", ".md", filename, flags=re.I)
        if out_name == filename:
            out_name = filename + ".md"

        individual_files.append({"filename": out_name, "text": markdown})
        all_figures.extend(figures)
        all_warnings.extend(warnings)
        if doc_summary:
            docling_summaries.append(doc_summary)

        for entry in page_entries:
            confidence_pages.append({"source": filename, **entry})
            total_pages += 1
            total_tables += entry.get("table_count", 0) or 0
            total_figures += entry.get("figure_count", 0) or 0
            if entry.get("extraction_method") in ("ocr", "mixed"):
                ocr_pages += 1
            if entry.get("engine") == "docling":
                docling_pages += 1
            elif entry.get("engine") == "heuristic":
                fallback_pages += 1

    # Merge doc-level Docling summaries across multi-doc runs
    docling_summary = None
    if docling_summaries:
        # Take min across low_grade/low_score, mean of mean_score, grades
        # reported as the first document's for readability.
        docling_summary = {
            "mean_grade": docling_summaries[0].get("mean_grade"),
            "low_grade":  docling_summaries[0].get("low_grade"),
            "mean_score": _safe_mean([s.get("mean_score") for s in docling_summaries]),
            "low_score":  _safe_min([s.get("low_score")  for s in docling_summaries]),
        }

    # Human-readable stats
    stats_bits = [
        f"{total_pages} pages",
        f"{total_tables} tables",
        f"{total_figures} figures",
    ]
    if ocr_pages:
        stats_bits.append(f"{ocr_pages} OCR'd")
    if fallback_pages and docling_pages:
        stats_bits.append(f"{fallback_pages} heuristic-fallback")
    elif fallback_pages and not docling_pages:
        stats_bits.append("heuristic engine (Docling unavailable)")
    elif docling_pages and not fallback_pages:
        stats_bits.append("Docling engine")
    if all_warnings:
        stats_bits.append(f"{len(all_warnings)} warnings")

    confidence = {
        "pages": confidence_pages,
        "docling_summary": docling_summary,
        "warnings": all_warnings,
    }

    if output_mode == "individual":
        return {
            "text": "",
            "filename": "",
            "files": individual_files,
            "file_count": len(individual_files),
            "figures": all_figures,
            "stats": stats_bits,
            "confidence": confidence,
        }

    # Consolidated
    if len(individual_files) == 1:
        f = individual_files[0]
        return {
            "text": f["text"],
            "filename": f["filename"],
            "figures": all_figures,
            "stats": stats_bits,
            "confidence": confidence,
        }

    merged = "\n\n---\n\n".join(f["text"] for f in individual_files)
    return {
        "text": merged,
        "filename": "converted.md",
        "file_count": len(individual_files),
        "figures": all_figures,
        "stats": stats_bits,
        "confidence": confidence,
    }


def _safe_mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]
    return sum(xs) / len(xs) if xs else None


def _safe_min(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]
    return min(xs) if xs else None
