"""MHTML → clean structured markdown.

Extracts content from MHTML web archives, stripping navigation, banners,
scripts, styles, and other non-content elements. Preserves headings, lists,
tables, and text structure as clean markdown suitable for knowledge bases
and LLM prompt context.

Handles CMS/ASP.NET pages where content lives inside layout tables by
detecting and flattening them into prose while preserving real data tables.

Pipeline: MHTML → MIME unwrap → HTML → content div detection
        → junk stripping → layout table flattening → markdown
"""

import email
import logging
import re
from typing import Dict, List, Optional

from lxml import etree
from lxml.html import fromstring, tostring

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scheme metadata (required by registry)
# ---------------------------------------------------------------------------
LABEL = "MHTML → Markdown"
ACCEPTS = [".mhtml", ".mht"]
MULTI_FILE = True
OUTPUT_OPTIONS = ["consolidated", "individual"]
RAW_INPUT = True  # receive raw bytes, not pre-extracted text

# ---------------------------------------------------------------------------
# MHTML → HTML extraction
# ---------------------------------------------------------------------------

def _unwrap_mhtml(raw: bytes) -> str:
    """Extract the main HTML part from an MHTML file."""
    text = raw.decode("utf-8", errors="ignore")
    msg = email.message_from_string(text)

    for part in msg.walk():
        if part.get_content_type() == "text/html":
            payload = part.get_payload(decode=True)
            if payload:
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="ignore")

    # Fallback: treat entire content as HTML
    return text


# ---------------------------------------------------------------------------
# HTML → clean markdown
# ---------------------------------------------------------------------------

# Tags whose entire subtree should be removed
_STRIP_TAGS = {
    "script", "style", "noscript", "iframe", "svg", "canvas",
    "nav", "header", "footer", "aside", "button", "input",
    "select", "textarea", "label", "fieldset", "legend",
    "menu", "menuitem",
}

# Class/id substrings that indicate non-content blocks
_JUNK_PATTERNS = re.compile(
    r"(sidebar|banner|breadcrumb|cookie|"
    r"advertisement|advert|social|share|related|comment|"
    r"popup|modal|overlay|toast|toolbar|ribbon|masthead|"
    r"skip-to|jump-to|back-to-top)",
    re.IGNORECASE,
)

# Class/id patterns that indicate navigation (separate so we can be precise)
_NAV_PATTERNS = re.compile(
    r"(^nav$|^nav\b|navbar|mainNav|nav-group|navmenu|"
    r"^menu$|^menu\b|mainMenu|footerContainer|footerNav)",
    re.IGNORECASE,
)

# Class/id patterns that likely contain the main content
_CONTENT_HINTS = re.compile(
    r"(content|article|main-body|mainWrap|theme-content|"
    r"post-body|entry-content|page-content|story-body|"
    r"col-md-9|col-md-8|col-lg-9|col-lg-8)",
    re.IGNORECASE,
)


def _find_content_root(doc):
    """Find the most likely content-bearing element.

    Strategy: look for elements with content-hint classes first,
    then fall back to the element with the most text that isn't
    a navigation or footer block.
    """
    # Try semantic elements first
    for tag in ("main", "article"):
        for el in doc.iter(tag):
            if len(el.text_content().strip()) > 200:
                return el

    # Try content-hint class/id — prefer the SMALLEST (most specific)
    # match to avoid picking up a parent that also includes nav chrome
    candidates = []
    for el in doc.iter("div", "section"):
        attrs = " ".join(filter(None, [
            el.get("class") or "",
            el.get("id") or "",
        ]))
        if attrs and _CONTENT_HINTS.search(attrs):
            text_len = len(el.text_content().strip())
            if text_len > 200:
                candidates.append((text_len, el))

    if candidates:
        # Sort ascending by text length — smallest content div first
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    # Fallback: largest text-bearing div
    best = None
    best_len = 0
    for el in doc.iter("div"):
        text_len = len(el.text_content().strip())
        if text_len > best_len:
            best = el
            best_len = text_len

    return best


def _is_layout_table(table) -> bool:
    """Detect if a table is used for page layout vs actual tabular data.

    Layout tables typically have:
    - Large colspan values (spanning "columns" that are really page regions)
    - Varying column counts per row
    - Rows that are mostly empty (spacer rows)
    - Numbered paragraph text in cells
    """
    rows = list(table.iter("tr"))
    if len(rows) < 2:
        return True

    has_large_colspan = False
    col_counts = []

    for row in rows:
        cells = list(row.iter("td", "th"))
        col_counts.append(len(cells))
        for cell in cells:
            cs = int(cell.get("colspan") or "1")
            if cs >= 4:
                has_large_colspan = True

    if has_large_colspan:
        return True

    # Highly variable column counts = layout
    if col_counts and (max(col_counts) - min(col_counts)) > 2:
        return True

    # Check for spacer rows (rows where all cells are empty/whitespace)
    empty_rows = 0
    for row in rows:
        cells = list(row.iter("td", "th"))
        if all(not (c.text_content().strip()) or
               c.text_content().strip() in ("\xa0", " ")
               for c in cells):
            empty_rows += 1

    # If more than 30% of rows are spacers, it's layout
    if len(rows) > 4 and empty_rows / len(rows) > 0.3:
        return True

    return False


def _flatten_layout_table(table) -> str:
    """Convert a layout table into structured paragraphs.

    Detects document headings from bold/underline formatting in full-width
    cells, and numbered paragraphs (1., 2.1, a), i), etc.).
    """
    parts: List[str] = []
    _NUM_RE = re.compile(r'^(\d+\.|\d+\.\d+|[a-z]\)|[ivx]+\))$')

    for row in table.iter("tr"):
        cells = list(row.iter("td", "th"))
        cell_texts = []
        for cell in cells:
            t = cell.text_content().strip()
            if t and t not in ("\xa0", " "):
                cell_texts.append(t)

        if not cell_texts:
            continue

        # --- Heading detection ---
        # Check if this row is a single full-width cell with bold/underline
        # (common CMS pattern for section headings in layout tables)
        if len(cell_texts) == 1:
            cell = cells[0] if cells else None
            if cell is not None:
                cs = int(cell.get("colspan") or "1")
                t = cell_texts[0]
                has_bold = any(c.tag in ("strong", "b") for c in cell.iter())
                has_underline = any(c.tag == "u" for c in cell.iter())
                is_short = len(t) < 80

                if is_short and has_bold and cs >= 4:
                    if has_underline:
                        # Bold + underline = top-level heading
                        parts.append(f"## {t}")
                    else:
                        # Bold only = sub-heading
                        parts.append(f"### {t}")
                    continue

            # Not a heading, just a standalone paragraph
            parts.append(cell_texts[0])
            continue

        first = cell_texts[0]

        if len(cell_texts) >= 2 and _NUM_RE.match(first):
            # Check if this is a numbered heading (e.g. "1." + "Scope")
            # Heading pattern: number cell + bold short text in large colspan
            second_cell = cells[1] if len(cells) > 1 else None
            body = " ".join(cell_texts[1:])
            is_heading = False

            if second_cell is not None and len(body) < 80:
                sc_bold = any(c.tag in ("strong", "b") for c in second_cell.iter())
                sc_cs = int(second_cell.get("colspan") or "1")
                if sc_bold and sc_cs >= 4:
                    is_heading = True

            if is_heading:
                parts.append(f"### {first} {body}")
            else:
                parts.append(f"{first} {body}")
        else:
            parts.append(" ".join(cell_texts))

    return "\n\n".join(parts)


def _data_table_to_markdown(table) -> str:
    """Convert a real data table to a markdown table."""
    rows = list(table.iter("tr"))
    if not rows:
        return ""

    md_rows: List[str] = []
    for i, row in enumerate(rows):
        cells = list(row.iter("td", "th"))
        cell_texts = [c.text_content().strip() for c in cells]
        md_rows.append("| " + " | ".join(cell_texts) + " |")

        # Add separator after first row (header)
        if i == 0:
            md_rows.append("| " + " | ".join("---" for _ in cells) + " |")

    return "\n".join(md_rows)


def _strip_junk(doc) -> None:
    """Remove non-content elements from the document tree in-place."""

    # Remove by tag name
    for tag in _STRIP_TAGS:
        for el in list(doc.iter(tag)):
            if el.getparent() is not None:
                el.getparent().remove(el)

    # Remove navigation blocks by class/id
    for el in list(doc.iter()):
        attrs = " ".join(filter(None, [
            el.get("class") or "",
            el.get("id") or "",
            el.get("role") or "",
        ]))
        if not attrs:
            continue
        if _JUNK_PATTERNS.search(attrs) or _NAV_PATTERNS.search(attrs):
            if el.getparent() is not None:
                el.getparent().remove(el)

    # Remove hidden elements
    for el in list(doc.iter()):
        style = el.get("style") or ""
        if "display:none" in style.replace(" ", "") or \
           "visibility:hidden" in style.replace(" ", ""):
            if el.getparent() is not None:
                el.getparent().remove(el)

    # Remove media elements
    for tag in ("img", "figure", "figcaption", "picture",
                "video", "audio", "source", "object", "embed"):
        for el in list(doc.iter(tag)):
            if el.getparent() is not None:
                el.getparent().remove(el)


def _html_to_markdown(html: str) -> str:
    """Convert HTML to structured markdown, stripping non-content."""
    try:
        doc = fromstring(html)
    except Exception:
        from lxml.html import document_fromstring
        doc = document_fromstring(html)

    # --- Phase 1: Strip junk from full document ---
    _strip_junk(doc)

    # --- Phase 2: Find the content root ---
    content = _find_content_root(doc)
    if content is None:
        content = doc

    # --- Phase 3: Process tables (flatten layout, preserve data) ---
    table_replacements: List[tuple] = []

    for table in list(content.iter("table")):
        if _is_layout_table(table):
            flat_text = _flatten_layout_table(table)
            table_replacements.append((table, flat_text, "layout"))
        else:
            md_table = _data_table_to_markdown(table)
            table_replacements.append((table, md_table, "data"))

    for table, replacement, kind in table_replacements:
        parent = table.getparent()
        if parent is None:
            continue
        new_div = etree.Element("div")
        for para in replacement.split("\n\n"):
            p = etree.SubElement(new_div, "p")
            p.text = para
        parent.replace(table, new_div)

    # --- Phase 4: Convert to markdown ---
    content_html = tostring(content, encoding="unicode", method="html")

    try:
        from markdownify import markdownify as md
        markdown = md(
            content_html,
            heading_style="ATX",
            bullets="-",
            strip=["a", "img", "figure", "figcaption",
                   "picture", "video", "audio", "source",
                   "object", "embed", "iframe"],
        ) or ""
    except ImportError:
        raise RuntimeError("markdownify is required — pip install markdownify")

    if not markdown:
        return ""

    return _clean_markdown(markdown)


def _clean_markdown(text: str) -> str:
    """Normalize whitespace and remove artifacts from converted markdown."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse runs of 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip pure decorative separators (but keep markdown HRs)
        if re.match(r"^[-_=~*]{4,}$", stripped) and stripped != "---":
            continue
        # Skip lines that are just non-breaking spaces or zero-width chars
        if stripped and all(c in "\u00a0\u200b\u200c\u200d\ufeff" for c in stripped):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)

    # Normalize multiple spaces within lines (preserve code indentation)
    out_lines = []
    for line in text.split("\n"):
        if line.startswith("    ") or line.startswith("\t"):
            out_lines.append(line)
        else:
            out_lines.append(re.sub(r"  +", " ", line).rstrip())

    text = "\n".join(out_lines).strip()
    return text


# ---------------------------------------------------------------------------
# Scheme entry point
# ---------------------------------------------------------------------------

def transform(files: Dict[str, bytes], output_mode: str) -> dict:
    """Transform MHTML file(s) into clean markdown.

    Args:
        files: {filename: raw_bytes} (RAW_INPUT=True)
        output_mode: "consolidated" or "individual"

    Returns:
        dict with "text" and/or "files" keys.
    """
    results = {}
    for fname, raw in sorted(files.items()):
        html = _unwrap_mhtml(raw)
        md = _html_to_markdown(html)
        results[fname] = md

    if output_mode == "individual":
        return {
            "merged": False,
            "files": [
                {"filename": fn, "text": txt}
                for fn, txt in results.items()
            ],
        }

    # Consolidated
    if len(results) == 1:
        fname, text = next(iter(results.items()))
        return {
            "text": text,
            "filename": fname.rsplit(".", 1)[0] + ".md",
        }

    merged_parts = []
    for fn, txt in results.items():
        merged_parts.append(f"# {fn}\n\n{txt}")

    return {
        "merged": True,
        "text": "\n\n---\n\n".join(merged_parts),
        "filename": "merged_content.md",
        "file_count": len(results),
    }
