"""MHTML → Markdown scheme.

Converts MHTML web archives into clean, structured markdown optimised for
LLM consumption (research, review, learning).  Preserves heading hierarchy,
tables, lists, and emphasis.  Includes YAML front-matter with source metadata.

Pipeline stages:
  1. MHTML extraction — pull HTML + envelope/head metadata
  2. Noise removal — strip scripts, nav, footers, hidden elements
  3. Layout table unwrapping — detect and flatten layout-only tables
  4. Semantic inference — recover headings/lists from inline styles
  5. Normalisation — whitespace, encoding, empty element cleanup
  6. Markdown conversion — markdownify + post-processing
"""

import email
import os
import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from lxml import etree
from lxml.html import tostring as html_tostring, fromstring as html_fromstring
from markdownify import markdownify as md, MarkdownConverter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scheme interface
# ---------------------------------------------------------------------------
LABEL = "MHTML → Markdown"
ACCEPTS = [".mhtml", ".mht"]
MULTI_FILE = True
OUTPUT_OPTIONS = ["consolidated", "individual"]
RAW_INPUT = True  # receive raw bytes, not pre-extracted text


# ---------------------------------------------------------------------------
# Stage 1: MHTML extraction
# ---------------------------------------------------------------------------
def _extract_mhtml(content: bytes) -> Tuple[str, Dict[str, str], Dict[str, Tuple[str, bytes]]]:
    """Extract HTML body, metadata, and images from MHTML bytes.

    Returns:
        (html_string, metadata_dict, images_dict)
        images_dict: {content_location_url: (media_type, raw_bytes)}
    """
    text_content = content.decode("utf-8", errors="ignore")
    msg = email.message_from_string(text_content)

    # Collect envelope metadata
    meta = {}
    subject = msg.get("Subject", "").strip()
    if subject:
        meta["title"] = subject
    location = msg.get("Content-Location", "").strip()
    if location:
        meta["source"] = location
    date_str = msg.get("Date", "").strip()
    if date_str:
        try:
            dt = email.utils.parsedate_to_datetime(date_str)
            meta["saved"] = dt.strftime("%Y-%m-%d")
        except Exception:
            meta["saved"] = date_str

    # Find the HTML part and collect image parts
    html_part = None
    images = {}
    for part in msg.walk():
        ct = part.get_content_type()
        if ct == "text/html" and html_part is None:
            payload = part.get_payload(decode=True)
            if payload:
                html_part = payload.decode("utf-8", errors="ignore")
        elif ct.startswith("image/"):
            payload = part.get_payload(decode=True)
            loc = part.get("Content-Location", "").strip()
            if payload and loc:
                images[loc] = (ct, payload)

    if not html_part:
        html_part = text_content

    # Extract <head> metadata to supplement envelope metadata
    meta = _extract_head_metadata(html_part, meta)

    return html_part, meta, images

    return html_part, meta


def _extract_head_metadata(html: str, meta: Dict[str, str]) -> Dict[str, str]:
    """Extract metadata from HTML <head> to fill gaps in envelope metadata."""
    try:
        doc = html_fromstring(html)
    except Exception:
        return meta

    # <title> — fallback for title
    if "title" not in meta:
        title_el = doc.find(".//title")
        if title_el is not None and title_el.text:
            meta["title"] = title_el.text.strip()

    # Resolve title: prefer shorter of envelope vs <title> (less likely
    # to have site suffix like " | Ministry of Finance")
    title_el = doc.find(".//title")
    if title_el is not None and title_el.text:
        html_title = title_el.text.strip()
        if "title" in meta and html_title and meta["title"]:
            if len(html_title) < len(meta["title"]):
                meta["title"] = html_title

    # <meta> tags
    for m in doc.findall(".//meta"):
        name = (m.get("name") or m.get("property") or "").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue

        if name == "description" and "description" not in meta:
            meta["description"] = content
        elif name == "og:description" and "description" not in meta:
            meta["description"] = content
        elif name == "og:site_name" and "site_name" not in meta:
            meta["site_name"] = content
        elif name == "og:title" and "title" not in meta:
            meta["title"] = content
        elif name == "og:url" and "source" not in meta:
            meta["source"] = content
        elif name == "author" and "author" not in meta:
            meta["author"] = content
        elif name in ("date", "article:published_time", "dcterms.date"):
            if "published" not in meta:
                meta["published"] = content

    # <link rel="canonical"> — fallback for source
    if "source" not in meta:
        for link in doc.findall(".//link"):
            if link.get("rel") == "canonical" and link.get("href"):
                meta["source"] = link.get("href").strip()
                break

    # Infer site_name from source URL domain
    if "site_name" not in meta and "source" in meta:
        src = meta["source"]
        m = re.match(r"https?://(?:www\.)?([^/]+)", src)
        if m:
            meta["site_name"] = m.group(1)

    return meta


# ---------------------------------------------------------------------------
# Stage 2: Noise removal
# ---------------------------------------------------------------------------
# Elements that never contain primary content
_STRIP_TAGS = {
    "script", "style", "noscript", "iframe", "svg", "object", "embed",
    "video", "audio", "canvas", "map", "input", "select",
    "textarea", "button",
}

# Tags to unwrap (keep children, remove the tag itself)
# ASP.NET wraps entire pages in <form>; stripping it would kill all content
_UNWRAP_TAGS = {"form"}

# Semantic HTML5 elements that are navigation/chrome, not content
_STRIP_SEMANTIC = {"nav", "footer", "aside", "header"}

# Class/ID substrings that signal non-content (case-insensitive)
_NOISE_PATTERNS = [
    "sidebar", "breadcrumb", "cookie", "menu", "pagination", "toolbar",
    "social-share", "share-buttons", "related-posts", "advertisement",
    "ad-wrapper", "sticky-header", "sticky-footer", "popup", "modal",
    "banner", "toast", "notification", "skip-nav", "back-to-top",
    # Government portal chrome
    "announcement", "search-dropdown", "radio-container", "search-source",
    "printer", "print-preview", "icon_printer", "icon_ask_question",
    "linkAskQuestion", "webchat", "footerContainer", "return-to-top",
    "navbar-header", "nav-group", "headerTitle", "rss-feed",
    "aspNetHidden",
]

_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)

# Inline style patterns indicating hidden elements
_HIDDEN_STYLE_RE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden", re.IGNORECASE
)


def _strip_noise(doc: etree._Element) -> etree._Element:
    """Remove non-content elements from the HTML tree."""
    removals = []

    for el in doc.iter():
        tag = el.tag if isinstance(el.tag, str) else ""
        tag_lower = tag.lower()

        # Strip known non-content tags
        if tag_lower in _STRIP_TAGS:
            removals.append(el)
            continue

        # Strip semantic nav/chrome elements
        if tag_lower in _STRIP_SEMANTIC:
            removals.append(el)
            continue

        # Strip elements hidden by inline style
        style = el.get("style", "")
        if style and _HIDDEN_STYLE_RE.search(style):
            removals.append(el)
            continue

        # Strip elements whose class or id match noise patterns
        classes = el.get("class", "")
        el_id = el.get("id", "")
        combined = f"{classes} {el_id}"
        if combined.strip() and _NOISE_RE.search(combined):
            removals.append(el)
            continue

    # Remove in reverse document order to avoid invalidating references
    for el in reversed(removals):
        parent = el.getparent()
        if parent is not None:
            # Preserve tail text (text after the element's closing tag)
            if el.tail and el.tail.strip():
                prev = el.getprevious()
                if prev is not None:
                    prev.tail = (prev.tail or "") + el.tail
                else:
                    parent.text = (parent.text or "") + el.tail
            parent.remove(el)

    # Unwrap tags that should be removed but whose children should be kept
    # (e.g. ASP.NET <form> wrapping the whole page)
    for tag_name in _UNWRAP_TAGS:
        for el in list(doc.iter(tag_name)):
            el.drop_tag()

    # Strip HTML comments
    for comment in doc.iter(etree.Comment):
        parent = comment.getparent()
        if parent is not None:
            if comment.tail and comment.tail.strip():
                prev = comment.getprevious()
                if prev is not None:
                    prev.tail = (prev.tail or "") + comment.tail
                else:
                    parent.text = (parent.text or "") + comment.tail
            parent.remove(comment)

    return doc


# ---------------------------------------------------------------------------
# Stage 2.5: Content root refinement
# ---------------------------------------------------------------------------
# Known CMS content wrapper class fragments (case-insensitive)
_CONTENT_CLASSES = [
    "content-wrapper", "theme-content", "article-content", "post-content",
    "entry-content", "main-content", "page-content", "content-body",
    "content-area", "content-main", "rich-text", "field-body",
]

_CONTENT_CLASS_RE = re.compile("|".join(_CONTENT_CLASSES), re.IGNORECASE)


def _find_content_root(doc: etree._Element) -> etree._Element:
    """Narrow the tree to the densest content region.

    Strategy:
    1. Check for known CMS content wrapper classes.
    2. Failing that, find the deepest element that still holds >60% of all text.
       This naturally skips thin chrome wrappers and finds the actual content.
    """
    total_text_len = len((doc.text_content() or "").strip())
    if total_text_len < 100:
        return doc

    # Strategy 1: known content wrapper classes
    for el in doc.iter():
        if not isinstance(el.tag, str):
            continue
        cls = el.get("class", "")
        if cls and _CONTENT_CLASS_RE.search(cls):
            el_len = len((el.text_content() or "").strip())
            if el_len > total_text_len * 0.4:
                return el

    # Strategy 2: find deepest element holding >60% of text
    # Never select table cells, table rows, or list items as content root
    _SKIP_ROOT_TAGS = {"td", "th", "tr", "li", "thead", "tbody", "tfoot"}
    best = doc
    best_depth = 0

    def _walk(el, depth):
        nonlocal best, best_depth
        if not isinstance(el.tag, str):
            return
        tag = el.tag.lower()
        el_len = len((el.text_content() or "").strip())
        # Must hold substantial portion of total text
        if el_len > total_text_len * 0.6:
            # Prefer deeper elements (more specific content region)
            if depth > best_depth and tag not in _SKIP_ROOT_TAGS:
                best = el
                best_depth = depth
            # Keep going deeper
            for child in el:
                _walk(child, depth + 1)

    _walk(doc, 0)

    return best


def _is_layout_table(table: etree._Element) -> bool:
    """Heuristic: return True if a <table> is used for layout, not data."""
    # If it has role="presentation" or role="none", it's definitively layout
    role = (table.get("role") or "").lower()
    if role in ("presentation", "none"):
        return True

    # If this table's own rows (not nested tables') have <th>, it's data
    for row in table.findall("./tr") + table.findall("./thead/tr") + table.findall("./tbody/tr"):
        if row.findall("th"):
            return False

    rows = (table.findall("./tr") + table.findall("./thead/tr")
            + table.findall("./tbody/tr"))
    if not rows:
        return True  # empty table, unwrap

    # Count columns, empty cells, and cells with block content
    col_counts = []
    empty_cells = 0
    total_cells = 0
    cells_with_blocks = 0
    total_text_len = 0
    block_tags = {
        "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "table", "blockquote", "article", "section",
    }

    for row in rows:
        cells = row.findall("td") + row.findall("th")
        col_counts.append(len(cells))
        for cell in cells:
            total_cells += 1
            text = (cell.text_content() or "").strip()
            if not text or text == "\xa0":
                empty_cells += 1
            total_text_len += len(text)
            for child in cell:
                if isinstance(child.tag, str) and child.tag.lower() in block_tags:
                    cells_with_blocks += 1
                    break

    if not col_counts:
        return True

    avg_cols = sum(col_counts) / len(col_counts)

    # Single-column tables are almost always layout
    if avg_cols < 1.5:
        return True

    # High empty cell ratio: tables used for layout/spacing have many
    # empty cells (e.g. indentation columns, spacer rows)
    if total_cells > 4:
        empty_ratio = empty_cells / total_cells
        if empty_ratio > 0.4:
            return True

    # border="0" with many rows and no th → strong layout signal
    border = table.get("border", "")
    if border == "0" and len(rows) > 5:
        return True

    # If >40% of cells contain block elements with few columns, likely layout
    if total_cells > 0 and avg_cols <= 3:
        if cells_with_blocks / total_cells > 0.4:
            return True

    # If average cell text is very long, it's likely prose in a layout table
    if total_cells > 0 and total_text_len / total_cells > 200:
        return True

    return False


def _unwrap_layout_tables(doc: etree._Element) -> etree._Element:
    """Replace layout tables with their cell contents in reading order.

    Smart merging: if a row has a short 'numbering' cell followed by a
    content cell, they are merged into a single paragraph.  Spacer rows
    (all empty cells) are dropped.
    """
    max_passes = 10
    for _ in range(max_passes):
        layout_tables = [
            t for t in doc.findall(".//table") if _is_layout_table(t)
        ]
        if not layout_tables:
            break

        for table in layout_tables:
            parent = table.getparent()
            if parent is None:
                continue

            container = etree.Element("div")
            direct_rows = (table.findall("./tr") + table.findall("./thead/tr")
                           + table.findall("./tbody/tr"))

            for row in direct_rows:
                cells = list(row.findall("td")) + list(row.findall("th"))

                # Collect non-empty cells (text or images)
                cell_data = []
                for cell in cells:
                    text = (cell.text_content() or "").strip()
                    has_img = cell.find(".//img") is not None
                    if has_img or (text and text != "\xa0"):
                        cell_data.append((cell, text))

                if not cell_data:
                    continue  # skip spacer rows

                # Single non-empty cell: output its children directly
                if len(cell_data) == 1:
                    cell = cell_data[0][0]
                    wrapper = etree.SubElement(container, "div")
                    if cell.text and cell.text.strip():
                        wrapper.text = cell.text
                    for child in list(cell):
                        wrapper.append(child)
                    continue

                # Multiple non-empty cells: check for numbering pattern
                first_text = cell_data[0][1]
                is_numbering = (
                    len(first_text) <= 10
                    and bool(re.match(
                        r"^(\d+\.?\d*\.?|[a-zA-Z][.)]\s*|[ivxIVX]+[.)]\s*)$",
                        first_text.strip()
                    ))
                )

                if is_numbering and len(cell_data) >= 2:
                    # Merge: prefix with number, then content from remaining cells
                    wrapper = etree.SubElement(container, "div")
                    prefix = first_text.strip()
                    if not prefix.endswith(".") and not prefix.endswith(")"):
                        prefix += "."
                    prefix += " "

                    # Get content from second cell (the main content cell)
                    main_cell = cell_data[1][0]
                    has_children = len(list(main_cell)) > 0

                    if has_children:
                        # Put prefix text, then move children
                        wrapper.text = prefix + (main_cell.text or "").lstrip()
                        for child in list(main_cell):
                            wrapper.append(child)
                    else:
                        wrapper.text = prefix + cell_data[1][1]

                    # Append any additional cells' content
                    for cell, text in cell_data[2:]:
                        extra = etree.SubElement(container, "div")
                        if list(cell):
                            if cell.text and cell.text.strip():
                                extra.text = cell.text
                            for child in list(cell):
                                extra.append(child)
                        else:
                            extra.text = text
                else:
                    # Not a numbering row: output each cell as separate block
                    for cell, text in cell_data:
                        wrapper = etree.SubElement(container, "div")
                        if list(cell):
                            if cell.text and cell.text.strip():
                                wrapper.text = cell.text
                            for child in list(cell):
                                wrapper.append(child)
                        else:
                            wrapper.text = text

            container.tail = table.tail
            parent.replace(table, container)

    return doc

    return doc


# ---------------------------------------------------------------------------
# Stage 4: Semantic inference — recover headings & lists from styling
# ---------------------------------------------------------------------------
_FONT_SIZE_RE = re.compile(r"font-size\s*:\s*([\d.]+)\s*(px|pt|em|rem)", re.I)
_FONT_WEIGHT_RE = re.compile(r"font-weight\s*:\s*(bold|[7-9]\d{2})", re.I)
_MARGIN_LEFT_RE = re.compile(r"margin-left\s*:\s*([\d.]+)\s*(px|pt|em|rem)", re.I)
_PADDING_LEFT_RE = re.compile(r"padding-left\s*:\s*([\d.]+)\s*(px|pt|em|rem)", re.I)
_TEXT_ALIGN_CENTER_RE = re.compile(r"text-align\s*:\s*center", re.I)

# Tags that should not be promoted to headings
_NO_PROMOTE = {
    "a", "td", "th", "li", "label", "figcaption", "caption",
    "h1", "h2", "h3", "h4", "h5", "h6",
}

# Tags that are inline — candidates for heading promotion
_INLINE_OR_BLOCK = {"div", "span", "p", "font", "b", "strong"}


def _get_font_size_px(style: str) -> Optional[float]:
    """Extract font-size from inline style, normalised to px."""
    m = _FONT_SIZE_RE.search(style)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "pt":
        val *= 1.333
    elif unit == "em" or unit == "rem":
        val *= 16.0
    return val


def _is_bold(el: etree._Element) -> bool:
    """Check if element is bold via tag or inline style."""
    tag = el.tag if isinstance(el.tag, str) else ""
    if tag.lower() in ("b", "strong"):
        return True
    style = el.get("style", "")
    return bool(_FONT_WEIGHT_RE.search(style))


def _has_only_bold_content(el: etree._Element) -> bool:
    """Check if element's entire text content comes from bold descendants.

    Returns True for patterns like:
        <div><span><strong><u>Heading</u></strong></span></div>
        <div><b>Heading</b></div>
        <div><strong>Heading Text</strong></div>
    """
    text = (el.text_content() or "").strip()
    if not text:
        return False

    # Check if element itself has direct text that's not inside bold tags
    if el.text and el.text.strip():
        # The element has direct text — not purely from bold children
        return False

    # Walk to find if all text lives inside <strong>/<b> descendants
    bold_text_len = 0
    for desc in el.iter():
        if not isinstance(desc.tag, str):
            continue
        tag = desc.tag.lower()
        if tag in ("strong", "b"):
            bold_text_len += len((desc.text_content() or "").strip())

    # All text should be inside bold tags (with tolerance for whitespace)
    return bold_text_len >= len(text) * 0.8


def _get_text_length(el: etree._Element) -> int:
    """Get total text content length of an element."""
    return len(el.text_content() or "")


def _is_block_like(el: etree._Element) -> bool:
    """Check if element is rendered as its own line (block or styled block)."""
    tag = (el.tag if isinstance(el.tag, str) else "").lower()
    if tag in ("div", "p", "h1", "h2", "h3", "h4", "h5", "h6",
               "blockquote", "section", "article", "header"):
        return True
    style = el.get("style", "")
    if re.search(r"display\s*:\s*block", style, re.I):
        return True
    return False


def _collect_font_sizes(doc: etree._Element) -> List[float]:
    """Collect all distinct font sizes used in the document for heading inference."""
    sizes = []
    for el in doc.iter():
        if not isinstance(el.tag, str):
            continue
        tag = el.tag.lower()
        if tag in _NO_PROMOTE:
            continue
        style = el.get("style", "")
        sz = _get_font_size_px(style)
        if sz and sz > 14:  # only consider sizes above base text
            text = (el.text_content() or "").strip()
            # Headings are typically short
            if text and len(text) < 200:
                sizes.append(sz)
    return sizes


def _build_size_to_heading_map(sizes: List[float]) -> Dict[float, str]:
    """Map distinct font sizes to heading levels.

    Groups sizes into distinct tiers (with 2px tolerance) and maps
    the largest to h1, next to h2, etc.  Max h4 to avoid over-nesting.
    """
    if not sizes:
        return {}

    # Cluster sizes with 2px tolerance
    unique = sorted(set(sizes), reverse=True)
    tiers = []
    for s in unique:
        if not tiers or abs(s - tiers[-1]) > 2:
            tiers.append(s)

    # Map: largest → h1, etc.  Cap at h4.
    mapping = {}
    for i, tier_size in enumerate(tiers[:4]):
        level = i + 1
        # Map all sizes within 2px of this tier
        for s in unique:
            if abs(s - tier_size) <= 2:
                mapping[s] = f"h{level}"

    return mapping


def _infer_semantics(doc: etree._Element) -> etree._Element:
    """Recover heading hierarchy and list structure from inline styles."""

    # --- Heading inference ---
    sizes = _collect_font_sizes(doc)
    size_map = _build_size_to_heading_map(sizes)

    # Also detect bold-only headings (bold + block + short text, no font-size)
    # When no size hierarchy exists, these are the primary structural cues
    # so they should be h2 (reserving h1 for explicit page titles).
    # When size headings exist, bold-only headings sit one level below.
    max_inferred = max((int(h[1]) for h in size_map.values()), default=0)
    bold_heading_level = min(max_inferred + 1, 5) if max_inferred else 2

    promotions = []

    for el in doc.iter():
        if not isinstance(el.tag, str):
            continue
        tag = el.tag.lower()
        if tag in _NO_PROMOTE:
            continue
        if tag not in _INLINE_OR_BLOCK:
            continue

        style = el.get("style", "")
        text = (el.text_content() or "").strip()
        if not text or len(text) > 200:
            continue

        # Check font-size based promotion
        sz = _get_font_size_px(style)
        if sz:
            # Find closest tier
            heading_tag = None
            for mapped_sz, htag in size_map.items():
                if abs(sz - mapped_sz) <= 2:
                    heading_tag = htag
                    break
            if heading_tag:
                promotions.append((el, heading_tag))
                continue

        # Bold + block-like + short text → heading (conservative)
        if _is_bold(el) and _is_block_like(el) and len(text) < 100:
            # Skip if it's a bold fragment INSIDE a paragraph (inline emphasis)
            # But allow standalone block divs inside larger containers
            parent = el.getparent()
            if parent is not None:
                parent_tag = (parent.tag if isinstance(parent.tag, str) else "").lower()
                # Only skip if parent is a paragraph-like element (inline context)
                # where this bold text is just emphasis within running prose
                if parent_tag == "p":
                    parent_text_len = _get_text_length(parent)
                    if parent_text_len > len(text) * 2:
                        continue
            promotions.append((el, f"h{bold_heading_level}"))
            continue

        # Block element whose entire content is bold (possibly via descendants)
        # Common in gov sites: <div><span><strong><u>Heading</u></strong></span></div>
        if _is_block_like(el) and len(text) < 100 and _has_only_bold_content(el):
            parent = el.getparent()
            if parent is not None:
                parent_tag = (parent.tag if isinstance(parent.tag, str) else "").lower()
                if parent_tag == "p":
                    parent_text_len = _get_text_length(parent)
                    if parent_text_len > len(text) * 2:
                        continue
            promotions.append((el, f"h{bold_heading_level}"))

    # Apply promotions
    for el, heading_tag in promotions:
        el.tag = heading_tag
        # Clear styling attributes — they're no longer needed
        for attr in ("style", "class", "align"):
            if attr in el.attrib:
                del el.attrib[attr]
        # Strip redundant formatting tags inside headings
        # (bold, underline, italic are implied by heading level)
        for fmt_tag in ("strong", "b", "u", "em", "i", "span", "font"):
            for fmt_el in list(el.iter(fmt_tag)):
                fmt_el.drop_tag()

    # --- <br><br> to paragraph breaks ---
    _convert_br_sequences(doc)

    # --- Indentation-based list inference ---
    _infer_lists_from_indentation(doc)

    return doc


def _convert_br_sequences(doc: etree._Element):
    """Convert sequences of 2+ <br> tags into paragraph breaks."""
    for el in list(doc.iter("br")):
        parent = el.getparent()
        if parent is None:
            continue

        # Check if next sibling is also a <br>
        nxt = el.getnext()
        if nxt is not None and isinstance(nxt.tag, str) and nxt.tag.lower() == "br":
            # Replace the first <br> with a <p> break
            # by splitting the parent's content at this point
            el.tag = "p"
            # Remove consecutive <br>s after this one
            while True:
                nxt = el.getnext()
                if nxt is not None and isinstance(nxt.tag, str) and nxt.tag.lower() == "br":
                    parent.remove(nxt)
                else:
                    break


def _infer_lists_from_indentation(doc: etree._Element):
    """Detect list-like patterns from margin/padding indentation."""
    # Look for sequences of sibling elements with consistent left indentation
    # and list-like markers (bullets, numbers, dashes, letters)
    _LIST_MARKER_RE = re.compile(
        r"^\s*(?:[\u2022\u2023\u25E6\u2043\u2219•●○◦‣⁃–—-]"  # bullet chars
        r"|\d+[.)]\s"                                           # numbered
        r"|[a-zA-Z][.)]\s"                                      # lettered
        r"|[ivxIVX]+[.)]\s"                                     # roman
        r")\s*"
    )

    for parent in doc.iter():
        if not isinstance(parent.tag, str):
            continue

        children = list(parent)
        if len(children) < 2:
            continue

        # Find runs of indented children with list markers
        run_start = None
        run_items = []

        for i, child in enumerate(children):
            if not isinstance(child.tag, str):
                continue
            style = child.get("style", "")
            margin = _MARGIN_LEFT_RE.search(style) or _PADDING_LEFT_RE.search(style)
            text = (child.text_content() or "").strip()

            if margin and text and _LIST_MARKER_RE.match(text):
                if run_start is None:
                    run_start = i
                run_items.append(child)
            else:
                if len(run_items) >= 2:
                    _convert_run_to_list(parent, run_items)
                run_start = None
                run_items = []

        # Handle run at end
        if len(run_items) >= 2:
            _convert_run_to_list(parent, run_items)


def _convert_run_to_list(parent: etree._Element, items: List[etree._Element]):
    """Convert a run of indented elements into a <ul> or <ol>."""
    _STRIP_MARKER_RE = re.compile(
        r"^\s*(?:[\u2022\u2023\u25E6\u2043\u2219•●○◦‣⁃–—-]"
        r"|\d+[.)]\s?"
        r"|[a-zA-Z][.)]\s?"
        r"|[ivxIVX]+[.)]\s?"
        r")\s*"
    )

    # Determine if ordered or unordered
    first_text = (items[0].text_content() or "").strip()
    is_ordered = bool(re.match(r"^\s*\d+[.)]\s", first_text))

    list_el = etree.Element("ol" if is_ordered else "ul")

    # Insert the list element before the first item
    first_item = items[0]
    parent.insert(list(parent).index(first_item), list_el)

    for item in items:
        li = etree.SubElement(list_el, "li")
        text = (item.text_content() or "").strip()
        text = _STRIP_MARKER_RE.sub("", text, count=1).strip()
        li.text = text
        parent.remove(item)


# ---------------------------------------------------------------------------
# Stage 5: Normalisation
# ---------------------------------------------------------------------------
def _normalise(doc: etree._Element) -> etree._Element:
    """Clean up whitespace, encoding artefacts, and empty elements."""

    # Pass 0: convert <u> to <strong>
    # Markdown has no underline — markdownify drops <u> entirely.
    # In government documents, <u> is used as a label/title marker with
    # the same semantic intent as <strong>.  Converting preserves emphasis.
    # (By this stage, <u> inside promoted headings was already stripped.)
    for el in list(doc.iter("u")):
        el.tag = "strong"

    # Pass 1: normalise text content
    for el in doc.iter():
        # Normalise element text
        if el.text:
            el.text = _normalise_text(el.text)
        # Normalise tail text
        if el.tail:
            el.tail = _normalise_text(el.tail)

    # Pass 2: remove empty elements (but not void elements like <br>, <hr>, <img>)
    _VOID_TAGS = {"br", "hr", "img", "col", "area", "base", "embed",
                   "input", "link", "meta", "param", "source", "track", "wbr"}

    # Multiple passes to handle nested empty elements
    for _ in range(3):
        removals = []
        for el in doc.iter():
            if not isinstance(el.tag, str):
                continue
            tag = el.tag.lower()
            if tag in _VOID_TAGS:
                continue
            # Keep if has children, text content, or is a structural element
            if len(el) > 0:
                continue
            if (el.text and el.text.strip()):
                continue
            if tag in ("td", "th", "li"):
                continue  # keep even if empty (table/list structure)
            # Element is empty — remove, preserving tail
            removals.append(el)

        for el in reversed(removals):
            parent = el.getparent()
            if parent is not None:
                if el.tail and el.tail.strip():
                    prev = el.getprevious()
                    if prev is not None:
                        prev.tail = (prev.tail or "") + el.tail
                    else:
                        parent.text = (parent.text or "") + el.tail
                parent.remove(el)

    return doc


def _normalise_text(text: str) -> str:
    """Normalise a text string: fix encoding artefacts, collapse whitespace."""
    # Replace non-breaking spaces and other Unicode spaces with regular space
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")  # zero-width space
    text = text.replace("\u200c", "")  # zero-width non-joiner
    text = text.replace("\u200d", "")  # zero-width joiner
    text = text.replace("\ufeff", "")  # BOM
    # Collapse multiple spaces (but preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Stage 5.5: Image description — replace <img> with text descriptions
# ---------------------------------------------------------------------------
_MIN_IMAGE_SIZE = 5000  # bytes — skip icons and UI chrome below this

_VISION_PROMPT = """Describe this diagram/figure for a text document. Your description will replace the image in a markdown document used for research and review.

Rules:
- Describe the structure, relationships, and flow — not just labels.
- For flowcharts: describe nodes, connections, conditions, and flow direction.
- For framework/hierarchy diagrams: describe the groupings, levels, and relationships.
- For tables rendered as images: recreate as a text table.
- Use plain structured text. No markdown headings or bullet formatting.
- Be concise but complete. Every piece of information in the diagram must be captured.
- Start directly with the description — no preamble like "This diagram shows..."."""


def _describe_images(
    doc: etree._Element,
    images: Dict[str, Tuple[str, bytes]],
    api_key: str = "",
) -> etree._Element:
    """Replace <img> elements with text descriptions from the vision API.

    For each significant image (>5KB), attempts to describe via Claude vision API.
    Falls back to OCR text extraction if the API call fails.
    Images below the size threshold (icons, UI chrome) are simply removed.
    """
    import base64
    import os
    import json
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError

    # Use passed key; fall back to env var for backward compat (direct CLI use)
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    logger.warning(f"[VISION] api_key={'set' if api_key else 'empty'}, images={len(images)} MHTML parts")

    img_elements = list(doc.findall(".//img"))
    if not img_elements:
        logger.warning(f"[VISION] No <img> tags found in content. Images dict has {len(images)} MHTML parts.")
        return doc

    logger.warning(f"[VISION] Found {len(img_elements)} <img> tags, {len(images)} MHTML image parts")

    for img in img_elements:
        src = img.get("src", "")
        alt = img.get("alt", "")
        parent = img.getparent()
        if parent is None:
            continue

        # Match src to MHTML image parts
        # MHTML Content-Location may be truncated vs the full src URL
        image_data = None
        media_type = None
        for loc, (mt, data) in images.items():
            if src.startswith(loc) or loc.startswith(src) or _urls_match(src, loc):
                image_data = data
                media_type = mt
                break

        # No matching image part, or too small (icon/UI element) — remove
        if image_data is None:
            _remove_element_preserve_tail(img)
            continue
        if len(image_data) < _MIN_IMAGE_SIZE:
            _remove_element_preserve_tail(img)
            continue

        logger.warning(f"[VISION] Describing image ({len(image_data)}B, {media_type}): {src[:80]}")

        # Attempt vision API description
        description = None
        if api_key:
            description = _describe_with_vision(image_data, media_type, api_key)

        # Fallback: OCR
        if description is None:
            description = _describe_with_ocr(image_data)

        # Replace <img> with description block
        desc_block = etree.Element("div")
        # Add figure label
        label = alt.strip() if alt.strip() else "Figure"
        label_el = etree.SubElement(desc_block, "strong")
        label_el.text = f"[{label}]"
        label_el.tail = "\n"
        # Add description text
        desc_p = etree.SubElement(desc_block, "p")
        desc_p.text = description

        desc_block.tail = img.tail
        parent.replace(img, desc_block)

    return doc


def _urls_match(url1: str, url2: str) -> bool:
    """Fuzzy URL matching for MHTML Content-Location vs img src."""
    # Strip query params and fragments for comparison
    def _core(u):
        u = u.split("?")[0].split("#")[0]
        return u.rstrip("/").lower()
    return _core(url1) == _core(url2)


def _describe_with_vision(
    image_data: bytes, media_type: str, api_key: str
) -> Optional[str]:
    """Send image to Claude vision API and get a text description."""
    import base64
    import json
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError

    # Validate media type — Anthropic only accepts these
    _SUPPORTED_MEDIA = {"image/png", "image/jpeg", "image/gif", "image/webp"}
    if media_type not in _SUPPORTED_MEDIA:
        logger.warning(f"Unsupported image media type: {media_type}, skipping vision")
        return None

    b64 = base64.b64encode(image_data).decode("ascii")

    body = json.dumps({
        "model": os.environ.get("VISION_MODEL", "claude-sonnet-4-6"),
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                },
                {"type": "text", "text": _VISION_PROMPT},
            ],
        }],
    }).encode("utf-8")

    req = Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            text_parts = [
                block["text"]
                for block in result.get("content", [])
                if block.get("type") == "text"
            ]
            description = "\n".join(text_parts).strip()
            if description:
                logger.info(f"Vision API described image ({len(image_data)}B)")
                return description
    except HTTPError as e:
        # Read the error response body for better diagnostics
        try:
            err_body = e.read().decode("utf-8", errors="ignore")[:500]
            logger.warning(f"Vision API {e.code}: {err_body}")
        except Exception:
            logger.warning(f"Vision API failed: {e}")
    except (URLError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Vision API failed: {e}")

    return None


def _describe_with_ocr(image_data: bytes) -> str:
    """Fallback: extract text labels from image via OCR."""
    import tempfile
    import os

    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_data))

        # Use doctr if available (already in the project's deps)
        from ..extracttext import _preprocess_image, _ocr_with_doctr
        preprocessed = _preprocess_image(img)
        ocr_text = _ocr_with_doctr([preprocessed]).strip()

        if ocr_text:
            return f"[OCR extracted text — diagram structure not captured]\n{ocr_text}"
    except Exception as e:
        logger.warning(f"OCR fallback failed: {e}")

    return "[Figure: image could not be described]"


def _remove_element_preserve_tail(el):
    """Remove an element from the tree, preserving its tail text."""
    parent = el.getparent()
    if parent is None:
        return
    if el.tail and el.tail.strip():
        prev = el.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + el.tail
        else:
            parent.text = (parent.text or "") + el.tail
    parent.remove(el)


# ---------------------------------------------------------------------------
# Stage 6: Markdown conversion + post-processing
# ---------------------------------------------------------------------------
class _CleanConverter(MarkdownConverter):
    """Custom markdownify converter with better defaults for LLM output."""

    def convert_table(self, el, text, parent_tags=None):
        """Ensure tables have proper spacing."""
        return "\n\n" + text.strip() + "\n\n"

    def convert_a(self, el, text, parent_tags=None):
        """Convert links: keep href if meaningful, otherwise just text."""
        href = el.get("href", "").strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            return text
        # Skip if link text is same as href (redundant)
        if text.strip() == href:
            return href
        return f"[{text.strip()}]({href})"


def _html_to_markdown(doc: etree._Element) -> str:
    """Convert cleaned HTML tree to markdown string."""
    # Serialize the tree back to HTML string for markdownify
    html_str = html_tostring(doc, encoding="unicode", method="html")

    # Run markdownify with our custom converter
    result = _CleanConverter(
        heading_style="atx",
        bullets="-",
        strong_em_symbol="*",
        newline_style="backslash",
        strip=["img"],  # strip images (not useful for LLM text consumption)
        wrap=False,
        wrap_width=0,
    ).convert(html_str)

    # Post-processing
    result = _postprocess_markdown(result)

    return result



def _postprocess_markdown(text: str) -> str:
    """Clean up common markdownify output quirks."""
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Ensure headings have blank lines before and after
    text = re.sub(r"([^\n])\n(#{1,6}\s)", r"\1\n\n\2", text)
    text = re.sub(r"(#{1,6}\s[^\n]+)\n([^\n#])", r"\1\n\n\2", text)

    # Fix table formatting:
    # 1. Remove blank lines BETWEEN table rows (markdownify sometimes adds these)
    text = re.sub(r"(\|[^\n]*)\n\n(\|)", r"\1\n\2", text)
    # Repeat to handle multiple consecutive blank lines within tables
    text = re.sub(r"(\|[^\n]*)\n\n(\|)", r"\1\n\2", text)
    # 2. Ensure blank line BEFORE first table row and AFTER last table row
    text = re.sub(r"([^\n|])\n(\|)", r"\1\n\n\2", text)
    text = re.sub(r"(\|[^\n]*)\n([^\n|])", r"\1\n\n\2", text)

    # Remove lines that are just whitespace
    text = "\n".join(
        line if line.strip() else "" for line in text.split("\n")
    )

    # Collapse again after all fixes
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Metadata formatting
# ---------------------------------------------------------------------------
def _format_frontmatter(meta: Dict[str, str]) -> str:
    """Format metadata as YAML front-matter."""
    if not meta:
        return ""

    lines = ["---"]

    # Ordered keys for consistent output
    key_order = ["title", "source", "site_name", "saved", "published",
                 "description", "author"]

    for key in key_order:
        if key in meta:
            val = meta[key]
            # Escape YAML special chars in values
            if any(c in val for c in ":#{}[]|>&*!"):
                val = f'"{val}"'
            elif val.startswith('"') or val.startswith("'"):
                val = f'"{val}"'
            lines.append(f"{key}: {val}")

    # Any remaining keys not in our order
    for key, val in meta.items():
        if key not in key_order:
            if any(c in val for c in ":#{}[]|>&*!"):
                val = f'"{val}"'
            lines.append(f"{key}: {val}")

    lines.append("---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def _convert_single(content: bytes, filename: str, api_key: str = "") -> Tuple[str, Dict[str, str]]:
    """Run the full 6-stage pipeline on a single MHTML file.

    Returns:
        (markdown_string, metadata_dict)
    """
    # Stage 1: Extract HTML + metadata + images
    html_str, meta, images = _extract_mhtml(content)

    # Parse into lxml tree
    try:
        doc = html_fromstring(html_str)
    except Exception:
        # If lxml can't parse, fall back to basic text extraction
        logger.warning(f"lxml failed to parse {filename}, returning raw text")
        from ..extracttext import _extract_from_mhtml
        text = _extract_from_mhtml(content)
        return text, meta

    # Find the best content root — prefer <body>, <main>, or <article>
    body = doc.find(".//body")
    if body is not None:
        doc = body

    for tag in ("main", "article"):
        el = doc.find(f".//{tag}")
        if el is not None:
            # Only use if it contains substantial content
            if _get_text_length(el) > _get_text_length(doc) * 0.5:
                doc = el
                break

    # Stage 2: Noise removal
    doc = _strip_noise(doc)

    # Stage 3: Layout table unwrapping (must happen before content root
    # detection, otherwise the finder may select a <td> inside a layout table)
    doc = _unwrap_layout_tables(doc)

    # Stage 3.5: Content root refinement
    # After stripping noise and unwrapping layout, narrow to the densest
    # content region.  Handles CMS pages where content is buried deep.
    doc = _find_content_root(doc)

    # Stage 4: Semantic inference
    doc = _infer_semantics(doc)

    # Stage 5: Normalisation
    doc = _normalise(doc)

    # Stage 5.5: Image description — replace <img> with text descriptions
    doc = _describe_images(doc, images, api_key=api_key)

    # Stage 6: Markdown conversion
    markdown = _html_to_markdown(doc)

    return markdown, meta


# ---------------------------------------------------------------------------
# Scheme interface: transform()
# ---------------------------------------------------------------------------
def transform(texts: Dict[str, bytes], output_mode: str = "consolidated", api_key: str = "") -> dict:
    """Transform MHTML bytes into structured markdown.

    Args:
        texts: {filename: raw_mhtml_bytes, ...}
        output_mode: "consolidated" or "individual"
        api_key: Anthropic API key for vision-based image descriptions

    Returns:
        Scheme result dict.
    """
    individual_files = []

    for filename, content in sorted(texts.items()):
        try:
            markdown, meta = _convert_single(content, filename, api_key=api_key)
        except Exception as e:
            logger.error(f"Conversion failed for {filename}: {e}")
            markdown = f"*Conversion failed: {e}*"
            meta = {}

        frontmatter = _format_frontmatter(meta)
        if frontmatter:
            full_md = frontmatter + "\n\n" + markdown
        else:
            full_md = markdown

        out_filename = re.sub(r"\.mhtml?$", ".md", filename, flags=re.I)

        individual_files.append({
            "filename": out_filename,
            "text": full_md,
            "metadata": meta,
        })

    if output_mode == "individual":
        return {
            "text": "",
            "filename": "",
            "files": individual_files,
            "file_count": len(individual_files),
        }

    # Consolidated: merge all files
    if len(individual_files) == 1:
        f = individual_files[0]
        return {
            "text": f["text"],
            "filename": f["filename"],
            "metadata": f.get("metadata", {}),
        }

    merged_parts = []
    for f in individual_files:
        merged_parts.append(f["text"])

    return {
        "text": "\n\n---\n\n".join(merged_parts),
        "filename": "converted.md",
        "file_count": len(individual_files),
    }
