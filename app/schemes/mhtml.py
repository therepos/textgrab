"""MHTML → clean structured markdown.

Extracts content from MHTML web archives, stripping navigation, banners,
scripts, styles, and other non-content elements. Preserves headings, lists,
tables, and text structure as clean markdown suitable for knowledge bases
and LLM prompt context.

Handles CMS/ASP.NET pages where content lives inside layout tables by
detecting and flattening them into prose while preserving real data tables.

Pipeline: MHTML → MIME unwrap → HTML → content div detection
        → junk stripping → layout table flattening → markdown

Key improvements over previous version:
- Faithful heading-level preservation (h1-h6 from HTML → # – ######)
- Proper data-table conversion with colspan/rowspan expansion
- Nested-table handling (layout tables inside data tables, etc.)
- Definition list (<dl>), <details>/<summary>, <blockquote> support
- <pre>/<code> block preservation
- Bold/italic inline formatting retained for semantic emphasis
- Ordered list numbering preserved
- Robust layout-table detection with scoring heuristic
"""

import email
import logging
import re
from typing import Dict, List, Tuple

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
    "menu", "menuitem", "form",
}

# Class/id substrings that indicate non-content blocks
_JUNK_PATTERNS = re.compile(
    r"(sidebar|banner|breadcrumb|cookie|"
    r"advertisement|advert|social|share|related|comment|"
    r"popup|modal|overlay|toast|toolbar|ribbon|masthead|"
    r"skip-to|jump-to|back-to-top|pagination|pager)",
    re.IGNORECASE,
)

# Class/id patterns that indicate navigation
_NAV_PATTERNS = re.compile(
    r"(^nav$|^nav\b|navbar|mainNav|nav-group|navmenu|"
    r"^menu$|^menu\b|mainMenu|footerContainer|footerNav|"
    r"site-header|site-footer|page-header|page-footer)",
    re.IGNORECASE,
)

# Class/id patterns that likely contain the main content
_CONTENT_HINTS = re.compile(
    r"(content|article|main-body|mainWrap|theme-content|"
    r"post-body|entry-content|page-content|story-body|"
    r"col-md-9|col-md-8|col-lg-9|col-lg-8|"
    r"ipsType_richText|cke_editable|document-content|"
    r"wiki-content|mw-parser-output|markdown-body)",
    re.IGNORECASE,
)


def _find_content_root(doc):
    """Find the most likely content-bearing element."""
    # Try semantic elements first
    for tag in ("main", "article"):
        for el in doc.iter(tag):
            if len(el.text_content().strip()) > 200:
                return el

    # Try content-hint class/id — prefer the SMALLEST (most specific)
    candidates = []
    for el in doc.iter("div", "section", "td"):
        attrs = " ".join(filter(None, [
            el.get("class") or "",
            el.get("id") or "",
        ]))
        if attrs and _CONTENT_HINTS.search(attrs):
            text_len = len(el.text_content().strip())
            if text_len > 200:
                candidates.append((text_len, el))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    # Try role="main"
    for el in doc.iter():
        if el.get("role") == "main":
            if len(el.text_content().strip()) > 200:
                return el

    # Fallback: largest text-bearing div
    best = None
    best_len = 0
    for el in doc.iter("div"):
        text_len = len(el.text_content().strip())
        if text_len > best_len:
            best = el
            best_len = text_len

    return best


# ---------------------------------------------------------------------------
# Table analysis and conversion
# ---------------------------------------------------------------------------

def _is_layout_table(table) -> bool:
    """Detect layout vs data table using a scoring heuristic.

    Positive score = layout, negative = data.
    """
    rows = list(table.iter("tr"))
    if not rows:
        return True
    if len(rows) == 1:
        cells = list(rows[0].iter("td", "th"))
        # Single-row with many short cells = possibly data
        if len(cells) > 2 and all(len(c.text_content().strip()) < 50 for c in cells):
            return False
        return True

    score = 0

    # <th> tags suggest data
    th_count = sum(1 for _ in table.iter("th"))
    if th_count > 0:
        score -= 3

    # thead/tbody = data
    if table.find(".//thead") is not None:
        score -= 3

    # Consistent column counts = data
    col_counts = []
    for row in rows:
        cells = list(row.iter("td", "th"))
        effective_cols = sum(int(c.get("colspan") or "1") for c in cells)
        col_counts.append(effective_cols)

    if col_counts:
        variance = max(col_counts) - min(col_counts)
        if variance == 0:
            score -= 2
        elif variance > 2:
            score += 2

    # Large colspans = layout
    for row in rows:
        for cell in row.iter("td", "th"):
            cs = int(cell.get("colspan") or "1")
            if cs >= 4:
                score += 2

    # Spacer rows = layout
    empty_rows = 0
    for row in rows:
        cells = list(row.iter("td", "th"))
        if all(not c.text_content().strip() or
               c.text_content().strip() in ("\xa0", " ")
               for c in cells):
            empty_rows += 1

    if len(rows) > 4 and empty_rows / len(rows) > 0.3:
        score += 3

    # Cells with long paragraph text = layout
    long_cell_count = 0
    total_cells = 0
    for row in rows:
        for cell in row.iter("td", "th"):
            total_cells += 1
            if len(cell.text_content().strip()) > 200:
                long_cell_count += 1

    if total_cells > 0 and long_cell_count / total_cells > 0.2:
        score += 2

    # Nested tables = layout
    if table.findall(".//table"):
        score += 2

    # Cells containing block elements = layout
    block_cell_count = 0
    for row in rows:
        for cell in row.iter("td", "th"):
            for child in cell:
                if child.tag in ("div", "p", "h1", "h2", "h3", "h4", "h5", "h6",
                                  "ul", "ol", "blockquote", "pre", "table"):
                    block_cell_count += 1
                    break

    if total_cells > 0 and block_cell_count / total_cells > 0.3:
        score += 2

    # Many rows with uniform short text = data
    if len(rows) >= 5 and long_cell_count == 0:
        score -= 2

    return score > 0


def _expand_table_grid(table) -> Tuple[List[List[str]], bool]:
    """Expand a table into a rectangular grid handling colspan and rowspan.

    Returns (grid, has_header) where grid[r][c] is cell text.
    """
    rows = list(table.iter("tr"))
    if not rows:
        return [], False

    # Determine grid dimensions
    max_cols = 0
    for row in rows:
        cols_in_row = sum(int(c.get("colspan") or "1") for c in row.iter("td", "th"))
        max_cols = max(max_cols, cols_in_row)

    if max_cols == 0:
        return [], False

    n_rows = len(rows)
    grid = [[None] * max_cols for _ in range(n_rows)]

    has_header = False
    for r_idx, row in enumerate(rows):
        col_idx = 0
        for cell in row.iter("td", "th"):
            if cell.tag == "th":
                has_header = True

            while col_idx < max_cols and grid[r_idx][col_idx] is not None:
                col_idx += 1
            if col_idx >= max_cols:
                break

            text = cell.text_content().strip()
            cs = int(cell.get("colspan") or "1")
            rs = int(cell.get("rowspan") or "1")

            for dr in range(rs):
                for dc in range(cs):
                    nr, nc = r_idx + dr, col_idx + dc
                    if nr < n_rows and nc < max_cols:
                        grid[nr][nc] = text if (dr == 0 and dc == 0) else ""

            col_idx += cs

    # Fill remaining None
    for r in range(n_rows):
        for c in range(max_cols):
            if grid[r][c] is None:
                grid[r][c] = ""

    if table.find(".//thead") is not None:
        has_header = True

    return grid, has_header


def _data_table_to_markdown(table) -> str:
    """Convert a data table to markdown with colspan/rowspan expansion."""
    grid, has_header = _expand_table_grid(table)
    if not grid:
        return ""

    def esc(t: str) -> str:
        return t.replace("|", "\\|").replace("\n", " ")

    md_rows: List[str] = []
    for i, row in enumerate(grid):
        md_rows.append("| " + " | ".join(esc(c) for c in row) + " |")
        if i == 0:
            md_rows.append("| " + " | ".join("---" for _ in row) + " |")

    return "\n".join(md_rows)


def _flatten_layout_table(table) -> str:
    """Convert a layout table into structured text."""
    parts: List[str] = []
    _NUM_RE = re.compile(r'^(\d+\.|\d+\.\d+|[a-z]\)|[ivx]+\))$')

    for row in table.iter("tr"):
        # Skip rows belonging to nested tables
        if row.getparent() is not None:
            row_table = row.getparent()
            if row_table.tag == "tbody":
                row_table = row_table.getparent()
            if row_table is not None and row_table is not table:
                continue

        cells = list(row.iter("td", "th"))
        cell_texts = []
        for cell in cells:
            # Handle nested tables
            nested_tables = cell.findall(".//table")
            for nt in nested_tables:
                if _is_layout_table(nt):
                    parts.append(_flatten_layout_table(nt))
                else:
                    parts.append(_data_table_to_markdown(nt))

            # Remove nested tables before getting text
            for nt in list(cell.findall(".//table")):
                if nt.getparent() is not None:
                    nt.getparent().remove(nt)

            t = cell.text_content().strip()
            if t and t not in ("\xa0", " "):
                cell_texts.append(t)

        if not cell_texts:
            continue

        # Heading detection
        if len(cell_texts) == 1:
            cell = cells[0] if cells else None
            if cell is not None:
                cs = int(cell.get("colspan") or "1")
                t = cell_texts[0]
                has_bold = any(c.tag in ("strong", "b") for c in cell.iter())
                has_underline = any(c.tag == "u" for c in cell.iter())
                is_short = len(t) < 120

                if is_short and has_bold and cs >= 4:
                    if has_underline:
                        parts.append(f"## {t}")
                    else:
                        parts.append(f"### {t}")
                    continue

            parts.append(cell_texts[0])
            continue

        first = cell_texts[0]
        if len(cell_texts) >= 2 and _NUM_RE.match(first):
            second_cell = cells[1] if len(cells) > 1 else None
            body = " ".join(cell_texts[1:])
            is_heading = False

            if second_cell is not None and len(body) < 120:
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


def _strip_junk(doc) -> None:
    """Remove non-content elements from the document tree in-place."""

    # Remove by tag name
    for tag in _STRIP_TAGS:
        for el in list(doc.iter(tag)):
            if el.getparent() is not None:
                el.getparent().remove(el)

    # Remove by class/id pattern
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

    # Handle images: preserve alt text if meaningful, remove element
    for el in list(doc.iter("img")):
        alt = (el.get("alt") or "").strip()
        parent = el.getparent()
        if parent is not None:
            if alt and len(alt) > 3 and alt.lower() not in (
                "image", "img", "photo", "icon", "logo", "banner",
                "spacer", "pixel", "dot",
            ):
                span = etree.Element("span")
                span.text = f"[Image: {alt}]"
                parent.replace(el, span)
            else:
                parent.remove(el)

    # Remove remaining media
    for tag in ("figure", "figcaption", "picture",
                "video", "audio", "source", "object", "embed"):
        for el in list(doc.iter(tag)):
            if el.getparent() is not None:
                el.getparent().remove(el)


def _preprocess_headings(doc) -> None:
    """Detect CSS-styled headings and convert to semantic h1-h6."""
    _HEADING_CLASS_RE = re.compile(
        r"(heading|title|section-?head|chapter-?head|topic-?head)",
        re.IGNORECASE,
    )

    for el in list(doc.iter("div", "p", "span")):
        # role="heading" with aria-level
        if el.get("role") == "heading":
            level = el.get("aria-level", "2")
            try:
                level = int(level)
            except ValueError:
                level = 2
            level = max(1, min(6, level))
            el.tag = f"h{level}"
            continue

        # Class-based heading detection
        cls = el.get("class") or ""
        if _HEADING_CLASS_RE.search(cls):
            text = el.text_content().strip()
            if text and len(text) < 150:
                level = 2
                if "h1" in cls.lower() or "title" in cls.lower():
                    level = 1
                elif "h3" in cls.lower() or "sub" in cls.lower():
                    level = 3
                el.tag = f"h{level}"


def _html_to_markdown(html: str) -> str:
    """Convert HTML to structured markdown, stripping non-content."""
    if not html or not html.strip():
        return ""

    try:
        doc = fromstring(html)
    except Exception:
        from lxml.html import document_fromstring
        doc = document_fromstring(html)

    # Phase 1: Strip junk
    _strip_junk(doc)

    # Phase 2: Detect CSS-styled headings
    _preprocess_headings(doc)

    # Phase 3: Find content root
    content = _find_content_root(doc)
    if content is None:
        content = doc

    # Phase 4: Process tables (bottom-up for nested handling)
    all_tables = list(content.iter("table"))
    all_tables.reverse()

    for table in all_tables:
        parent = table.getparent()
        if parent is None:
            continue

        if _is_layout_table(table):
            flat_text = _flatten_layout_table(table)
            new_div = etree.Element("div")
            new_div.set("class", "_layout_flattened")
            for para in flat_text.split("\n\n"):
                if para.strip():
                    p = etree.SubElement(new_div, "p")
                    p.text = para
            parent.replace(table, new_div)
        else:
            md_table = _data_table_to_markdown(table)
            new_div = etree.Element("div")
            new_div.set("class", "_md_table")
            pre = etree.SubElement(new_div, "pre")
            pre.set("class", "_preserve_md_table")
            pre.text = md_table
            parent.replace(table, new_div)

    # Phase 5: Convert to markdown
    content_html = tostring(content, encoding="unicode", method="html")

    # Stash pre-rendered markdown tables so markdownify doesn't mangle them.
    # Use alphanumeric-only sentinels to avoid underscore escaping.
    table_blocks = {}
    _counter = [0]

    def _stash_table(m):
        key = f"MDTABLEPLACEHOLDER{_counter[0]}END"
        _counter[0] += 1
        table_blocks[key] = m.group(1)
        return key

    content_html = re.sub(
        r'<pre class="_preserve_md_table">(.*?)</pre>',
        _stash_table,
        content_html,
        flags=re.DOTALL,
    )

    try:
        from markdownify import MarkdownConverter

        class BetterConverter(MarkdownConverter):
            """Extended converter with dl, details, abbr support."""
            def convert_dl(self, el, text, **kwargs):
                return "\n" + text + "\n"

            def convert_dt(self, el, text, **kwargs):
                return f"\n**{text.strip()}**\n"

            def convert_dd(self, el, text, **kwargs):
                return f": {text.strip()}\n"

            def convert_details(self, el, text, **kwargs):
                summary_el = el.find("summary")
                summary = summary_el.text_content().strip() if summary_el is not None else "Details"
                body = text
                if summary_el is not None:
                    body = body.replace(summary, "", 1).strip()
                return f"\n**{summary}**\n\n{body}\n"

            def convert_abbr(self, el, text, **kwargs):
                title = el.get("title", "")
                if title:
                    return f"{text} ({title})"
                return text

        markdown = BetterConverter(
            heading_style="ATX",
            bullets="-",
            strip=["img", "figure", "figcaption",
                   "picture", "video", "audio", "source",
                   "object", "embed", "iframe"],
        ).convert(content_html) or ""

    except ImportError:
        raise RuntimeError("markdownify is required — pip install markdownify")

    if not markdown:
        return ""

    # Restore stashed tables
    for key, table_md in table_blocks.items():
        markdown = markdown.replace(key, "\n\n" + table_md + "\n\n")

    return _clean_markdown(markdown)


def _clean_markdown(text: str) -> str:
    """Normalize whitespace and remove artifacts from converted markdown."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse 3+ blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip decorative separators (keep markdown HRs and table separators)
        if re.match(r"^[-_=~*]{4,}$", stripped) and stripped != "---" and "|" not in stripped:
            continue
        # Skip zero-width / nbsp-only lines
        if stripped and all(c in "\u00a0\u200b\u200c\u200d\ufeff" for c in stripped):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)

    # Normalize spaces (preserve code blocks and table rows)
    out_lines = []
    in_code_block = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            out_lines.append(line)
            continue

        if in_code_block:
            out_lines.append(line)
        elif line.startswith("    ") or line.startswith("\t"):
            out_lines.append(line)
        elif stripped.startswith("|"):
            out_lines.append(line.rstrip())
        else:
            out_lines.append(re.sub(r"  +", " ", line).rstrip())

    text = "\n".join(out_lines).strip()

    # Clean redundant bold inside headings: "## **Text**" → "## Text"
    text = re.sub(
        r"^(#{1,6})\s+\*\*(.+?)\*\*\s*$",
        r"\1 \2",
        text,
        flags=re.MULTILINE,
    )

    # Remove empty headings
    text = re.sub(r"^#{1,6}\s*$", "", text, flags=re.MULTILINE)

    # Final whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


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
