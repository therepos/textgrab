"""MHTML → clean structured markdown.

Extracts content from MHTML web archives, stripping navigation, banners,
scripts, styles, and other non-content elements. Preserves headings, lists,
tables, and text structure as clean markdown suitable for knowledge bases
and LLM prompt context.

Pipeline: MHTML → MIME unwrap → HTML → content extraction → markdown
"""

import email
import logging
import re
from typing import Dict, Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scheme metadata (required by registry)
# ---------------------------------------------------------------------------
LABEL = "MHTML → Clean Markdown"
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

    # Walk MIME parts, prefer text/html
    for part in msg.walk():
        ct = part.get_content_type()
        if ct == "text/html":
            payload = part.get_payload(decode=True)
            if payload:
                # Detect charset from Content-Type header
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
    "nav", "header", "footer", "aside", "form", "button", "input",
    "select", "textarea", "label", "fieldset", "legend",
    "menu", "menuitem",
}

# Class/id substrings that indicate non-content blocks
_JUNK_PATTERNS = re.compile(
    r"(nav|menu|sidebar|footer|header|banner|breadcrumb|cookie|"
    r"advertisement|advert|social|share|related|comment|"
    r"popup|modal|overlay|toast|toolbar|ribbon|masthead|"
    r"skip-to|jump-to|back-to-top)",
    re.IGNORECASE,
)


def _html_to_markdown(html: str) -> str:
    """Convert HTML to structured markdown, stripping non-content."""
    try:
        from lxml import etree
        from lxml.html import fromstring, tostring
    except ImportError:
        raise RuntimeError("lxml is required — pip install lxml")

    try:
        doc = fromstring(html)
    except Exception:
        # Malformed HTML fallback
        from lxml.html import document_fromstring
        doc = document_fromstring(html)

    # --- Phase 1: Remove junk elements ---

    # Remove by tag name
    for tag in _STRIP_TAGS:
        for el in doc.iter(tag):
            el.getparent().remove(el) if el.getparent() is not None else None

    # Remove by class/id heuristics
    for el in doc.iter():
        attrs = " ".join(filter(None, [
            el.get("class", ""),
            el.get("id", ""),
            el.get("role", ""),
        ]))
        if attrs and _JUNK_PATTERNS.search(attrs):
            if el.getparent() is not None:
                el.getparent().remove(el)

    # Remove hidden elements
    for el in doc.iter():
        style = el.get("style", "")
        if "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", ""):
            if el.getparent() is not None:
                el.getparent().remove(el)

    # Remove images, figures (we only want text content)
    for tag in ("img", "figure", "figcaption", "picture", "video", "audio", "source", "object", "embed"):
        for el in doc.iter(tag):
            if el.getparent() is not None:
                el.getparent().remove(el)

    # --- Phase 2: Try trafilatura for content extraction ---
    cleaned_html = tostring(doc, encoding="unicode", method="html")
    markdown = None

    try:
        import trafilatura
        extracted = trafilatura.extract(
            cleaned_html,
            include_tables=True,
            include_links=False,
            include_images=False,
            include_comments=False,
            output_format="txt",
            favor_precision=False,
            favor_recall=True,
        )
        if extracted and len(extracted.strip()) > 100:
            markdown = extracted
    except Exception as e:
        log.warning(f"trafilatura extraction failed, falling back to markdownify: {e}")

    # --- Phase 3: Fallback to markdownify ---
    if not markdown:
        try:
            from markdownify import markdownify as md
            markdown = md(
                cleaned_html,
                heading_style="ATX",
                bullets="-",
                strip=["a"],
                convert=["h1", "h2", "h3", "h4", "h5", "h6",
                         "p", "br", "li", "ul", "ol",
                         "table", "thead", "tbody", "tr", "th", "td",
                         "blockquote", "pre", "code",
                         "strong", "em", "b", "i",
                         "dl", "dt", "dd", "hr"],
            )
        except ImportError:
            raise RuntimeError("markdownify is required — pip install markdownify")

    if not markdown:
        return ""

    # --- Phase 4: Clean up ---
    return _clean_markdown(markdown)


def _clean_markdown(text: str) -> str:
    """Normalize whitespace and remove artifacts from converted markdown."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse runs of 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that are only whitespace or dashes/underscores (decorative)
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

    # Normalize multiple spaces within lines (but preserve leading whitespace for code)
    out_lines = []
    for line in text.split("\n"):
        if line.startswith("    ") or line.startswith("\t"):
            out_lines.append(line)  # preserve code blocks
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
