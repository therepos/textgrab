"""SSO Statute scheme — converts Singapore Statutes Online MHTML content
into structured markdown with hierarchical coding for AI consumption.

Handles:
  - Acts (e.g. Companies Act, Charities Act, Societies Act)
  - Subsidiary Legislation / Regulations (e.g. Charities (IPC) Regulations)

All SSO pages share the same HTML template with consistent CSS classes:
  prov1/prov1Hdr/prov1Txt  — sections
  prov2Txt/prov2TxtIL      — subsections
  p1No/pTxt               — paragraphs (a),(b)
  p2No/p2Txt              — sub-paragraphs (i),(ii)
  p3No/p3Txt              — sub-sub-paragraphs (A),(B)
  partNo/partHdr          — Part headings
  divtitle                — Division headings (Companies Act)
  amendNote               — amendment references
  prov1Rep                — repealed sections (omitted)
"""

import re
import html as html_mod
import email
from typing import Dict, List, Tuple, Optional, Any
from html.parser import HTMLParser

# --- Required interface ---
LABEL = "SSO Statute"
ACCEPTS = [".mhtml", ".mht"]
MULTI_FILE = True
OUTPUT_OPTIONS = ["consolidated", "individual"]

# ---------------------------------------------------------------------------
# Prefix configuration — maps title keywords to short prefixes
# ---------------------------------------------------------------------------
PREFIX_MAP = [
    # Subsidiary legislation first (more specific matches)
    ("Charities (Accounts and Annual Report) Regulations", "CA-AAR"),
    ("Charities (Electronic Transactions Service) Regulations", "CA-ETS"),
    ("Charities (Fund-Raising Appeals", "CA-FRA"),
    ("Charities (Fund-raising Appeals", "CA-FRA"),
    ("Charities (Institutions of A Public Character) Regulations", "CA-IPCR"),
    ("Charities (Large Charities) Regulations", "CA-LCR"),
    ("Charities (Registration of Charities) Regulations", "CA-RCR"),
    # Acts
    ("Charities Act", "CA"),
    ("Societies Act", "SA"),
    ("Companies Act", "CoA"),
]


def _detect_prefix(title: str) -> Tuple[str, str]:
    """Detect prefix and clean title from the SSO page title.

    Returns (prefix, clean_title).
    """
    # Strip " - Singapore Statutes Online" suffix
    clean = re.sub(r"\s*-\s*Singapore Statutes Online\s*$", "", title).strip()
    for keyword, prefix in PREFIX_MAP:
        if keyword.lower() in clean.lower():
            return prefix, clean
    # Fallback: generate from initials
    words = [w for w in clean.split() if w[0].isupper()]
    fallback = "".join(w[0] for w in words[:4])
    return fallback, clean


# ---------------------------------------------------------------------------
# HTML Parser — extracts structured provisions from SSO HTML
# ---------------------------------------------------------------------------
class SSOParser(HTMLParser):
    """Event-driven HTML parser that builds a structured document from SSO HTML.

    Produces a list of 'elements' in document order, each tagged with its type:
      part, division, section_header, section_start, subsection,
      subsection_inline, paragraph, sub_paragraph, sub_sub_paragraph,
      definition, amendment, repealed, proviso, schedule_header,
      schedule_content, text
    """

    # Tags whose closing inserts a logical break
    BLOCK_TAGS = {
        "p", "div", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "dt", "dd", "blockquote",
    }

    def __init__(self):
        super().__init__()
        self.elements: List[Dict[str, Any]] = []
        self._class_stack: List[str] = []
        self._current_classes: str = ""
        self._text_buf: List[str] = []
        self._skip = False
        self._in_toc = False
        self._tag_stack: List[str] = []

    def _flush_text(self):
        text = "".join(self._text_buf).strip()
        text = re.sub(r"\s+", " ", text)
        self._text_buf = []
        return text

    def _flush_into_last(self):
        """Flush accumulated text into the most recent element."""
        if self.elements and self._text_buf:
            text = self._flush_text()
            if text:
                last = self.elements[-1]
                if last["text"]:
                    last["text"] += " " + text
                else:
                    last["text"] = text

    def handle_starttag(self, tag, attrs):
        self._tag_stack.append(tag)
        attr_dict = dict(attrs)
        classes = attr_dict.get("class", "")
        self._current_classes = classes
        tag_id = attr_dict.get("id", "")

        if tag in ("script", "style"):
            self._skip = True
            return

        # Skip Table of Contents
        if "TocParagraph" in classes:
            self._in_toc = True
            return

        # Part number
        if "partNo" in classes:
            self._flush_into_last()
            self.elements.append({"type": "part_start", "text": "", "id": tag_id})
            return

        # Part header
        if "partHdr" in classes:
            self._flush_into_last()
            self.elements.append({"type": "part_header", "text": "", "id": tag_id})
            return

        # Division title
        if "divtitle" in classes:
            self._flush_into_last()
            self.elements.append({"type": "division", "text": "", "id": tag_id})
            return

        # Section header (title of section)
        if "prov1Hdr" in classes:
            self._flush_into_last()
            self.elements.append({
                "type": "section_header", "text": "", "id": tag_id,
            })
            return

        # Repealed section
        if "prov1Rep" in classes:
            self._flush_into_last()
            self.elements.append({"type": "repealed", "text": "", "id": tag_id})
            return

        # Section body start
        if "prov1Txt" in classes:
            self._flush_into_last()
            self.elements.append({"type": "section_body", "text": ""})
            return

        # Subsection
        if "prov2Txt" in classes and "prov2TxtIL" not in classes:
            self._flush_into_last()
            self.elements.append({"type": "subsection", "text": ""})
            return

        # Subsection inline (first subsection, follows section number)
        if "prov2TxtIL" in classes:
            self._flush_into_last()
            self.elements.append({"type": "subsection_inline", "text": ""})
            return

        # Paragraph (a),(b) level
        if "p1No" in classes:
            self._flush_into_last()
            self.elements.append({"type": "p1_num", "text": ""})
            return
        if "pTxt" in classes:
            self._flush_into_last()
            self.elements.append({"type": "p1_text", "text": ""})
            return

        # Sub-paragraph (i),(ii) level
        if "p2No" in classes:
            self._flush_into_last()
            self.elements.append({"type": "p2_num", "text": ""})
            return
        if "p2Txt" in classes:
            self._flush_into_last()
            self.elements.append({"type": "p2_text", "text": ""})
            return

        # Sub-sub-paragraph (A),(B) level
        if "p3No" in classes:
            self._flush_into_last()
            self.elements.append({"type": "p3_num", "text": ""})
            return
        if "p3Txt" in classes:
            self._flush_into_last()
            self.elements.append({"type": "p3_text", "text": ""})
            return

        # Definition paragraph
        if "p1DefNo" in classes:
            self._flush_into_last()
            self.elements.append({"type": "def_num", "text": ""})
            return
        if "pDefTxt" in classes:
            self._flush_into_last()
            self.elements.append({"type": "def_text", "text": ""})
            return

        # Amendment note
        if "amendNote" in classes:
            self._flush_into_last()
            self.elements.append({"type": "amendment", "text": ""})
            return

        # Deleted/repealed definitions
        if "definitionRepealed" in classes:
            self._flush_into_last()
            self.elements.append({"type": "repealed_def", "text": ""})
            return

        # Proviso
        if "proviso" in classes:
            self._flush_into_last()
            self.elements.append({"type": "proviso", "text": ""})
            return

        # Schedule headers
        if "sHdr" in classes:
            self._flush_into_last()
            self.elements.append({"type": "schedule_header", "text": ""})
            return
        if "scHdr" in classes:
            self._flush_into_last()
            self.elements.append({"type": "schedule_subheader", "text": ""})
            return
        if "sGrpHdrCaps" in classes:
            self._flush_into_last()
            self.elements.append({"type": "schedule_group", "text": ""})
            return

        # Schedule body text
        if "s1Txt" in classes or "tailSTxt" in classes:
            self._flush_into_last()
            self.elements.append({"type": "schedule_text", "text": ""})
            return
        if "sTxtNoLeft" in classes or "sIndentNo" in classes:
            self._flush_into_last()
            self.elements.append({"type": "schedule_num", "text": ""})
            return

        # Schedule provision paragraphs
        if "sProvP1No" in classes:
            self._flush_into_last()
            self.elements.append({"type": "schedule_p1_num", "text": ""})
            return
        if "sProvP1" in classes and "sProvP1No" not in classes:
            self._flush_into_last()
            self.elements.append({"type": "schedule_p1_text", "text": ""})
            return

        # Indent text in schedules
        if "pIndentTxt" in classes:
            self._flush_into_last()
            self.elements.append({"type": "schedule_text", "text": ""})
            return

    def handle_endtag(self, tag):
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()

        if tag in ("script", "style"):
            self._skip = False
            return

        if self._in_toc and tag in ("tr", "div"):
            self._in_toc = False
            return

        # Flush text into the most recent element
        self._flush_into_last()

    def handle_data(self, data):
        if self._skip or self._in_toc:
            return
        self._text_buf.append(data)

    def handle_entityref(self, name):
        if self._skip or self._in_toc:
            return
        char = html_mod.unescape(f"&{name};")
        self._text_buf.append(char)

    def handle_charref(self, name):
        if self._skip or self._in_toc:
            return
        char = html_mod.unescape(f"&#{name};")
        self._text_buf.append(char)


# ---------------------------------------------------------------------------
# Document builder — converts parsed elements into structured document
# ---------------------------------------------------------------------------
def _build_document(elements: List[Dict], prefix: str, title: str) -> Tuple[dict, List[dict]]:
    """Build structured document and glossary from parsed HTML elements.

    Returns (document, glossary).
    """
    doc = {
        "title": title,
        "prefix": prefix,
        "parts": [],
        "sections": [],
        "schedules": [],
    }
    glossary = []

    current_part_num = ""
    current_part_name = ""
    current_division = ""
    current_section_num = ""
    current_section_title = ""
    in_schedule = False
    schedule_buf = []

    i = 0
    while i < len(elements):
        el = elements[i]
        etype = el["type"]
        text = el.get("text", "").strip()

        # --- REPEALED: skip ---
        if etype in ("repealed", "repealed_def"):
            # Skip until next section or part
            i += 1
            while i < len(elements) and elements[i]["type"] not in (
                "section_header", "part_start", "part_header", "division",
                "schedule_header",
            ):
                i += 1
            continue

        # --- AMENDMENTS: collect for glossary ---
        if etype == "amendment":
            # Store amendment against current section
            amend_text = re.sub(r"[\[\]]", "", text).strip()
            if glossary and amend_text:
                # Attach to most recent glossary entry
                if glossary[-1].get("amended_by"):
                    glossary[-1]["amended_by"] += "; " + amend_text
                else:
                    glossary[-1]["amended_by"] = amend_text
            i += 1
            continue

        # --- SCHEDULE CONTENT ---
        if etype == "schedule_header":
            in_schedule = True
            schedule_buf.append(f"\n### {text}\n")
            i += 1
            continue

        if in_schedule and etype in (
            "schedule_subheader", "schedule_group", "schedule_text",
            "schedule_num", "schedule_p1_num", "schedule_p1_text",
        ):
            if etype == "schedule_subheader":
                schedule_buf.append(f"\n**{text}**\n")
            elif etype == "schedule_group":
                schedule_buf.append(f"\n#### {text}\n")
            elif etype == "schedule_num":
                schedule_buf.append(f"{text} ")
            elif etype == "schedule_text":
                schedule_buf.append(f"{text}\n")
            elif etype == "schedule_p1_num":
                schedule_buf.append(f"\n> {text} ")
            elif etype == "schedule_p1_text":
                schedule_buf.append(f"{text}\n")
            i += 1
            continue

        # If we hit a non-schedule element after being in schedule, close it
        if in_schedule and etype not in (
            "schedule_header", "schedule_subheader", "schedule_group",
            "schedule_text", "schedule_num", "schedule_p1_num",
            "schedule_p1_text", "amendment",
        ):
            if schedule_buf:
                doc["schedules"].append("".join(schedule_buf))
                schedule_buf = []
            in_schedule = False
            # Don't increment i — re-process this element

        # --- PART ---
        if etype == "part_start":
            current_part_num = text.strip()
            i += 1
            continue

        if etype == "part_header":
            current_part_name = text.strip()
            i += 1
            continue

        # --- DIVISION ---
        if etype == "division":
            current_division = text.strip()
            i += 1
            continue

        # --- SECTION HEADER (title) ---
        if etype == "section_header":
            current_section_title = text.strip()
            # Extract section number from the id: "pr7-" → "7", "pr5A-" → "5A"
            sec_id = el.get("id", "")
            m = re.match(r"pr(\d+[A-Z]*)-", sec_id)
            if m:
                current_section_num = m.group(1)
            i += 1
            continue

        # --- SECTION BODY ---
        if etype == "section_body":
            # Extract section number from <strong>N.</strong> pattern
            m = re.match(r"(\d+[A-Z]*)\.\s*(.*)", text)
            if m:
                sec_num = m.group(1)
                remainder = m.group(2).strip()
                current_section_num = sec_num

                section = {
                    "num": sec_num,
                    "title": current_section_title,
                    "part_num": current_part_num,
                    "part_name": current_part_name,
                    "division": current_division,
                    "subsections": [],
                }

                code = f"{prefix}-s{sec_num}"
                glossary_path = _build_path(
                    current_part_num, current_part_name,
                    current_division, current_section_title,
                )

                glossary.append({
                    "code": code,
                    "legal_cite": f"s.{sec_num}",
                    "type": "Section",
                    "path": glossary_path,
                    "amended_by": "",
                })

                # Check if remainder has inline subsection: "—(1) ..."
                if remainder:
                    inline_m = re.match(r"—\((\d+)\)\s*(.*)", remainder)
                    if inline_m:
                        ss_num = inline_m.group(1)
                        ss_text = inline_m.group(2).strip()
                        sub = {"num": ss_num, "text": ss_text, "paragraphs": []}
                        section["subsections"].append(sub)
                        sub_code = f"{prefix}-s{sec_num}({ss_num})"
                        glossary.append({
                            "code": sub_code,
                            "legal_cite": f"s.{sec_num}({ss_num})",
                            "type": "Subsection",
                            "path": f"{glossary_path} > ({ss_num})",
                            "amended_by": "",
                        })
                    elif not remainder.startswith("—"):
                        # Simple section with no subsections, text directly
                        sub = {"num": "", "text": remainder, "paragraphs": []}
                        section["subsections"].append(sub)

                doc["sections"].append(section)
            i += 1
            continue

        # --- SUBSECTION ---
        if etype in ("subsection", "subsection_inline"):
            if not doc["sections"]:
                i += 1
                continue
            section = doc["sections"][-1]

            # Parse subsection number: "(2) text..." or "—(1) text..."
            m = re.match(r"—?\((\d+[A-Z]*)\)\s*(.*)", text)
            if m:
                ss_num = m.group(1)
                ss_text = m.group(2).strip()
            else:
                ss_num = ""
                ss_text = text

            sub = {"num": ss_num, "text": ss_text, "paragraphs": []}
            section["subsections"].append(sub)

            if ss_num:
                sec_num = section["num"]
                sub_code = f"{prefix}-s{sec_num}({ss_num})"
                glossary_path = _build_path(
                    section["part_num"], section["part_name"],
                    section["division"], section["title"],
                )
                glossary.append({
                    "code": sub_code,
                    "legal_cite": f"s.{sec_num}({ss_num})",
                    "type": "Subsection",
                    "path": f"{glossary_path} > ({ss_num})",
                    "amended_by": "",
                })
            i += 1
            continue

        # --- PARAGRAPH (a),(b) ---
        if etype == "p1_num":
            # Next element should be p1_text
            p_label = _clean_para_label(text)
            p_text = ""
            if i + 1 < len(elements) and elements[i + 1]["type"] == "p1_text":
                p_text = elements[i + 1]["text"].strip()
                i += 2
            else:
                i += 1

            para = {"label": p_label, "text": p_text, "sub_paragraphs": []}

            if doc["sections"] and doc["sections"][-1]["subsections"]:
                current_sub = doc["sections"][-1]["subsections"][-1]
                current_sub["paragraphs"].append(para)

                sec_num = doc["sections"][-1]["num"]
                ss_num = current_sub["num"]
                ss_part = f"({ss_num})" if ss_num else ""
                p_code = f"{prefix}-s{sec_num}{ss_part}({p_label})"
                glossary_path = _build_path(
                    doc["sections"][-1]["part_num"],
                    doc["sections"][-1]["part_name"],
                    doc["sections"][-1]["division"],
                    doc["sections"][-1]["title"],
                )
                ss_path = f" > ({ss_num})" if ss_num else ""
                glossary.append({
                    "code": p_code,
                    "legal_cite": f"s.{sec_num}{ss_part}({p_label})",
                    "type": "Paragraph",
                    "path": f"{glossary_path}{ss_path} > ({p_label})",
                    "amended_by": "",
                })
            continue

        # --- DEFINITION paragraph ---
        if etype == "def_num":
            p_label = _clean_para_label(text)
            p_text = ""
            if i + 1 < len(elements) and elements[i + 1]["type"] == "def_text":
                p_text = elements[i + 1]["text"].strip()
                i += 2
            else:
                i += 1

            para = {"label": p_label, "text": p_text, "sub_paragraphs": []}
            if doc["sections"] and doc["sections"][-1]["subsections"]:
                doc["sections"][-1]["subsections"][-1]["paragraphs"].append(para)
            continue

        # --- SUB-PARAGRAPH (i),(ii) ---
        if etype == "p2_num":
            sp_label = _clean_para_label(text)
            sp_text = ""
            if i + 1 < len(elements) and elements[i + 1]["type"] == "p2_text":
                sp_text = elements[i + 1]["text"].strip()
                i += 2
            else:
                i += 1

            subpara = {"label": sp_label, "text": sp_text, "sub_sub_paragraphs": []}

            # Attach to last paragraph of last subsection
            if doc["sections"] and doc["sections"][-1]["subsections"]:
                current_sub = doc["sections"][-1]["subsections"][-1]
                if current_sub["paragraphs"]:
                    current_sub["paragraphs"][-1]["sub_paragraphs"].append(subpara)

                    sec_num = doc["sections"][-1]["num"]
                    ss_num = current_sub["num"]
                    p_label = current_sub["paragraphs"][-1]["label"]
                    ss_part = f"({ss_num})" if ss_num else ""
                    sp_code = f"{prefix}-s{sec_num}{ss_part}({p_label})({sp_label})"
                    glossary_path = _build_path(
                        doc["sections"][-1]["part_num"],
                        doc["sections"][-1]["part_name"],
                        doc["sections"][-1]["division"],
                        doc["sections"][-1]["title"],
                    )
                    ss_path = f" > ({ss_num})" if ss_num else ""
                    glossary.append({
                        "code": sp_code,
                        "legal_cite": f"s.{sec_num}{ss_part}({p_label})({sp_label})",
                        "type": "Sub-paragraph",
                        "path": f"{glossary_path}{ss_path} > ({p_label})({sp_label})",
                        "amended_by": "",
                    })
            continue

        # --- SUB-SUB-PARAGRAPH (A),(B) ---
        if etype == "p3_num":
            ssp_label = _clean_para_label(text)
            ssp_text = ""
            if i + 1 < len(elements) and elements[i + 1]["type"] == "p3_text":
                ssp_text = elements[i + 1]["text"].strip()
                i += 2
            else:
                i += 1

            # Attach to last sub-paragraph
            if doc["sections"] and doc["sections"][-1]["subsections"]:
                current_sub = doc["sections"][-1]["subsections"][-1]
                if current_sub["paragraphs"] and current_sub["paragraphs"][-1]["sub_paragraphs"]:
                    current_sub["paragraphs"][-1]["sub_paragraphs"][-1].setdefault(
                        "sub_sub_paragraphs", []
                    ).append({"label": ssp_label, "text": ssp_text})
            continue

        # --- PROVISO ---
        if etype == "proviso":
            # Treat proviso text as continuation of current subsection
            if doc["sections"] and doc["sections"][-1]["subsections"]:
                current_sub = doc["sections"][-1]["subsections"][-1]
                current_sub["text"] += " " + text
            i += 1
            continue

        i += 1

    # Flush any remaining schedule
    if schedule_buf:
        doc["schedules"].append("".join(schedule_buf))

    return doc, glossary


def _build_path(part_num: str, part_name: str, division: str, section_title: str) -> str:
    """Build a breadcrumb path string."""
    parts = []
    if part_num:
        clean_num = re.sub(r"^PART\s*", "", part_num, flags=re.IGNORECASE).strip()
        parts.append(f"Part {clean_num}")
        if part_name:
            parts[-1] += f" {part_name}"
    if division:
        parts.append(division)
    if section_title:
        parts.append(section_title)
    return " > ".join(parts) if parts else section_title or ""


def _clean_para_label(text: str) -> str:
    """Extract clean paragraph label from '(a)' or '(ii)' style text."""
    text = re.sub(r"<[^>]+>", "", text)  # strip any residual HTML
    text = text.strip()
    m = re.match(r"\(([^)]+)\)", text)
    if m:
        return m.group(1).strip()
    return text.strip("(). ")


# ---------------------------------------------------------------------------
# Markdown generator
# ---------------------------------------------------------------------------
def _generate_markdown(doc: dict) -> str:
    """Generate structured markdown from parsed document."""
    prefix = doc["prefix"]
    md = []

    md.append(f"# {doc['title']} — AI Reference Document")
    md.append("")
    md.append(f"> **Prefix:** `{prefix}`")

    # Detect structure description
    has_parts = any(s["part_num"] for s in doc["sections"])
    has_divisions = any(s["division"] for s in doc["sections"])
    if has_parts and has_divisions:
        structure = "Part → Division → Section → Subsection → Paragraph → Sub-paragraph"
    elif has_parts:
        structure = "Part → Section → Subsection → Paragraph → Sub-paragraph"
    else:
        structure = "Section → Subsection → Paragraph → Sub-paragraph"

    md.append(f"> **Structure:** {structure}")
    md.append(f"> **Citation format:** `{prefix}-s1(1)(a)` = {doc['title']}, section 1, subsection (1), paragraph (a)")
    md.append("")

    current_part = ""
    current_div = ""

    for section in doc["sections"]:
        # Emit Part heading if changed
        part_key = f"{section['part_num']}|{section['part_name']}"
        if part_key != current_part and section["part_num"]:
            current_part = part_key
            clean_num = re.sub(r"^PART\s*", "", section["part_num"], flags=re.IGNORECASE).strip()
            md.append(f"## Part {clean_num}: {section['part_name']}")
            md.append("")

        # Emit Division heading if changed
        if section["division"] and section["division"] != current_div:
            current_div = section["division"]
            md.append(f"### {current_div}")
            md.append("")

        # Section heading
        sec_num = section["num"]
        sec_title = section["title"]
        md.append(f"### {prefix}-s{sec_num} — {sec_title}")
        md.append("")

        # Subsections
        for sub in section["subsections"]:
            ss_num = sub["num"]
            ss_text = sub["text"]

            if ss_num:
                md.append(f"**{prefix}-s{sec_num}({ss_num})**")
            else:
                md.append(f"**{prefix}-s{sec_num}**")
            md.append(ss_text)
            md.append("")

            # Paragraphs
            for para in sub["paragraphs"]:
                md.append(f"> **({para['label']})** {para['text']}")
                md.append(">")

                # Sub-paragraphs
                for sp in para.get("sub_paragraphs", []):
                    md.append(f"> > **({sp['label']})** {sp['text']}")
                    md.append("> >")

                    # Sub-sub-paragraphs
                    for ssp in sp.get("sub_sub_paragraphs", []):
                        md.append(f"> > > **({ssp['label']})** {ssp['text']}")
                        md.append("> > >")

            md.append("")

    # Schedules
    if doc["schedules"]:
        md.append("## Appendix: Schedules")
        md.append("")
        for sched in doc["schedules"]:
            md.append(sched)
            md.append("")

    return "\n".join(md)


def _generate_glossary(glossary: List[dict], has_amendments: bool = True) -> str:
    """Generate glossary markdown table."""
    md = []
    md.append("## Glossary")
    md.append("")

    if has_amendments:
        md.append("| Code | Legal Cite | Type | Path | Amended by |")
        md.append("|------|-----------|------|------|------------|")
        for e in glossary:
            amended = e.get("amended_by", "")
            md.append(
                f"| {e['code']} | {e['legal_cite']} | {e['type']} "
                f"| {e['path']} | {amended} |"
            )
    else:
        md.append("| Code | Legal Cite | Type | Path |")
        md.append("|------|-----------|------|------|")
        for e in glossary:
            md.append(
                f"| {e['code']} | {e['legal_cite']} | {e['type']} | {e['path']} |"
            )

    return "\n".join(md)


def _generate_master_index(all_glossaries: List[Tuple[str, str, List[dict]]]) -> str:
    """Generate cross-act master index from multiple glossaries.

    all_glossaries: [(prefix, title, glossary_entries), ...]
    """
    md = []
    md.append("# Singapore Statutes — Cross-Act Master Index")
    md.append("")
    md.append("> This index spans all uploaded legislation. Use the unique prefix")
    md.append("> to disambiguate when a section number appears in more than one Act.")
    md.append("")
    md.append("| Code | Legislation | Legal Cite | Section Title |")
    md.append("|------|------------|-----------|---------------|")

    for prefix, title, glossary in all_glossaries:
        # Only include section-level entries in master index
        for e in glossary:
            if e["type"] == "Section":
                # Extract section title from path (last segment)
                sec_title = e["path"].rsplit(" > ", 1)[-1] if " > " in e["path"] else e["path"]
                md.append(
                    f"| {e['code']} | {title} | {e['legal_cite']} | {sec_title} |"
                )

    return "\n".join(md)


# ---------------------------------------------------------------------------
# MHTML extraction (mirrors extracttext._extract_from_mhtml but returns HTML)
# ---------------------------------------------------------------------------
def _extract_html_from_mhtml(raw_bytes: bytes) -> str:
    """Extract the HTML part from MHTML content."""
    text_content = raw_bytes.decode("utf-8", errors="ignore")
    msg = email.message_from_string(text_content)
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            payload = part.get_payload(decode=True)
            if payload:
                return payload.decode("utf-8", errors="ignore")
    # Fallback: treat entire content as HTML
    return text_content


def _extract_title(html: str) -> str:
    """Extract page title from HTML."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        return title
    return "Unknown Legislation"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def transform(texts: Dict[str, str], output_mode: str = "consolidated") -> dict:
    """Transform extracted MHTML text into structured statute markdown.

    Note: Unlike imclassic which works on stripped text, this scheme needs
    the raw MHTML bytes to parse HTML structure. The `texts` dict contains
    the text extracted by extracttext, but we need to re-read the files.
    Since extracttext passes us text (not HTML), we work with what we get
    by reconstructing from the text extraction. However, for best results,
    this scheme should receive the raw content.

    In practice, the textgrab pipeline calls extracttext first, then passes
    the result to the scheme. For MHTML files, extracttext strips HTML.
    We handle this by accepting either raw HTML or pre-stripped text.
    If the input looks like HTML (contains tags), we parse it as HTML.
    Otherwise, we attempt a text-based fallback.
    """
    all_glossaries = []
    individual_files = []

    for filename, raw_text in sorted(texts.items()):
        # Determine if we have HTML or stripped text
        has_html = bool(re.search(r"<(?:div|table|td|span)\s", raw_text[:5000]))

        if has_html:
            html_content = raw_text
        else:
            # Text was already stripped by extracttext.
            # We need the original HTML. Try to reconstruct from MHTML.
            # The scheme will need the raw bytes — check if the file is
            # available in the upload directory.
            import os
            upload_path = f"/tmp/textgrab_uploads/{filename}"
            alt_paths = [
                upload_path,
                f"/mnt/user-data/uploads/{filename}",
            ]
            html_content = None
            for path in alt_paths:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        html_content = _extract_html_from_mhtml(f.read())
                    break

            if not html_content:
                # Last resort: try treating the raw text as partial HTML
                html_content = raw_text

        # Extract title and prefix
        page_title = _extract_title(html_content)
        prefix, clean_title = _detect_prefix(page_title)

        # Parse HTML
        parser = SSOParser()
        parser.feed(html_content)

        # Flush any remaining text
        if parser._text_buf and parser.elements:
            text = parser._flush_text()
            if text:
                parser.elements[-1]["text"] += " " + text

        # Build document
        doc, glossary = _build_document(parser.elements, prefix, clean_title)
        doc["prefix"] = prefix

        # Generate markdown
        has_amendments = any(e.get("amended_by") for e in glossary)
        body_md = _generate_markdown(doc)
        glossary_md = _generate_glossary(glossary, has_amendments)
        full_md = body_md + "\n\n" + glossary_md

        # Stats
        section_count = len(doc["sections"])
        schedule_count = len(doc["schedules"])
        glossary_count = len(glossary)

        all_glossaries.append((prefix, clean_title, glossary))
        individual_files.append({
            "filename": filename.replace(".mhtml", ".md").replace(".mht", ".md"),
            "text": full_md,
            "stats": {
                "prefix": prefix,
                "title": clean_title,
                "sections": section_count,
                "schedules": schedule_count,
                "glossary_entries": glossary_count,
            },
        })

    if output_mode == "individual":
        return {
            "text": "",
            "filename": "",
            "files": individual_files,
            "total_entries": sum(len(g) for _, _, g in all_glossaries),
        }

    # Consolidated
    consolidated = []
    for f in individual_files:
        consolidated.append(f["text"])
        consolidated.append("")

    # Add master index
    if len(all_glossaries) > 1:
        master_index = _generate_master_index(all_glossaries)
        consolidated.append(master_index)

    stats_summary = []
    for f in individual_files:
        s = f["stats"]
        stats_summary.append(
            f"{s['prefix']} ({s['title']}): "
            f"{s['sections']} sections / {s['schedules']} schedules"
        )

    return {
        "text": "\n".join(consolidated),
        "filename": "sso_statutes_complete.md",
        "stats": stats_summary,
        "total_entries": sum(len(g) for _, _, g in all_glossaries),
    }
