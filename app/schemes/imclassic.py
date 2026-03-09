"""IM Classic scheme — converts Government Instruction Manual MHTML content
into structured markdown with hierarchical coding.

Handles the standard IM format: Sections → Purpose → Operating Principles →
Topics → Rules / Good Practice Guides / FAQs.

Works across all IM domains (Procurement, Revenue, Asset Management, etc.)
as long as they follow the classic IM structure.
"""

import re
import json
from typing import Dict, List, Tuple, Any

# --- Required interface ---
LABEL = "IM Classic"
ACCEPTS = [".mhtml", ".mht"]
MULTI_FILE = True
OUTPUT_OPTIONS = ["consolidated", "individual"]


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
GUID_PATTERN = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"
)
TOPIC_LINE = re.compile(r"^(.+?)SB[\s\xa0]+Full[\s\xa0]+Compliance$")
RULE_REF = re.compile(r"^(S\d+|EP\d*|OP)\s+(R\d+(?:\.\d+)*)$")
GUIDE_REF = re.compile(r"^(S\d+|EP\d*|OP)\s+(G\d+(?:\.\d+)*)$")
RULE_REF_AMEND = re.compile(
    r"^(S\d+|EP\d*|OP)\s+(R\d+(?:\.\d+)*)\s+\[?(Amended|Inserted|Added)"
)
GUIDE_REF_AMEND = re.compile(
    r"^(S\d+|EP\d*|OP)\s+(G\d+(?:\.\d+)*)\s+\[?(Amended|Inserted|Added)"
)
PRINCIPLE_REF = re.compile(r"^(P\d+)$")
FAQ_REF = re.compile(r"^\[(S\d+|EP\d*|OP)\s+(FAQ\s+[\d\.]+)\]")
ANS_REF = re.compile(r"^\[(S\d+|EP\d*|OP)\s+((?:Ans|ANS)\s+[\d\.]+)\]")
SECTION_HEADER = re.compile(
    r"^Instruction Manuals\s*-\s*PrintSection$", re.IGNORECASE
)

# Revenue-style patterns (paragraph-numbered rules, no stage prefix on FAQs)
PARA_RULE_REF = re.compile(r"^#(\d+)$")
PARA_RULE_REF_AMEND = re.compile(
    r"^#(\d+)\s*[-:]\s*(Amended|Inserted|Added)\s+", re.IGNORECASE
)
PARA_RULE_INLINE = re.compile(r"^Paragraph\s*:\s*#(\d+)")
NONSTAGE_FAQ_REF = re.compile(r"^\[FAQ\s+(\d+)\]")
NONSTAGE_ANS_REF = re.compile(r"^\[Ans\s+(\d+)\]", re.IGNORECASE)
NAMED_PRINCIPLE = re.compile(r"^(P\d+)\s+(.+)$")
FAQ_COLON = re.compile(r"^FAQ:$")


def _is_noise(line: str) -> bool:
    if GUID_PATTERN.match(line):
        return True
    if SECTION_HEADER.match(line):
        return True
    if line == "SECTION":
        return True
    if re.match(
        r"^\d+\)\s*#?\d+\s+(Amended|Inserted|Added)\s+by\s+FC",
        line, re.IGNORECASE,
    ):
        return True
    if re.match(
        r"^\[S\d+\s+FAQ\s*&\s*Ans\s+[\d\.]+\s*(to\s+FAQ|Amended|Inserted)",
        line,
    ):
        return True
    return False


def _is_amendment_line(line: str) -> bool:
    if re.match(
        r"^(?:S\d+|EP\d*|OP)\s+[RG]\d+[\.\d]*"
        r"(\s+and\s+[RG]\d+[\.\d]*)?\s+(Amended|Inserted|Added)\s+",
        line, re.IGNORECASE,
    ):
        return True
    if re.match(
        r"^#\s*[\d,\s]+\s*(Amended|Inserted|Added)\s+(by|via)\s+FC",
        line, re.IGNORECASE,
    ):
        return True
    if re.match(
        r"^#\d+\s*[-:]\s*(Amended|Inserted|Added)\s+",
        line, re.IGNORECASE,
    ):
        return True
    if re.match(
        r"^#\d+\s*(to|and|&)\s*#\d+\s*(Amended|Inserted|Added|removed)\s+",
        line, re.IGNORECASE,
    ):
        return True
    if re.match(
        r"^#\d+\s*-\s*#\d+\s*(amended|removed)\s+via",
        line, re.IGNORECASE,
    ):
        return True
    if re.match(
        r"^FAQs?\s+\d+\s*[-–]\s*\d+\s+(inserted|amended|added)\s+via",
        line, re.IGNORECASE,
    ):
        return True
    if re.match(
        r"^Paragraph\s*:\s*#\d+",
        line, re.IGNORECASE,
    ):
        return True
    if re.match(r"^\[Request for Proposal", line):
        return True
    return False


def _flush_node(node, buffer):
    if node and buffer:
        text = " ".join(buffer)
        if node.get("type") == "FAQ":
            if not node.get("answer"):
                node["answer"] = text
        else:
            node["text"] = text
    return []


def _detect_stage_info(lines: List[str]) -> Tuple[str, str, str]:
    """Try to detect stage number, name, and code prefix from the content.
    
    Returns:
        (stage_num, stage_name, code_prefix)
        For stage-based IMs: ("02", "Determining Procurement Approach", "IM-P")
        For non-stage IMs: ("3G", "Revenue Contracting Procedures", "IM-P")
    """
    # Known non-stage IM mappings (section name -> (code_suffix, display_name))
    NON_STAGE_IMS = {
        "Revenue Contracting Procedures": ("3G", "Revenue Contracting Procedures"),
        "Asset Management": ("3H", "Asset Management"),
    }

    for line in lines[:15]:
        m = TOPIC_LINE.match(line)
        if m:
            name = m.group(1).strip()
            # Check for non-stage IM
            for key, (suffix, display) in NON_STAGE_IMS.items():
                if key.lower() in name.lower():
                    return suffix, display, "IM-P"
            # Stage-based IM
            if name.startswith("Stage "):
                parts = name.split(" - ", 1)
                if len(parts) == 2:
                    num = parts[0].replace("Stage ", "").strip().zfill(2)
                    return num, parts[1].strip(), "IM-P"
            # Fallback for other non-stage sections
            return "00", name, "IM-P"
    return "00", "Unknown", "IM-P"


def _parse_im_text(lines: List[str], stage_num: str, stage_name: str,
                   code_prefix: str = "IM-P"):
    """Parse IM text lines into structured document and glossary.
    
    Handles two formats:
    - Procurement-style: rules as 'S2 R1.1', FAQs as '[S2 FAQ1.1]'
    - Revenue-style: rules as '#41', FAQs as '[FAQ 3]' / '[Ans 3]'
    """
    document = {
        "stage": stage_num,
        "stage_name": stage_name,
        "purpose": [],
        "principles": [],
        "topics": [],
    }

    # Detect if this is a paragraph-numbered IM (Revenue-style)
    para_count = sum(1 for l in lines if PARA_RULE_REF.match(l))
    staged_count = sum(1 for l in lines if RULE_REF.match(l))
    is_para_style = para_count > staged_count

    glossary = []
    phase = "start"
    current_topic = None
    current_type = None
    current_node = None
    text_buf = []
    topic_counter = 0

    prefix = f"{code_prefix}{stage_num}"

    i = 0
    while i < len(lines):
        line = lines[i]

        if _is_noise(line):
            i += 1
            continue
        if _is_amendment_line(line):
            i += 1
            continue

        # Phase transitions
        if line == "PURPOSE":
            text_buf = _flush_node(current_node, text_buf)
            phase = "purpose"
            i += 1
            continue

        if line == "OPERATING PRINCIPLES":
            text_buf = _flush_node(current_node, text_buf)
            phase = "principles"
            current_node = None
            i += 1
            continue

        if line == "TOPIC":
            text_buf = _flush_node(current_node, text_buf)
            phase = "topics"
            current_node = None
            i += 1
            continue

        # Topic header
        topic_match = TOPIC_LINE.match(line)
        if topic_match:
            topic_name = topic_match.group(1).strip()
            if topic_name.startswith("Stage ") or topic_name in (
                stage_name, f"{stage_name}"
            ):
                i += 1
                continue
            text_buf = _flush_node(current_node, text_buf)
            phase = "topics"
            topic_counter += 1
            current_topic = {
                "number": topic_counter,
                "name": topic_name,
                "rules": [],
                "guides": [],
                "faqs": [],
            }
            document["topics"].append(current_topic)
            current_type = None
            current_node = None
            i += 1
            continue

        # Content type markers
        if line == "Rules":
            text_buf = _flush_node(current_node, text_buf)
            current_type = "rules"
            current_node = None
            i += 1
            continue

        if line.startswith("Good Practice Guid"):
            text_buf = _flush_node(current_node, text_buf)
            current_type = "guides"
            current_node = None
            i += 1
            continue

        if line.startswith("FAQ/Case Stud") or line == "FAQ/Case Studies":
            text_buf = _flush_node(current_node, text_buf)
            current_type = "faq"
            current_node = None
            i += 1
            continue

        if FAQ_COLON.match(line):
            # Revenue-style "FAQ:" marker within FAQ/Case Studies
            text_buf = _flush_node(current_node, text_buf)
            current_type = "faq"
            current_node = None
            i += 1
            continue

        if line == "Resources" or line == "Appendices":
            text_buf = _flush_node(current_node, text_buf)
            current_type = "resources"
            current_node = None
            i += 1
            continue

        # PURPOSE phase
        if phase == "purpose":
            if re.match(r"^\d+$", line):
                i += 1
                continue
            document["purpose"].append(line)
            i += 1
            continue

        # PRINCIPLES phase
        if phase == "principles":
            # Named principle (Revenue-style): "P1 Underlying Principles"
            np_match = NAMED_PRINCIPLE.match(line)
            if np_match:
                text_buf = _flush_node(current_node, text_buf)
                p_id = np_match.group(1)
                p_name = np_match.group(2).strip()
                code = f"{prefix}-{p_id}"
                current_node = {
                    "code": code, "id": p_id, "name": p_name,
                    "text": "", "type": "Principle",
                }
                document["principles"].append(current_node)
                glossary.append({
                    "code": code,
                    "path": f"{stage_name} > Operating Principles > {p_id}: {p_name}",
                    "type": "Operating Principle",
                    "original_ref": p_id,
                })
                i += 1
                continue

            # Simple principle (Procurement-style): "P1"
            p_match = PRINCIPLE_REF.match(line)
            if p_match:
                text_buf = _flush_node(current_node, text_buf)
                p_id = p_match.group(1)
                code = f"{prefix}-{p_id}"
                current_node = {"code": code, "id": p_id, "text": "", "type": "Principle"}
                document["principles"].append(current_node)
                glossary.append({
                    "code": code,
                    "path": f"Stage {stage_num} {stage_name} > Operating Principles > {p_id}",
                    "type": "Operating Principle",
                    "original_ref": p_id,
                })
                i += 1
                continue

            if current_node:
                text_buf.append(line)
                i += 1
                continue
            else:
                i += 1
                continue

        # TOPICS phase: Rule references (Procurement-style: "S2 R1.1")
        r_match = RULE_REF.match(line)
        if r_match and phase == "topics":
            text_buf = _flush_node(current_node, text_buf)
            stage_ref = r_match.group(1)
            rule_id = r_match.group(2)

            if not current_topic:
                topic_counter += 1
                current_topic = {
                    "number": topic_counter, "name": "General",
                    "rules": [], "guides": [], "faqs": [],
                }
                document["topics"].append(current_topic)

            code = f"{prefix}-T{current_topic['number']}-{rule_id}"
            current_node = {
                "code": code,
                "ref": f"{stage_ref} {rule_id}",
                "text": "",
                "type": "Rule",
            }
            current_topic["rules"].append(current_node)
            glossary.append({
                "code": code,
                "path": f"Stage {stage_num} {stage_name} > {current_topic['name']} > Rule {rule_id}",
                "type": "Rule",
                "original_ref": f"{stage_ref} {rule_id}",
            })
            i += 1
            continue

        # TOPICS phase: Paragraph-style rules (Revenue-style: "#41")
        p_rule_match = PARA_RULE_REF.match(line)
        if p_rule_match and phase == "topics" and current_type == "rules" and is_para_style:
            text_buf = _flush_node(current_node, text_buf)
            para_num = p_rule_match.group(1)

            if not current_topic:
                topic_counter += 1
                current_topic = {
                    "number": topic_counter, "name": "General",
                    "rules": [], "guides": [], "faqs": [],
                }
                document["topics"].append(current_topic)

            code = f"{prefix}-T{current_topic['number']}-R{para_num}"
            current_node = {
                "code": code,
                "ref": f"#{para_num}",
                "text": "",
                "type": "Rule",
            }
            current_topic["rules"].append(current_node)
            glossary.append({
                "code": code,
                "path": f"{stage_name} > {current_topic['name']} > Rule #{para_num}",
                "type": "Rule",
                "original_ref": f"#{para_num}",
            })
            i += 1
            continue

        # Guide references
        g_match = GUIDE_REF.match(line)
        if g_match and phase == "topics":
            text_buf = _flush_node(current_node, text_buf)
            stage_ref = g_match.group(1)
            guide_id = g_match.group(2)

            if not current_topic:
                topic_counter += 1
                current_topic = {
                    "number": topic_counter, "name": "General",
                    "rules": [], "guides": [], "faqs": [],
                }
                document["topics"].append(current_topic)

            code = f"{prefix}-T{current_topic['number']}-{guide_id}"
            current_node = {
                "code": code,
                "ref": f"{stage_ref} {guide_id}",
                "text": "",
                "type": "Good Practice Guide",
            }
            current_topic["guides"].append(current_node)
            glossary.append({
                "code": code,
                "path": f"Stage {stage_num} {stage_name} > {current_topic['name']} > Guide {guide_id}",
                "type": "Good Practice Guide",
                "original_ref": f"{stage_ref} {guide_id}",
            })
            i += 1
            continue

        # FAQ references (Procurement-style: "[S2 FAQ1.1]")
        faq_match = FAQ_REF.match(line)
        ans_match = ANS_REF.match(line)

        # Revenue-style FAQ: "[FAQ 3]"
        ns_faq_match = NONSTAGE_FAQ_REF.match(line) if not faq_match else None
        ns_ans_match = NONSTAGE_ANS_REF.match(line) if not ans_match else None

        if (faq_match or ns_faq_match) and phase == "topics":
            text_buf = _flush_node(current_node, text_buf)

            if not current_topic:
                topic_counter += 1
                current_topic = {
                    "number": topic_counter, "name": "General",
                    "rules": [], "guides": [], "faqs": [],
                }
                document["topics"].append(current_topic)

            if faq_match:
                stage_ref = faq_match.group(1)
                faq_id = faq_match.group(2).replace(" ", "")
                code = f"{prefix}-T{current_topic['number']}-{faq_id}"
                ref_str = f"{stage_ref} {faq_id}"
            else:
                faq_num = ns_faq_match.group(1)
                faq_id = f"FAQ{faq_num}"
                code = f"{prefix}-T{current_topic['number']}-{faq_id}"
                ref_str = f"FAQ {faq_num}"

            i += 1
            q_lines = []
            while i < len(lines):
                if _is_noise(lines[i]):
                    i += 1
                    continue
                if (ANS_REF.match(lines[i]) or FAQ_REF.match(lines[i])
                        or NONSTAGE_ANS_REF.match(lines[i])
                        or NONSTAGE_FAQ_REF.match(lines[i])
                        or TOPIC_LINE.match(lines[i])):
                    break
                if lines[i].startswith("[S") and ("Ans" in lines[i] or "FAQ" in lines[i]):
                    break
                if lines[i] in ("Rules", "Resources", "Appendices") or lines[i].startswith("Good Practice") or lines[i].startswith("FAQ/Case"):
                    break
                if FAQ_COLON.match(lines[i]):
                    break
                q_lines.append(lines[i])
                i += 1

            current_node = {
                "code": code,
                "ref": ref_str,
                "question": " ".join(q_lines),
                "answer": "",
                "type": "FAQ",
            }
            current_topic["faqs"].append(current_node)
            text_buf = []

            glossary.append({
                "code": code,
                "path": f"{stage_name} > {current_topic['name']} > {faq_id}",
                "type": "FAQ",
                "original_ref": ref_str,
            })
            continue

        if (ans_match or ns_ans_match) and current_node and current_node.get("type") == "FAQ":
            if text_buf and not current_node.get("question"):
                current_node["question"] = " ".join(text_buf)
            text_buf = []
            i += 1
            continue

        # Content accumulation
        if current_node and phase == "topics":
            if re.match(r"^[SG]\d+$", line) and len(line) <= 3:
                i += 1
                continue
            if re.match(r"^[GR]\d+\.\d+$", line) and len(line) <= 5:
                i += 1
                continue
            if re.match(r"^\[.*(?:Inserted|Amended|Added)\s+(?:by|via)\s+FC.*\]$", line):
                i += 1
                continue
            text_buf.append(line)

        i += 1

    _flush_node(current_node, text_buf)
    return document, glossary


def _generate_markdown(doc: dict) -> str:
    """Generate structured markdown from parsed document."""
    md = []
    md.append(f"# IM — Stage {doc['stage']}: {doc['stage_name']}")
    md.append("")

    if doc["purpose"]:
        md.append("## Purpose")
        md.append("")
        for p in doc["purpose"]:
            md.append(p)
            md.append("")

    if doc["principles"]:
        md.append("## Operating Principles")
        md.append("")
        for p in doc["principles"]:
            md.append(f"### {p['code']} [{p['id']}]")
            md.append("")
            md.append(p.get("text", ""))
            md.append("")

    for topic in doc["topics"]:
        md.append(f"## Topic {topic['number']}: {topic['name']}")
        md.append("")

        if topic["rules"]:
            md.append("### Rules")
            md.append("")
            for r in topic["rules"]:
                md.append(f"**{r['code']}** [{r['ref']}]")
                md.append("")
                if r.get("text"):
                    md.append(r["text"])
                    md.append("")

        if topic["guides"]:
            md.append("### Good Practice Guides")
            md.append("")
            for g in topic["guides"]:
                md.append(f"**{g['code']}** [{g['ref']}]")
                md.append("")
                if g.get("text"):
                    md.append(g["text"])
                    md.append("")

        if topic["faqs"]:
            md.append("### FAQ / Case Studies")
            md.append("")
            for f in topic["faqs"]:
                md.append(f"**{f['code']}** [{f['ref']}]")
                md.append("")
                if f.get("question"):
                    md.append(f"**Q:** {f['question']}")
                    md.append("")
                if f.get("answer"):
                    md.append(f"**A:** {f['answer']}")
                    md.append("")

    return "\n".join(md)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def transform(texts: Dict[str, str], output_mode: str = "consolidated") -> dict:
    """Transform extracted MHTML text into structured IM markdown.

    Args:
        texts: {filename: raw_extracted_text, ...}
        output_mode: "consolidated" or "individual"

    Returns:
        {"text": str, "filename": str} for consolidated
        {"text": str, "filename": str, "files": [...]} for individual
    """
    all_glossary = []
    all_documents = []
    individual_files = []

    for filename, raw_text in sorted(texts.items()):
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        stage_num, stage_name, code_prefix = _detect_stage_info(lines)
        doc, glossary = _parse_im_text(lines, stage_num, stage_name, code_prefix)
        all_documents.append(doc)
        all_glossary.extend(glossary)

        md = _generate_markdown(doc)
        topics = len(doc["topics"])
        rules = sum(len(t["rules"]) for t in doc["topics"])
        guides = sum(len(t["guides"]) for t in doc["topics"])
        faqs = sum(len(t["faqs"]) for t in doc["topics"])

        individual_files.append({
            "filename": filename.replace(".mhtml", ".md").replace(".mht", ".md"),
            "text": md,
            "stats": {
                "stage": stage_num,
                "stage_name": stage_name,
                "topics": topics,
                "rules": rules,
                "guides": guides,
                "faqs": faqs,
                "glossary_entries": len(glossary),
            },
        })

    # Build glossary markdown
    glossary_md = [
        "", "---", "",
        "# Glossary & Code Reference", "",
        "| Code | Original Ref | Type | Full Path |",
        "|------|-------------|------|-----------|",
    ]
    for e in sorted(all_glossary, key=lambda x: x["code"]):
        glossary_md.append(
            f"| {e['code']} | {e.get('original_ref', '')} | {e['type']} | {e['path']} |"
        )

    if output_mode == "individual":
        return {
            "text": "",
            "filename": "",
            "files": individual_files,
            "glossary": "\n".join(glossary_md),
            "total_entries": len(all_glossary),
        }

    # Consolidated
    consolidated = [
        "# Government Instruction Manual (Complete)", "",
        "Structured with hierarchical codes for precise reference.",
        "Glossary at the end maps each code to its readable path.", "",
        "---", "",
    ]
    for f in individual_files:
        consolidated.append(f["text"])
        consolidated.extend(["", "---", ""])

    consolidated.append("\n".join(glossary_md))

    stats_summary = []
    for f in individual_files:
        s = f["stats"]
        stats_summary.append(
            f"Stage {s['stage']} ({s['stage_name']}): "
            f"{s['rules']}R / {s['guides']}G / {s['faqs']}FAQ"
        )

    return {
        "text": "\n".join(consolidated),
        "filename": "im_complete.md",
        "stats": stats_summary,
        "total_entries": len(all_glossary),
    }
