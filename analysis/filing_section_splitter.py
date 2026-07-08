from __future__ import annotations

import re

SECTION_PATTERNS: list[tuple[str, str]] = [
    ("Business", r"(?:^|\n)\s*(?:item\s+1\.?\s*)business\b"),
    ("Risk Factors", r"(?:^|\n)\s*(?:part\s+ii[, ]*)?(?:item\s+1a\.?\s*)risk factors\b"),
    ("Properties", r"(?:^|\n)\s*(?:item\s+2\.?\s*)properties\b"),
    ("Legal Proceedings", r"(?:^|\n)\s*(?:part\s+ii[, ]*)?(?:item\s+1\.?\s*)legal proceedings\b"),
    ("MD&A", r"(?:^|\n)\s*(?:part\s+i[, ]*)?(?:item\s+2\.?|item\s+7\.?)\s*management'?s discussion and analysis\b"),
    ("Liquidity and Capital Resources", r"(?:^|\n).{0,80}liquidity and capital resources\b"),
    ("Market Risk", r"(?:^|\n)\s*(?:item\s+7a\.?|item\s+3\.?)\s*quantitative and qualitative disclosures\b"),
    ("Financial Statements", r"(?:^|\n)\s*(?:item\s+8\.?|item\s+1\.?)\s*financial statements\b"),
    ("Controls and Procedures", r"(?:^|\n)\s*(?:item\s+9a\.?|item\s+4\.?)\s*controls and procedures\b"),
    ("Executive Compensation", r"(?:^|\n).{0,80}(executive compensation|compensation discussion and analysis|pay versus performance)\b"),
    ("Security Ownership", r"(?:^|\n).{0,80}security ownership\b"),
    ("Notes", r"(?:^|\n).{0,80}(notes to consolidated financial statements|revenue recognition|segment information)\b"),
    ("Business Combinations", r"(?:^|\n).{0,80}(business combinations|goodwill and intangible assets|share repurchases|stock-based compensation)\b"),
]


def detect_section_headings(text: str) -> list[dict]:
    """
    Detect common SEC section headings.
    """
    if not text:
        return []
    matches = []
    seen = set()
    for section, pattern in SECTION_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            key = (section, match.start())
            if key in seen:
                continue
            seen.add(key)
            heading = re.sub(r"\s+", " ", match.group(0)).strip()
            matches.append({"section": section, "start": match.start(), "end": match.end(), "heading_text": heading})
    matches.sort(key=lambda item: item["start"])
    return matches


def split_filing_into_sections(text: str) -> dict:
    """
    Split filing text into named sections with a broad fallback.
    """
    if not text:
        return {"Unknown": "", "_warnings": ["Clause extraction unavailable: empty filing text."]}
    headings = detect_section_headings(text)
    if not headings:
        return {"Unknown": text, "_warnings": ["Section detection failed; using full filing text."]}
    sections: dict[str, str] = {"_warnings": []}
    for index, heading in enumerate(headings):
        start = heading["end"]
        stop = headings[index + 1]["start"] if index + 1 < len(headings) else len(text)
        section_text = text[start:stop].strip()
        if not section_text:
            continue
        current = sections.get(heading["section"], "")
        sections[heading["section"]] = f"{current}\n\n{section_text}".strip() if current else section_text
    if not any(key != "_warnings" and value for key, value in sections.items()):
        return {"Unknown": text, "_warnings": ["Section detection produced no usable sections."]}
    return sections
