from __future__ import annotations

import re


def analyze_management_and_board(filing_texts: dict, submissions: dict) -> dict:
    """
    Analyze management credibility, founder involvement, CEO style, board quality, and governance.
    """
    text = " ".join((filing_texts or {}).values()).lower()
    strengths = []
    red_flags = []
    style = "Unknown"
    if "founder" in text:
        strengths.append("Founder involvement mentioned in filings.")
        style = "Founder-compounder"
    if "independent director" in text or "independent board" in text:
        strengths.append("Board independence language present.")
    if "material weakness" in text:
        red_flags.append("Material weakness language detected.")
    if "acquisition" in text and style == "Unknown":
        style = "Acquisition-led CEO"
    if re.search(r"\bguidance\b|\boutlook\b", text) and style == "Unknown":
        style = "Financial operator"
    score = 5 + min(len(strengths), 3) - min(len(red_flags) * 2, 4)
    return {
        "management_score": max(min(score, 10), 1),
        "style": style,
        "summary": "Management assessment is filing-text based and should be manually reviewed.",
        "strengths": strengths,
        "red_flags": red_flags,
        "model_implications": ["Use management score to adjust scenario probability and confidence."],
    }

