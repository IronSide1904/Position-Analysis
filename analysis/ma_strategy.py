from __future__ import annotations

import re

import pandas as pd


def analyze_ma_strategy(filing_texts: dict, historicals: pd.DataFrame) -> dict:
    """
    Analyze past, present, and future M&A strategy.
    """
    rows = []
    red_flags = []
    for filing_type, text in (filing_texts or {}).items():
        for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text or "")):
            lower = sentence.lower()
            if any(k in lower for k in ["acquisition", "business combination", "goodwill", "intangible", "impairment"]):
                rows.append({"filing": filing_type, "event": sentence[:500], "topic": "M&A disclosure"})
                if "impairment" in lower:
                    red_flags.append("Filing mentions impairment tied to acquired assets or goodwill.")
    timeline = pd.DataFrame(rows)
    if timeline.empty:
        return {
            "ma_quality_score": 5,
            "classification": "Insufficient data",
            "timeline": timeline,
            "summary": "No reliable M&A disclosures were extracted.",
            "red_flags": [],
            "dcf_implications": ["Do not assume acquisition synergies without evidence."],
        }
    score = 6 - min(len(red_flags), 3)
    return {
        "ma_quality_score": max(score, 1),
        "classification": "Neutral" if not red_flags else "Revenue-padding",
        "timeline": timeline.head(20),
        "summary": "M&A language was found; review whether revenue contribution, margins, and goodwill support value creation.",
        "red_flags": red_flags,
        "dcf_implications": ["Stress test goodwill/intangible risk and acquisition-related margin pressure."],
    }

