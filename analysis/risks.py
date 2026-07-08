from __future__ import annotations

import re

import pandas as pd


def analyze_risks_and_thesis_breakers(filing_texts: dict, clauses: pd.DataFrame, historicals: pd.DataFrame) -> dict:
    """
    Translate filing risk factors into investment consequences and measurable thesis breakers.
    """
    top_risks = []
    text = " ".join((filing_texts or {}).values())
    for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)):
        lower = sentence.lower()
        if "risk" in lower or "competition" in lower or "litigation" in lower or "supply" in lower:
            top_risks.append(sentence[:400])
        if len(top_risks) >= 8:
            break
    if clauses is not None and not clauses.empty:
        top_risks.extend(clauses[clauses["topic"].eq("Risk")]["clause_text"].head(5).tolist())
    thesis_breakers = [
        "Revenue growth falls below the modeled CAGR for two consecutive reporting periods.",
        "OCF conversion remains weak while CAPEX intensity rises.",
        "Moat evidence fails to support terminal value assumptions.",
    ]
    return {
        "risk_score": max(1, 7 - min(len(top_risks), 6)),
        "top_risks": top_risks or ["Not enough data available."],
        "thesis_breakers": thesis_breakers,
        "bear_case_implications": ["Lower terminal multiple, higher WACC, and wider margin of safety."],
    }

