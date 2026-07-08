from __future__ import annotations

import re

import pandas as pd


TOPIC_RULES = [
    ("Revenue", ["revenue", "sales", "customer demand", "backlog"], "revenue_growth"),
    ("Backlog", ["backlog", "remaining performance obligation", "rpo"], "revenue_growth"),
    ("Gross Margin", ["gross margin", "cost of revenue", "cost of sales"], "gross_margin"),
    ("OPEX", ["operating expense", "sales and marketing", "research and development"], "opex_ratio"),
    ("NOPAT", ["operating income", "operating margin", "tax"], "nopat_margin"),
    ("OCF", ["operating cash flow", "cash provided by operating"], "ocf_margin"),
    ("Maintenance CAPEX", ["maintenance capital", "replacement", "capital expenditures"], "maintenance_capex_pct_revenue"),
    ("Growth CAPEX", ["capacity expansion", "new facility", "growth capital", "capital expenditures"], "growth_capex_pct_revenue"),
    ("Working Capital", ["inventory", "receivable", "deferred revenue", "contract asset"], "working_capital_pct_revenue"),
    ("Debt", ["debt", "credit facility", "notes payable"], "wacc"),
    ("Dilution", ["dilution", "shares outstanding", "equity award"], "diluted_shares"),
    ("SBC", ["stock-based compensation", "share-based compensation"], "diluted_shares"),
    ("M&A", ["acquisition", "business combination", "goodwill", "intangible"], "scenario_probability"),
    ("Moat", ["competition", "competitive", "patent", "network", "switching"], "terminal_multiple"),
    ("Risk", ["risk factor", "legal proceeding", "litigation"], "wacc"),
    ("Management Credibility", ["guidance", "outlook", "executive compensation"], "scenario_probability"),
]

SECTION_HINTS = {
    "MD&A": ["management discussion", "md&a"],
    "Liquidity and Capital Resources": ["liquidity and capital resources"],
    "Risk Factors": ["risk factors"],
    "Executive Compensation": ["executive compensation", "compensation discussion"],
    "Business Combinations": ["business combination", "acquisition"],
    "Segment Notes": ["segment"],
}


def _sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text or "")
    return re.split(r"(?<=[.!?])\s+", compact)


def _section(sentence: str) -> str:
    lower = sentence.lower()
    for section, hints in SECTION_HINTS.items():
        if any(h in lower for h in hints):
            return section
    return "Filing text"


def _classify(sentence: str):
    lower = sentence.lower()
    for topic, keywords, model_line in TOPIC_RULES:
        if any(k in lower for k in keywords):
            direction = "positive" if any(w in lower for w in ["increase", "improve", "growth", "higher", "expanded"]) else "negative" if any(w in lower for w in ["decline", "risk", "decrease", "lower", "impairment"]) else "mixed"
            confidence = "Medium" if len(sentence) > 80 else "Low"
            return topic, model_line, direction, confidence
    return None, None, None, None


def extract_relevant_clauses(filing_texts: dict) -> pd.DataFrame:
    """
    Return table of clauses relevant to valuation.
    """
    rows = []
    for filing_type, text in (filing_texts or {}).items():
        for sentence in _sentences(text)[:2500]:
            topic, model_line, direction, confidence = _classify(sentence)
            if not topic:
                continue
            rows.append(
                {
                    "source": "SEC filing",
                    "filing_type": filing_type,
                    "section": _section(sentence),
                    "clause_text": sentence[:900],
                    "topic": topic,
                    "model_line_affected": model_line,
                    "direction": direction,
                    "confidence": confidence,
                    "suggested_assumption_change": "review manually" if confidence == "Low" else f"Consider {direction} adjustment to {model_line}.",
                    "evidence_grade": "Reported" if filing_type in {"10-K", "10-Q", "DEF 14A"} else "Proxy-based",
                }
            )
    columns = [
        "source",
        "filing_type",
        "section",
        "clause_text",
        "topic",
        "model_line_affected",
        "direction",
        "confidence",
        "suggested_assumption_change",
        "evidence_grade",
    ]
    return pd.DataFrame(rows, columns=columns)

