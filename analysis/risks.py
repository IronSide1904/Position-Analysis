from __future__ import annotations

import re

import pandas as pd


RISK_KEYWORDS = {
    "competition",
    "competitive",
    "customer",
    "demand",
    "litigation",
    "regulatory",
    "supply",
    "cybersecurity",
    "security",
    "depend",
    "concentration",
    "margin",
    "revenue",
    "market",
    "economic",
    "acquisition",
    "integration",
    "goodwill",
    "impairment",
    "debt",
    "liquidity",
}


def _looks_like_xbrl_fragment(text: str) -> bool:
    lower = text.lower()
    if any(token in lower for token in ["http://", "https://", "us-gaap:", "fasb.org", "0000", "#"]):
        return True
    digit_count = sum(char.isdigit() for char in text)
    slash_count = text.count("/") + text.count("#")
    return digit_count > 25 or slash_count > 4


def _clean_risk_text(text: str) -> str | None:
    clean = re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n-•​")
    clean = re.sub(r"^item\s+\d+[a-z]?\.?\s*", "", clean, flags=re.IGNORECASE)
    clean = clean.strip()
    if not clean or len(clean) < 35 or _looks_like_xbrl_fragment(clean):
        return None
    words = re.findall(r"[A-Za-z]{3,}", clean)
    if len(words) < 7:
        return None
    lower = clean.lower()
    if "risk factors" == lower or "market risk" == lower or "quantitative and qualitative disclosures about market risk" in lower:
        return None
    if not any(keyword in lower for keyword in RISK_KEYWORDS):
        return None
    return clean[:320]


def _risk_explanation(risk_text: str) -> str:
    lower = risk_text.lower()
    if any(token in lower for token in ["competition", "competitive", "pricing"]):
        return "Competitive pressure can reduce pricing power, margins, growth durability, and terminal multiple support."
    if any(token in lower for token in ["customer", "demand", "concentration"]):
        return "Customer or demand risk can reduce revenue visibility and make the modeled growth case less reliable."
    if any(token in lower for token in ["litigation", "regulatory", "compliance"]):
        return "Legal or regulatory risk can raise costs, increase WACC, and reduce valuation confidence."
    if any(token in lower for token in ["acquisition", "integration", "goodwill", "impairment"]):
        return "M&A risk can pressure margins, create impairment risk, and weaken confidence in acquired growth."
    if any(token in lower for token in ["supply", "inventory", "vendor"]):
        return "Supply-chain risk can pressure revenue timing, gross margin, and working-capital needs."
    if any(token in lower for token in ["debt", "liquidity", "interest"]):
        return "Balance-sheet risk can raise discount-rate assumptions and reduce equity value after net debt."
    return "This risk should be translated into scenario probability, WACC, margins, or growth assumptions before sizing."


def analyze_risks_and_thesis_breakers(filing_texts: dict, clauses: pd.DataFrame, historicals: pd.DataFrame) -> dict:
    """
    Translate filing risk factors into investment consequences and measurable thesis breakers.
    """
    top_risks = []
    seen = set()
    text = " ".join((filing_texts or {}).values())
    for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)):
        lower = sentence.lower()
        if "risk" in lower or "competition" in lower or "litigation" in lower or "supply" in lower:
            clean = _clean_risk_text(sentence)
            if clean and clean.lower() not in seen:
                top_risks.append(clean)
                seen.add(clean.lower())
        if len(top_risks) >= 8:
            break
    if clauses is not None and not clauses.empty:
        for clause in clauses[clauses["topic"].eq("Risk")]["clause_text"].head(8).tolist():
            clean = _clean_risk_text(clause)
            if clean and clean.lower() not in seen:
                top_risks.append(clean)
                seen.add(clean.lower())
            if len(top_risks) >= 8:
                break
    risk_rows = [
        {
            "risk": risk,
            "explanation": _risk_explanation(risk),
            "model_line": "Revenue / margins" if any(token in risk.lower() for token in ["competition", "customer", "demand", "supply"]) else "WACC / confidence",
            "review_action": "Review latest 10-K/10-Q risk factor language and decide whether Base, Bear, or User Case assumptions should change.",
        }
        for risk in top_risks
    ]
    thesis_breakers = [
        "Revenue growth falls below the modeled CAGR for two consecutive reporting periods.",
        "OCF conversion remains weak while CAPEX intensity rises.",
        "Moat evidence fails to support terminal value assumptions.",
    ]
    return {
        "risk_score": max(1, 10 - min(len(top_risks), 8)),
        "top_risks": top_risks or ["Risk extraction requires clean filing evidence; use manual review until risk text is available."],
        "risk_rows": risk_rows,
        "thesis_breakers": thesis_breakers,
        "bear_case_implications": ["Lower terminal multiple, higher WACC, and wider margin of safety."],
    }
