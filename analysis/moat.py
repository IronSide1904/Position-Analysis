from __future__ import annotations

import pandas as pd


MOAT_SOURCES = [
    "Switching costs",
    "Network effects",
    "Scale advantage",
    "Brand / trust",
    "Technology / IP / patents",
    "Distribution advantage",
    "Data advantage",
    "Regulatory barriers",
    "Cost advantage",
    "Ecosystem / platform lock-in",
    "Customer embeddedness",
    "Capital intensity / entry barriers",
]


def _score_source(source: str, text: str, historicals: pd.DataFrame) -> dict:
    lower = text.lower()
    keywords = {
        "Switching costs": ["switching", "mission critical", "embedded"],
        "Network effects": ["network effect", "marketplace", "ecosystem"],
        "Scale advantage": ["scale", "large customer base", "cost advantage"],
        "Brand / trust": ["brand", "trust", "reputation"],
        "Technology / IP / patents": ["patent", "proprietary", "technology"],
        "Distribution advantage": ["distribution", "channel", "partner"],
        "Data advantage": ["data", "dataset", "analytics"],
        "Regulatory barriers": ["regulatory", "certification", "license"],
        "Cost advantage": ["low cost", "cost advantage", "efficiency"],
        "Ecosystem / platform lock-in": ["platform", "integration", "ecosystem"],
        "Customer embeddedness": ["long-term contract", "embedded", "retention"],
        "Capital intensity / entry barriers": ["capital intensive", "manufacturing capacity", "facility"],
    }
    hits = [k for k in keywords[source] if k in lower]
    score = 3 + min(len(hits) * 2, 4)
    evidence = "; ".join(hits) if hits else "No direct KPI found; proxy-based assessment only."
    confidence = "Medium" if hits else "Low"
    return {
        "moat_source": source,
        "score_1_to_10": score,
        "evidence": evidence,
        "counter_evidence": "Manual review required for churn, market share, pricing power, and peer margin durability.",
        "confidence": confidence,
        "model_implication": "Can support higher terminal multiple only with measurable evidence." if hits else "Do not raise terminal assumptions from this source alone.",
    }


def run_new_entrant_test(dataset: dict, filing_texts: dict, peers: pd.DataFrame | None = None) -> dict:
    """
    Test how difficult it would be for a new entrant to compete with the company.
    """
    text = " ".join((filing_texts or {}).values()).lower()
    advantages = []
    risks = []
    for word, label in [
        ("certification", "Certifications or approvals may slow entrants."),
        ("integration", "Product integrations may increase switching friction."),
        ("scale", "Scale may improve unit economics."),
        ("brand", "Brand/trust may reduce customer switching."),
        ("patent", "IP protection may block imitation."),
    ]:
        if word in text:
            advantages.append(label)
    if "competition" in text or "competitive" in text:
        risks.append("Filings emphasize competitive pressure.")
    score = 4 + min(len(advantages), 5) - min(len(risks), 2)
    return {
        "entry_barrier_score": max(min(score, 10), 1),
        "summary": "Entrant test is based on filing evidence and should be validated against market structure.",
        "entrant_risks": risks,
        "incumbent_advantages": advantages,
        "model_implications": ["Higher terminal value requires strong entry barriers and durable margins."],
    }


def analyze_moat(dataset: dict, historicals: pd.DataFrame, filing_texts: dict, peers: pd.DataFrame | None = None, clauses: pd.DataFrame | None = None) -> dict:
    """
    Analyze competitive advantage and classify moat strength.
    """
    text = " ".join((filing_texts or {}).values())
    source_rows = [_score_source(source, text, historicals) for source in MOAT_SOURCES]
    sources = pd.DataFrame(source_rows)
    direct = sources[sources["confidence"] != "Low"]
    if direct.empty:
        score = 3
        classification = "Unknown / insufficient data"
        confidence = "Low"
    else:
        score = float(sources["score_1_to_10"].mean())
        classification = "Wide moat" if score >= 8 else "Narrow moat" if score >= 6 else "Emerging moat" if score >= 4.5 else "Weak moat"
        confidence = "Medium"
    entrant = run_new_entrant_test(dataset, filing_texts, peers)
    red_flags = []
    if classification in {"Weak moat", "Unknown / insufficient data"}:
        red_flags.append("Moat evidence is weak or unproven; avoid aggressive terminal assumptions.")
    return {
        "moat_score": round(score, 1),
        "classification": classification,
        "summary": "Moat assessment ties claims to filing keywords and available proxies; missing KPIs are not inferred.",
        "moat_sources": sources,
        "new_entrant_test": entrant,
        "evidence": direct["evidence"].tolist() if not direct.empty else [],
        "counter_evidence": ["Missing direct evidence for retention, churn, pricing power, market share, and peer margin gap."],
        "red_flags": red_flags,
        "peer_context": "Peer context unavailable." if peers is None or peers.empty else "Peer context included.",
        "dcf_implications": ["Moat should affect terminal growth, terminal multiple, WACC, long-term margins, and margin-of-safety requirement."],
        "terminal_value_implication": "Do not rely on high terminal value unless moat evidence improves." if score < 5 else "Moat evidence can support moderate terminal assumptions.",
        "confidence": confidence,
    }

