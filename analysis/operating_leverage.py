from __future__ import annotations

import pandas as pd


def analyze_operating_leverage(historicals: pd.DataFrame, peers: pd.DataFrame | None) -> dict:
    """
    Analyze scalability and operating leverage.
    """
    if historicals is None or historicals.empty:
        return {"score": 4, "classification": "Unknown", "summary": "Not enough data available.", "metrics": {}, "red_flags": []}
    row = historicals.iloc[-1]
    revenue = row.get("Revenue") or 0
    opex = row.get("OPEX") or 0
    operating_margin = row.get("EBIT", 0) / revenue if revenue else None
    opex_ratio = opex / revenue if revenue else None
    score = 5
    if operating_margin is not None:
        score += 2 if operating_margin > 0.2 else 1 if operating_margin > 0.1 else -1
    if opex_ratio is not None and opex_ratio < 0.35:
        score += 1
    classification = "Premium leverage" if score >= 8 else "Average leverage" if score >= 6 else "Weak leverage" if score >= 4 else "Deteriorating"
    return {
        "score": max(min(score, 10), 1),
        "classification": classification,
        "summary": f"{classification} based on reported operating margin and OPEX ratio.",
        "metrics": {"operating_margin": operating_margin, "opex_ratio": opex_ratio},
        "red_flags": [] if score >= 5 else ["Operating margin does not yet show clear scalability."],
    }

