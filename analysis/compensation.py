from __future__ import annotations

import pandas as pd


def analyze_compensation_alignment(filing_texts: dict, historicals: pd.DataFrame) -> dict:
    """
    Analyze management compensation, SBC, dilution, and shareholder alignment.
    """
    proxy = filing_texts.get("DEF 14A", "") if filing_texts else ""
    revenue = historicals["Revenue"].iloc[-1] if historicals is not None and not historicals.empty and "Revenue" in historicals else 0
    sbc = historicals["SBC"].iloc[-1] if historicals is not None and not historicals.empty and "SBC" in historicals else 0
    shares = historicals["Diluted Shares"].iloc[-1] if historicals is not None and not historicals.empty and "Diluted Shares" in historicals else None
    sbc_pct_revenue = sbc / revenue if revenue else None
    red_flags = []
    if sbc_pct_revenue and sbc_pct_revenue > 0.1:
        red_flags.append("SBC exceeds 10% of revenue.")
    if not proxy:
        red_flags.append("Proxy unavailable; compensation alignment requires manual review.")
    score = 6 - min(len(red_flags), 4)
    compensation_table = pd.DataFrame(
        [{"metric": "Proxy availability", "value": "Available" if proxy else "Unavailable"}, {"metric": "Diluted shares", "value": shares}]
    )
    sbc_table = pd.DataFrame([{"metric": "SBC", "value": sbc}, {"metric": "SBC as % revenue", "value": sbc_pct_revenue}])
    return {
        "alignment_score": max(score, 1),
        "summary": "Alignment view emphasizes SBC, dilution, and proxy availability.",
        "compensation_table": compensation_table,
        "sbc_table": sbc_table,
        "red_flags": red_flags,
        "model_implications": ["Heavy SBC should reduce per-share fair value and confidence."],
    }

