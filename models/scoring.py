from __future__ import annotations

import pandas as pd


WEIGHTS = {
    "Business quality": 0.15,
    "Financial quality": 0.15,
    "Operating leverage": 0.10,
    "Moat": 0.10,
    "Management": 0.10,
    "Capital allocation / M&A": 0.10,
    "SBC / alignment": 0.08,
    "Valuation": 0.12,
    "Positioning / price action": 0.10,
}


def score_investment(inputs: dict) -> dict:
    dcf = inputs.get("dcf", {})
    reverse = inputs.get("reverse_dcf", {})
    moat = inputs.get("moat", {})
    management = inputs.get("management", {})
    ma = inputs.get("ma", {})
    alignment = inputs.get("alignment", {})
    quality = inputs.get("quality", {})
    leverage = inputs.get("operating_leverage", {})

    upside = dcf.get("upside_downside_pct") or 0
    valuation_score = min(max(5 + upside * 20, 1), 10)
    if reverse.get("market_case") == "Extreme Bull":
        valuation_score = min(valuation_score, 5)
    moat_score = moat.get("moat_score") or 4
    if moat_score < 4:
        valuation_score = min(valuation_score, 6)

    raw = {
        "Business quality": quality.get("quality_score", 5),
        "Financial quality": quality.get("quality_score", 5),
        "Operating leverage": leverage.get("score", 5),
        "Moat": moat_score,
        "Management": management.get("management_score", 5),
        "Capital allocation / M&A": ma.get("ma_quality_score", 5),
        "SBC / alignment": alignment.get("alignment_score", 5),
        "Valuation": valuation_score,
        "Positioning / price action": inputs.get("price_action_score", 5),
    }
    scorecard = pd.DataFrame(
        [{"category": k, "score_1_to_10": v, "weight": WEIGHTS[k], "weighted_score": v * 10 * WEIGHTS[k]} for k, v in raw.items()]
    )
    total = float(scorecard["weighted_score"].sum())
    evidence_ok = moat.get("confidence") not in {"Low", "Unknown"} and quality.get("quality_score", 5) >= 5
    if total >= 72 and upside > 0.15 and evidence_ok:
        recommendation = "Buy"
    elif total >= 50:
        recommendation = "Watchlist"
    else:
        recommendation = "Avoid"
    conviction = "Very High" if total >= 82 and evidence_ok else "High" if total >= 72 else "Medium" if total >= 55 else "Low"
    return {
        "total_score": total,
        "recommendation": recommendation,
        "conviction": conviction,
        "position_size_guidance": "Starter position only unless valuation, evidence quality, and risk controls all improve." if recommendation != "Buy" else "Consider normal sizing subject to portfolio risk limits.",
        "scorecard": scorecard,
    }

