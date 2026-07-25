from __future__ import annotations

import re

import pandas as pd


UNAVAILABLE = "Unavailable"


def _clip(text: object, limit: int = 420) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return UNAVAILABLE
    return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "..."


def _sentences(text: object, count: int = 2, limit: int = 420) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return UNAVAILABLE
    parts = re.split(r"(?<=[.!?])\s+", clean)
    return _clip(" ".join(parts[:count]), limit)


def _top_clause_rows(clauses: pd.DataFrame | None, limit: int = 6) -> list[dict]:
    if clauses is None or clauses.empty:
        return []
    rows = []
    for _, row in clauses.head(limit).iterrows():
        topic = row.get("Topic") or row.get("topic") or row.get("Model Line") or "Clause"
        model_line = row.get("Model Line") or row.get("model_line") or row.get("Assumption") or "Manual review"
        direction = row.get("Direction") or row.get("Action") or row.get("Assumption Signal") or "Review"
        rows.append(
            {
                "Clause/Event": _clip(row.get("Clause") or row.get("event") or row.get("Text") or row.get("sentence") or topic, 180),
                "Source": row.get("Source") or row.get("filing") or row.get("Filing") or "SEC / evidence table",
                "Topic": topic,
                "Story implication": _clip(row.get("Implication") or row.get("Action") or direction, 180),
                "Affected assumption": model_line,
                "Suggested direction": direction,
                "Confidence": row.get("Confidence") or row.get("confidence") or "Medium",
                "Action": "Review for User Case",
            }
        )
    return rows


def _growth_driver_rows(dataset: dict, clauses: pd.DataFrame | None, ma_analysis: dict | None, social_buzz: dict | None) -> list[dict]:
    description = " ".join(
        str(dataset.get(key) or "")
        for key in ["company_description", "sector", "industry", "company"]
    ).lower()
    drivers = []
    if any(token in description for token in ["subscription", "saas", "cloud", "software"]):
        drivers.append(("Recurring software / subscription demand", "Company profile / industry classification", "Revenue Growth, OCF Margin", "Revenue up; OCF depends on retention and billing quality", "Medium"))
    if any(token in description for token in ["retail", "consumer", "store", "brand"]):
        drivers.append(("Brand, product mix, pricing, and channel productivity", "Company profile / industry classification", "Revenue Growth, Gross Margin, OPEX % Revenue", "Revenue and margins depend on unit demand, pricing, and marketing intensity", "Medium"))
    if any(token in description for token in ["data center", "gpu", "ai", "infrastructure"]):
        drivers.append(("AI infrastructure capacity and utilization", "Company profile / industry classification", "Revenue Growth, Growth CAPEX, Debt/Dilution", "Revenue up but CAPEX and funding risk may rise", "Medium"))
    if clauses is not None and not clauses.empty:
        drivers.append(("Filing clauses with model impact", "Extracted clause table", "Revenue Growth, OPEX % Revenue, OCF Margin, CAPEX, Terminal Multiple", "Review clause-specific direction before changing User Case", "Medium"))
    if ma_analysis and ma_analysis.get("classification") not in {None, "Insufficient data"}:
        drivers.append(("M&A / acquired business contribution", "M&A disclosure analysis", "Revenue Growth, OPEX % Revenue, OCF Margin, SBC/Dilution, Terminal Multiple", "Revenue may rise; integration and goodwill risk may offset", "Medium"))
    if social_buzz:
        drivers.append(("Social/news buzz signal", "Configured buzz source", "Revenue Growth, Terminal Multiple", "Review whether buzz is fundamental or temporary", "Low"))
    if not drivers:
        drivers.append(("Baseline demand, pricing, margin leverage, and reinvestment", "Company profile and historical financials", "Revenue Growth, OPEX % Revenue, OCF Margin, Growth CAPEX", "Use General driver template until specific evidence is loaded", "Low"))
    return [
        {
            "Driver": driver,
            "Evidence": evidence,
            "Affected assumption": assumption,
            "Direction": direction,
            "Confidence": confidence,
            "Manual review needed?": "Yes",
        }
        for driver, evidence, assumption, direction, confidence in drivers[:6]
    ]


def build_pa11_story(
    dataset: dict,
    filing_texts: dict | None = None,
    clauses: pd.DataFrame | None = None,
    news_items: list[dict] | None = None,
    events: list[dict] | None = None,
    peer_data: pd.DataFrame | None = None,
    social_buzz: dict | None = None,
    ma_analysis: dict | None = None,
    management_analysis: dict | None = None,
    moat_analysis: dict | None = None,
) -> dict:
    """
    Build the main PA-11 company story and assumption map.
    """
    dataset = dataset or {}
    company = dataset.get("company") or dataset.get("ticker") or "Company"
    sector = dataset.get("sector") or UNAVAILABLE
    industry = dataset.get("industry") or UNAVAILABLE
    description = dataset.get("company_description") or ""
    clause_rows = _top_clause_rows(clauses)
    growth_rows = _growth_driver_rows(dataset, clauses, ma_analysis, social_buzz)
    news_text = UNAVAILABLE
    if news_items:
        news_text = _clip("; ".join(_sentences(item.get("title") or item.get("summary") or item, 1, 140) for item in news_items[:4]), 420)
    event_text = UNAVAILABLE
    if events:
        event_text = _clip("; ".join(_sentences(item.get("title") or item.get("event") or item, 1, 140) for item in events[:4]), 420)
    latest_updates = news_text if news_text != UNAVAILABLE else event_text
    if latest_updates == UNAVAILABLE:
        latest_updates = "Dashboard has not fetched recent news/social data yet."
    ma_summary = (ma_analysis or {}).get("summary") or "No clear M&A impact found. Manual review: check business combinations note, goodwill/intangibles, 8-Ks, and MD&A."
    management_summary = (management_analysis or {}).get("summary") or "Management story unavailable. Load SEC evidence for deeper founder, board, and governance context."
    moat_context = (moat_analysis or {}).get("terminal_value_implication") or (moat_analysis or {}).get("classification") or "Moat/risk context unavailable."
    peer_context = "Peer data unavailable. Add peers or fetch peer data." if peer_data is None or peer_data.empty else f"Peer set loaded with {len(peer_data)} rows; use peer medians to anchor terminal multiple and scenario checks."
    buzz_context = "Social/news buzz unavailable." if not social_buzz and not news_items else _clip(str(social_buzz or news_text), 300)

    assumption_map = [
        {
            "assumption": "Revenue Growth",
            "current_model_value": "Review current User Case",
            "story_signal": growth_rows[0]["Direction"] if growth_rows else "Demand signal unclear.",
            "evidence": growth_rows[0]["Evidence"] if growth_rows else "Manual review",
            "suggested_action": "Review revenue growth in User Case; do not change Base Case automatically.",
            "confidence": growth_rows[0]["Confidence"] if growth_rows else "Low",
        },
        {
            "assumption": "OPEX % Revenue",
            "current_model_value": "Review current User Case",
            "story_signal": "Growth investments, integration costs, and operating leverage determine whether margins improve.",
            "evidence": "Company story, M&A analysis, and clause map.",
            "suggested_action": "Review OPEX % Revenue if story implies new product, M&A, or scaling cost changes.",
            "confidence": "Medium" if clause_rows else "Low",
        },
        {
            "assumption": "OCF Margin",
            "current_model_value": "Review current User Case",
            "story_signal": "Cash conversion may diverge from EBIT due to working capital, deferred revenue, inventory, SBC, or prepayments.",
            "evidence": "Cash-flow history and clause map.",
            "suggested_action": "Review OCF Margin and working capital treatment before raising FCF.",
            "confidence": "Medium",
        },
        {
            "assumption": "Growth CAPEX",
            "current_model_value": "Review current User Case",
            "story_signal": "Capacity expansion, product investment, or infrastructure needs can require higher reinvestment.",
            "evidence": "Company profile, CAPEX history, and filing clauses.",
            "suggested_action": "Review growth CAPEX % revenue alongside revenue growth.",
            "confidence": "Medium",
        },
        {
            "assumption": "Terminal Multiple",
            "current_model_value": "Review selected multiple",
            "story_signal": moat_context,
            "evidence": "Moat, risk, peer, and reverse DCF context.",
            "suggested_action": "Anchor terminal multiple to peer/sector medians and durability evidence.",
            "confidence": "Medium" if moat_context != "Moat/risk context unavailable." else "Low",
        },
    ]
    manual_review = [
        "Fetch latest 8-K, earnings call, company IR news, press releases, and trusted news if latest events matter.",
        "Review relevant clauses before applying any story signal to User Case.",
        "Check whether M&A created revenue, products, customers, goodwill/intangibles, debt, dilution, or integration risk.",
        "Validate terminal multiple against peer median, sector median, moat, capital intensity, cyclicality, and management credibility.",
    ]
    return {
        "company_one_liner": f"{company} operates in {sector} / {industry}.",
        "what_they_do": _sentences(description or f"{company} operates in {sector} / {industry}.", 2, 420),
        "how_they_make_money": "Revenue model should be verified from filings: identify product mix, recurring versus transactional revenue, pricing, customers, and segment economics.",
        "product_story": _sentences(description or "Product and segment details unavailable from loaded sources.", 3, 520),
        "industry_positioning": f"Industry context: {sector} / {industry}. {peer_context}",
        "growth_driver_story": _clip("; ".join(row["Driver"] for row in growth_rows), 420),
        "growth_drivers": growth_rows,
        "ma_effect_on_growth": _clip(ma_summary, 520),
        "new_drivers_or_changes": "Review clauses/events for new products, backlog/contracts, pricing, capacity expansion, product launches, M&A, and guidance changes.",
        "latest_updates": latest_updates,
        "peer_context": peer_context,
        "social_buzz_context": buzz_context,
        "moat_and_risk_context": _clip(moat_context, 420),
        "management_context": _clip(management_summary, 420),
        "assumption_map": assumption_map,
        "relevant_clauses": clause_rows,
        "key_questions_for_user": [
            "Which story driver is strong enough to change User Case revenue growth?",
            "Does growth require higher OPEX, working capital, CAPEX, debt, or dilution?",
            "Is terminal multiple supported by moat and peer evidence?",
        ],
        "manual_review_items": manual_review,
        "sources_used": ["Company profile", "SEC clauses" if clause_rows else "Dashboard metadata", "Peer data" if peer_data is not None and not peer_data.empty else "Peer data unavailable"],
    }
