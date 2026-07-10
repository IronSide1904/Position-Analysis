from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _clean_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text


def _sentences(text: str, limit: int = 3) -> str:
    clean = _clean_text(text)
    if not clean:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", clean)
    return " ".join(parts[:limit]).strip()


def _filing_excerpt(filing_texts: dict | None) -> tuple[str, list[str]]:
    if not filing_texts:
        return "", []
    sources: list[str] = []
    for preferred in ["10-K", "10-K/A", "20-F", "40-F", "10-Q", "10-Q/A"]:
        text = filing_texts.get(preferred)
        if text:
            sources.append(f"SEC {preferred}")
            return _sentences(text, 3), sources
    for form, text in filing_texts.items():
        if text:
            sources.append(f"SEC {form}")
            return _sentences(text, 3), sources
    return "", sources


def _peer_context(peers: pd.DataFrame | None, dataset: dict) -> tuple[str, list[str]]:
    if peers is None or peers.empty:
        sector = dataset.get("sector") or "sector"
        industry = dataset.get("industry") or "industry"
        return f"Peer data unavailable; use sector/industry context only: {sector} / {industry}.", []
    labels = []
    for _, row in peers.head(6).iterrows():
        ticker = row.get("ticker") or row.get("Ticker")
        company = row.get("company") or row.get("Company")
        labels.append(f"{ticker} ({company})" if ticker and company else str(ticker or company))
    return f"Available peer set: {', '.join([label for label in labels if label])}.", ["Peer comparison table"]


def _buzz_context(news_items: list[dict] | None, social_buzz: dict | None, web_context: dict | None) -> tuple[str, list[str]]:
    sources = []
    snippets = []
    if news_items:
        for item in news_items[:3]:
            headline = item.get("title") or item.get("headline") or item.get("summary")
            if headline:
                snippets.append(_clean_text(headline))
        if snippets:
            sources.append("News items")
    if social_buzz:
        summary = social_buzz.get("summary") or social_buzz.get("sentiment") or social_buzz.get("mentions")
        if summary:
            snippets.append(f"Social buzz: {_clean_text(summary)}")
            sources.append("Social buzz")
    if web_context:
        summary = web_context.get("summary") or web_context.get("context")
        if summary:
            snippets.append(_clean_text(summary))
            sources.append("Web context")
    if not snippets:
        return "Social/news buzz unavailable.", sources
    return " ".join(snippets[:4]), sources


def _manual_review_questions(has_description: bool, has_peers: bool, has_buzz: bool) -> list[dict]:
    items = []
    if not has_description:
        items.append(
            {
                "Question": "Need to verify product-level revenue split.",
                "Why it matters": "Revenue CAGR and gross margin depend on which products or services are driving growth.",
                "Where to look": "10-K business section, revenue recognition note, investor presentation.",
                "Suggested keywords": "revenue by product, platform, services, customers, backlog, RPO",
                "Model assumption affected": "Revenue CAGR / Gross margin",
            }
        )
    items.append(
        {
            "Question": "Need to verify organic vs acquired growth.",
            "Why it matters": "Acquisition-led growth can overstate durable organic demand and distort margins.",
            "Where to look": "MD&A, business combinations note, earnings call transcript.",
            "Suggested keywords": "organic growth, acquisition, integration, acquired revenue",
            "Model assumption affected": "Revenue CAGR / OPEX % Revenue / Terminal Multiple",
        }
    )
    items.append(
        {
            "Question": "Need to verify customer concentration and contract durability.",
            "Why it matters": "Concentrated or short-duration revenue raises forecast risk.",
            "Where to look": "10-K risk factors, customer footnotes, revenue recognition note.",
            "Suggested keywords": "major customer, concentration, contract, renewal, churn",
            "Model assumption affected": "Revenue CAGR / OCF Margin / WACC",
        }
    )
    if not has_buzz:
        items.append(
            {
                "Question": "Need to verify social/news buzz.",
                "Why it matters": "Recent events may explain estimate changes, sentiment, or execution risk.",
                "Where to look": "Company IR, press releases, earnings transcripts, credible news sources.",
                "Suggested keywords": "guidance, demand, partnership, product launch, litigation",
                "Model assumption affected": "Revenue CAGR / Terminal Multiple",
            }
        )
    if not has_peers:
        items.append(
            {
                "Question": "Need to verify peer positioning.",
                "Why it matters": "Peer growth, margins, and multiples anchor whether assumptions are aggressive.",
                "Where to look": "Peer financials, sector comps, competitor filings.",
                "Suggested keywords": "peer revenue growth, gross margin, EV/Sales, EV/EBITDA, FCF yield",
                "Model assumption affected": "Terminal Multiple / Margins / WACC",
            }
        )
    return items


def build_company_story_summary(
    dataset: dict,
    filing_texts: dict | None = None,
    peers: pd.DataFrame | None = None,
    news_items: list[dict] | None = None,
    social_buzz: dict | None = None,
    web_context: dict | None = None,
) -> dict:
    """
    Build a business/product/industry summary to support assumption-setting.
    """
    dataset = dataset or {}
    company = dataset.get("company") or dataset.get("ticker") or "Company"
    sector = dataset.get("sector") or "sector unavailable"
    industry = dataset.get("industry") or "industry unavailable"
    description = _sentences(dataset.get("company_description"), 4)
    filing_summary, filing_sources = _filing_excerpt(filing_texts)
    business_summary = description or filing_summary
    sources_used = []
    if description:
        sources_used.append("Company description from provider snapshot")
    sources_used.extend(filing_sources)

    if not business_summary:
        business_summary = f"{company} operates in {sector} / {industry}, but a business description was not available."

    peer_text, peer_sources = _peer_context(peers, dataset)
    buzz_text, buzz_sources = _buzz_context(news_items, social_buzz, web_context)
    sources_used.extend(peer_sources)
    sources_used.extend(buzz_sources)

    product_story = filing_summary or description or "Product and service detail unavailable from loaded sources."
    how_money = (
        f"{company} appears to earn revenue through products/services described in loaded company materials. "
        "Verify product mix, recurring versus transactional revenue, pricing, and customer concentration before changing growth or margin assumptions."
    )
    industry_positioning = f"Industry context: {sector} / {industry}. {peer_text}"
    has_buzz = buzz_text != "Social/news buzz unavailable."
    assumption_implications = [
        {
            "assumption": "Revenue CAGR",
            "implication": "Driven by market growth, product demand, backlog/RPO, pricing, customer expansion, competition, and organic versus acquired growth.",
            "confidence": "Medium" if description or filing_summary else "Low",
        },
        {
            "assumption": "OPEX % Revenue",
            "implication": "Depends on operating leverage, sales efficiency, R&D intensity, integration costs, and G&A scale.",
            "confidence": "Medium" if filing_summary else "Low",
        },
        {
            "assumption": "OCF Margin",
            "implication": "Depends on billing model, collections, working capital timing, deferred revenue, inventory, and cash conversion quality.",
            "confidence": "Low",
        },
        {
            "assumption": "CAPEX",
            "implication": "Depends on asset intensity, equipment, infrastructure, capacity expansion, acquired intangibles, and maintenance needs.",
            "confidence": "Medium" if sector != "sector unavailable" else "Low",
        },
        {
            "assumption": "Terminal Multiple",
            "implication": "Depends on moat durability, peer multiples, growth runway, margin stability, reinvestment needs, and industry structure.",
            "confidence": "Medium" if peers is not None and not peers.empty else "Low",
        },
    ]

    manual_review = _manual_review_questions(bool(description or filing_summary), peers is not None and not peers.empty, has_buzz)
    return {
        "business_summary": business_summary,
        "how_they_make_money": how_money,
        "product_story": product_story,
        "industry_positioning": industry_positioning,
        "peer_context": peer_text,
        "buzz_context": buzz_text,
        "assumption_implications": assumption_implications,
        "manual_review_questions": manual_review,
        "sources_used": sources_used or ["Dashboard metadata only"],
    }
