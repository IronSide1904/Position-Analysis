from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

from models.business_drivers import infer_business_driver_profile
from models.driver_templates import get_driver_template


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


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _profile_name(business_driver_profile: dict | None, dataset: dict, filing_texts: dict | None, peer_data: pd.DataFrame | None) -> tuple[str, str, str]:
    profile = business_driver_profile or infer_business_driver_profile(dataset, filing_texts, peer_data)
    return (
        str(profile.get("profile") or "General"),
        str(profile.get("confidence") or "Low"),
        str(profile.get("reason") or "Driver profile inferred from loaded metadata."),
    )


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


PROFILE_STORIES = {
    "AI Infrastructure / Data Center": {
        "business_model_type": "AI Infrastructure / Data Center",
        "economic_engine": "Capacity x utilization x revenue per unit of compute, funded by growth CAPEX, debt/equity, and customer prepayments.",
        "make_money": "The business monetizes contracted compute capacity. Revenue depends on available GPU/data-center capacity, utilization, chip generation mix, customer duration, and price per compute unit.",
        "product_story": "Products are best understood as infrastructure capacity: GPU clusters, data-center power/cooling, customer contracts, and managed compute services.",
        "revenue": ["Energized data-center capacity", "GPU/chip availability", "Utilization", "Revenue per GW or compute unit", "Customer contract duration"],
        "margin": ["Power cost", "Cooling cost", "Chip generation efficiency", "Data-center operating cost", "Utilization"],
        "opex": ["Data-center operations", "Energy procurement", "Engineering/support headcount", "Customer onboarding", "Financing overhead"],
        "ocf": ["Customer prepayments", "Working capital timing", "Interest burden", "Ramp timing", "Deferred revenue quality"],
        "capex": ["GPU/chip prices", "Blackwell/Rubin mix", "Data-center build cost", "Power/cooling infrastructure", "Capacity expansion"],
        "balance": ["Debt funding", "Equity issuance", "SBC dilution", "Customer-funded prepayments", "Net debt"],
        "terminal": ["Capacity durability", "Contract duration", "Energy access", "Chip-cycle risk", "Peer infrastructure multiples"],
        "competitive": "Competitive position depends on scarce power, GPU access, contract quality, financing terms, utilization, and ability to refresh chips without destroying FCF.",
    },
    "Consumer Brand / Retail": {
        "business_model_type": "Retail / Membership Retail",
        "economic_engine": "Traffic and merchandise sales at scale, supported by membership/loyalty economics, supplier bargaining power, and working-capital efficiency.",
        "make_money": "The business earns revenue from merchandise sales, pricing/mix, store or channel productivity, and potentially high-margin membership or loyalty economics.",
        "product_story": "Products are retail assortments, private-label or branded goods, store/channel experience, and customer loyalty programs.",
        "revenue": ["Membership growth", "Renewal rate", "Traffic", "Basket size", "Store/channel expansion"],
        "margin": ["Supplier bargaining power", "Private-label mix", "Merchandise margin", "Shrink/freight", "Pricing discipline"],
        "opex": ["Store labor", "Fulfillment/logistics", "Marketing", "Technology", "Operating leverage"],
        "ocf": ["Inventory turns", "Payable terms", "Membership cash collection", "Working-capital efficiency", "Seasonality"],
        "capex": ["New stores", "Logistics capacity", "Technology", "International expansion", "Maintenance remodels"],
        "balance": ["Buybacks", "Lease obligations", "Inventory funding", "Debt capacity", "Dividend/buyback policy"],
        "terminal": ["Renewal durability", "Scale moat", "Supplier power", "FCF consistency", "Peer retail premium/discount"],
        "competitive": "Competitive position depends on scale purchasing power, customer loyalty, membership renewal, traffic durability, inventory efficiency, and pricing trust.",
    },
    "SaaS / Software": {
        "business_model_type": "SaaS / Software",
        "economic_engine": "Recurring revenue growth driven by customer acquisition, retention, expansion, pricing, and operating leverage.",
        "make_money": "The business monetizes software subscriptions, usage, seats, services, or platform access. Revenue durability depends on retention, expansion, pricing, and customer growth.",
        "product_story": "Products are software workflows, cloud services, data/platform modules, implementation services, and customer success support.",
        "revenue": ["ARR/subscription growth", "Net revenue retention", "Customer growth", "ARPU/ACV", "Expansion revenue"],
        "margin": ["Gross retention", "Hosting cost", "Support cost", "Product mix", "Pricing power"],
        "opex": ["S&M efficiency", "R&D intensity", "G&A leverage", "Customer acquisition cost", "Implementation cost"],
        "ocf": ["Deferred revenue", "Billing terms", "Renewals", "Working capital", "Collections"],
        "capex": ["Capitalized software", "Cloud infrastructure", "Platform investment", "Security/compliance investment", "Maintenance spend"],
        "balance": ["SBC dilution", "Buybacks", "Net cash/debt", "Acquisition funding", "Share count growth"],
        "terminal": ["NRR durability", "Churn risk", "Rule-of-40 profile", "Moat/product depth", "Peer software multiples"],
        "competitive": "Competitive position depends on retention, product depth, switching costs, pricing power, sales efficiency, and ability to scale R&D/S&M.",
    },
    "Industrial / Hardware": {
        "business_model_type": "Industrial / Hardware",
        "economic_engine": "Orders and backlog convert into revenue through production capacity, pricing, utilization, working capital, and CAPEX.",
        "make_money": "The business monetizes units, equipment, hardware, services, or systems. Revenue depends on orders, backlog conversion, production capacity, ASP, and utilization.",
        "product_story": "Products are manufactured goods, hardware systems, components, services, or equipment sold through production and delivery cycles.",
        "revenue": ["Backlog", "Book-to-bill", "Orders", "Units shipped", "ASP/mix"],
        "margin": ["Production utilization", "Input costs", "Pricing", "Warranty/service burden", "Mix"],
        "opex": ["Engineering", "Sales/service footprint", "Manufacturing overhead", "Quality/warranty", "Scale leverage"],
        "ocf": ["Inventory build", "Receivables growth", "Customer deposits", "Payables", "Backlog conversion timing"],
        "capex": ["Production capacity", "Tooling", "Maintenance CAPEX", "Growth CAPEX", "Automation"],
        "balance": ["Inventory financing", "Receivables", "Debt", "Buybacks", "Share issuance"],
        "terminal": ["Backlog durability", "Cycle risk", "Service mix", "CAPEX intensity", "Peer industrial multiples"],
        "competitive": "Competitive position depends on backlog quality, utilization, product differentiation, supplier/input costs, service attachment, and capital intensity.",
    },
    "Marketplace / Platform": {
        "business_model_type": "Marketplace / Platform",
        "economic_engine": "Network activity x monetization rate, supported by scale, liquidity, trust, and platform efficiency.",
        "make_money": "The business monetizes GMV, transactions, users, listings, payments, or advertising through take rate, fees, and value-added services.",
        "product_story": "Products are marketplace liquidity, platform tools, buyer/seller services, payments, ads, trust and safety, and data/commerce workflows.",
        "revenue": ["GMV", "Take rate", "Active users", "Transactions per user", "Buyer/seller growth"],
        "margin": ["Take-rate mix", "Payment cost", "Trust/safety cost", "Platform automation", "Seller services mix"],
        "opex": ["Marketing intensity", "Trust and safety", "Product engineering", "Customer support", "International expansion"],
        "ocf": ["Payment timing", "Seller payouts", "Deferred fees", "Working capital", "Platform capex"],
        "capex": ["Platform investment", "Data/security", "Payments infrastructure", "Trust systems", "Growth initiatives"],
        "balance": ["SBC", "Buybacks", "Acquisition currency", "Net cash/debt", "Regulatory reserves"],
        "terminal": ["Network effects", "Liquidity moat", "Take-rate durability", "Regulatory risk", "Peer platform multiples"],
        "competitive": "Competitive position depends on network liquidity, trust, take-rate discipline, user growth, switching costs, and platform efficiency.",
    },
    "Financial / Fintech": {
        "business_model_type": "Financial / Fintech",
        "economic_engine": "Assets, deposits, loans, transactions, or AUM generate spread income and fees, constrained by credit risk, capital, and regulation.",
        "make_money": "The business earns spreads, fees, payment economics, AUM fees, lending income, or insurance/financial-service revenue.",
        "product_story": "Products are financial accounts, loans, payments, insurance, investing, banking, or platform services.",
        "revenue": ["AUM/deposits/loan growth", "Transaction volume", "Fee rate", "NIM/spread", "Customer growth"],
        "margin": ["Credit losses", "Funding cost", "Fee mix", "Loss reserves", "Scale"],
        "opex": ["Efficiency ratio", "Compliance", "Technology", "Sales/service", "Fraud/risk operations"],
        "ocf": ["Capital requirements", "Credit cycle", "Reserve changes", "Deposit flows", "Receivable/loan growth"],
        "capex": ["Technology platform", "Compliance systems", "Branch/network investment", "Product expansion", "Data/security"],
        "balance": ["Capital ratio", "Book value growth", "Leverage", "Liquidity", "Dilution"],
        "terminal": ["ROE durability", "Credit quality", "Regulatory risk", "Capital needs", "P/B or earnings multiple"],
        "competitive": "Competitive position depends on funding advantage, customer acquisition cost, credit quality, regulatory position, efficiency ratio, and ROE.",
    },
    "Energy / Commodity": {
        "business_model_type": "Energy / Commodity",
        "economic_engine": "Production volume x realized commodity price, less operating cost and reinvestment required to sustain production.",
        "make_money": "The business monetizes production, reserves, realized prices, processing/spread economics, or energy services.",
        "product_story": "Products are commodities, energy output, reserves, generation capacity, or related services.",
        "revenue": ["Production volume", "Realized price", "Commodity cycle", "Hedging", "Reserve additions"],
        "margin": ["Operating cost per unit", "Transportation", "Royalty/taxes", "Mix", "Hedge pricing"],
        "opex": ["Lease operating cost", "Maintenance", "Labor/services", "Fuel/power", "Regulatory cost"],
        "ocf": ["Commodity price", "Hedges", "Working capital", "Taxes", "Production timing"],
        "capex": ["Maintenance CAPEX", "Growth drilling/project CAPEX", "Decline rate", "Reserve life", "Infrastructure"],
        "balance": ["Net debt", "Hedge obligations", "Dividend/buyback policy", "Asset sales", "Funding risk"],
        "terminal": ["Reserve life", "Cost curve", "Cycle risk", "Decline rate", "FCF yield"],
        "competitive": "Competitive position depends on cost curve, reserve life, balance sheet, hedging, decline rate, and reinvestment discipline.",
    },
    "Biotech / Pharma": {
        "business_model_type": "Biotech / Pharma",
        "economic_engine": "Probability-weighted pipeline value plus commercial product revenue, constrained by clinical risk, patent life, R&D spend, and cash runway.",
        "make_money": "The business monetizes approved products, royalties, partnerships, or pipeline assets if trials/regulatory outcomes succeed.",
        "product_story": "Products are commercial drugs, pipeline candidates, therapeutic platforms, royalties, or licensing partnerships.",
        "revenue": ["Commercial product revenue", "Pipeline probability", "Peak sales", "Launch timing", "Partnership milestones"],
        "margin": ["Gross-to-net", "Manufacturing cost", "Royalty burden", "Product mix", "Pricing/reimbursement"],
        "opex": ["R&D spend", "Clinical trial cost", "SG&A launch ramp", "Regulatory cost", "Platform investment"],
        "ocf": ["Cash burn", "Milestones", "Launch timing", "Inventory/build", "Receivables"],
        "capex": ["Manufacturing scale-up", "Lab/platform investment", "Clinical infrastructure", "Maintenance spend", "Capacity"],
        "balance": ["Cash runway", "Dilution", "Debt/royalty financing", "Partnership funding", "Milestone obligations"],
        "terminal": ["Patent life", "Pipeline depth", "Clinical risk", "LOE risk", "Probability-weighted value"],
        "competitive": "Competitive position depends on clinical probability, patent runway, differentiation, reimbursement, cash runway, and pipeline breadth.",
    },
    "Real Estate / REIT": {
        "business_model_type": "Real Estate / REIT",
        "economic_engine": "Rental income and property value driven by occupancy, rent, NOI, cap rates, leverage, and AFFO.",
        "make_money": "The business monetizes rent, occupancy, property services, development spreads, and asset value/NAV.",
        "product_story": "Products are property portfolios, leases, tenant relationships, development pipeline, and asset-management capability.",
        "revenue": ["Occupancy", "Rent per unit/sq ft", "Same-store NOI growth", "Lease spreads", "Development additions"],
        "margin": ["Property operating cost", "NOI margin", "Tenant mix", "Maintenance", "Taxes/insurance"],
        "opex": ["G&A", "Property management", "Leasing cost", "Maintenance", "Scale efficiency"],
        "ocf": ["FFO/AFFO", "Rent collections", "Straight-line rent", "Tenant improvements", "Leasing commissions"],
        "capex": ["Maintenance CAPEX", "Development CAPEX", "Tenant improvements", "Redevelopment", "Acquisitions"],
        "balance": ["Debt maturity", "Interest expense", "Leverage", "Dividend payout", "Equity issuance"],
        "terminal": ["Cap rates", "NAV", "AFFO durability", "Leverage", "Property quality"],
        "competitive": "Competitive position depends on asset quality, occupancy, rent growth, cap rates, leverage maturity, AFFO quality, and dividend coverage.",
    },
    "Advertising / Media / Ad-Tech": {
        "business_model_type": "Advertising / Media / Ad-Tech",
        "economic_engine": "Traffic or audience supply monetized through pricing, fill rate, advertiser demand, take rate, and platform efficiency.",
        "make_money": "The business monetizes impressions, traffic, audience, content, ad inventory, software fees, or take rates from advertisers/publishers.",
        "product_story": "Products are audience reach, ad-tech tools, publisher/advertiser services, content, data, and measurement workflows.",
        "revenue": ["Traffic/impressions", "CPM/CPC/CPA", "Fill rate", "Take rate", "Advertiser demand"],
        "margin": ["Traffic acquisition cost", "Publisher payouts", "Data/hosting cost", "Content cost", "Mix"],
        "opex": ["Sales", "Product/engineering", "Traffic quality", "Trust/safety", "Marketing"],
        "ocf": ["Collections", "Publisher payout timing", "Working capital", "Advertiser concentration", "Seasonality"],
        "capex": ["Platform/data investment", "Content investment", "Infrastructure", "Measurement tools", "Security/privacy"],
        "balance": ["SBC", "Acquisitions", "Net cash/debt", "Buybacks", "Regulatory reserves"],
        "terminal": ["Audience durability", "Take-rate durability", "Privacy/regulatory risk", "Platform moat", "Peer media/ad-tech multiples"],
        "competitive": "Competitive position depends on traffic quality, advertiser demand, fill rate, take rate, data advantage, and regulatory/privacy exposure.",
    },
    "General": {
        "business_model_type": "General",
        "economic_engine": "Revenue growth, margin structure, cash conversion, reinvestment needs, balance-sheet risk, and terminal value durability.",
        "make_money": "The business model is not specific enough from loaded data. Treat revenue, margin, cash conversion, reinvestment, debt, dilution, and terminal multiple as the core review loop.",
        "product_story": "Product/segment detail is unavailable or not specific enough; load filings, peers, and management commentary for a sharper driver story.",
        "revenue": ["Revenue growth source", "Pricing", "Volume", "Customer demand", "Mix"],
        "margin": ["Gross margin structure", "Input costs", "Pricing power", "Mix", "Scale"],
        "opex": ["Operating leverage", "Sales/R&D/G&A intensity", "Integration cost", "Fixed cost", "Efficiency"],
        "ocf": ["Cash conversion", "Working capital", "Collections", "Deferred revenue/prepayments", "One-time items"],
        "capex": ["Maintenance needs", "Growth CAPEX", "Capacity", "Technology", "Capital intensity"],
        "balance": ["Debt", "Dilution", "Buybacks", "Liquidity", "Net debt"],
        "terminal": ["Moat durability", "Peer multiples", "Cyclicality", "Capital intensity", "Management credibility"],
        "competitive": "Competitive position needs manual review against peers, margins, growth, cash conversion, reinvestment, and moat evidence.",
    },
}


def _story_template(profile: str) -> dict:
    return PROFILE_STORIES.get(profile) or PROFILE_STORIES.get("General", {})


def _driver_rows(template: dict) -> list[dict]:
    categories = [
        ("Revenue", "core_revenue_drivers", "Revenue Growth"),
        ("Margin", "core_margin_drivers", "Gross Margin / OPEX % Revenue"),
        ("OPEX", "core_opex_drivers", "OPEX % Revenue"),
        ("Cash Conversion", "core_ocf_drivers", "OCF Margin / Working Capital"),
        ("CAPEX / Reinvestment", "core_capex_drivers", "Growth CAPEX / D&A"),
        ("Balance Sheet / Dilution", "core_dilution_or_balance_sheet_drivers", "Net Debt / Diluted Shares / WACC"),
        ("Terminal Value", "core_terminal_value_drivers", "Terminal Multiple"),
    ]
    key_map = {
        "core_revenue_drivers": "revenue",
        "core_margin_drivers": "margin",
        "core_opex_drivers": "opex",
        "core_ocf_drivers": "ocf",
        "core_capex_drivers": "capex",
        "core_dilution_or_balance_sheet_drivers": "balance",
        "core_terminal_value_drivers": "terminal",
    }
    rows = []
    for category, key, assumptions in categories:
        for driver in template.get(key_map[key], [])[:5]:
            rows.append(
                {
                    "driver": driver,
                    "driver_category": category,
                    "story_signal": f"{driver} is a core {category.lower()} driver for this business model.",
                    "affected_assumptions": [item.strip() for item in assumptions.split("/")],
                    "direction": "Review",
                    "evidence": "Business-driver template plus loaded company/filing context.",
                    "confidence": "Medium",
                    "suggested_action": f"Review User Case {assumptions}.",
                }
            )
    return rows


def _driver_to_assumption_map(template: dict, profile: str, clause_rows: list[dict], ma_analysis: dict | None, social_buzz: dict | None) -> list[dict]:
    rows = _driver_rows(template)
    if clause_rows:
        rows.append(
            {
                "driver": "Filing clause signal",
                "driver_category": "Evidence / Manual Review",
                "story_signal": "Extracted filing clauses may point to model-changing revenue, margin, cash conversion, reinvestment, or risk drivers.",
                "affected_assumptions": ["Revenue Growth", "OPEX % Revenue", "OCF Margin", "Growth CAPEX", "Terminal Multiple"],
                "direction": "Review",
                "evidence": "SEC clause map.",
                "confidence": "Medium",
                "suggested_action": "Review clauses before changing User Case.",
            }
        )
    if ma_analysis and ma_analysis.get("classification") not in {None, "Insufficient data"}:
        rows.append(
            {
                "driver": "M&A / acquired business contribution",
                "driver_category": "M&A",
                "story_signal": "Acquisitions can add products, customers, revenue, goodwill/intangibles, integration cost, debt, or dilution.",
                "affected_assumptions": ["Revenue Growth", "Gross Margin", "OPEX % Revenue", "OCF Margin", "Growth CAPEX", "Terminal Multiple"],
                "direction": "Mixed",
                "evidence": "M&A disclosure analysis.",
                "confidence": "Medium",
                "suggested_action": "Review acquired revenue quality and integration cost before adjusting User Case.",
            }
        )
    if social_buzz:
        rows.append(
            {
                "driver": "News / social buzz signal",
                "driver_category": "External signal",
                "story_signal": "Buzz can affect sentiment, but it needs evidence before it changes fundamentals.",
                "affected_assumptions": ["Revenue Growth", "Terminal Multiple"],
                "direction": "Review",
                "evidence": "Configured news/social source.",
                "confidence": "Low",
                "suggested_action": "Do not adjust numeric assumptions until buzz is tied to revenue, margin, or cash-flow evidence.",
            }
        )
    for row in rows:
        row.setdefault("sotp_line_affected", "Relevant segment revenue/margin if SOTP segment exists")
        row.setdefault("multiples_implication", "Higher quality/durability can support a premium; weaker cash conversion or higher reinvestment can require a discount.")
        row.setdefault("manual_review_needed", "Yes")
    return rows


def _assumption_map(driver_rows: list[dict]) -> list[dict]:
    targets = {
        "Revenue Growth": ["Revenue Growth", "revenue"],
        "Gross Margin": ["Gross Margin", "margin"],
        "OPEX % Revenue": ["OPEX % Revenue", "opex"],
        "OCF Margin": ["OCF Margin", "cash", "Working Capital"],
        "Growth CAPEX": ["Growth CAPEX", "CAPEX", "reinvestment"],
        "Diluted Shares": ["Diluted Shares", "dilution", "equity"],
        "WACC / Net Debt": ["WACC", "Debt", "Net Debt"],
        "Terminal Multiple": ["Terminal Multiple", "terminal"],
    }
    rows = []
    for assumption, needles in targets.items():
        matches = []
        for item in driver_rows:
            text = " ".join([str(item.get("driver")), str(item.get("driver_category")), " ".join(_as_list(item.get("affected_assumptions")))])
            if any(needle.lower() in text.lower() for needle in needles):
                matches.append(item)
        if not matches:
            continue
        rows.append(
            {
                "assumption": assumption,
                "current_model_value": "Review current User Case",
                "story_signal": _clip("; ".join(match.get("driver", "") for match in matches[:3]), 220),
                "evidence": _clip("; ".join(match.get("evidence", "") for match in matches[:2]), 180),
                "suggested_action": f"Review {assumption} in User Case; story signals do not auto-change Base/Bull/Bear.",
                "confidence": "Medium" if any(match.get("confidence") == "Medium" for match in matches) else "Low",
            }
        )
    return rows


def _peer_context(profile: str, peer_data: pd.DataFrame | None, sector: str, industry: str, template: dict) -> tuple[str, dict]:
    if peer_data is None or peer_data.empty:
        summary = "Peer data unavailable. Add peers or enable peer fetch before anchoring valuation premium/discount."
        peer_count = 0
    else:
        peer_count = len(peer_data)
        summary = f"Peer set loaded with {peer_count} rows. Compare growth, margins, OCF conversion, CAPEX intensity, leverage, and valuation multiples through the {profile} driver lens."
    theme = (
        f"{sector} / {industry}: evaluate whether the sector tailwind actually improves the model drivers, "
        "or only improves narrative sentiment."
    )
    implications = [
        "Do not adjust revenue growth without checking the paired cost, cash conversion, and reinvestment drivers.",
        "Terminal multiple should be anchored to peer/sector multiples only after comparing moat, cash conversion, CAPEX intensity, and balance-sheet risk.",
    ]
    return summary, {
        "sector_summary": f"Sector context: {sector} / {industry}.",
        "theme_summary": theme,
        "peer_positioning": summary,
        "relative_strengths": template.get("terminal", [])[:3],
        "relative_weaknesses": template.get("capex", [])[:2] + template.get("balance", [])[:2],
        "assumption_implications": implications,
    }


def _latest_updates(news_items: list[dict] | None, events: list[dict] | None) -> str:
    if news_items:
        return _clip("; ".join(_sentences(item.get("title") or item.get("summary") or item, 1, 140) for item in news_items[:4]), 420)
    if events:
        return _clip("; ".join(_sentences(item.get("title") or item.get("event") or item, 1, 140) for item in events[:4]), 420)
    return "Dashboard has not fetched recent news/social data yet."


def _driver_reflection(driver_rows: list[dict]) -> list[dict]:
    reflected = []
    for item in driver_rows[:18]:
        assumptions = ", ".join(_as_list(item.get("affected_assumptions")))
        reflected.append(
            {
                "Story point": item.get("story_signal"),
                "Driver affected": item.get("driver"),
                "DCF line affected": assumptions,
                "SOTP line affected": item.get("sotp_line_affected"),
                "Multiples implication": item.get("multiples_implication"),
                "Confidence": item.get("confidence"),
                "Evidence": item.get("evidence"),
                "Manual review needed?": item.get("manual_review_needed", "Yes"),
            }
        )
    return reflected


def _sources_used(clause_rows: list[dict], peer_data: pd.DataFrame | None, business_driver_profile: dict | None, driver_template: dict | None) -> list[str]:
    sources = ["Company profile", "Business driver profile"]
    if business_driver_profile:
        sources.append("Business-driver inference")
    if driver_template:
        sources.append("Driver template")
    sources.append("SEC clauses" if clause_rows else "SEC clauses unavailable")
    sources.append("Peer data" if peer_data is not None and not peer_data.empty else "Peer data unavailable")
    return sources


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
    business_driver_profile: dict | None = None,
    driver_template: dict | None = None,
) -> dict:
    """
    Build the main PA-11 business-model-specific story and assumption map.

    The output explains the economic engine, driver map, and model implications.
    It recommends review only; it does not change numeric assumptions.
    """
    dataset = dataset or {}
    company = dataset.get("company") or dataset.get("ticker") or "Company"
    sector = dataset.get("sector") or UNAVAILABLE
    industry = dataset.get("industry") or UNAVAILABLE
    description = dataset.get("company_description") or ""
    profile, profile_confidence, profile_reason = _profile_name(business_driver_profile, dataset, filing_texts, peer_data)
    selected_template = driver_template or get_driver_template(profile)
    story_template = _story_template(profile)
    clause_rows = _top_clause_rows(clauses)
    driver_rows = _driver_to_assumption_map(story_template, profile, clause_rows, ma_analysis, social_buzz)
    assumption_rows = _assumption_map(driver_rows)
    peer_summary, sector_theme_peer_context = _peer_context(profile, peer_data, sector, industry, story_template)
    moat_context = (moat_analysis or {}).get("terminal_value_implication") or (moat_analysis or {}).get("classification") or "Moat/risk context unavailable."
    ma_summary = (ma_analysis or {}).get("summary") or "No clear M&A impact found. Manual review: check business combinations note, goodwill/intangibles, 8-Ks, and MD&A."
    management_summary = (management_analysis or {}).get("summary") or "Management story unavailable. Load SEC evidence for deeper founder, board, and governance context."
    buzz_context = "Social/news buzz unavailable." if not social_buzz and not news_items else _clip(str(social_buzz or _latest_updates(news_items, events)), 300)
    driver_counts = Counter(row["driver_category"] for row in driver_rows)
    manual_review = [
        "Review the driver profile. If the detected profile is wrong, change the Business Driver Profile in the DCF Model tab.",
        "Check whether the top revenue driver also requires higher OPEX, working capital, CAPEX, debt, or dilution.",
        "Anchor terminal multiple to peer/sector multiples only after comparing moat, cash conversion, CAPEX intensity, cyclicality, and balance-sheet risk.",
        "Fetch latest 8-K, earnings call, company IR news, press releases, and trusted news if latest events matter.",
    ]
    if selected_template.get("manual_review_questions"):
        manual_review.extend(selected_template.get("manual_review_questions", [])[:4])

    economic_engine = story_template.get("economic_engine", PROFILE_STORIES["General"]["economic_engine"])
    what_they_do = _sentences(description or f"{company} operates in {sector} / {industry}.", 2, 420)
    product_story = _sentences(description or story_template.get("product_story"), 3, 520)
    economic_summary = (
        f"{company} should be analyzed as a {story_template.get('business_model_type', profile)} business. "
        f"Economic engine: {economic_engine}"
    )
    growth_story = (
        "Growth is not one generic CAGR input. It should be tied to "
        + ", ".join(story_template.get("revenue", [])[:4])
        + ", then checked against margin, cash conversion, reinvestment, and dilution drivers."
    )

    return {
        "company_one_liner": f"{company} operates in {sector} / {industry}.",
        "business_model_type": story_template.get("business_model_type", profile),
        "business_model_confidence": profile_confidence,
        "business_model_reason": profile_reason,
        "economic_engine_summary": _clip(economic_summary, 520),
        "what_they_do": what_they_do,
        "how_they_make_money": _clip(story_template.get("make_money"), 520),
        "product_or_service_story": product_story,
        "product_story": product_story,
        "core_revenue_drivers": story_template.get("revenue", []),
        "core_margin_drivers": story_template.get("margin", []),
        "core_opex_drivers": story_template.get("opex", []),
        "core_ocf_drivers": story_template.get("ocf", []),
        "core_capex_drivers": story_template.get("capex", []),
        "core_dilution_or_balance_sheet_drivers": story_template.get("balance", []),
        "core_terminal_value_drivers": story_template.get("terminal", []),
        "industry_theme_context": sector_theme_peer_context.get("theme_summary"),
        "peer_positioning_context": peer_summary,
        "competitive_dynamics": story_template.get("competitive"),
        "moat_context": _clip(moat_context, 420),
        "industry_positioning": f"{sector_theme_peer_context.get('sector_summary')} {peer_summary}",
        "sector_theme_peer_context": sector_theme_peer_context,
        "growth_driver_story": _clip(growth_story, 620),
        "growth_drivers": [
            {
                "Driver": row.get("driver"),
                "Evidence": row.get("evidence"),
                "Affected assumption": ", ".join(_as_list(row.get("affected_assumptions"))),
                "Direction": row.get("direction"),
                "Confidence": row.get("confidence"),
                "Manual review needed?": row.get("manual_review_needed", "Yes"),
            }
            for row in driver_rows[:8]
        ],
        "ma_effect_on_growth": _clip(ma_summary, 520),
        "new_drivers_or_changes": "Review clauses/events for new products, backlog/contracts, pricing, capacity expansion, product launches, M&A, guidance changes, financing changes, and new risk drivers.",
        "latest_updates": _latest_updates(news_items, events),
        "social_buzz_context": buzz_context,
        "moat_and_risk_context": _clip(moat_context, 420),
        "management_context": _clip(management_summary, 420),
        "driver_to_assumption_map": driver_rows,
        "driver_reflection_map": _driver_reflection(driver_rows),
        "assumption_map": assumption_rows,
        "relevant_clauses": clause_rows,
        "key_questions_for_user": [
            f"Is {profile} the right economic-engine profile for this company?",
            "Which revenue driver is strong enough to change User Case revenue growth?",
            "Does growth require higher OPEX, working capital, CAPEX, debt, or dilution?",
            "Is terminal multiple supported by moat, peer, and cash-conversion evidence?",
        ],
        "manual_review_items": manual_review,
        "sources_used": _sources_used(clause_rows, peer_data, business_driver_profile, selected_template),
        "driver_category_counts": dict(driver_counts),
    }
