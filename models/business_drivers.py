from __future__ import annotations

import re

import pandas as pd

from models.driver_templates import PROFILE_ALIASES, SUPPORTED_DRIVER_PROFILES


PROFILE_KEYWORDS = {
    "AI Infrastructure / Data Center": [
        "ai infrastructure",
        "data center",
        "datacenter",
        "gpu",
        "blackwell",
        "rubin",
        "nvidia",
        "accelerated computing",
        "cloud infrastructure",
        "compute capacity",
        "energized",
        "megawatt",
        "gigawatt",
        "gw",
    ],
    "SaaS / Software": ["software", "saas", "subscription", "arr", "cloud software", "net retention", "seat"],
    "Semiconductor": ["semiconductor", "chip", "wafer", "foundry", "fabless", "memory", "gpu", "asic"],
    "Marketplace / Platform": ["marketplace", "platform", "gmv", "take rate", "transaction", "network"],
    "Consumer Brand / Retail": ["consumer", "brand", "retail", "apparel", "restaurant", "same-store", "store", "omnichannel"],
    "Financial / Fintech": ["bank", "fintech", "loan", "deposit", "aum", "net interest", "payment", "insurance"],
    "Industrial / Hardware": ["industrial", "hardware", "equipment", "manufacturing", "factory", "aerospace"],
    "Energy / Commodity": ["energy", "oil", "gas", "commodity", "mining", "utility", "solar", "power"],
    "Biotech / Pharma": ["biotech", "pharma", "drug", "clinical", "fda", "pipeline", "therapy"],
    "Real Estate / REIT": ["reit", "real estate", "occupancy", "rent", "property", "cap rate", "noi"],
    "Advertising / Media / Ad-Tech": ["advertising", "ad-tech", "ad tech", "media", "publisher", "impressions", "cpm", "cpc", "fill rate", "advertiser"],
}


def _clean_text(*parts: object) -> str:
    return " ".join(str(part or "") for part in parts).lower()


def _filing_text_blob(filing_texts: dict | None) -> str:
    if not filing_texts:
        return ""
    parts = []
    for value in filing_texts.values():
        if isinstance(value, dict):
            parts.extend(str(item or "") for item in value.values())
        else:
            parts.append(str(value or ""))
    return " ".join(parts).lower()


def infer_business_driver_profile(dataset: dict, filing_texts: dict | None = None, peer_data: pd.DataFrame | None = None) -> dict:
    """
    Infer the company's business-driver profile from loaded metadata and filing text.
    """
    dataset = dataset or {}
    override = dataset.get("business_driver_profile_override") or dataset.get("driver_profile_override")
    override = PROFILE_ALIASES.get(override, override)
    if override in SUPPORTED_DRIVER_PROFILES:
        return {
            "profile": override,
            "confidence": "High",
            "reason": "User override selected.",
            "key_evidence": ["User override selected."],
            "alternative_profiles": [item for item in SUPPORTED_DRIVER_PROFILES if item != override][:3],
        }

    peers = dataset.get("peer_group") or dataset.get("peers") or []
    peer_text = " ".join(str(item) for item in peers) if not isinstance(peers, str) else peers
    peer_frame_text = ""
    if isinstance(peer_data, pd.DataFrame) and not peer_data.empty:
        peer_frame_text = " ".join(
            str(value or "")
            for column in peer_data.columns[:8]
            for value in peer_data[column].head(20).tolist()
        )
    text = _clean_text(
        dataset.get("ticker"),
        dataset.get("company"),
        dataset.get("sector"),
        dataset.get("industry"),
        dataset.get("company_description"),
        peer_text,
        peer_frame_text,
        _filing_text_blob(filing_texts),
    )
    scores = []
    for profile, keywords in PROFILE_KEYWORDS.items():
        score = 0
        hits = []
        for keyword in keywords:
            pattern = r"(?<![a-z0-9])" + re.escape(keyword.lower()) + r"(?![a-z0-9])"
            matches = re.findall(pattern, text)
            if matches:
                score += len(matches)
                hits.append(keyword)
        if score:
            scores.append((score, profile, hits[:4]))
    if not scores:
        return {
            "profile": "General",
            "confidence": "Low",
            "reason": "No clear business-driver keywords found in loaded metadata.",
            "key_evidence": [],
            "alternative_profiles": ["Industrial / Hardware", "SaaS / Software", "Consumer Brand / Retail"],
        }

    scores.sort(reverse=True)
    best_score, profile, hits = scores[0]
    alternatives = [item[1] for item in scores[1:4]]
    confidence = "High" if best_score >= 4 else "Medium" if best_score >= 2 else "Low"
    return {
        "profile": profile,
        "confidence": confidence,
        "reason": f"Matched driver keywords: {', '.join(hits)}.",
        "key_evidence": hits,
        "alternative_profiles": alternatives or ["General"],
    }
