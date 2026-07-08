from __future__ import annotations

import math
from typing import Any

import pandas as pd


CAPACITY_TERMS = [
    "capacity expansion",
    "facility expansion",
    "new facility",
    "new facilities",
    "manufacturing capacity",
    "data center",
    "datacenter",
    "infrastructure build-out",
    "infrastructure buildout",
    "automation",
    "fleet expansion",
    "equipment purchases",
    "capital project",
]
MA_TERMS = ["acquisition", "business combination", "goodwill", "intangible", "acquired intangible", "amortization of acquired"]
MAINTENANCE_TERMS = ["maintenance capital", "replacement", "sustaining capital", "aging equipment", "safety", "compliance"]
WORKING_CAPITAL_RISK_TERMS = ["receivable", "contract asset", "inventory", "payables stretch", "working capital timing"]
WORKING_CAPITAL_SUPPORT_TERMS = ["deferred revenue", "contract liabilities", "remaining performance obligation", "rpo"]
SBC_TERMS = ["stock-based compensation", "share-based compensation", "rsu", "equity award"]
ONE_TIME_TERMS = ["restructuring", "impairment", "integration", "legal settlement", "one-time", "non-recurring"]


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    numerator = _safe_float(numerator)
    denominator = _safe_float(denominator)
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _latest(historicals: pd.DataFrame, column: str, default: float | None = None) -> float | None:
    if historicals is None or historicals.empty or column not in historicals:
        return default
    series = pd.to_numeric(historicals[column], errors="coerce").dropna()
    if series.empty:
        return default
    return float(series.iloc[-1])


def _ratio_series(historicals: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    if historicals is None or historicals.empty or numerator not in historicals or denominator not in historicals:
        return pd.Series(dtype=float)
    numer = pd.to_numeric(historicals[numerator], errors="coerce").abs()
    denom = pd.to_numeric(historicals[denominator], errors="coerce").abs().replace(0, math.nan)
    return (numer / denom).dropna()


def _trend(historicals: pd.DataFrame, column: str) -> float | None:
    if historicals is None or historicals.empty or column not in historicals:
        return None
    values = pd.to_numeric(historicals[column], errors="coerce").dropna()
    if len(values) < 2:
        return None
    start = values.iloc[0]
    end = values.iloc[-1]
    if start in (0, None) or pd.isna(start):
        return None
    return float((end / abs(start)) - 1)


def _clauses_text(clauses: pd.DataFrame | None) -> str:
    if clauses is None or clauses.empty:
        return ""
    pieces: list[str] = []
    for column in ["topic", "subtopic", "section", "model_line_affected", "suggested_assumption_change", "clause_text"]:
        if column in clauses:
            pieces.extend(str(value) for value in clauses[column].dropna().head(120).tolist())
    return " ".join(pieces).lower()


def _has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _evidence_matches(text: str, terms: list[str], limit: int = 3) -> list[str]:
    matches = [term for term in terms if term in text]
    return matches[:limit]


def _quality_from_score(score: float) -> str:
    if score >= 8:
        return "High"
    if score >= 5:
        return "Medium"
    return "Low"


def _confidence(evidence_count: int, has_financials: bool = True, penalty: int = 0) -> str:
    score = evidence_count + (1 if has_financials else 0) - penalty
    if score >= 4:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


def _fmt_money(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Data unavailable"
    value = float(value)
    for suffix, divisor in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(value) >= divisor:
            return f"${value / divisor:,.2f}{suffix}"
    return f"${value:,.2f}"


def _fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Data unavailable"
    return f"{float(value):.1%}"


def infer_business_profile(dataset: dict, clauses: pd.DataFrame | None = None) -> dict:
    """
    Infer the company business profile using sector, industry, segment notes,
    filing clauses, and available company metadata.
    """
    company = dataset.get("company") if isinstance(dataset.get("company"), dict) else {}
    sector = str(dataset.get("sector") or company.get("sector") or "").lower()
    industry = str(dataset.get("industry") or company.get("industry") or "").lower()
    description = str(company.get("longBusinessSummary") or company.get("description") or "").lower()
    clause_text = _clauses_text(clauses)
    blob = " ".join([sector, industry, description, clause_text])
    evidence: list[str] = []

    business_model = "Other"
    if _has_any(blob, ["saas", "software", "cloud", "subscription"]):
        business_model = "SaaS"
        evidence.append("Software/SaaS or cloud language found.")
    elif _has_any(blob, ["data center", "datacenter", "hosting", "infrastructure"]):
        business_model = "Data Center"
        evidence.append("Data center or infrastructure language found.")
    elif _has_any(blob, ["marketplace", "platform"]):
        business_model = "Marketplace"
        evidence.append("Marketplace/platform language found.")
    elif _has_any(blob, ["manufacturing", "manufacturer", "machinery", "industrial", "equipment"]):
        business_model = "Manufacturing"
        evidence.append("Manufacturing/industrial language found.")
    elif _has_any(blob, ["hardware", "semiconductor", "devices", "electronics"]):
        business_model = "Hardware"
        evidence.append("Hardware/semiconductor/device language found.")
    elif _has_any(blob, ["defense", "aerospace", "contractor"]):
        business_model = "Defense"
        evidence.append("Defense/aerospace language found.")
    elif _has_any(blob, ["bank", "insurance", "financial services", "asset management"]):
        business_model = "Financial"
        evidence.append("Financial-services language found.")
    elif _has_any(blob, ["retail", "ecommerce", "e-commerce", "store"]):
        business_model = "Retail"
        evidence.append("Retail/e-commerce language found.")
    elif _has_any(blob, ["services", "consulting", "outsourcing"]):
        business_model = "Services"
        evidence.append("Services language found.")

    if business_model in {"SaaS", "Marketplace", "Services", "Financial"}:
        asset_intensity = "Low"
    elif business_model in {"Manufacturing", "Data Center", "Defense", "Retail"}:
        asset_intensity = "High"
    elif business_model == "Hardware":
        asset_intensity = "Medium"
    else:
        asset_intensity = "Unknown"

    acquisition_hits = _evidence_matches(blob, MA_TERMS)
    capacity_hits = _evidence_matches(blob, CAPACITY_TERMS)
    maintenance_hits = _evidence_matches(blob, MAINTENANCE_TERMS)
    wc_risk_hits = _evidence_matches(blob, WORKING_CAPITAL_RISK_TERMS)
    wc_support_hits = _evidence_matches(blob, WORKING_CAPITAL_SUPPORT_TERMS)

    if acquisition_hits:
        capex_profile = "Acquisition-heavy"
        evidence.append(f"M&A/intangible evidence: {', '.join(acquisition_hits)}.")
    elif capacity_hits:
        capex_profile = "Growth-capex-heavy"
        evidence.append(f"Growth CAPEX evidence: {', '.join(capacity_hits)}.")
    elif asset_intensity == "Low":
        capex_profile = "Asset-light"
        evidence.append("Business model appears asset-light.")
    elif maintenance_hits or asset_intensity == "High":
        capex_profile = "Maintenance-heavy"
        if maintenance_hits:
            evidence.append(f"Maintenance CAPEX evidence: {', '.join(maintenance_hits)}.")
    else:
        capex_profile = "Unknown"

    if wc_support_hits:
        working_capital_profile = "Contract-liability-supported"
        evidence.append(f"Cash collection support evidence: {', '.join(wc_support_hits)}.")
    elif _has_any(blob, ["inventory", "hardware", "retail", "manufacturing"]):
        working_capital_profile = "Inventory-heavy"
        evidence.append("Inventory or goods-based working capital language found.")
    elif wc_risk_hits:
        working_capital_profile = "Receivables-heavy"
        evidence.append(f"Working-capital risk evidence: {', '.join(wc_risk_hits)}.")
    elif business_model in {"SaaS", "Marketplace"}:
        working_capital_profile = "Negative working capital"
        evidence.append("Asset-light/subscription model may receive cash before revenue recognition.")
    else:
        working_capital_profile = "Unknown"

    if business_model in {"SaaS", "Marketplace"}:
        margin_profile = "High gross margin"
    elif business_model in {"Retail", "Manufacturing", "Hardware"}:
        margin_profile = "Low gross margin"
    elif _has_any(blob, ["mix shift", "product mix", "segment mix"]):
        margin_profile = "Mix-shift-driven"
        evidence.append("Mix-shift language found.")
    elif _has_any(blob, ["cyclical", "cycle", "commodity"]):
        margin_profile = "Cyclical"
        evidence.append("Cyclical/commodity language found.")
    else:
        margin_profile = "Unknown"

    acquisition_intensity = "High" if len(acquisition_hits) >= 2 else "Medium" if acquisition_hits else "Low"
    return {
        "asset_intensity": asset_intensity,
        "business_model": business_model,
        "capex_profile": capex_profile,
        "working_capital_profile": working_capital_profile,
        "margin_profile": margin_profile,
        "acquisition_intensity": acquisition_intensity,
        "confidence": _confidence(len(evidence), has_financials=bool(sector or industry or description or clause_text)),
        "evidence": evidence[:10],
    }


def interpret_depreciation_amortization(
    historicals: pd.DataFrame,
    clauses: pd.DataFrame,
    business_profile: dict,
) -> dict:
    """
    Decide whether D&A is a good proxy for maintenance CAPEX.
    """
    clause_text = _clauses_text(clauses)
    capex_da_ratios = _ratio_series(historicals, "Total CAPEX", "D&A")
    median_ratio = float(capex_da_ratios.median()) if not capex_da_ratios.empty else None
    capex_tracks_da = median_ratio is not None and 0.7 <= median_ratio <= 1.3
    profile = business_profile or {}
    warnings: list[str] = []
    adjustments: list[str] = []
    reasons: list[str] = []

    acquisition_distortion = profile.get("acquisition_intensity") in {"Medium", "High"} or _has_any(clause_text, MA_TERMS)
    growth_distortion = profile.get("capex_profile") == "Growth-capex-heavy" or _has_any(clause_text, CAPACITY_TERMS)
    asset_heavy = profile.get("asset_intensity") == "High"
    asset_light = profile.get("asset_intensity") == "Low" or profile.get("business_model") in {"SaaS", "Marketplace", "Services"}

    if capex_tracks_da:
        reasons.append(f"Historical CAPEX/D&A median is {median_ratio:.2f}x.")
    elif median_ratio is not None:
        reasons.append(f"Historical CAPEX/D&A median is {median_ratio:.2f}x, so the line items do not track closely.")

    if asset_heavy and capex_tracks_da and not acquisition_distortion and not growth_distortion:
        proxy = True
        method = "D&A proxy"
        reasons.append("Asset-heavy profile with no major filing evidence of growth CAPEX or acquired-intangible distortion.")
        confidence = _confidence(len(reasons), has_financials=True)
    else:
        proxy = False
        method = "manual review"
        warnings.append("D&A may not represent maintenance CAPEX for this company.")
        if asset_light:
            adjustments.append("Use CAPEX history or an industry maintenance-CAPEX proxy instead of D&A alone.")
            reasons.append("Asset-light/software-like models often include amortization that is not maintenance reinvestment.")
        if acquisition_distortion:
            adjustments.append("Separate acquired intangible amortization from operating maintenance reinvestment.")
            warnings.append("Acquisition or intangible amortization can depress EBIT while overstating maintenance CAPEX.")
            reasons.append("M&A/intangible evidence is present.")
        if growth_distortion:
            adjustments.append("Classify current CAPEX into growth versus maintenance before normalizing FCF.")
            warnings.append("Capacity expansion or infrastructure build-out may make current CAPEX a growth investment.")
            reasons.append("Growth CAPEX evidence is present.")
        if median_ratio is not None and median_ratio > 1.5:
            adjustments.append("Stress test whether growth CAPEX is recurring or discretionary.")
            warnings.append("Total CAPEX has been materially above D&A.")
            method = "% of revenue"
        if median_ratio is not None and median_ratio < 0.6:
            adjustments.append("Check whether D&A is amortization-heavy or whether the company is underinvesting.")
            warnings.append("Total CAPEX has been materially below D&A.")
            method = "industry proxy"
        confidence = _confidence(len(reasons) + len(adjustments), has_financials=median_ratio is not None, penalty=1)

    return {
        "da_as_maintenance_capex_proxy": proxy,
        "recommended_maintenance_capex_method": method,
        "reason": " ".join(reasons) or "Not enough evidence to validate D&A as a maintenance CAPEX proxy.",
        "adjustments": adjustments,
        "warnings": warnings,
        "confidence": confidence,
        "capex_da_ratio": median_ratio,
        "reliability": "High" if proxy and confidence == "High" else "Medium" if proxy or confidence == "Medium" else "Low",
    }


def interpret_capex(historicals: pd.DataFrame, clauses: pd.DataFrame, business_profile: dict) -> dict:
    """
    Classify CAPEX into maintenance CAPEX, growth CAPEX, and uncertain CAPEX.
    """
    clause_text = _clauses_text(clauses)
    latest_revenue = _latest(historicals, "Revenue", 0) or 0
    latest_capex = abs(_latest(historicals, "Total CAPEX", 0) or 0)
    latest_da = abs(_latest(historicals, "D&A", 0) or 0)
    capex_da_ratios = _ratio_series(historicals, "Total CAPEX", "D&A")
    median_ratio = float(capex_da_ratios.median()) if not capex_da_ratios.empty else None
    capex_pct_revenue = _safe_div(latest_capex, latest_revenue)
    da_read = interpret_depreciation_amortization(historicals, clauses, business_profile)
    evidence = list((business_profile or {}).get("evidence", [])[:3])
    warnings: list[str] = []
    implications: list[str] = []

    growth_evidence = _evidence_matches(clause_text, CAPACITY_TERMS)
    maintenance_evidence = _evidence_matches(clause_text, MAINTENANCE_TERMS)
    if growth_evidence:
        evidence.append(f"Growth CAPEX clauses: {', '.join(growth_evidence)}.")
    if maintenance_evidence:
        evidence.append(f"Maintenance CAPEX clauses: {', '.join(maintenance_evidence)}.")

    if da_read["da_as_maintenance_capex_proxy"]:
        maintenance = min(latest_capex, latest_da) if latest_da else latest_capex
        method = "D&A proxy"
    elif (business_profile or {}).get("capex_profile") == "Growth-capex-heavy" or growth_evidence or (median_ratio is not None and median_ratio > 1.5):
        maintenance = min(latest_capex, latest_da if latest_da else latest_capex * 0.5)
        method = "D&A floor plus growth-CAPEX evidence"
        warnings.append("Do not treat all current CAPEX as maintenance; a growth component is likely.")
        implications.append("Near-term FCF is lower, but normalized owner earnings should separate capacity investment from maintenance.")
    elif (business_profile or {}).get("capex_profile") == "Asset-light" or (median_ratio is not None and median_ratio < 0.6):
        maintenance = latest_capex
        method = "CAPEX history / asset-light review"
        warnings.append("D&A may be amortization-heavy; maintenance CAPEX is better anchored to actual CAPEX history.")
        implications.append("Avoid penalizing maintenance reinvestment with acquired-intangible amortization unless the business truly requires it.")
    else:
        maintenance = min(latest_capex, latest_da) if latest_da else latest_capex * 0.6
        method = "manual review"
        warnings.append("Maintenance/growth CAPEX split is not disclosed; user review is required.")

    if maintenance_evidence and latest_da:
        maintenance = max(maintenance, min(latest_capex, latest_da))
    growth = max(latest_capex - maintenance, 0)
    uncertain = 0.0
    if method == "manual review":
        uncertain = max(latest_capex * 0.25, 0)
        maintenance = max(maintenance - uncertain / 2, 0)
        growth = max(latest_capex - maintenance - uncertain, 0)

    growth_share = _safe_div(growth, latest_capex) or 0
    if latest_capex == 0:
        classification = "Unclear"
    elif growth_share >= 0.55:
        classification = "Growth-heavy"
    elif growth_share <= 0.25:
        classification = "Maintenance-heavy"
    else:
        classification = "Mixed"

    implications.append(f"Review maintenance CAPEX assumption versus latest estimated maintenance CAPEX of {_fmt_money(maintenance)}.")
    if capex_pct_revenue is not None:
        implications.append(f"Latest total CAPEX is {_fmt_pct(capex_pct_revenue)} of revenue.")

    return {
        "maintenance_capex_estimate": maintenance,
        "growth_capex_estimate": growth,
        "uncertain_capex": uncertain,
        "method": method,
        "classification": classification,
        "evidence": evidence,
        "warnings": warnings + da_read.get("warnings", []),
        "dcf_implications": implications,
        "confidence": da_read.get("confidence", "Medium"),
        "capex_pct_revenue": capex_pct_revenue,
        "growth_share_of_capex": growth_share,
    }


def interpret_ocf_quality(historicals: pd.DataFrame, clauses: pd.DataFrame, business_profile: dict) -> dict:
    """
    Analyze whether OCF is high-quality and sustainable.
    """
    clause_text = _clauses_text(clauses)
    latest_ocf = _latest(historicals, "OCF", 0) or 0
    latest_nopat = _latest(historicals, "NOPAT", 0) or 0
    ocf_to_nopat = _safe_div(latest_ocf, latest_nopat)
    revenue_trend = _trend(historicals, "Revenue")
    ocf_trend = _trend(historicals, "OCF")
    score = 7.0
    drivers: list[str] = []
    red_flags: list[str] = []
    suggestions: list[str] = []
    implications: list[str] = []

    if ocf_to_nopat is not None:
        drivers.append(f"OCF/NOPAT is {ocf_to_nopat:.2f}x.")
        if ocf_to_nopat < 0.7:
            score -= 2
            red_flags.append("OCF conversion trails NOPAT.")
            suggestions.append("Review working-capital investment and lower OCF margin if conversion does not normalize.")
        elif ocf_to_nopat > 1.4:
            score -= 0.5
            drivers.append("OCF exceeds NOPAT; verify whether this is durable cash conversion or timing.")
        else:
            score += 1

    if revenue_trend is not None and ocf_trend is not None and revenue_trend > 0.1 and ocf_trend < -0.05:
        score -= 1.5
        red_flags.append("OCF is moving against revenue growth.")
        suggestions.append("Stress test working-capital investment.")

    if _has_any(clause_text, WORKING_CAPITAL_SUPPORT_TERMS):
        score += 1
        drivers.append("Deferred revenue / contract liability evidence supports cash collection.")
    if _has_any(clause_text, ["receivable", "contract asset"]):
        score -= 1.5
        red_flags.append("Receivables or contract assets may be pressuring cash conversion.")
    if "inventory" in clause_text:
        score -= 1.5
        red_flags.append("Inventory build can pressure OCF and create demand risk.")
    if _has_any(clause_text, ["payable", "payables"]):
        score -= 0.75
        red_flags.append("Payables timing may temporarily support OCF.")
    if _has_any(clause_text, ["acquisition", "integration"]):
        score -= 0.5
        red_flags.append("M&A or integration activity may distort OCF comparability.")

    score = max(1, min(10, round(score)))
    quality = _quality_from_score(score)
    if red_flags:
        implications.append("DCF OCF margin and working-capital assumptions require user confirmation before changing valuation.")
    else:
        implications.append("OCF quality appears usable as a base-case input, subject to normal sensitivity checks.")

    return {
        "ocf_quality_score": int(score),
        "quality": quality,
        "summary": "High-quality OCF read." if quality == "High" else "OCF needs interpretation through working capital and cash conversion.",
        "working_capital_drivers": drivers,
        "red_flags": red_flags,
        "adjusted_ocf_suggestions": suggestions,
        "dcf_implications": implications,
        "confidence": _confidence(len(drivers) + len(red_flags), has_financials=latest_ocf != 0),
        "ocf_to_nopat": ocf_to_nopat,
    }


def interpret_nopat_quality(historicals: pd.DataFrame, clauses: pd.DataFrame, business_profile: dict) -> dict:
    """
    Analyze whether NOPAT reflects true operating economics.
    """
    clause_text = _clauses_text(clauses)
    latest_revenue = _latest(historicals, "Revenue", 0) or 0
    latest_sbc = _latest(historicals, "SBC", 0) or 0
    latest_ebit = _latest(historicals, "EBIT", 0) or 0
    latest_nopat = _latest(historicals, "NOPAT", 0) or 0
    sbc_pct_revenue = _safe_div(latest_sbc, latest_revenue)
    score = 7.0
    adjustments: list[str] = []
    warnings: list[str] = []
    implications: list[str] = []

    if _has_any(clause_text, MA_TERMS) or (business_profile or {}).get("acquisition_intensity") in {"Medium", "High"}:
        score -= 1.5
        adjustments.append("Identify acquired intangible amortization before treating GAAP EBIT/NOPAT as normalized.")
        warnings.append("M&A amortization may depress NOPAT without representing maintenance reinvestment.")
    if _has_any(clause_text, ONE_TIME_TERMS):
        score -= 1.5
        adjustments.append("Separate temporary restructuring, impairment, legal, or integration costs from recurring cost structure.")
        warnings.append("One-time or transition costs may distort operating income.")
    if sbc_pct_revenue is not None and sbc_pct_revenue > 0.05:
        score -= 1.5
        adjustments.append("Review SBC as a recurring compensation and dilution cost; do not automatically add it back.")
        warnings.append("SBC is material relative to revenue.")
    if _has_any(clause_text, ["mix shift", "product mix", "segment mix"]):
        score -= 0.75
        adjustments.append("Review segment mix before carrying forward latest NOPAT margin.")
    if latest_ebit < 0 < latest_revenue:
        score -= 1
        warnings.append("Negative EBIT makes normalized NOPAT margin uncertain.")
    if latest_nopat > 0 and latest_ebit <= 0:
        score -= 0.5
        warnings.append("NOPAT and EBIT direction conflict; tax or normalization assumptions need review.")

    score = max(1, min(10, round(score)))
    quality = _quality_from_score(score)
    implications.append("DCF NOPAT margin should stay as a user-confirmed assumption when NOPAT quality is Medium/Low.")
    if quality == "High":
        implications.append("Reported EBIT/NOPAT appears broadly usable for base-case normalization.")

    return {
        "nopat_quality_score": int(score),
        "quality": quality,
        "summary": "NOPAT appears clean enough for base-case normalization." if quality == "High" else "NOPAT needs adjustment review before being treated as economic earnings.",
        "adjustments": adjustments,
        "warnings": warnings,
        "dcf_implications": implications,
        "confidence": _confidence(len(adjustments) + len(warnings), has_financials=latest_revenue != 0),
        "sbc_pct_revenue": sbc_pct_revenue,
    }


def build_accounting_interpretation(dataset: dict, historicals: pd.DataFrame, clauses: pd.DataFrame | None = None) -> dict:
    business_profile = infer_business_profile(dataset, clauses)
    da = interpret_depreciation_amortization(historicals, clauses if clauses is not None else pd.DataFrame(), business_profile)
    capex = interpret_capex(historicals, clauses if clauses is not None else pd.DataFrame(), business_profile)
    ocf = interpret_ocf_quality(historicals, clauses if clauses is not None else pd.DataFrame(), business_profile)
    nopat = interpret_nopat_quality(historicals, clauses if clauses is not None else pd.DataFrame(), business_profile)

    distortion = "none"
    if any("SBC" in warning or "SBC" in adjustment for warning in nopat.get("warnings", []) for adjustment in nopat.get("adjustments", [""])):
        distortion = "SBC"
    elif any("amortization" in text.lower() for text in nopat.get("warnings", []) + da.get("warnings", [])):
        distortion = "M&A amortization"
    elif any("inventory" in text.lower() for text in ocf.get("red_flags", [])):
        distortion = "inventory"
    elif any("receivable" in text.lower() for text in ocf.get("red_flags", [])):
        distortion = "receivables"
    elif capex.get("classification") == "Growth-heavy":
        distortion = "growth CAPEX"
    elif ocf.get("red_flags"):
        distortion = "working capital"

    confidence_score = 0
    for item in [da, capex, ocf, nopat]:
        confidence_score += {"High": 2, "Medium": 1, "Low": 0}.get(item.get("confidence"), 1)
    valuation_confidence = "High" if confidence_score >= 7 else "Medium" if confidence_score >= 4 else "Low"
    if da.get("reliability") == "Low" or ocf.get("quality") == "Low" or nopat.get("quality") == "Low":
        valuation_confidence = "Low"

    warnings = []
    warnings.extend(da.get("warnings", []))
    warnings.extend(capex.get("warnings", []))
    warnings.extend(ocf.get("red_flags", []))
    warnings.extend(nopat.get("warnings", []))
    warnings = list(dict.fromkeys(warnings))

    return {
        "business_profile": business_profile,
        "depreciation_amortization": da,
        "capex": capex,
        "ocf": ocf,
        "nopat": nopat,
        "cards": {
            "D&A Reliability as Maintenance CAPEX Proxy": da.get("reliability", "Low"),
            "OCF Quality": ocf.get("quality", "Medium"),
            "CAPEX Classification": capex.get("classification", "Unclear"),
            "NOPAT Quality": nopat.get("quality", "Medium"),
            "Main Accounting Distortion": distortion,
        },
        "warnings": warnings,
        "valuation_confidence": valuation_confidence,
    }


def build_accounting_interpretation_table(interpretation: dict, historicals: pd.DataFrame) -> pd.DataFrame:
    if not interpretation:
        return pd.DataFrame()
    da = interpretation.get("depreciation_amortization", {})
    capex = interpretation.get("capex", {})
    ocf = interpretation.get("ocf", {})
    nopat = interpretation.get("nopat", {})
    profile = interpretation.get("business_profile", {})
    business_logic = (
        f"{profile.get('business_model', 'Unknown')} model; "
        f"{profile.get('asset_intensity', 'Unknown')} asset intensity; "
        f"{profile.get('capex_profile', 'Unknown')} CAPEX profile."
    )

    latest_da = _latest(historicals, "D&A")
    latest_capex = _latest(historicals, "Total CAPEX")
    latest_ocf = _latest(historicals, "OCF")
    latest_nopat = _latest(historicals, "NOPAT")
    latest_fcf = _latest(historicals, "FCF")
    evidence = "; ".join(profile.get("evidence", [])[:3])

    rows = [
        {
            "Metric": "D&A",
            "Reported Value": _fmt_money(latest_da),
            "Economic Interpretation": "Usable maintenance CAPEX proxy" if da.get("da_as_maintenance_capex_proxy") else "Weak maintenance CAPEX proxy",
            "Business Logic": business_logic,
            "Clause Evidence": evidence,
            "Model Impact": "Use D&A proxy only after review." if da.get("da_as_maintenance_capex_proxy") else "Do not use D&A blindly as maintenance CAPEX.",
            "Confidence": da.get("confidence"),
            "Suggested Action": da.get("recommended_maintenance_capex_method"),
        },
        {
            "Metric": "CAPEX",
            "Reported Value": _fmt_money(latest_capex),
            "Economic Interpretation": capex.get("classification", "Unclear"),
            "Business Logic": f"Estimated maintenance {_fmt_money(capex.get('maintenance_capex_estimate'))}; growth {_fmt_money(capex.get('growth_capex_estimate'))}.",
            "Clause Evidence": "; ".join(capex.get("evidence", [])[:3]),
            "Model Impact": "; ".join(capex.get("dcf_implications", [])[:2]),
            "Confidence": capex.get("confidence"),
            "Suggested Action": capex.get("method"),
        },
        {
            "Metric": "OCF",
            "Reported Value": _fmt_money(latest_ocf),
            "Economic Interpretation": f"{ocf.get('quality', 'Medium')} quality cash conversion",
            "Business Logic": "; ".join(ocf.get("working_capital_drivers", [])[:2]) or profile.get("working_capital_profile"),
            "Clause Evidence": "; ".join(ocf.get("red_flags", [])[:2]),
            "Model Impact": "; ".join(ocf.get("dcf_implications", [])[:2]),
            "Confidence": ocf.get("confidence"),
            "Suggested Action": "; ".join(ocf.get("adjusted_ocf_suggestions", [])[:2]) or "Use as-is with sensitivity checks.",
        },
        {
            "Metric": "NOPAT",
            "Reported Value": _fmt_money(latest_nopat),
            "Economic Interpretation": f"{nopat.get('quality', 'Medium')} quality operating earnings",
            "Business Logic": nopat.get("summary"),
            "Clause Evidence": "; ".join(nopat.get("warnings", [])[:2]),
            "Model Impact": "; ".join(nopat.get("dcf_implications", [])[:2]),
            "Confidence": nopat.get("confidence"),
            "Suggested Action": "; ".join(nopat.get("adjustments", [])[:2]) or "Use as-is with normalized margin review.",
        },
        {
            "Metric": "FCF",
            "Reported Value": _fmt_money(latest_fcf),
            "Economic Interpretation": "Reported FCF is an input, not the economic conclusion.",
            "Business Logic": "FCF must be read through OCF quality and maintenance/growth CAPEX classification.",
            "Clause Evidence": interpretation.get("cards", {}).get("Main Accounting Distortion"),
            "Model Impact": f"Valuation confidence: {interpretation.get('valuation_confidence', 'Medium')}.",
            "Confidence": interpretation.get("valuation_confidence", "Medium"),
            "Suggested Action": "Confirm assumption changes manually before valuation changes.",
        },
    ]
    return pd.DataFrame(rows)
