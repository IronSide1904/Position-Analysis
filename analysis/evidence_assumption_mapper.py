from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone

import pandas as pd


UNAVAILABLE = "Unavailable"

TOPIC_LABELS = {
    "GUIDANCE_OUTLOOK": "Guidance / Outlook",
    "BACKLOG_RPO_BOOKINGS": "Backlog / RPO / Bookings",
    "SBC_DILUTION_BUYBACKS": "SBC / Dilution / Buybacks",
    "M_AND_A": "M&A",
    "CAPEX": "CAPEX",
    "REVENUE_GROWTH": "Revenue Growth",
    "MARGIN_COSTS": "Margins / Costs",
    "OCF_WORKING_CAPITAL": "OCF / Working Capital",
    "DEBT_LIQUIDITY": "Debt / Liquidity",
    "MANAGEMENT_CREDIBILITY": "Management Credibility",
    "MOAT_COMPETITION": "Moat / Competition",
    "RISK_FACTORS": "Risk Factors",
}

MODEL_LINE_LABELS = {
    "revenue_growth": "Revenue Growth",
    "revenue_cagr": "Revenue Growth",
    "scenario_probability": "Scenario Probability",
    "growth_capex_pct_revenue": "Growth CAPEX % Revenue",
    "maintenance_capex_pct_revenue": "Maintenance CAPEX % Revenue",
    "total_capex_pct_revenue": "Total CAPEX % Revenue",
    "sbc": "SBC % Revenue",
    "sbc_pct_revenue": "SBC % Revenue",
    "ocf_margin": "OCF Margin",
    "fcf_margin": "FCF Margin",
    "nopat_margin": "NOPAT Margin",
    "operating_margin": "NOPAT Margin",
    "gross_margin": "Gross Margin",
    "wacc": "WACC",
    "terminal_multiple": "Terminal Multiple",
    "diluted_shares": "Diluted Shares",
    "working_capital_pct_revenue": "Working Capital % Revenue",
}

DCF_LINE_ALIASES = {
    "revenue_growth": "revenue_cagr",
    "sbc": "sbc_pct_revenue",
    "operating_margin": "nopat_margin",
}

IMPACT_COLUMNS = [
    "evidence_id",
    "ticker",
    "form",
    "filing_date",
    "section",
    "topic",
    "topic_label",
    "subtopic",
    "clause_text",
    "evidence_summary",
    "model_line",
    "model_line_label",
    "implied_value_status",
    "assumption_signal",
    "assumption_signal_help",
    "period",
    "implied_value",
    "implied_value_display",
    "implied_range_low",
    "implied_range_high",
    "implied_range_display",
    "current_dcf_value",
    "current_dcf_value_display",
    "delta",
    "delta_vs_current_dcf",
    "delta_display",
    "direction",
    "confidence",
    "evidence_grade",
    "method",
    "calculation",
    "reason",
    "numeric_signals",
    "confidence_reason",
    "exact_value_note",
    "base_value",
    "base_value_display",
    "suggested_action",
    "recommended_case",
    "source_url",
    "review_status",
    "user_note",
    "extraction_method",
    "warnings",
    "where_to_verify",
    "search_keywords",
]

ASSUMPTION_SIGNAL_HELP = {
    "Calculated %": "Direct numeric guidance was converted into a DCF assumption.",
    "Implied Range": "A disclosed numeric range was converted into an implied DCF range.",
    "Estimated Range": "No exact percentage was disclosed; the range is an economic review estimate.",
    "Qualitative Support": "The clause supports reviewing the assumption, but it does not provide a precise model input.",
    "Revenue Visibility Support": "Backlog, RPO, bookings, demand, pipeline, customer growth, or retention improves revenue visibility without setting an exact growth rate.",
    "Margin Expansion Support": "The clause supports potential margin improvement without a precise margin percentage.",
    "Margin Pressure Warning": "The clause points to cost, mix, or operating pressure that may reduce margins.",
    "CAPEX Increase Warning": "The clause points to facility, data center, infrastructure, equipment, capacity, or similar investment needs.",
    "Near-Term FCF Pressure": "The clause may pressure near-term free cash flow without setting a precise FCF margin.",
    "SBC / Dilution Warning": "SBC, equity awards, options, RSUs, dilution, or share-count pressure may affect per-share value.",
    "Bear Case Support": "The clause makes downside scenario assumptions more relevant.",
    "Bull Case Support": "The clause makes upside scenario assumptions more relevant.",
    "Risk Reduction Support": "The clause points to lower risk, better visibility, or stronger confidence in assumptions.",
    "Risk Increase Warning": "The clause points to higher risk or weaker confidence in assumptions.",
    "Not Enough Evidence": "The clause does not contain enough usable information to classify the assumption impact.",
}


def _clean_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _finite(value) -> bool:
    try:
        return value is not None and not pd.isna(value) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_float(value) -> float | None:
    if _finite(value):
        return float(value)
    return None


def _pct_display(value) -> str:
    return f"{float(value) * 100:.1f}%" if _finite(value) else UNAVAILABLE


def _pts_display(value) -> str:
    if not _finite(value):
        return "No exact % available"
    sign = "+" if float(value) >= 0 else ""
    return f"{sign}{float(value) * 100:.1f} pts"


def _money_display(value) -> str:
    if not _finite(value):
        return UNAVAILABLE
    amount = float(value)
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 1_000_000_000:
        return f"{sign}${amount / 1_000_000_000:.2f}B"
    if amount >= 1_000_000:
        return f"{sign}${amount / 1_000_000:.1f}M"
    if amount >= 1_000:
        return f"{sign}${amount / 1_000:.1f}K"
    return f"{sign}${amount:,.0f}"


def _value_display(value, unit: str | None = None) -> str:
    if isinstance(value, str):
        return value or UNAVAILABLE
    if not _finite(value):
        return UNAVAILABLE
    if unit in {"money"}:
        return _money_display(value)
    if unit in {"percent", "basis_points"}:
        return _pct_display(value)
    return f"{float(value):,.1f}"


def _assumption_value_display(model_line: str, value) -> str:
    if not _finite(value):
        return UNAVAILABLE
    canonical_line = DCF_LINE_ALIASES.get(model_line, model_line)
    if canonical_line == "terminal_multiple":
        return f"{float(value):.1f}x"
    return _pct_display(value)


def _period_from_text(text: str) -> str:
    match = re.search(r"\b(?:FY|fiscal(?: year)?\s*)?(20\d{2})\b", text, flags=re.I)
    if match:
        return f"FY{match.group(1)}E"
    if re.search(r"\bnext year\b|\bfollowing year\b", text, flags=re.I):
        return "Next FY"
    if re.search(r"\bquarter|Q[1-4]\b", text, flags=re.I):
        return "Next Quarter"
    return "Forward period"


def _money_to_float(raw_number: str, suffix: str | None) -> float:
    number = float(str(raw_number).replace(",", ""))
    scale = 1.0
    suffix = (suffix or "").lower()
    if suffix.startswith("b"):
        scale = 1_000_000_000.0
    elif suffix.startswith("m"):
        scale = 1_000_000.0
    elif suffix.startswith("k"):
        scale = 1_000.0
    return number * scale


def _percent_to_float(raw_number: str) -> float:
    return float(str(raw_number).replace(",", "")) / 100.0


def _detect_metric(text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ["free cash flow", "fcf"]):
        return "fcf"
    if any(token in lower for token in ["operating cash flow", "cash provided by operating", "ocf"]):
        return "ocf"
    if any(token in lower for token in ["capital expenditure", "capital expenditures", "capex", "capital spending"]):
        return "capex"
    if any(token in lower for token in ["stock-based compensation", "share-based compensation", "sbc"]):
        return "sbc"
    if "gross margin" in lower:
        return "gross_margin"
    if any(token in lower for token in ["operating margin", "operating income margin", "ebit margin"]):
        return "operating_margin"
    if "margin" in lower:
        return "gross_margin"
    if any(token in lower for token in ["backlog", "rpo", "remaining performance obligation", "bookings"]):
        return "backlog"
    if any(token in lower for token in ["revenue growth", "sales growth"]):
        return "revenue_growth"
    if any(token in lower for token in ["revenue", "sales"]):
        return "revenue"
    return "qualitative"


def extract_numeric_guidance(clause_text: str) -> dict:
    """
    Extract numeric values from guidance clauses.

    Detects money, percentage, basis-point, range, midpoint, and period clues.
    Returns a directional low-confidence object when no exact number exists.
    """
    text = _clean_text(clause_text)
    lower = text.lower()
    metric = _detect_metric(text)
    period = _period_from_text(text)

    bps_match = re.search(r"([+-]?\d[\d,]*(?:\.\d+)?)\s*(?:basis points|bps)\b", lower)
    if bps_match:
        return {
            "metric": metric if metric != "qualitative" else "operating_margin",
            "bps_change": float(bps_match.group(1).replace(",", "")),
            "unit": "basis_points",
            "period": period,
            "confidence": "Medium",
        }

    money_matches = re.findall(
        r"\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(billion|million|thousand|bn|mm|m|b|k)\b",
        lower,
        flags=re.I,
    )
    if money_matches:
        values = [_money_to_float(number, suffix) for number, suffix in money_matches]
        low = min(values)
        high = max(values)
        midpoint = sum(values[:2]) / 2 if len(values) >= 2 else values[0]
        return {
            "metric": metric,
            "low": low,
            "high": high,
            "midpoint": midpoint,
            "unit": "money",
            "period": period,
            "confidence": "Medium" if len(values) >= 2 else "High",
        }

    percent_matches = re.findall(r"([+-]?\d[\d,]*(?:\.\d+)?)\s*(?:%|percent)", lower)
    if percent_matches:
        values = [_percent_to_float(value) for value in percent_matches]
        low = min(values)
        high = max(values)
        midpoint = sum(values[:2]) / 2 if len(values) >= 2 else values[0]
        confidence = "High" if metric in {"revenue_growth", "gross_margin", "operating_margin", "ocf", "fcf"} else "Medium"
        return {
            "metric": metric if metric != "revenue" else "revenue_growth",
            "low": low,
            "high": high,
            "midpoint": midpoint,
            "unit": "percent",
            "period": period,
            "confidence": confidence,
        }

    direction = "Directional negative" if any(word in lower for word in ["decline", "decrease", "lower", "pressure", "weaker"]) else "Directional positive" if any(word in lower for word in ["increase", "growth", "improve", "strong", "expand", "higher"]) else "Directional mixed"
    return {
        "metric": metric,
        "implied_value_display": direction,
        "unit": "directional",
        "period": period,
        "confidence": "Low",
        "method": "Qualitative guidance only; no numeric value extracted.",
    }


def extract_numeric_signals(clause_text: str) -> dict:
    """
    Extract numeric signals before assigning direction-only labels.
    """
    guidance = extract_numeric_guidance(clause_text)
    unit = guidance.get("unit")
    if unit == "directional":
        return {
            "has_numeric_signal": False,
            "metric": guidance.get("metric"),
            "period": guidance.get("period"),
            "confidence": guidance.get("confidence"),
            "reason": "No extractable numeric value.",
        }
    return {
        "has_numeric_signal": True,
        "metric": guidance.get("metric"),
        "value_low": guidance.get("low"),
        "value_high": guidance.get("high"),
        "midpoint": guidance.get("midpoint"),
        "bps_change": guidance.get("bps_change"),
        "unit": unit,
        "period": guidance.get("period"),
        "confidence": guidance.get("confidence"),
    }


def _latest_financial_value(financials, column: str) -> float | None:
    if financials is None:
        return None
    if isinstance(financials, pd.DataFrame):
        if financials.empty or column not in financials:
            return None
        values = pd.to_numeric(financials[column], errors="coerce").dropna()
        values = values[values != 0]
        return float(values.iloc[-1]) if not values.empty else None
    if isinstance(financials, dict):
        value = financials.get(column) or financials.get(column.lower()) or financials.get(column.replace(" ", "_").lower())
        return _as_float(value)
    return None


def _base_revenue(financials, dcf_assumptions: dict) -> float | None:
    return (
        _latest_financial_value(financials, "Revenue")
        or _as_float(dcf_assumptions.get("revenue"))
        or _as_float(dcf_assumptions.get("ltm_revenue"))
        or _as_float(dcf_assumptions.get("forecast_revenue"))
    )


def _forecast_revenue(financials, dcf_assumptions: dict) -> float | None:
    base = _base_revenue(financials, dcf_assumptions)
    if base is None:
        return None
    return base * (1 + float(dcf_assumptions.get("revenue_cagr", dcf_assumptions.get("revenue_growth", 0)) or 0))


def _current_margin(financials, dcf_assumptions: dict, model_line: str) -> float | None:
    canonical = DCF_LINE_ALIASES.get(model_line, model_line)
    if canonical in dcf_assumptions and dcf_assumptions.get(canonical) is not None:
        return _as_float(dcf_assumptions.get(canonical))
    revenue = _base_revenue(financials, dcf_assumptions)
    if not revenue:
        return None
    column_map = {
        "gross_margin": "Gross Profit",
        "nopat_margin": "NOPAT",
        "ocf_margin": "OCF",
        "fcf_margin": "FCF",
        "sbc_pct_revenue": "SBC",
    }
    numerator = _latest_financial_value(financials, column_map.get(canonical, ""))
    return float(numerator) / revenue if numerator is not None and revenue else None


def _model_line_for_guidance(extracted_guidance: dict, fallback_model_line: str | None = None) -> str:
    metric = extracted_guidance.get("metric")
    fallback = fallback_model_line or "scenario_probability"
    mapping = {
        "revenue": "revenue_growth",
        "revenue_growth": "revenue_growth",
        "gross_margin": "gross_margin",
        "operating_margin": "nopat_margin",
        "capex": "growth_capex_pct_revenue" if fallback == "growth_capex_pct_revenue" else "total_capex_pct_revenue",
        "sbc": "sbc_pct_revenue",
        "ocf": "ocf_margin",
        "fcf": "fcf_margin",
        "backlog": "revenue_growth",
    }
    return mapping.get(metric, fallback)


def _range_display(low: float | None, high: float | None, unit: str = "percent") -> str:
    if low is None or high is None:
        return UNAVAILABLE
    if unit == "points":
        return f"{low * 100:+.1f} to {high * 100:+.1f} pts"
    return f"{_pct_display(low)} - {_pct_display(high)}"


def _estimated_range_result(
    model_line: str,
    low_delta: float,
    high_delta: float,
    direction: str,
    reason: str,
    period: str = "Forward period",
    confidence: str = "Low",
) -> dict:
    return {
        "model_line": model_line,
        "period": period,
        "implied_value": None,
        "implied_range_low": low_delta,
        "implied_range_high": high_delta,
        "unit": "estimated_points",
        "base_value": None,
        "method": "Estimated range from clause economics and current DCF baseline.",
        "calculation": f"Estimated impact range: {_range_display(low_delta, high_delta, unit='points')}.",
        "confidence": confidence,
        "direction": direction,
        "warnings": [],
        "implied_value_status": "Estimated Range",
        "implied_value_display": _range_display(low_delta, high_delta, unit="points"),
        "reason": reason,
        "confidence_reason": "Low confidence because no exact guidance value was disclosed.",
        "exact_value_note": "No exact percentage was disclosed; range is a review estimate.",
    }


def _clamp_pct_assumption(model_line: str, value: float) -> float:
    canonical_line = DCF_LINE_ALIASES.get(model_line, model_line)
    bounds = {
        "revenue_cagr": (-0.50, 1.00),
        "gross_margin": (0.00, 1.00),
        "nopat_margin": (-0.50, 0.80),
        "ocf_margin": (-0.50, 0.80),
        "fcf_margin": (-0.50, 0.80),
        "wacc": (0.01, 0.35),
        "sbc_pct_revenue": (0.00, 0.80),
        "growth_capex_pct_revenue": (0.00, 0.80),
        "maintenance_capex_pct_revenue": (0.00, 0.80),
        "total_capex_pct_revenue": (0.00, 0.80),
        "working_capital_pct_revenue": (-0.50, 0.80),
    }.get(canonical_line, (-1.00, 1.00))
    return min(max(float(value), bounds[0]), bounds[1])


def _current_assumption_value(model_line: str, dcf_assumptions: dict) -> float | None:
    canonical_line = DCF_LINE_ALIASES.get(model_line, model_line)
    return _as_float((dcf_assumptions or {}).get(canonical_line))


def _estimated_assumption_range_result(
    model_line: str,
    dcf_assumptions: dict,
    low_delta: float,
    high_delta: float,
    direction: str,
    reason: str,
    period: str = "Forward period",
    confidence: str = "Low",
) -> dict:
    current = _current_assumption_value(model_line, dcf_assumptions)
    base = current if current is not None else 0.0
    low_value = _clamp_pct_assumption(model_line, base + min(low_delta, high_delta))
    high_value = _clamp_pct_assumption(model_line, base + max(low_delta, high_delta))
    midpoint = (low_value + high_value) / 2
    range_display = _range_display(low_value, high_value)
    delta_display = _range_display(low_value - base, high_value - base, unit="points")
    return {
        "model_line": model_line,
        "period": period,
        "implied_value": midpoint,
        "implied_range_low": low_value,
        "implied_range_high": high_value,
        "unit": "estimated_percent",
        "base_value": None,
        "method": "Estimated assumption range from clause economics and current DCF baseline.",
        "calculation": f"Estimated assumption range: {range_display}; estimated change versus current DCF: {delta_display}.",
        "confidence": confidence,
        "direction": direction,
        "warnings": [],
        "implied_value_status": "Estimated Range",
        "implied_value_display": range_display,
        "reason": reason,
        "confidence_reason": "Low confidence because no exact DCF assumption percentage was disclosed.",
        "exact_value_note": "Estimated range only; review before applying to a DCF case.",
    }


def _scenario_support_result(
    model_line: str,
    direction: str,
    reason: str,
    period: str = "Forward period",
    confidence: str = "Medium",
) -> dict:
    return {
        "model_line": model_line,
        "period": period,
        "implied_value": None,
        "implied_range_low": None,
        "implied_range_high": None,
        "unit": "scenario_support",
        "base_value": None,
        "method": "Scenario support only; no direct DCF percentage assigned.",
        "calculation": "Evidence supports scenario review but does not mechanically set a DCF assumption.",
        "confidence": confidence,
        "direction": direction,
        "warnings": [],
        "implied_value_status": "Scenario Support",
        "implied_value_display": "Scenario support",
        "reason": reason,
        "confidence_reason": "Scenario support because conversion to a model percentage requires more data.",
        "exact_value_note": "No exact percentage available.",
    }


def map_clause_to_multi_line_impacts(clause, financials, dcf_assumptions):
    """
    Return one row per model-line impact, not one vague row per clause.
    """
    row = clause if isinstance(clause, dict) else dict(clause)
    text = _clean_text(row.get("clause_text"))
    lower = text.lower()
    topic = row.get("topic")
    fallback_line = row.get("model_line_affected") or "scenario_probability"
    guidance = extract_numeric_guidance(text)
    signal = extract_numeric_signals(text)
    period = guidance.get("period") or "Forward period"

    if guidance.get("metric") == "backlog":
        backlog_growth = _as_float(guidance.get("midpoint"))
        low_delta = max(0.005, min((backlog_growth or 0.20) * 0.05, 0.03))
        high_delta = max(low_delta + 0.005, min((backlog_growth or 0.20) * 0.15, 0.05))
        return [
            _estimated_assumption_range_result("revenue_growth", dcf_assumptions, low_delta, high_delta, "Increase", "Backlog/RPO growth improves revenue visibility; only a conservative fraction is mapped into estimated revenue CAGR.", period, "Low") | {"numeric_signals": signal},
            _scenario_support_result("scenario_probability", "Bull support", "Backlog visibility can support Base/Bull scenario weighting.", period, "Medium") | {"implied_value_display": "Base/Bull support", "numeric_signals": signal},
        ]

    if signal.get("has_numeric_signal"):
        model_line = _model_line_for_guidance(guidance, fallback_line)
        implied = calculate_implied_assumption(guidance, model_line, financials, dcf_assumptions)
        status = "Range" if implied.get("implied_range_low") is not None and implied.get("implied_range_high") is not None and implied.get("implied_range_low") != implied.get("implied_range_high") else "Calculated"
        implied.update(
            {
                "model_line": model_line,
                "implied_value_status": status,
                "reason": "Numeric signal was extracted before directional classification.",
                "numeric_signals": signal,
                "confidence_reason": "High when metric and period are clear; Medium when base value or mapping requires assumption.",
                "exact_value_note": "Exact or range-based numeric signal was available.",
            }
        )
        return [implied]

    if topic == "CAPEX" or any(word in lower for word in ["capital expenditures", "capex", "facility expansion", "data center", "infrastructure", "manufacturing equipment", "automation", "capacity"]):
        return [
            _estimated_range_result("growth_capex_pct_revenue", 0.01, 0.03, "Increase", "Clause mentions expansion or infrastructure investment, which usually raises growth CAPEX intensity.", period, "Low"),
            _scenario_support_result("fcf_margin", "Decrease", "Higher growth CAPEX creates near-term FCF pressure.", period, "Medium") | {"implied_value_display": "Near-term FCF pressure"},
            _estimated_assumption_range_result("revenue_growth", dcf_assumptions, 0.005, 0.02, "Increase", "Capacity investment may support medium-term revenue, but no revenue amount was disclosed.", period, "Low"),
            _estimated_range_result("maintenance_capex_pct_revenue", 0.00, 0.01, "Increase", "A larger asset base can raise future maintenance CAPEX needs.", period, "Low"),
        ]

    if topic == "BACKLOG_RPO_BOOKINGS" or any(word in lower for word in ["backlog", "rpo", "remaining performance obligation", "bookings"]):
        return [
            _estimated_assumption_range_result("revenue_growth", dcf_assumptions, 0.01, 0.03, "Increase", "Backlog/RPO growth improves revenue visibility, but no direct revenue conversion was disclosed.", period, "Low"),
            _scenario_support_result("scenario_probability", "Bull support", "Backlog visibility can support Base/Bull scenario weighting.", period, "Medium") | {"implied_value_display": "Base/Bull support"},
        ]

    if topic == "SBC_DILUTION_BUYBACKS" or any(word in lower for word in ["stock-based compensation", "share-based compensation", "rsu", "equity awards", "dilution"]):
        current_sbc = float(dcf_assumptions.get("sbc_pct_revenue", 0.0) or 0.0)
        return [
            _estimated_range_result("sbc_pct_revenue", max(current_sbc, 0.02), max(current_sbc + 0.03, 0.05), "Increase", "SBC language without dollars uses current SBC intensity as the baseline review range.", period, "Low"),
            _scenario_support_result("diluted_shares", "Increase", "Equity awards can dilute per-share value unless offset by buybacks.", period, "Medium") | {"implied_value_display": "Per-share value pressure"},
        ]

    if topic == "OCF_WORKING_CAPITAL" or any(word in lower for word in ["receivable", "inventory", "contract asset", "deferred revenue", "contract liabilities", "payables", "working capital"]):
        if any(word in lower for word in ["deferred revenue", "contract liabilities"]):
            return [_scenario_support_result("ocf_margin", "Increase", "Deferred revenue or contract liabilities can support OCF and revenue visibility.", period, "Medium") | {"implied_value_display": "OCF support"}]
        if "inventory" in lower:
            return [
                _scenario_support_result("ocf_margin", "Decrease", "Inventory build can pressure cash conversion.", period, "Medium") | {"implied_value_display": "Cash conversion pressure"},
                _scenario_support_result("scenario_probability", "Decrease", "Inventory build can add demand risk if not matched by orders.", period, "Low") | {"implied_value_display": "Demand-risk review"},
            ]
        return [_scenario_support_result("ocf_margin", "Decrease", "Working-capital growth may pressure operating cash flow.", period, "Medium") | {"implied_value_display": "OCF margin pressure"}]

    if topic == "M_AND_A" or any(word in lower for word in ["acquisition", "acquired", "business combination", "goodwill", "intangible", "integration"]):
        return [
            _estimated_assumption_range_result("revenue_growth", dcf_assumptions, 0.01, 0.04, "Increase", "Acquisitions can add revenue if acquired revenue is material.", period, "Low"),
            _estimated_assumption_range_result("nopat_margin", dcf_assumptions, -0.03, -0.01, "Decrease", "Integration costs and amortization can reduce near-term NOPAT quality.", period, "Low"),
            _scenario_support_result("terminal_multiple", "Mixed", "M&A can improve scale but raises integration and goodwill risk.", period, "Low") | {"implied_value_display": "Quality review"},
            _estimated_assumption_range_result("wacc", dcf_assumptions, 0.005, 0.015, "Increase", "Debt-financed acquisitions can increase financing risk.", period, "Low"),
        ]

    if topic == "DEBT_LIQUIDITY" or any(word in lower for word in ["debt", "credit facility", "notes payable", "liquidity", "covenant", "refinanc", "interest expense", "leverage"]):
        if any(word in lower for word in ["improve", "strong", "reduc", "repay", "lower", "refinanc"]):
            return [
                _estimated_assumption_range_result("wacc", dcf_assumptions, -0.015, -0.005, "Decrease", "Improved liquidity or lower debt risk can reduce the DCF risk premium.", period, "Low")
            ]
        return [
            _estimated_assumption_range_result("wacc", dcf_assumptions, 0.005, 0.02, "Increase", "Higher leverage, debt, or liquidity risk can raise the DCF risk premium.", period, "Low")
        ]

    if topic in {"REVENUE_GROWTH", "GUIDANCE_OUTLOOK"} or fallback_line in {"revenue_growth", "revenue_cagr"} and any(word in lower for word in ["demand", "pipeline", "customer", "retention", "growth", "strong", "weak", "pressure", "slow"]):
        if any(word in lower for word in ["decline", "decrease", "lower", "pressure", "weaker", "weak", "slow"]):
            return [
                _estimated_assumption_range_result("revenue_growth", dcf_assumptions, -0.03, -0.01, "Decrease", "Qualitative revenue language points to a lower revenue CAGR review range.", period, "Low")
            ]
        return [
            _estimated_assumption_range_result("revenue_growth", dcf_assumptions, 0.01, 0.03, "Increase", "Qualitative revenue language points to a higher revenue CAGR review range.", period, "Low")
        ]

    if any(word in lower for word in ["increase", "growth", "improve", "strong", "expand", "higher", "decline", "decrease", "lower", "pressure", "weaker"]):
        direction = "Decrease" if any(word in lower for word in ["decline", "decrease", "lower", "pressure", "weaker"]) else "Increase"
        return [
            {
                "model_line": fallback_line,
                "period": period,
                "implied_value": None,
                "implied_range_low": None,
                "implied_range_high": None,
                "unit": "directional",
                "base_value": None,
                "method": "Directional fallback after numeric and estimated interpretation failed.",
                "calculation": "No numeric, estimated-range, or scenario-support translation was available.",
                "confidence": "Low",
                "direction": direction,
                "warnings": [],
                "implied_value_status": "Directional Only",
                "implied_value_display": f"Directional {direction.lower()}",
                "reason": "Fallback directional classification.",
                "numeric_signals": signal,
                "confidence_reason": "Low confidence because the clause lacks usable numerical or economic mapping detail.",
                "exact_value_note": "No exact percentage available.",
            }
        ]

    return [
        {
            "model_line": fallback_line,
            "period": period,
            "implied_value": None,
            "implied_range_low": None,
            "implied_range_high": None,
            "unit": "unclear",
            "base_value": None,
            "method": "Clause is too vague or boilerplate for DCF translation.",
            "calculation": "Manual review required.",
            "confidence": "Manual Review",
            "direction": "Unknown",
            "warnings": ["Clause is too vague for a model-line implication."],
            "implied_value_status": "Unclear",
            "implied_value_display": "Manual review required",
            "reason": "No specific model implication could be inferred.",
            "numeric_signals": signal,
            "confidence_reason": "Manual review because extraction failed or clause is boilerplate.",
            "exact_value_note": "No exact percentage available.",
        }
    ]


def calculate_implied_assumption(
    extracted_guidance: dict,
    model_line: str,
    financials,
    dcf_assumptions: dict,
) -> dict:
    """
    Convert extracted guidance into a DCF-compatible assumption.
    """
    unit = extracted_guidance.get("unit")
    metric = extracted_guidance.get("metric")
    canonical_line = DCF_LINE_ALIASES.get(model_line, model_line)
    method = extracted_guidance.get("method") or "Qualitative guidance only; no numeric value extracted."
    result = {
        "model_line": model_line,
        "period": extracted_guidance.get("period") or "Forward period",
        "implied_value": None,
        "implied_range_low": None,
        "implied_range_high": None,
        "unit": unit,
        "base_value": None,
        "method": method,
        "calculation": method,
        "confidence": extracted_guidance.get("confidence") or "Low",
        "direction": "Mixed",
        "warnings": [],
    }

    if unit == "percent":
        midpoint = _as_float(extracted_guidance.get("midpoint"))
        low = _as_float(extracted_guidance.get("low"))
        high = _as_float(extracted_guidance.get("high"))
        result.update(
            {
                "implied_value": midpoint,
                "implied_range_low": low,
                "implied_range_high": high,
                "method": "Used management percentage guidance midpoint.",
                "calculation": f"Guidance midpoint = {_pct_display(midpoint)}.",
                "direction": "Increase" if midpoint is not None and midpoint > float(dcf_assumptions.get(canonical_line, 0) or 0) else "Decrease",
            }
        )
        return result

    if unit == "basis_points":
        current = _current_margin(financials, dcf_assumptions, canonical_line)
        bps = _as_float(extracted_guidance.get("bps_change"))
        if current is not None and bps is not None:
            implied = current + bps / 10000.0
            result.update(
                {
                    "implied_value": implied,
                    "base_value": current,
                    "method": "Added basis-point guidance to the current DCF margin.",
                    "calculation": f"Current margin {_pct_display(current)} + {bps:.0f} bps = {_pct_display(implied)}.",
                    "direction": "Increase" if bps >= 0 else "Decrease",
                }
            )
        else:
            result["warnings"].append("Current margin unavailable for basis-point conversion.")
        return result

    if unit == "money" and metric in {"revenue"}:
        revenue = _base_revenue(financials, dcf_assumptions)
        midpoint = _as_float(extracted_guidance.get("midpoint"))
        low = _as_float(extracted_guidance.get("low"))
        high = _as_float(extracted_guidance.get("high"))
        if revenue and midpoint is not None:
            implied = midpoint / revenue - 1
            result.update(
                {
                    "implied_value": implied,
                    "implied_range_low": low / revenue - 1 if low is not None else None,
                    "implied_range_high": high / revenue - 1 if high is not None else None,
                    "base_value": revenue,
                    "method": "Calculated from management revenue guidance midpoint versus latest revenue base.",
                    "calculation": f"Guidance midpoint {_money_display(midpoint)} / base revenue {_money_display(revenue)} - 1 = {_pct_display(implied)}.",
                    "direction": "Increase" if implied >= 0 else "Decrease",
                }
            )
        else:
            result["warnings"].append("Revenue base unavailable for guidance conversion.")
        return result

    if unit == "money" and metric in {"capex", "sbc", "ocf", "fcf"}:
        revenue = _forecast_revenue(financials, dcf_assumptions)
        midpoint = _as_float(extracted_guidance.get("midpoint"))
        if revenue and midpoint is not None:
            implied = midpoint / revenue
            if metric == "capex" and canonical_line == "growth_capex_pct_revenue":
                maintenance = float(dcf_assumptions.get("maintenance_capex_pct_revenue", 0) or 0)
                implied = max(implied - maintenance, 0)
                method = "Calculated guided CAPEX as a percentage of forecast revenue, net of maintenance CAPEX."
                calc = f"({_money_display(midpoint)} / forecast revenue {_money_display(revenue)}) - maintenance CAPEX {_pct_display(maintenance)} = {_pct_display(implied)}."
            else:
                method = f"Calculated guided {metric.upper()} as a percentage of forecast revenue."
                calc = f"{_money_display(midpoint)} / forecast revenue {_money_display(revenue)} = {_pct_display(implied)}."
            result.update(
                {
                    "implied_value": implied,
                    "base_value": revenue,
                    "method": method,
                    "calculation": calc,
                    "direction": "Increase" if implied >= float(dcf_assumptions.get(canonical_line, 0) or 0) else "Decrease",
                }
            )
        else:
            result["warnings"].append("Forecast revenue unavailable for dollar guidance conversion.")
        return result

    if metric == "backlog":
        result.update(
            {
                "implied_value": None,
                "method": "Backlog/RPO evidence is directional; revenue conversion data is required before setting revenue growth.",
                "calculation": "No exact percentage assigned because backlog growth is not mechanically equal to revenue growth.",
                "direction": "Increase",
                "confidence": "Medium" if unit == "percent" else "Low",
            }
        )
        return result

    return result


def _evidence_id(row: dict, model_line: str, period: str) -> str:
    seed = "|".join(
        [
            str(row.get("ticker") or ""),
            str(row.get("form") or row.get("filing_type") or ""),
            str(row.get("section") or ""),
            str(row.get("topic") or ""),
            str(row.get("clause_text") or ""),
            str(model_line or ""),
            str(period or ""),
        ]
    )
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def _source_plan(topic: str, model_line: str, source_url: str | None) -> tuple[str, str]:
    line_keywords = {
        "revenue_growth": "guidance, outlook, revenue, growth, backlog, RPO",
        "revenue_cagr": "guidance, outlook, revenue, growth, backlog, RPO",
        "gross_margin": "gross margin, margin, costs, outlook",
        "nopat_margin": "operating income, operating margin, costs, outlook",
        "ocf_margin": "operating cash flow, cash provided by operating activities",
        "fcf_margin": "free cash flow, FCF, cash flow margin",
        "growth_capex_pct_revenue": "capital expenditures, capex, capital spending, capacity",
        "maintenance_capex_pct_revenue": "maintenance capital, replacement, capex",
        "sbc_pct_revenue": "stock-based compensation, share-based compensation, equity awards",
    }
    where = "SEC 10-K / 10-Q; earnings release; investor presentation; earnings call transcript; company investor relations"
    if source_url:
        where = f"{where}; source filing link available"
    return where, line_keywords.get(DCF_LINE_ALIASES.get(model_line, model_line), "guidance, outlook, model assumption")


def _suggested_action(implied_value, confidence: str, direction: str, status: str | None = None) -> str:
    if _finite(implied_value):
        return "Compare to User Case" if confidence in {"Low", "Medium"} else "Apply to User Case"
    if status == "Estimated Range":
        return "Compare to User Case"
    if status == "Scenario Support":
        return "Create Note"
    if status == "Unclear":
        return "Needs Source Verification"
    if direction in {"Increase", "Decrease"}:
        return "Create Note / Review Scenario Probability"
    return "Manual Review"


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _assumption_signal(
    topic: str,
    model_line: str,
    status: str,
    direction: str,
    clause_text: str,
    implied_display: str,
) -> str:
    lower = f"{clause_text} {implied_display}".lower()
    canonical_line = DCF_LINE_ALIASES.get(model_line, model_line)
    direction_lower = str(direction or "").lower()

    if status == "Calculated":
        return "Calculated %"
    if status == "Range":
        return "Implied Range"
    if status == "Unclear":
        return "Not Enough Evidence"

    if canonical_line in {"revenue_growth", "revenue_cagr"}:
        if topic in {"BACKLOG_RPO_BOOKINGS", "REVENUE_GROWTH"} or _contains_any(
            lower,
            ("backlog", "rpo", "remaining performance obligation", "booking", "demand", "pipeline", "customer growth", "retention", "visibility"),
        ):
            return "Revenue Visibility Support"
        if topic == "M_AND_A" or _contains_any(lower, ("acquisition", "acquired")):
            return "Bull Case Support" if "decrease" not in direction_lower else "Bear Case Support"

    if canonical_line == "wacc":
        if "decrease" in direction_lower or _contains_any(lower, ("reduction", "mitigate", "improve", "strong", "repay", "lower debt")):
            return "Risk Reduction Support"
        return "Risk Increase Warning"

    capex_context = topic == "CAPEX" or canonical_line in {
        "growth_capex_pct_revenue",
        "maintenance_capex_pct_revenue",
        "total_capex_pct_revenue",
    } or _contains_any(lower, ("capex", "capital expenditure", "facility", "data center", "infrastructure", "equipment", "capacity"))
    fcf_context = canonical_line == "fcf_margin" or "free cash flow" in lower or "fcf" in lower
    sbc_context = canonical_line in {"sbc_pct_revenue", "diluted_shares"} or topic == "SBC_DILUTION_BUYBACKS" or _contains_any(
        lower,
        ("stock-based compensation", "share-based compensation", "sbc", "rsu", "option", "equity award", "dilution", "share count"),
    )
    margin_context = canonical_line in {"gross_margin", "nopat_margin", "operating_margin", "ocf_margin"} or _contains_any(
        lower,
        ("margin", "cost", "profitability", "operating income"),
    )
    revenue_visibility_context = topic == "BACKLOG_RPO_BOOKINGS" or canonical_line in {"revenue_growth", "revenue_cagr"} and _contains_any(
        lower,
        ("backlog", "rpo", "remaining performance obligation", "booking", "demand", "pipeline", "customer growth", "retention", "visibility"),
    )
    risk_context = topic in {"RISK_FACTORS", "DEBT_LIQUIDITY", "MANAGEMENT_CREDIBILITY", "MOAT_COMPETITION"} or canonical_line in {
        "wacc",
        "terminal_multiple",
        "scenario_probability",
    } or _contains_any(lower, ("risk", "uncertain", "weak", "competition", "liquidity", "leverage", "debt"))
    positive_context = _contains_any(lower, ("strong", "increase", "growth", "improve", "expand", "higher", "visibility", "retention"))
    negative_context = _contains_any(lower, ("decline", "decrease", "lower", "pressure", "weak", "risk", "uncertain", "headwind"))

    if canonical_line == "terminal_multiple":
        if "decrease" in direction_lower or negative_context:
            return "Risk Increase Warning"
        if "increase" in direction_lower or positive_context:
            return "Bull Case Support"
        return "Qualitative Support"

    if sbc_context:
        return "SBC / Dilution Warning"
    if fcf_context and (capex_context or "pressure" in lower or "decrease" in direction_lower):
        return "Near-Term FCF Pressure"
    if canonical_line in {"revenue_growth", "revenue_cagr"} and capex_context:
        return "Revenue Visibility Support"
    if capex_context:
        return "CAPEX Increase Warning"
    if margin_context:
        if "decrease" in direction_lower or negative_context:
            return "Margin Pressure Warning"
        if "increase" in direction_lower or positive_context:
            return "Margin Expansion Support"
    if "bull" in direction_lower:
        return "Bull Case Support"
    if "bear" in direction_lower:
        return "Bear Case Support"
    if revenue_visibility_context:
        return "Revenue Visibility Support"
    if risk_context:
        if "decrease" in direction_lower or _contains_any(lower, ("reduction", "mitigate", "visibility", "improve", "strong")):
            return "Risk Reduction Support"
        return "Risk Increase Warning"
    if negative_context or "decrease" in direction_lower:
        return "Bear Case Support"
    if positive_context or "increase" in direction_lower:
        return "Bull Case Support" if canonical_line == "scenario_probability" else "Qualitative Support"
    if status == "Estimated Range":
        return "Estimated Range"
    return "Qualitative Support" if status in {"Scenario Support", "Directional Only"} else "Not Enough Evidence"


def build_evidence_assumption_impacts(
    clauses_df,
    dcf_assumptions,
    financials,
    business_profile=None,
    peer_data=None,
):
    """
    Convert extracted clauses into model-line assumption impacts.

    Returns a DataFrame with evidence summary, model line affected, implied value,
    range, current DCF value, delta, confidence, evidence grade, and action.
    """
    if clauses_df is None or len(clauses_df) == 0:
        return pd.DataFrame(columns=IMPACT_COLUMNS)

    assumptions = dcf_assumptions or {}
    rows = []
    for clause in pd.DataFrame(clauses_df).to_dict("records"):
        clause_text = _clean_text(clause.get("clause_text"))
        topic = clause.get("topic") or "Evidence"
        topic_label = TOPIC_LABELS.get(topic, str(topic).replace("_", " ").title())
        source_url = clause.get("source_url") or clause.get("filing_url") or clause.get("document_url")
        evidence_grade = clause.get("evidence_grade") or ("Guided" if topic == "GUIDANCE_OUTLOOK" else "Reported")
        impact_items = map_clause_to_multi_line_impacts(clause, financials, assumptions)

        for implied in impact_items:
            model_line = implied.get("model_line") or clause.get("model_line_affected") or "scenario_probability"
            canonical_line = DCF_LINE_ALIASES.get(model_line, model_line)
            current = _as_float(assumptions.get(canonical_line))
            implied_value = _as_float(implied.get("implied_value"))
            delta = implied_value - current if implied_value is not None and current is not None else None
            direction = implied.get("direction") or clause.get("direction") or "Unknown"
            confidence = implied.get("confidence") or clause.get("confidence") or "Low"
            status = implied.get("implied_value_status")
            range_low = _as_float(implied.get("implied_range_low"))
            range_high = _as_float(implied.get("implied_range_high"))
            if not status:
                status = "Range" if range_low is not None and range_high is not None and range_low != range_high else "Calculated" if implied_value is not None else "Directional Only"
            if implied.get("implied_value_display"):
                implied_display = implied.get("implied_value_display")
            elif implied_value is not None:
                implied_display = _pct_display(implied_value)
            elif status == "Unclear":
                implied_display = "Manual review required"
            else:
                implied_display = "No exact % available"
            range_display = (
                f"{_pct_display(range_low)} - {_pct_display(range_high)}"
                if status == "Range" and range_low is not None and range_high is not None
                else implied_display
            )
            model_label = MODEL_LINE_LABELS.get(model_line, model_line.replace("_", " ").title())
            assumption_signal = _assumption_signal(topic, model_line, status, direction, clause_text, implied_display)
            where, keywords = _source_plan(topic, model_line, source_url)
            suggested_action = _suggested_action(implied_value, confidence, direction, status=status)
            if implied_value is not None and current is not None:
                comparison = "above" if implied_value > current else "below" if implied_value < current else "in line with"
                summary = f"{topic_label} implies {model_label} {comparison} the current DCF assumption."
            elif status == "Estimated Range":
                summary = f"{topic_label} gives an estimated range for {model_label}."
            elif status == "Scenario Support":
                summary = f"{topic_label} provides {assumption_signal.lower()} for {model_label}."
            elif status == "Directional Only":
                summary = f"{topic_label} provides {assumption_signal.lower()} for {model_label}."
            else:
                summary = f"{topic_label} needs review before changing {model_label}."

            rows.append(
                {
                    "evidence_id": _evidence_id(clause, model_line, implied.get("period")),
                    "ticker": clause.get("ticker"),
                    "form": clause.get("form") or clause.get("filing_type"),
                    "filing_date": clause.get("filing_date"),
                    "section": clause.get("section"),
                    "topic": topic,
                    "topic_label": topic_label,
                    "subtopic": clause.get("subtopic"),
                    "clause_text": clause_text,
                    "evidence_summary": summary,
                    "model_line": model_line,
                    "model_line_label": model_label,
                    "implied_value_status": status,
                    "assumption_signal": assumption_signal,
                    "assumption_signal_help": ASSUMPTION_SIGNAL_HELP.get(assumption_signal),
                    "period": implied.get("period"),
                    "implied_value": implied_value,
                    "implied_value_display": implied_display,
                    "implied_range_low": range_low,
                    "implied_range_high": range_high,
                    "implied_range_display": range_display,
                    "current_dcf_value": current,
                    "current_dcf_value_display": _assumption_value_display(model_line, current),
                    "delta": delta,
                    "delta_vs_current_dcf": delta,
                    "delta_display": _pts_display(delta) if delta is not None else "n.m.",
                    "direction": direction,
                    "confidence": confidence,
                    "evidence_grade": evidence_grade,
                    "method": implied.get("method"),
                    "calculation": implied.get("calculation"),
                    "reason": implied.get("reason"),
                    "numeric_signals": implied.get("numeric_signals") or extract_numeric_signals(clause_text),
                    "confidence_reason": implied.get("confidence_reason"),
                    "exact_value_note": implied.get("exact_value_note"),
                    "base_value": implied.get("base_value"),
                    "base_value_display": _money_display(implied.get("base_value")) if _finite(implied.get("base_value")) else UNAVAILABLE,
                    "suggested_action": suggested_action,
                    "recommended_case": "User Case",
                    "source_url": source_url,
                    "review_status": clause.get("review_status") or "Unreviewed",
                    "user_note": clause.get("user_note") or "",
                    "extraction_method": clause.get("extraction_method"),
                    "warnings": "; ".join(implied.get("warnings") or []),
                    "where_to_verify": where,
                    "search_keywords": keywords,
                }
            )

    impact_df = pd.DataFrame(rows, columns=IMPACT_COLUMNS)
    if impact_df.empty:
        return impact_df
    return impact_df.drop_duplicates(subset=["topic", "clause_text", "model_line", "period"]).reset_index(drop=True)


def build_assumption_update_from_impact(impact: dict, scenario: str = "User Case", status: str = "Pending", user_note: str = "") -> dict:
    """
    Create a pending assumption update log entry from an evidence impact.
    This does not mutate any DCF case by itself.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evidence_id": impact.get("evidence_id"),
        "scenario": scenario,
        "case": scenario,
        "model_line": impact.get("model_line"),
        "period": impact.get("period"),
        "old_value": impact.get("current_dcf_value"),
        "new_value": impact.get("implied_value"),
        "source": f"{impact.get('topic') or 'Evidence'} clause",
        "evidence_source": f"{impact.get('topic_label') or impact.get('topic') or 'Evidence'} clause",
        "evidence_summary": impact.get("evidence_summary"),
        "method": impact.get("method"),
        "confidence": impact.get("confidence"),
        "user_note": user_note or impact.get("user_note") or "",
        "status": status,
    }


def unique_filter_values(frame: pd.DataFrame, column: str) -> list[str]:
    if frame is None or frame.empty or column not in frame:
        return []
    values = [str(value) for value in frame[column].dropna().unique().tolist() if str(value).strip()]
    return sorted(values)


def impact_status_summary(frame: pd.DataFrame) -> dict:
    statuses = ["Calculated", "Range", "Estimated Range", "Scenario Support", "Directional Only", "Unclear"]
    if frame is None or frame.empty or "implied_value_status" not in frame:
        return {status: 0 for status in statuses} | {"total": 0, "directional_only_share": 0.0, "needs_review": 0}
    counts = frame["implied_value_status"].value_counts()
    total = int(len(frame))
    needs_review = frame[
        frame["implied_value_status"].isin(["Directional Only", "Unclear"])
        | frame["confidence"].isin(["Low", "Manual Review"])
    ]
    return {
        **{status: int(counts.get(status, 0)) for status in statuses},
        "total": total,
        "directional_only_share": float(counts.get("Directional Only", 0)) / max(total, 1),
        "needs_review": int(len(needs_review)),
    }
