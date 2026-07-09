from __future__ import annotations

from copy import deepcopy


ANALYSIS_SCHEMA_VERSION = "1.0"
DASHBOARD_VERSION = "0.1.0"


def default_analysis_payload() -> dict:
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "dashboard": "PA-11R Hybrid",
        "dashboard_version": DASHBOARD_VERSION,
        "analysis_id": "",
        "ticker": "",
        "company_name": "",
        "created_at": "",
        "updated_at": "",
        "analysis_name": "",
        "description": "",
        "tags": [],
        "user_profile": {"mode": "Investor", "time_horizon": "Long-term", "risk_level": "Medium"},
        "decision": {
            "investment_view": "",
            "swing_view": "",
            "market_regime": "",
            "final_rating": "",
            "conviction": "",
            "position_size_guidance": "",
            "summary": "",
        },
        "market_snapshot_at_save": {
            "price": None,
            "market_cap": None,
            "enterprise_value": None,
            "shares_outstanding": None,
            "net_debt": None,
            "date": None,
        },
        "data_sources": {
            "sec": {"available": False, "cik": None, "latest_10k": None, "latest_10q": None, "latest_proxy": None, "warnings": []},
            "finviz": {"available": False, "fields_used": [], "warnings": []},
            "yfinance": {"available": False, "warnings": []},
        },
        "dcf": {
            "valuation_basis": "OCF-based FCF",
            "scenario_assumptions": {"Bear Case": {}, "Base Case": {}, "Bull Case": {}, "User Case": {}, "Market-Implied Case": {}},
            "scenario_outputs": {"Bear Case": {}, "Base Case": {}, "Bull Case": {}, "User Case": {}, "Market-Implied Case": {}},
            "selected_case": "User Case",
            "assumption_update_log": [],
        },
        "sotp": {"enabled": True, "segment_assumptions": {}, "scenario_outputs": {}, "manual_segments": [], "whole_vs_sum_conclusion": ""},
        "multiples": {
            "selected_multiple_basis": "Normalized Year",
            "peer_set": [],
            "scenario_implied_multiples": {},
            "peer_medians": {},
            "sector_medians": {},
            "user_notes": "",
        },
        "evidence": {"clause_mappings": [], "applied_evidence": [], "ignored_evidence": [], "manual_review_items": []},
        "business_quality": {"moat": {}, "risks": {}, "thesis_breakers": [], "user_notes": ""},
        "management_capital_allocation": {"management": {}, "ma_strategy": {}, "compensation_sbc": {}, "user_notes": ""},
        "user_notes": {"general": "", "valuation": "", "thesis": "", "risks": "", "manual_review": ""},
    }


def deep_merge_defaults(payload: dict) -> dict:
    merged = default_analysis_payload()

    def merge(target: dict, source: dict) -> dict:
        for key, value in (source or {}).items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                target[key] = merge(target[key], value)
            else:
                target[key] = deepcopy(value)
        return target

    return merge(merged, payload or {})


def validate_analysis_payload(payload: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return False, ["Payload is not a JSON object."]
    for key in ["schema_version", "dashboard", "analysis_id", "ticker", "created_at", "updated_at", "dcf", "sotp", "multiples"]:
        if key not in payload:
            errors.append(f"Missing required key: {key}")
    if payload.get("dashboard") != "PA-11R Hybrid":
        errors.append("This file does not look like a PA-11R saved analysis.")
    if payload.get("schema_version") not in {ANALYSIS_SCHEMA_VERSION}:
        errors.append(f"Unsupported schema version: {payload.get('schema_version')}")
    return not errors, errors


def migrate_analysis_payload(payload: dict) -> dict:
    payload = deep_merge_defaults(payload or {})
    payload["schema_version"] = ANALYSIS_SCHEMA_VERSION
    payload["dashboard"] = "PA-11R Hybrid"
    return payload
