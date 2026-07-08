from __future__ import annotations

import re
from datetime import datetime, timezone


def _row(model_line: str, direction: str, timeframe: str, change: str, action: str) -> dict:
    return {
        "model_line_affected": model_line,
        "direction": direction,
        "timeframe": timeframe,
        "suggested_assumption_change": change,
        "dashboard_action": action,
    }


def map_clause_to_model_lines(topic: str, subtopic: str, clause_text: str) -> list[dict]:
    """
    Map a clause to affected model lines.
    """
    text = (clause_text or "").lower()
    subtopic = subtopic or "General"
    if topic == "CAPEX":
        if "maintenance" in subtopic.lower():
            return [_row("maintenance_capex_pct_revenue", "Increase", "Medium-term", "Review normalized maintenance CAPEX and FCF.", "Review DCF")]
        if "automation" in subtopic.lower():
            return [
                _row("growth_capex_pct_revenue", "Increase", "Near-term", "Review growth CAPEX for automation investment.", "Review DCF"),
                _row("gross_margin", "Increase", "Medium-term", "Check whether automation supports margin expansion.", "Update scenario"),
                _row("opex_ratio", "Decrease", "Medium-term", "Check whether productivity lowers OPEX ratio.", "Update scenario"),
            ]
        return [
            _row("growth_capex_pct_revenue", "Increase", "Near-term", "Review growth CAPEX and delayed revenue benefit.", "Review DCF"),
            _row("revenue_growth", "Increase", "Medium-term", "Check whether capacity expansion supports future revenue.", "Update scenario"),
            _row("ocf_margin", "Decrease", "Near-term", "Review possible FCF drag from upfront investment.", "Flag risk"),
        ]
    if topic == "BACKLOG_RPO_BOOKINGS":
        direction = "Decrease" if any(word in text for word in ["decline", "decrease", "lower"]) else "Increase"
        return [
            _row("revenue_growth", direction, "Near-term", "Review revenue visibility from backlog/RPO.", "Update scenario"),
            _row("scenario_probability", "Mixed" if direction == "Increase" else "Decrease", "Medium-term", "Review bull/base/bear probabilities.", "Update scenario"),
        ]
    if topic == "REVENUE_GROWTH":
        direction = "Decrease" if any(word in text for word in ["decreased", "decline", "churn"]) else "Increase"
        return [_row("revenue_growth", direction, "Near-term", "Review modeled revenue growth.", "Review DCF")]
    if topic == "MARGIN_COSTS":
        direction = "Decrease" if any(word in text for word in ["pressure", "higher costs", "inflation", "lower"]) else "Increase"
        return [
            _row("gross_margin", direction, "Near-term", "Review gross margin assumption.", "Review DCF"),
            _row("terminal_multiple", "Decrease" if direction == "Decrease" else "Mixed", "Long-term", "Review whether margin trend changes business quality.", "Update scenario"),
        ]
    if topic == "OCF_WORKING_CAPITAL":
        if "inventory" in text:
            return [
                _row("working_capital_pct_revenue", "Increase", "Near-term", "Review working capital investment from inventory build.", "Flag risk"),
                _row("ocf_margin", "Decrease", "Near-term", "Review cash conversion and possible demand risk.", "Review DCF"),
            ]
        if "receivable" in text:
            return [
                _row("working_capital_pct_revenue", "Increase", "Near-term", "Review receivables and revenue quality.", "Flag risk"),
                _row("ocf_margin", "Decrease", "Near-term", "Review OCF conversion.", "Review DCF"),
            ]
        return [_row("ocf_margin", "Mixed", "Near-term", "Review OCF quality and working capital drivers.", "Review DCF")]
    if topic == "M_AND_A":
        if "impairment" in text:
            return [
                _row("terminal_multiple", "Decrease", "Long-term", "Review goodwill/intangible impairment risk.", "Flag risk"),
                _row("scenario_probability", "Decrease", "Medium-term", "Review bear-case probability.", "Update scenario"),
            ]
        return [
            _row("revenue_growth", "Increase", "Near-term", "Review acquired revenue contribution.", "Review DCF"),
            _row("gross_margin", "Mixed", "Medium-term", "Review acquisition margin mix.", "Manual review"),
            _row("debt", "Mixed", "Near-term", "Review acquisition financing source.", "Manual review"),
        ]
    if topic == "SBC_DILUTION_BUYBACKS":
        if any(word in text for word in ["repurchase", "buyback"]):
            return [_row("diluted_shares", "Decrease", "Medium-term", "Check if buybacks reduce net share count or only offset SBC.", "Manual review")]
        return [
            _row("sbc", "Increase", "Near-term", "Review SBC as a share of revenue and FCF.", "Flag risk"),
            _row("diluted_shares", "Increase", "Medium-term", "Review dilution and per-share value.", "Review DCF"),
        ]
    if topic == "DEBT_LIQUIDITY":
        if "liquidity" in subtopic.lower() or "cash and cash equivalents" in text:
            return [
                _row("net_debt", "Decrease", "Near-term", "Review whether cash/liquidity offsets debt risk.", "Review DCF"),
                _row("wacc", "Mixed", "Medium-term", "Review financing risk and liquidity buffer together.", "Manual review"),
            ]
        return [
            _row("net_debt", "Increase", "Near-term", "Review net debt and liquidity runway.", "Review DCF"),
            _row("wacc", "Increase", "Medium-term", "Review financing risk and cost of capital.", "Flag risk"),
        ]
    if topic == "GUIDANCE_OUTLOOK":
        model_line = "revenue_growth" if "revenue" in text or "sales" in text else "gross_margin" if "margin" in text else "growth_capex_pct_revenue" if "capital" in text else "scenario_probability"
        return [_row(model_line, "Mixed", "Medium-term", "Guidance requires user review before changing assumptions.", "Manual review")]
    if topic == "MOAT_COMPETITION":
        return [
            _row("terminal_multiple", "Decrease" if "competition" in text else "Increase", "Long-term", "Review moat durability and terminal multiple.", "Update scenario"),
            _row("wacc", "Increase" if "competition" in text else "Mixed", "Long-term", "Review risk premium if moat is weak.", "Flag risk"),
        ]
    if topic == "RISK_FACTORS":
        return [
            _row("wacc", "Increase", "Medium-term", "Review risk premium.", "Flag risk"),
            _row("terminal_multiple", "Decrease", "Long-term", "Review thesis durability.", "Update scenario"),
            _row("scenario_probability", "Decrease", "Medium-term", "Review bear-case probability.", "Update scenario"),
        ]
    return [_row("scenario_probability", "Unknown", "Unknown", "Review manually; evidence is unclear.", "Manual review")]


def assign_evidence_grade(clause_text: str, section: str, source_form: str) -> str:
    """
    Assign a deterministic evidence grade.
    """
    text = (clause_text or "").lower()
    section_lower = (section or "").lower()
    if any(word in text for word in ["expect", "expects", "anticipate", "forecast", "target", "outlook", "plan", "intend"]):
        return "Guided"
    if any(word in text for word in ["scenario", "sensitivity", "bull", "bear", "base case"]):
        return "Scenario-based"
    if any(word in text for word in ["proxy", "estimate", "derived", "calculated"]):
        return "Calculated"
    if "compensation" in section_lower or source_form == "DEF 14A":
        return "Proxy-based"
    if re.search(r"\$?\d[\d,.]*\s*(million|billion|%|percent)?", text) or source_form in {"10-K", "10-Q"}:
        return "Reported"
    return "Unknown"


def build_assumption_update_from_clause(clause_row: dict, old_value=None, new_value=None, scenario: str = "Base") -> dict:
    """
    Create a pending DCF assumption-update object. This never applies changes.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "timestamp": timestamp,
        "ticker": clause_row.get("ticker"),
        "assumption": clause_row.get("model_line_affected"),
        "old_value": old_value,
        "new_value": new_value,
        "reason": f"Clause from {clause_row.get('form')}: {str(clause_row.get('clause_text') or '')[:260]}",
        "linked_clause": clause_row.get("clause_text"),
        "source_url": clause_row.get("source_url"),
        "confidence": clause_row.get("confidence"),
        "scenario": scenario,
        "status": "Pending user input",
    }
