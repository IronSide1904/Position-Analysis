from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Any


@dataclass
class AssumptionEstimate:
    value: float | None
    method: str
    evidence_grade: str
    confidence: str
    warning: str | None
    source: str | None
    is_real_zero: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EVIDENCE_ORDER = {
    "Reported": 5,
    "Calculated": 4,
    "Reconstructed": 3,
    "Proxy-based": 2,
    "Estimated": 2,
    "Peer-derived": 2,
    "Business-profile estimate": 1,
    "Manual review": 0,
}

CONFIDENCE_ORDER = {"High": 3, "Medium": 2, "Low": 1}


def _num(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _clean(values: list[float] | None) -> list[float]:
    return [float(value) for value in values or [] if _num(value) is not None]


def _normalized_average(values: list[float] | None) -> float | None:
    clean = _clean(values)
    if not clean:
        return None
    if len(clean) <= 2:
        return sum(clean) / len(clean)
    ordered = sorted(clean)
    trimmed = ordered[1:-1] or ordered
    return sum(trimmed) / len(trimmed)


def _profile_text(business_profile: dict | None) -> str:
    profile = business_profile or {}
    return " ".join(str(profile.get(key) or "") for key in ["profile", "stock_profile", "sector", "industry", "business_model"]).lower()


def _profile_default(metric: str, business_profile: dict | None) -> tuple[float, str, str]:
    text = _profile_text(business_profile)
    is_software = any(token in text for token in ["software", "saas", "cloud", "platform", "application"])
    is_hardware = any(token in text for token in ["hardware", "industrial", "manufacturing", "electronics", "semiconductor", "equipment"])
    is_infra = any(token in text for token in ["data center", "infrastructure", "telecom", "utility"])
    is_acq = any(token in text for token in ["acquisition", "m&a", "roll-up"])

    if metric == "maintenance_capex":
        if is_software:
            return 0.015, "Business-profile estimate for asset-light software.", "Low"
        if is_infra:
            return 0.06, "Business-profile estimate for infrastructure replacement needs.", "Low"
        if is_hardware:
            return 0.035, "Business-profile estimate for industrial/hardware asset base.", "Low"
        if is_acq:
            return 0.025, "Business-profile estimate adjusted for acquisition-heavy amortization risk.", "Low"
        return 0.03, "General business-profile CAPEX estimate.", "Low"

    if metric == "sbc":
        if is_software:
            return 0.08, "Software/SaaS business-profile SBC estimate.", "Low"
        if is_hardware:
            return 0.015, "Industrial/hardware business-profile SBC estimate.", "Low"
        return 0.02, "General business-profile SBC estimate.", "Low"

    if metric == "working_capital":
        if is_software:
            return -0.01, "Software profile can benefit from deferred revenue and upfront billing.", "Low"
        if is_hardware:
            return 0.03, "Hardware/inventory profile usually requires working-capital investment.", "Low"
        return 0.01, "General business-profile working-capital estimate.", "Low"

    if metric == "da":
        if is_software:
            return 0.025, "Software profile D&A estimate; review acquired amortization.", "Low"
        if is_infra:
            return 0.07, "Infrastructure profile D&A estimate.", "Low"
        if is_hardware:
            return 0.035, "Industrial/hardware profile D&A estimate.", "Low"
        return 0.03, "General business-profile D&A estimate.", "Low"

    return 0.0, "No profile rule available.", "Low"


def _cap_confidence(a: str, b: str) -> str:
    return a if CONFIDENCE_ORDER.get(a, 0) <= CONFIDENCE_ORDER.get(b, 0) else b


def estimate_maintenance_capex(
    revenue: float | None,
    total_capex: float | None,
    depreciation_amortization: float | None,
    business_profile: dict | None,
    historical_capex_pct_revenue: list[float] | None = None,
    clauses=None,
) -> AssumptionEstimate:
    revenue = _num(revenue)
    total_capex = abs(_num(total_capex)) if _num(total_capex) is not None else None
    da = abs(_num(depreciation_amortization)) if _num(depreciation_amortization) is not None else None

    if total_capex == 0:
        return AssumptionEstimate(0.0, "Reported total CAPEX was zero; maintenance CAPEX cannot exceed it.", "Reported", "High", None, "SEC companyfacts / cash flow statement", True)

    if da is not None and da == 0:
        return AssumptionEstimate(0.0, "Reported D&A was zero and used as maintenance proxy.", "Reported", "Medium", "Review if total CAPEX exists; zero D&A can understate maintenance needs.", "SEC companyfacts / cash flow statement", True)

    history_pct = _normalized_average(historical_capex_pct_revenue)
    history_value = revenue * history_pct if revenue and history_pct is not None else None
    text = _profile_text(business_profile)
    acq_or_software = any(token in text for token in ["software", "saas", "cloud", "acquisition", "m&a", "roll-up"])
    infra = any(token in text for token in ["data center", "infrastructure"])

    if da is not None and total_capex is not None:
        if acq_or_software:
            value = min(da, total_capex)
            warning = "D&A may include acquired intangible amortization; review if acquisition-heavy or asset-light."
            return AssumptionEstimate(value, "Lower of D&A and total CAPEX due D&A distortion risk.", "Proxy-based", "Medium", warning, "SEC companyfacts + calculated history")
        if infra and history_value is not None:
            return AssumptionEstimate(max(da, min(history_value, total_capex)), "D&A proxy with infrastructure replacement floor.", "Proxy-based", "Medium", "D&A may understate future replacement CAPEX.", "SEC companyfacts + calculated history")
        return AssumptionEstimate(min(da, total_capex), "D&A proxy capped at total CAPEX.", "Proxy-based", "Medium", "D&A is a proxy, not a disclosed maintenance-growth split.", "SEC companyfacts + calculated history")

    if da is not None:
        return AssumptionEstimate(da, "D&A proxy because maintenance CAPEX split is not disclosed.", "Proxy-based", "Medium", "D&A may not represent true maintenance CAPEX.", "SEC companyfacts / cash flow statement")

    if total_capex is not None and history_value is not None:
        return AssumptionEstimate(min(total_capex, history_value), "Historical normalized total CAPEX floor.", "Estimated", "Low", "Maintenance/growth split is not disclosed.", "Calculated history")

    if revenue and history_pct is not None:
        return AssumptionEstimate(revenue * history_pct, "Historical normalized CAPEX intensity.", "Estimated", "Low", "Maintenance/growth split is not disclosed.", "Calculated history")

    if revenue:
        pct, method, confidence = _profile_default("maintenance_capex", business_profile)
        return AssumptionEstimate(revenue * pct, method, "Business-profile estimate", confidence, "No direct maintenance CAPEX or D&A found.", "Business profile")

    return AssumptionEstimate(None, "No defensible maintenance CAPEX estimate.", "Manual review", "Low", "Revenue, D&A, and total CAPEX are missing.", None)


def estimate_growth_capex(
    revenue: float | None,
    total_capex: float | None,
    maintenance_capex_estimate: AssumptionEstimate,
    historical_total_capex_pct_revenue: list[float] | None = None,
    business_profile: dict | None = None,
) -> AssumptionEstimate:
    revenue = _num(revenue)
    total_capex = abs(_num(total_capex)) if _num(total_capex) is not None else None
    maintenance = _num(maintenance_capex_estimate.value)
    warning = None

    if total_capex is None and revenue:
        history_pct = _normalized_average(historical_total_capex_pct_revenue)
        if history_pct is not None:
            total_capex = revenue * history_pct
            total_method = "estimated total CAPEX from historical intensity"
        else:
            pct, method, _confidence = _profile_default("maintenance_capex", business_profile)
            total_capex = revenue * max(pct * 1.25, pct + 0.01)
            total_method = method
    else:
        total_method = "reported total CAPEX"

    if total_capex is not None and maintenance is not None:
        value = max(total_capex - maintenance, 0.0)
        if value == 0.0 and ("D&A" in maintenance_capex_estimate.method or maintenance >= total_capex):
            warning = "D&A proxy exceeds total CAPEX; maintenance/growth split requires review."
        evidence = "Calculated" if maintenance_capex_estimate.evidence_grade in {"Reported", "Calculated"} else "Estimated"
        confidence = _cap_confidence("Medium", maintenance_capex_estimate.confidence)
        return AssumptionEstimate(value, f"Total CAPEX minus maintenance CAPEX ({total_method}).", evidence, confidence, warning, "SEC companyfacts + calculated history", value == 0.0)

    return AssumptionEstimate(None, "No defensible growth CAPEX estimate.", "Manual review", "Low", "Total CAPEX or maintenance CAPEX is missing.", None)


def estimate_sbc_pct_revenue(
    revenue: float | None,
    sbc_raw: float | None = None,
    historical_sbc_pct: list[float] | None = None,
    business_profile: dict | None = None,
    peer_sbc_pct: list[float] | None = None,
) -> AssumptionEstimate:
    revenue = _num(revenue)
    sbc = _num(sbc_raw)
    if revenue and sbc is not None:
        return AssumptionEstimate(sbc / revenue, "SBC divided by revenue.", "Calculated", "High", "High SBC may dilute per-share value.", "SEC companyfacts / cash flow statement", sbc == 0.0)

    historical = _normalized_average(historical_sbc_pct)
    if historical is not None:
        return AssumptionEstimate(historical, "Historical average SBC % revenue.", "Estimated", "Medium", "SBC was not found for the latest period; using company history.", "Calculated history")

    peers = _clean(peer_sbc_pct)
    if peers:
        return AssumptionEstimate(median(peers), "Peer median SBC % revenue.", "Peer-derived", "Low", "Peer SBC may not match company compensation policy.", "Peer data")

    pct, method, confidence = _profile_default("sbc", business_profile)
    return AssumptionEstimate(pct, method, "Business-profile estimate", confidence, "SBC missing; do not treat this as reported zero.", "Business profile")


def estimate_working_capital_pct_revenue(
    revenue: float | None,
    cash_flow_statement=None,
    balance_sheet=None,
    historical_wc_pct: list[float] | None = None,
    business_profile: dict | None = None,
    peer_wc_pct: list[float] | None = None,
) -> AssumptionEstimate:
    revenue = _num(revenue)
    cfs = cash_flow_statement or {}
    change_wc = _num(cfs.get("change_in_working_capital") if isinstance(cfs, dict) else None)
    if revenue and change_wc is not None:
        return AssumptionEstimate(change_wc / revenue, "Change in working capital divided by revenue.", "Calculated", "High", None, "Cash flow statement", change_wc == 0.0)

    bs = balance_sheet or {}
    if revenue and isinstance(bs, dict):
        operating_assets = sum(_num(bs.get(key)) or 0.0 for key in ["receivables", "inventory", "contract_assets"])
        operating_liabilities = sum(_num(bs.get(key)) or 0.0 for key in ["payables", "accrued_expenses", "contract_liabilities", "deferred_revenue"])
        has_any = any(_num(bs.get(key)) is not None for key in ["receivables", "inventory", "contract_assets", "payables", "accrued_expenses", "contract_liabilities", "deferred_revenue"])
        if has_any:
            return AssumptionEstimate((operating_assets - operating_liabilities) / revenue, "Operating working capital reconstructed from balance sheet lines.", "Reconstructed", "Medium", "Review receivables, inventory, payables, and deferred revenue.", "Balance sheet")

    historical = _normalized_average(historical_wc_pct)
    if historical is not None:
        return AssumptionEstimate(historical, "Historical average working-capital intensity.", "Estimated", "Medium", "Working capital can swing with timing.", "Calculated history")

    peers = _clean(peer_wc_pct)
    if peers:
        return AssumptionEstimate(median(peers), "Peer median working-capital intensity.", "Peer-derived", "Low", "Peer working capital may not match billing/inventory model.", "Peer data")

    pct, method, confidence = _profile_default("working_capital", business_profile)
    return AssumptionEstimate(pct, method, "Business-profile estimate", confidence, "Working capital missing; value is not reported fact.", "Business profile")


def estimate_ocf_margin(
    revenue: float | None,
    operating_cash_flow: float | None = None,
    nopat: float | None = None,
    depreciation_amortization: float | None = None,
    change_in_working_capital: float | None = None,
    historical_ocf_margin: list[float] | None = None,
) -> AssumptionEstimate:
    revenue = _num(revenue)
    ocf = _num(operating_cash_flow)
    if revenue and ocf is not None:
        return AssumptionEstimate(ocf / revenue, "Operating cash flow divided by revenue.", "Calculated", "High", None, "Cash flow statement", ocf == 0.0)

    nopat = _num(nopat)
    da = _num(depreciation_amortization)
    wc = _num(change_in_working_capital)
    if revenue and nopat is not None and da is not None and wc is not None:
        return AssumptionEstimate((nopat + da - wc) / revenue, "NOPAT + D&A - change in working capital, divided by revenue.", "Reconstructed", "Medium", "Reconstructed OCF should be checked against cash-flow statement timing.", "Income statement + cash-flow bridge")

    historical = _normalized_average(historical_ocf_margin)
    if historical is not None:
        return AssumptionEstimate(historical, "Historical average OCF margin.", "Estimated", "Medium", "OCF may be distorted by working-capital timing.", "Calculated history")

    return AssumptionEstimate(None, "No defensible OCF margin estimate.", "Manual review", "Low", "OCF, bridge inputs, and historical OCF margin are missing.", None)


def estimate_nopat_margin(
    revenue: float | None,
    ebit: float | None = None,
    gross_profit: float | None = None,
    opex: float | None = None,
    tax_rate: float | None = None,
    historical_nopat_margin: list[float] | None = None,
) -> AssumptionEstimate:
    revenue = _num(revenue)
    tax = _num(tax_rate)
    if tax is None:
        tax = 0.21
        tax_evidence = "normalized tax assumption"
    else:
        tax_evidence = "reported/selected tax rate"

    ebit_value = _num(ebit)
    if ebit_value is None and _num(gross_profit) is not None and _num(opex) is not None:
        ebit_value = _num(gross_profit) - _num(opex)
        method = f"Gross profit minus OPEX with {tax_evidence}."
        evidence = "Reconstructed"
        confidence = "Medium"
    else:
        method = f"EBIT times one minus tax rate with {tax_evidence}."
        evidence = "Calculated"
        confidence = "High" if _num(ebit) is not None else "Medium"

    if revenue and ebit_value is not None:
        return AssumptionEstimate((ebit_value * (1 - tax)) / revenue, method, evidence, confidence, None, "Income statement", ebit_value == 0.0)

    historical = _normalized_average(historical_nopat_margin)
    if historical is not None:
        return AssumptionEstimate(historical, "Historical average NOPAT margin.", "Estimated", "Medium", "NOPAT margin estimated from history.", "Calculated history")

    return AssumptionEstimate(None, "No defensible NOPAT margin estimate.", "Manual review", "Low", "EBIT/revenue and historical NOPAT margin are missing.", None)


def estimate_da_pct_revenue(
    revenue: float | None,
    depreciation_amortization: float | None = None,
    historical_da_pct: list[float] | None = None,
    business_profile: dict | None = None,
) -> AssumptionEstimate:
    revenue = _num(revenue)
    da = _num(depreciation_amortization)
    if revenue and da is not None:
        return AssumptionEstimate(abs(da) / revenue, "D&A divided by revenue.", "Calculated", "High", "D&A may include acquired intangible amortization.", "SEC companyfacts / cash flow statement", da == 0.0)

    historical = _normalized_average(historical_da_pct)
    if historical is not None:
        return AssumptionEstimate(historical, "Historical average D&A % revenue.", "Estimated", "Medium", "D&A estimated from history.", "Calculated history")

    pct, method, confidence = _profile_default("da", business_profile)
    return AssumptionEstimate(pct, method, "Business-profile estimate", confidence, "D&A missing; estimate should be reviewed.", "Business profile")


def run_assumption_sanity_checks(assumption_table, business_profile, financials) -> list[dict]:
    assumptions = assumption_table or {}
    dcf = financials or {}
    profile_text = _profile_text(business_profile)
    warnings: list[dict] = []

    def add(metric, reason, severity="Medium"):
        warnings.append({"Metric": metric, "Severity": severity, "Reason": reason})

    maintenance = _num(assumptions.get("maintenance_capex_pct_revenue"))
    growth = _num(assumptions.get("growth_capex_pct_revenue"))
    total_capex = _num(assumptions.get("total_capex_pct_revenue"))
    revenue_growth = _num(assumptions.get("revenue_cagr"))
    sbc = _num(assumptions.get("sbc_pct_revenue"))
    wc = _num(assumptions.get("working_capital_pct_revenue"))
    da = _num(assumptions.get("depreciation_amortization_pct_revenue"))
    tv_weight = _num(dcf.get("terminal_value_weight_pct"))

    if maintenance == 0 and total_capex and total_capex > 0:
        add("Maintenance CAPEX", "Maintenance CAPEX is zero while total CAPEX exists.", "High")
    if "software" in profile_text and sbc == 0 and not assumptions.get("_sbc_real_zero"):
        add("SBC % Revenue", "Software/SaaS profile should not show SBC as zero unless explicitly reported.", "High")
    if wc == 0 and not assumptions.get("_working_capital_real_zero"):
        add("Working Capital % Revenue", "Working capital is zero; confirm this is not a fallback artifact.")
    if da == 0 and not assumptions.get("_da_real_zero"):
        add("D&A % Revenue", "D&A is zero without explicit reported-zero evidence.")
    if growth == 0 and revenue_growth and revenue_growth > 0.08:
        add("Growth CAPEX", "Growth CAPEX is zero while revenue growth remains elevated.")
    if tv_weight and tv_weight > 0.65:
        add("Terminal Value Weight", "Terminal value exceeds 65% of enterprise value.", "High")

    return warnings
