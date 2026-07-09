from __future__ import annotations

import math
from typing import Any

import pandas as pd


SOTP_SCENARIOS = ["Bear Case", "Base Case", "Bull Case", "User Case", "Market-Implied Case"]
VALUATION_METHODS = ["EV/Revenue", "EV/EBITDA", "EV/NOPAT", "EV/OCF", "EV/FCF"]


SECTOR_MULTIPLE_FALLBACKS = {
    "technology": {"EV/Revenue": 5.0, "EV/EBITDA": 18.0, "EV/NOPAT": 24.0, "EV/OCF": 20.0, "EV/FCF": 22.0},
    "healthcare": {"EV/Revenue": 4.0, "EV/EBITDA": 15.0, "EV/NOPAT": 22.0, "EV/OCF": 18.0, "EV/FCF": 20.0},
    "industrials": {"EV/Revenue": 2.0, "EV/EBITDA": 11.0, "EV/NOPAT": 16.0, "EV/OCF": 14.0, "EV/FCF": 16.0},
    "consumer cyclical": {"EV/Revenue": 1.6, "EV/EBITDA": 10.0, "EV/NOPAT": 15.0, "EV/OCF": 13.0, "EV/FCF": 15.0},
    "default": {"EV/Revenue": 2.5, "EV/EBITDA": 12.0, "EV/NOPAT": 18.0, "EV/OCF": 16.0, "EV/FCF": 18.0},
}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        number = float(value)
        if not math.isfinite(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    num = _safe_float(numerator)
    den = _safe_float(denominator)
    if num is None or den in (None, 0):
        return None
    return num / den


def _latest(historicals: pd.DataFrame | None, column: str, default: float | None = None) -> float | None:
    if historicals is None or historicals.empty or column not in historicals:
        return default
    series = pd.to_numeric(historicals[column], errors="coerce").dropna()
    if series.empty:
        return default
    return float(series.iloc[-1])


def _sector_fallback(sector: str | None) -> dict:
    sector_key = str(sector or "").lower()
    for key, values in SECTOR_MULTIPLE_FALLBACKS.items():
        if key != "default" and key in sector_key:
            return values
    return SECTOR_MULTIPLE_FALLBACKS["default"]


def peer_multiple_for_method(peer_multiples: pd.DataFrame | dict | None, method: str, sector: str | None = None) -> float | None:
    if isinstance(peer_multiples, dict):
        return _safe_float(peer_multiples.get(method) or peer_multiples.get(method.replace("/", "_").lower()))
    if isinstance(peer_multiples, pd.DataFrame) and not peer_multiples.empty:
        candidates = [method, method.replace("/", "_").lower(), method.replace("/", "").lower()]
        for column in candidates:
            if column in peer_multiples:
                series = pd.to_numeric(peer_multiples[column], errors="coerce").replace([math.inf, -math.inf], pd.NA).dropna()
                if not series.empty:
                    return float(series.median())
    return _sector_fallback(sector).get(method)


def build_default_segment_data(
    historicals: pd.DataFrame | None,
    dataset: dict | None = None,
    assumptions: dict | None = None,
) -> pd.DataFrame:
    assumptions = assumptions or {}
    dataset = dataset or {}
    revenue = _latest(historicals, "Revenue", 0.0) or 0.0
    gross_margin = _latest(historicals, "Gross Margin", assumptions.get("gross_margin", 0.45)) or assumptions.get("gross_margin", 0.45)
    ocf_margin = _safe_div(_latest(historicals, "OCF"), revenue) or assumptions.get("ocf_margin", 0.16)
    nopat_margin = _safe_div(_latest(historicals, "NOPAT"), revenue) or assumptions.get("nopat_margin", 0.12)
    capex_intensity = abs(_safe_div(_latest(historicals, "Total CAPEX"), revenue) or assumptions.get("maintenance_capex_pct_revenue", 0.03))
    terminal_multiple = assumptions.get("terminal_multiple", 15.0)
    method = "EV/FCF" if ocf_margin and capex_intensity and ocf_margin > capex_intensity else "EV/NOPAT"
    description = str(dataset.get("company_description") or "").lower()
    rows = [
        {
            "Segment": "Core business",
            "Revenue": revenue,
            "Revenue Growth": assumptions.get("revenue_cagr", 0.08),
            "Gross Margin": gross_margin,
            "OPEX % Revenue": max(float(gross_margin or 0.45) - float(assumptions.get("operating_margin", 0.15) or 0.15), 0.0),
            "OCF Margin": ocf_margin,
            "NOPAT Margin": nopat_margin,
            "CAPEX % Revenue": capex_intensity,
            "Reinvestment Need": assumptions.get("growth_capex_pct_revenue", 0.02),
            "Valuation Method": method,
            "Selected Multiple": terminal_multiple,
            "Discount / Premium": 0.0,
            "Confidence": "Low" if revenue <= 0 else "Medium",
            "Source": "Manual builder fallback",
        }
    ]
    if any(token in description for token in ["software", "platform", "subscription", "cloud"]) and revenue > 0:
        rows[0]["Revenue"] = revenue * 0.75
        rows[0]["Segment"] = "Platform / recurring core"
        rows[0]["Valuation Method"] = "EV/Revenue"
        rows[0]["Selected Multiple"] = min(max(float(terminal_multiple or 15.0) / 3.0, 4.0), 10.0)
        rows.append(
            {
                "Segment": "Services / implementation",
                "Revenue": revenue * 0.25,
                "Revenue Growth": max(float(assumptions.get("revenue_cagr", 0.08) or 0.08) - 0.02, -0.1),
                "Gross Margin": max(float(gross_margin or 0.45) - 0.12, 0.1),
                "OPEX % Revenue": rows[0]["OPEX % Revenue"],
                "OCF Margin": max(float(ocf_margin or 0.16) - 0.04, 0.01),
                "NOPAT Margin": max(float(nopat_margin or 0.12) - 0.04, 0.01),
                "CAPEX % Revenue": capex_intensity,
                "Reinvestment Need": assumptions.get("growth_capex_pct_revenue", 0.02),
                "Valuation Method": "EV/EBITDA",
                "Selected Multiple": peer_multiple_for_method(None, "EV/EBITDA", dataset.get("sector")),
                "Discount / Premium": -0.1,
                "Confidence": "Low",
                "Source": "Manual split from business description",
            }
        )
    return pd.DataFrame(rows)


def normalize_segment_table(segment_data: pd.DataFrame | None, assumptions: dict | None = None) -> pd.DataFrame:
    assumptions = assumptions or {}
    if segment_data is None or segment_data.empty:
        return pd.DataFrame()
    rename = {
        "segment": "Segment",
        "revenue": "Revenue",
        "growth": "Revenue Growth",
        "margin": "NOPAT Margin",
        "multiple": "Selected Multiple",
        "method": "Valuation Method",
        "discount_premium": "Discount / Premium",
        "confidence": "Confidence",
    }
    frame = segment_data.rename(columns={k: v for k, v in rename.items() if k in segment_data.columns}).copy()
    defaults = {
        "Segment": "Segment",
        "Revenue": 0.0,
        "Revenue Growth": assumptions.get("revenue_cagr", 0.08),
        "Gross Margin": assumptions.get("gross_margin", 0.45),
        "OPEX % Revenue": max(float(assumptions.get("gross_margin", 0.45) or 0.45) - float(assumptions.get("operating_margin", 0.15) or 0.15), 0.0),
        "OCF Margin": assumptions.get("ocf_margin", 0.16),
        "NOPAT Margin": assumptions.get("nopat_margin", 0.12),
        "CAPEX % Revenue": assumptions.get("maintenance_capex_pct_revenue", 0.03),
        "Reinvestment Need": assumptions.get("growth_capex_pct_revenue", 0.02),
        "Valuation Method": "EV/NOPAT",
        "Selected Multiple": assumptions.get("terminal_multiple", 15.0),
        "Peer Multiple": None,
        "Market-Implied Multiple": None,
        "Discount / Premium": 0.0,
        "Confidence": "Manual Review",
        "Source": "Manual",
    }
    for column, default in defaults.items():
        if column not in frame:
            frame[column] = default
        frame[column] = frame[column].fillna(default)
    frame["Valuation Method"] = frame["Valuation Method"].where(frame["Valuation Method"].isin(VALUATION_METHODS), "EV/NOPAT")
    return frame[list(defaults.keys())]


def _segment_basis(row: pd.Series, method: str) -> float:
    revenue = _safe_float(row.get("Revenue"), 0.0) or 0.0
    gross_margin = _safe_float(row.get("Gross Margin"), 0.45) or 0.45
    opex_ratio = _safe_float(row.get("OPEX % Revenue"), 0.30) or 0.30
    ocf_margin = _safe_float(row.get("OCF Margin"), 0.16) or 0.16
    nopat_margin = _safe_float(row.get("NOPAT Margin"), 0.12) or 0.12
    capex_ratio = abs(_safe_float(row.get("CAPEX % Revenue"), 0.03) or 0.03)
    if method == "EV/Revenue":
        return revenue
    if method == "EV/EBITDA":
        return revenue * max(gross_margin - opex_ratio + capex_ratio, 0.0)
    if method == "EV/NOPAT":
        return revenue * max(nopat_margin, 0.0)
    if method == "EV/OCF":
        return revenue * max(ocf_margin, 0.0)
    if method == "EV/FCF":
        return revenue * max(ocf_margin - capex_ratio, 0.0)
    return revenue * max(nopat_margin, 0.0)


def _net_debt(market_data: dict | None, assumptions: dict | None) -> float:
    market_data = market_data or {}
    assumptions = assumptions or {}
    if assumptions.get("net_debt") is not None:
        return float(assumptions.get("net_debt") or 0.0)
    debt = _safe_float(market_data.get("debt"), 0.0) or 0.0
    cash = _safe_float(market_data.get("cash"), 0.0) or 0.0
    return debt - cash


def _shares(market_data: dict | None, assumptions: dict | None) -> float | None:
    assumptions = assumptions or {}
    market_data = market_data or {}
    return _safe_float(assumptions.get("diluted_shares") or market_data.get("shares_outstanding"))


def _whole_vs_parts(dcf_ev: float | None, sotp_ev: float | None, market_ev: float | None) -> tuple[str, str]:
    if sotp_ev in (None, 0):
        return "SOTP unavailable", "Segment data is insufficient for a whole-versus-parts read."
    gap = _safe_div((dcf_ev or 0) - sotp_ev, sotp_ev) if dcf_ev is not None else None
    market_gap = _safe_div((market_ev or 0) - sotp_ev, sotp_ev) if market_ev is not None else None
    if gap is not None and gap > 0.15:
        return "Whole > Sum of Parts", "DCF is materially above SOTP; synergies, shared platform economics, or operating leverage must justify the premium."
    if gap is not None and gap < -0.15:
        return "Whole < Sum of Parts", "SOTP is materially above DCF; this may indicate hidden segment value or a conglomerate discount."
    if market_gap is not None and market_gap > 0.15:
        return "Overvalued Consolidated Story", "Market EV is above SOTP; identify which segment must justify the premium."
    return "Whole ~= Sum of Parts", "DCF, market EV, and segment value are close enough that assumption quality matters more than method selection."


def _scenario_adjustments(scenario: str) -> dict:
    return {
        "Bear Case": {"growth": -0.03, "margin": -0.03, "capex": 0.02, "multiple": -0.20, "discount": -0.05},
        "Base Case": {"growth": 0.0, "margin": 0.0, "capex": 0.0, "multiple": 0.0, "discount": 0.0},
        "Bull Case": {"growth": 0.04, "margin": 0.03, "capex": -0.01, "multiple": 0.20, "discount": 0.05},
        "User Case": {"growth": 0.0, "margin": 0.0, "capex": 0.0, "multiple": 0.0, "discount": 0.0},
        "Market-Implied Case": {"growth": 0.0, "margin": 0.0, "capex": 0.0, "multiple": 0.0, "discount": 0.0},
    }.get(scenario, {})


def run_sotp(
    segment_data: pd.DataFrame | None,
    market_data: dict | None = None,
    assumptions: dict | None = None,
    scenario: str = "Base Case",
    dcf_output: dict | None = None,
    peer_multiples: pd.DataFrame | dict | None = None,
    sector: str | None = None,
) -> dict:
    """
    Segment-level valuation model. Backward compatible with the old call shape:
    run_sotp(segment_data, {"default_margin": ..., "default_multiple": ...}).
    """
    if assumptions is None and isinstance(market_data, dict) and ("default_margin" in market_data or "default_multiple" in market_data):
        assumptions = {"nopat_margin": market_data.get("default_margin"), "terminal_multiple": market_data.get("default_multiple")}
        market_data = {}
    market_data = market_data or {}
    assumptions = assumptions or {}
    segment_data = normalize_segment_table(segment_data, assumptions)
    if segment_data.empty:
        return {
            "available": False,
            "scenario": scenario,
            "segments": pd.DataFrame(),
            "segment_table": pd.DataFrame(),
            "enterprise_value": None,
            "net_debt": _net_debt(market_data, assumptions),
            "equity_value": None,
            "fair_value_per_share": None,
            "upside_downside_pct": None,
            "sotp_vs_dcf_gap_pct": None,
            "whole_vs_sum_interpretation": "Segment data unavailable from filings. Use the manual segment builder.",
            "warnings": ["Segment data unavailable from filings. Manual segment builder is active."],
            "summary": "Manual segment assumptions required; SEC segment data is unavailable.",
        }
    adjustments = _scenario_adjustments(scenario)
    rows = []
    total_ev = 0.0
    warnings = []
    reverse = None
    if scenario == "Market-Implied Case":
        reverse = run_reverse_sotp(market_data, segment_data, assumptions, peer_multiples)
    for _, raw in segment_data.iterrows():
        row = raw.copy()
        method = str(row.get("Valuation Method") or "EV/NOPAT")
        revenue = (_safe_float(row.get("Revenue"), 0.0) or 0.0) * (1 + (_safe_float(row.get("Revenue Growth"), 0.0) or 0.0) + adjustments.get("growth", 0.0))
        row["Revenue"] = max(revenue, 0.0)
        row["OCF Margin"] = max((_safe_float(row.get("OCF Margin"), 0.16) or 0.16) + adjustments.get("margin", 0.0), 0.0)
        row["NOPAT Margin"] = max((_safe_float(row.get("NOPAT Margin"), 0.12) or 0.12) + adjustments.get("margin", 0.0), 0.0)
        row["CAPEX % Revenue"] = max((_safe_float(row.get("CAPEX % Revenue"), 0.03) or 0.03) + adjustments.get("capex", 0.0), 0.0)
        peer_multiple = _safe_float(row.get("Peer Multiple")) or peer_multiple_for_method(peer_multiples, method, sector)
        selected_multiple = _safe_float(row.get("Selected Multiple"), peer_multiple or assumptions.get("terminal_multiple", 15.0)) or 0.0
        selected_multiple = max(selected_multiple * (1 + adjustments.get("multiple", 0.0)), 0.0)
        market_implied_multiple = None
        if reverse is not None and not reverse.get("segments", pd.DataFrame()).empty:
            match = reverse["segments"][reverse["segments"]["Segment"].astype(str) == str(row.get("Segment"))]
            if not match.empty:
                method_map = {
                    "EV/Revenue": "Market-Implied EV/Revenue",
                    "EV/OCF": "Market-Implied EV/OCF",
                    "EV/NOPAT": "Market-Implied EV/NOPAT",
                    "EV/FCF": "Market-Implied EV/FCF",
                }
                market_implied_multiple = _safe_float(match.iloc[0].get(method_map.get(method, "Market-Implied EV/Revenue")))
                if market_implied_multiple is not None:
                    selected_multiple = market_implied_multiple
        discount = (_safe_float(row.get("Discount / Premium"), 0.0) or 0.0) + adjustments.get("discount", 0.0)
        basis = _segment_basis(row, method)
        segment_ev = basis * selected_multiple * (1 + discount)
        if method == "EV/Revenue" and (row.get("NOPAT Margin") is None or float(row.get("NOPAT Margin") or 0) <= 0):
            warnings.append(f"{row.get('Segment')}: EV/Revenue used because profit basis is unavailable or negative; review margin normalization.")
        total_ev += segment_ev
        rows.append(
            {
                "Segment": row.get("Segment"),
                "Revenue": row.get("Revenue"),
                "Revenue Growth": row.get("Revenue Growth"),
                "Gross Margin": row.get("Gross Margin"),
                "OPEX % Revenue": row.get("OPEX % Revenue"),
                "OCF Margin": row.get("OCF Margin"),
                "NOPAT Margin": row.get("NOPAT Margin"),
                "CAPEX % Revenue": row.get("CAPEX % Revenue"),
                "Valuation Method": method,
                "Selected Multiple": selected_multiple,
                "Peer Multiple": peer_multiple,
                "Market-Implied Multiple": market_implied_multiple,
                "Segment EV": segment_ev,
                "% of Total EV": None,
                "Confidence": row.get("Confidence"),
                "Reason for Premium / Discount": _segment_premium_reason(selected_multiple, peer_multiple, row.get("Confidence")),
            }
        )
    segments = pd.DataFrame(rows)
    if total_ev:
        segments["% of Total EV"] = segments["Segment EV"] / total_ev
    net_debt = _net_debt(market_data, assumptions)
    equity_value = total_ev - net_debt
    share_count = _shares(market_data, assumptions)
    fair_value = equity_value / share_count if share_count else None
    price = _safe_float(market_data.get("price"))
    upside = fair_value / price - 1 if fair_value is not None and price else None
    dcf_ev = _safe_float((dcf_output or {}).get("enterprise_value"))
    dcf_fv = _safe_float((dcf_output or {}).get("fair_value_per_share"))
    gap = (total_ev / dcf_ev - 1) if dcf_ev else None
    market_ev = _safe_float(market_data.get("enterprise_value")) or ((_safe_float(market_data.get("market_cap")) or 0.0) + net_debt if market_data.get("market_cap") else None)
    conclusion, interpretation = _whole_vs_parts(dcf_ev, total_ev, market_ev)
    return {
        "available": True,
        "scenario": scenario,
        "segments": segments,
        "segment_table": segments,
        "enterprise_value": total_ev,
        "net_debt": net_debt,
        "equity_value": equity_value,
        "fair_value_per_share": fair_value,
        "upside_downside_pct": upside,
        "sotp_vs_dcf_gap_pct": gap,
        "dcf_fair_value_per_share": dcf_fv,
        "current_price": price,
        "current_market_ev": market_ev,
        "whole_vs_sum": conclusion,
        "whole_vs_sum_interpretation": interpretation,
        "warnings": list(dict.fromkeys(warnings)),
        "summary": f"{scenario}: {conclusion}. {interpretation}",
    }


def _segment_premium_reason(selected: float | None, peer: float | None, confidence: str | None) -> str:
    if selected is None or peer is None:
        return "Peer reference unavailable; treat selected multiple as manual."
    gap = selected / peer - 1 if peer else None
    confidence_text = str(confidence or "Manual Review")
    if gap is not None and gap > 0.15:
        return f"Premium requires stronger growth, margins, moat, or cash conversion evidence. Confidence: {confidence_text}."
    if gap is not None and gap < -0.15:
        return f"Discount reflects weaker confidence, lower margins, higher cyclicality, or higher reinvestment. Confidence: {confidence_text}."
    return f"In line with peer reference. Confidence: {confidence_text}."


def run_sotp_scenarios(
    segment_data: pd.DataFrame | None,
    market_data: dict | None,
    assumptions: dict | None,
    dcf_output: dict | None = None,
    peer_multiples: pd.DataFrame | dict | None = None,
    sector: str | None = None,
) -> dict[str, dict]:
    return {
        scenario: run_sotp(segment_data, market_data, assumptions, scenario, dcf_output, peer_multiples, sector)
        for scenario in SOTP_SCENARIOS
    }


def sotp_summary_table(scenarios: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for scenario, result in scenarios.items():
        rows.append(
            {
                "Scenario": scenario,
                "SOTP EV": result.get("enterprise_value"),
                "Equity Value": result.get("equity_value"),
                "Fair Value / Share": result.get("fair_value_per_share"),
                "Upside / Downside": result.get("upside_downside_pct"),
                "SOTP vs DCF Gap": result.get("sotp_vs_dcf_gap_pct"),
                "Whole vs Sum": result.get("whole_vs_sum"),
                "Interpretation": result.get("whole_vs_sum_interpretation"),
            }
        )
    return pd.DataFrame(rows)


def run_reverse_sotp(
    market_data: dict,
    segment_data: pd.DataFrame,
    base_segment_assumptions: dict,
    peer_multiples: pd.DataFrame | dict | None = None,
) -> dict:
    """
    Estimate segment values or multiples implied by the current enterprise value.

    This is an allocation model, not a reported fact.
    """
    segments = normalize_segment_table(segment_data, base_segment_assumptions)
    if segments.empty:
        return {
            "available": False,
            "segments": pd.DataFrame(),
            "enterprise_value": None,
            "warning": "Market-implied SOTP unavailable because segment data is missing.",
        }
    net_debt = _net_debt(market_data, base_segment_assumptions)
    market_ev = _safe_float(market_data.get("enterprise_value"))
    if market_ev is None:
        market_cap = _safe_float(market_data.get("market_cap"))
        market_ev = market_cap + net_debt if market_cap is not None else None
    if market_ev is None:
        return {
            "available": False,
            "segments": pd.DataFrame(),
            "enterprise_value": None,
            "warning": "Current EV is unavailable; cannot estimate market-implied segment multiples.",
        }
    revenue_total = pd.to_numeric(segments["Revenue"], errors="coerce").fillna(0).sum()
    profit_proxy = (pd.to_numeric(segments["Revenue"], errors="coerce").fillna(0) * pd.to_numeric(segments["OCF Margin"], errors="coerce").fillna(0)).clip(lower=0)
    profit_total = profit_proxy.sum()
    rows = []
    for idx, row in segments.iterrows():
        revenue = _safe_float(row.get("Revenue"), 0.0) or 0.0
        revenue_share = revenue / revenue_total if revenue_total else 1 / len(segments)
        profit_share = float(profit_proxy.iloc[idx] / profit_total) if profit_total else revenue_share
        allocation_weight = (revenue_share + profit_share) / 2
        implied_ev = market_ev * allocation_weight
        ocf = revenue * (_safe_float(row.get("OCF Margin"), 0.0) or 0.0)
        nopat = revenue * (_safe_float(row.get("NOPAT Margin"), 0.0) or 0.0)
        capex = revenue * abs(_safe_float(row.get("CAPEX % Revenue"), 0.0) or 0.0)
        fcf = max(ocf - capex, 0.0)
        base_ev = _segment_basis(row, str(row.get("Valuation Method") or "EV/NOPAT")) * (_safe_float(row.get("Selected Multiple"), 0.0) or 0.0)
        peer_multiple = _safe_float(row.get("Peer Multiple")) or peer_multiple_for_method(peer_multiples, str(row.get("Valuation Method") or "EV/NOPAT"))
        implied_revenue_multiple = _safe_div(implied_ev, revenue)
        implied_ocf_multiple = _safe_div(implied_ev, ocf)
        implied_nopat_multiple = _safe_div(implied_ev, nopat)
        implied_fcf_multiple = _safe_div(implied_ev, fcf)
        premium_discount = implied_revenue_multiple / peer_multiple - 1 if implied_revenue_multiple is not None and peer_multiple else None
        rows.append(
            {
                "Segment": row.get("Segment"),
                "Revenue": revenue,
                "Revenue Share": revenue_share,
                "Profit Share": profit_share,
                "Base Segment EV": base_ev,
                "Market-Implied Segment EV": implied_ev,
                "Market-Implied EV/Revenue": implied_revenue_multiple,
                "Market-Implied EV/OCF": implied_ocf_multiple,
                "Market-Implied EV/NOPAT": implied_nopat_multiple,
                "Market-Implied EV/FCF": implied_fcf_multiple,
                "Peer Median Multiple": peer_multiple,
                "Premium / Discount vs Peers": premium_discount,
                "Interpretation": _reverse_sotp_interpretation(implied_revenue_multiple, peer_multiple),
            }
        )
    return {
        "available": True,
        "enterprise_value": market_ev,
        "segments": pd.DataFrame(rows),
        "warning": "Market-implied SOTP is an allocation model, not a reported fact. Use it to understand what expectations the current stock price may already reflect.",
    }


def _reverse_sotp_interpretation(implied_multiple: float | None, peer_multiple: float | None) -> str:
    if implied_multiple is None or peer_multiple is None:
        return "Implied multiple or peer reference unavailable."
    gap = implied_multiple / peer_multiple - 1 if peer_multiple else None
    if gap is not None and gap > 0.25:
        return "Market pricing implies a material premium; segment needs strong growth, margin, or moat evidence."
    if gap is not None and gap < -0.25:
        return "Market pricing implies a discount; could reflect execution risk or hidden value if assumptions prove conservative."
    return "Market pricing is broadly in line with peer reference."


def sotp_assumption_comparison(base: pd.DataFrame, user: pd.DataFrame, market_implied: pd.DataFrame | None = None) -> pd.DataFrame:
    base = normalize_segment_table(base)
    user = normalize_segment_table(user)
    market_implied = market_implied if market_implied is not None else pd.DataFrame()
    rows = []
    for _, row in user.iterrows():
        segment = row.get("Segment")
        base_row = base[base["Segment"].astype(str) == str(segment)]
        implied_row = market_implied[market_implied["Segment"].astype(str) == str(segment)] if not market_implied.empty else pd.DataFrame()
        for metric in ["Revenue Growth", "OCF Margin", "NOPAT Margin", "CAPEX % Revenue", "Selected Multiple", "Discount / Premium"]:
            base_value = base_row.iloc[0].get(metric) if not base_row.empty else None
            market_value = implied_row.iloc[0].get("Market-Implied EV/Revenue") if metric == "Selected Multiple" and not implied_row.empty else None
            user_value = row.get(metric)
            rows.append(
                {
                    "Segment": segment,
                    "Assumption": metric,
                    "User Case": user_value,
                    "Base Case": base_value,
                    "Market-Implied": market_value,
                    "Delta vs Base": (_safe_float(user_value) or 0) - (_safe_float(base_value) or 0) if base_value is not None else None,
                    "Delta vs Market-Implied": (_safe_float(user_value) or 0) - (_safe_float(market_value) or 0) if market_value is not None else None,
                    "Source Badge": row.get("Source") or "Manual",
                }
            )
    return pd.DataFrame(rows)
