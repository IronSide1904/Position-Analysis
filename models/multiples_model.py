from __future__ import annotations

import math
from typing import Any

import pandas as pd


MULTIPLE_METRICS = [
    "P/E",
    "Forward P/E",
    "P/B",
    "P/S",
    "EV/Revenue",
    "EV/EBITDA",
    "EV/EBIT",
    "EV/NOPAT",
    "EV/OCF",
    "EV/FCF",
    "FCF Yield",
    "Earnings Yield",
    "OCF Yield",
    "Sales Multiple",
    "PEG",
]


SECTOR_MEDIANS = {
    "technology": {"P/E": 28.0, "P/B": 7.0, "P/S": 6.0, "EV/Revenue": 5.5, "EV/EBITDA": 18.0, "EV/EBIT": 22.0, "EV/NOPAT": 24.0, "EV/OCF": 20.0, "EV/FCF": 22.0, "FCF Yield": 0.035, "OCF Yield": 0.045},
    "healthcare": {"P/E": 24.0, "P/B": 4.5, "P/S": 4.0, "EV/Revenue": 4.2, "EV/EBITDA": 15.0, "EV/EBIT": 20.0, "EV/NOPAT": 22.0, "EV/OCF": 18.0, "EV/FCF": 20.0, "FCF Yield": 0.04, "OCF Yield": 0.05},
    "industrials": {"P/E": 18.0, "P/B": 3.2, "P/S": 1.8, "EV/Revenue": 2.0, "EV/EBITDA": 11.0, "EV/EBIT": 15.0, "EV/NOPAT": 16.0, "EV/OCF": 14.0, "EV/FCF": 16.0, "FCF Yield": 0.055, "OCF Yield": 0.07},
    "consumer cyclical": {"P/E": 17.0, "P/B": 3.0, "P/S": 1.5, "EV/Revenue": 1.7, "EV/EBITDA": 10.0, "EV/EBIT": 14.0, "EV/NOPAT": 15.0, "EV/OCF": 13.0, "EV/FCF": 15.0, "FCF Yield": 0.06, "OCF Yield": 0.075},
    "default": {"P/E": 20.0, "P/B": 3.5, "P/S": 2.5, "EV/Revenue": 2.8, "EV/EBITDA": 12.0, "EV/EBIT": 16.0, "EV/NOPAT": 18.0, "EV/OCF": 16.0, "EV/FCF": 18.0, "FCF Yield": 0.05, "OCF Yield": 0.065},
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
    value = num / den
    return value if math.isfinite(value) else None


def _latest(historicals: pd.DataFrame | None, column: str, default: float | None = None) -> float | None:
    if historicals is None or historicals.empty or column not in historicals:
        return default
    series = pd.to_numeric(historicals[column], errors="coerce").replace([math.inf, -math.inf], pd.NA).dropna()
    if series.empty:
        return default
    return float(series.iloc[-1])


def sector_median_multiples(sector: str | None, industry: str | None = None) -> dict:
    text = f"{sector or ''} {industry or ''}".lower()
    for key, values in SECTOR_MEDIANS.items():
        if key != "default" and key in text:
            return dict(values)
    return dict(SECTOR_MEDIANS["default"])


def calculate_current_multiples(historicals: pd.DataFrame | None, market_data: dict | None) -> dict:
    market_data = market_data or {}
    market_cap = _safe_float(market_data.get("market_cap"))
    ev = _safe_float(market_data.get("enterprise_value"))
    if ev is None and market_cap is not None:
        debt = _safe_float(market_data.get("debt"), 0.0) or 0.0
        cash = _safe_float(market_data.get("cash"), 0.0) or 0.0
        ev = market_cap + debt - cash
    revenue = _latest(historicals, "Revenue")
    ebitda = _latest(historicals, "EBITDA")
    ebit = _latest(historicals, "EBIT")
    nopat = _latest(historicals, "NOPAT")
    ocf = _latest(historicals, "OCF")
    fcf = _latest(historicals, "FCF")
    net_income = _latest(historicals, "Net Income")
    book_value = _latest(historicals, "Book Value")
    values = {
        "P/E": _safe_float(market_data.get("pe")) or _safe_div(market_cap, net_income),
        "Forward P/E": _safe_float(market_data.get("forward_pe")),
        "P/B": _safe_float(market_data.get("pb")) or _safe_div(market_cap, book_value),
        "P/S": _safe_float(market_data.get("ps")) or _safe_div(market_cap, revenue),
        "EV/Revenue": _safe_div(ev, revenue),
        "EV/EBITDA": _safe_div(ev, ebitda),
        "EV/EBIT": _safe_div(ev, ebit),
        "EV/NOPAT": _safe_div(ev, nopat),
        "EV/OCF": _safe_div(ev, ocf),
        "EV/FCF": _safe_float(market_data.get("pfcf")) or _safe_div(ev, fcf),
        "FCF Yield": _safe_div(fcf, market_cap),
        "Earnings Yield": _safe_div(net_income, market_cap),
        "OCF Yield": _safe_div(ocf, market_cap),
        "Sales Multiple": _safe_float(market_data.get("ps")) or _safe_div(market_cap, revenue),
        "PEG": _safe_float(market_data.get("peg")),
    }
    return {metric: values.get(metric) for metric in MULTIPLE_METRICS}


def _peer_column_candidates(metric: str) -> list[str]:
    base = metric.replace("/", "_").replace(" ", "_").replace("-", "_").lower()
    compact = metric.replace("/", "").replace(" ", "").replace("-", "").lower()
    aliases = {
        "EV/Revenue": ["ev_sales", "ev_revenue", "enterprise_value_to_revenue"],
        "EV/EBITDA": ["ev_ebitda"],
        "EV/EBIT": ["ev_ebit"],
        "EV/NOPAT": ["ev_nopat"],
        "EV/OCF": ["ev_ocf"],
        "EV/FCF": ["ev_fcf"],
        "P/E": ["pe", "trailing_pe"],
        "Forward P/E": ["forward_pe"],
        "P/B": ["pb", "price_to_book"],
        "P/S": ["ps", "price_to_sales"],
        "FCF Yield": ["fcf_yield"],
        "OCF Yield": ["ocf_yield"],
        "PEG": ["peg"],
    }
    return [base, compact, *aliases.get(metric, [])]


def peer_median_multiples(peer_df: pd.DataFrame | None, sector: str | None = None, industry: str | None = None) -> tuple[dict, list[str]]:
    sector_values = sector_median_multiples(sector, industry)
    warnings = []
    medians: dict[str, float | None] = {}
    if peer_df is None or peer_df.empty:
        warnings.append("Peer set limited. Sector median may be less reliable.")
        return {metric: sector_values.get(metric) for metric in MULTIPLE_METRICS}, warnings
    for metric in MULTIPLE_METRICS:
        value = None
        for column in _peer_column_candidates(metric):
            if column in peer_df:
                series = pd.to_numeric(peer_df[column], errors="coerce").replace([math.inf, -math.inf], pd.NA).dropna()
                if not series.empty:
                    value = float(series.median())
                    break
        medians[metric] = value if value is not None else sector_values.get(metric)
    provided = sum(1 for metric in MULTIPLE_METRICS for column in _peer_column_candidates(metric) if column in peer_df)
    if provided < 4:
        warnings.append("Peer set limited. Sector median may be less reliable.")
    return medians, warnings


def _basis_values_from_forecast(dcf_output: dict | None, historicals: pd.DataFrame | None, basis: str = "Normalized Year") -> dict:
    dcf_output = dcf_output or {}
    forecast = dcf_output.get("forecast_table", pd.DataFrame())
    basis = basis or "Normalized Year"
    if forecast is not None and not forecast.empty and basis in {"Next Year", "Final Forecast Year", "Normalized Year"}:
        row = forecast.iloc[0] if basis == "Next Year" else forecast.iloc[-1]
        revenue = _safe_float(row.get("Revenue"))
        ebitda = _safe_float(row.get("EBITDA")) or (_safe_float(row.get("NOPAT")) or 0) / 0.79
        ebit = _safe_float(row.get("EBIT")) or (_safe_float(row.get("NOPAT")) or 0) / 0.79
        nopat = _safe_float(row.get("NOPAT"))
        ocf = _safe_float(row.get("OCF"))
        fcf = _safe_float(row.get("FCF"))
        net_income = nopat
        return {"Revenue": revenue, "EBITDA": ebitda, "EBIT": ebit, "NOPAT": nopat, "OCF": ocf, "FCF": fcf, "Net Income": net_income}
    return {
        "Revenue": _latest(historicals, "Revenue"),
        "EBITDA": _latest(historicals, "EBITDA"),
        "EBIT": _latest(historicals, "EBIT"),
        "NOPAT": _latest(historicals, "NOPAT"),
        "OCF": _latest(historicals, "OCF"),
        "FCF": _latest(historicals, "FCF"),
        "Net Income": _latest(historicals, "Net Income"),
    }


def calculate_scenario_implied_multiples(
    scenario_outputs: dict[str, dict],
    historicals: pd.DataFrame | None,
    market_data: dict | None,
    basis: str = "Normalized Year",
) -> dict[str, dict]:
    market_data = market_data or {}
    rows = {}
    book_value = _latest(historicals, "Book Value")
    for scenario, output in scenario_outputs.items():
        values = _basis_values_from_forecast(output, historicals, basis)
        ev = _safe_float(output.get("enterprise_value"))
        equity = _safe_float(output.get("equity_value"))
        if equity is None and output.get("fair_value_per_share") is not None:
            shares = _safe_float(market_data.get("shares_outstanding"))
            equity = _safe_float(output.get("fair_value_per_share")) * shares if shares else None
        scenario_values = {
            "P/E": _safe_div(equity, values.get("Net Income")),
            "P/B": _safe_div(equity, book_value),
            "P/S": _safe_div(equity, values.get("Revenue")),
            "EV/Revenue": _safe_div(ev, values.get("Revenue")),
            "EV/EBITDA": _safe_div(ev, values.get("EBITDA")),
            "EV/EBIT": _safe_div(ev, values.get("EBIT")),
            "EV/NOPAT": _safe_div(ev, values.get("NOPAT")),
            "EV/OCF": _safe_div(ev, values.get("OCF")),
            "EV/FCF": _safe_div(ev, values.get("FCF")),
            "FCF Yield": _safe_div(values.get("FCF"), equity),
            "OCF Yield": _safe_div(values.get("OCF"), equity),
        }
        rows[scenario] = scenario_values
    return rows


def multiples_interpretation(metric: str, scenario_value: float | None, peer_value: float | None, moat_score: float | None = None) -> str:
    if scenario_value is None or peer_value is None:
        return "Unavailable; missing company, scenario, or peer metric."
    if "Yield" in metric:
        gap = scenario_value - peer_value
        if gap < -0.02:
            return "Lower yield than peers; valuation requires stronger growth, moat, or cash conversion."
        if gap > 0.02:
            return "Higher yield than peers; valuation may embed discount or execution concern."
        return "Broadly in line with peer yield."
    gap = scenario_value / peer_value - 1 if peer_value else None
    if gap is not None and gap > 0.25:
        quality = "Moat support is stronger" if moat_score and moat_score >= 7 else "Needs stronger moat, growth, margin, or OCF evidence"
        return f"Premium to peers. {quality} before accepting this case."
    if gap is not None and gap < -0.25:
        return "Discount to peers; check whether risk, lower growth, cyclicality, dilution, or reinvestment explains it."
    return "In line with peers; assumptions do not require a major relative-valuation stretch."


def build_multiples_table(
    current_multiples: dict,
    scenario_multiples: dict[str, dict],
    peer_medians: dict,
    sector_medians: dict,
    moat_score: float | None = None,
) -> pd.DataFrame:
    rows = []
    scenario_labels = ["Bear Case", "Base Case", "Bull Case", "User Case", "Market-Implied Case", "SOTP Case"]
    for metric in MULTIPLE_METRICS:
        peer = peer_medians.get(metric)
        current = current_multiples.get(metric)
        row = {
            "Metric": metric,
            "Current Company": current,
            "Peer Median": peer,
            "Sector Median": sector_medians.get(metric),
            "Premium / Discount vs Peer": current / peer - 1 if current is not None and peer else None,
            "Interpretation": multiples_interpretation(metric, scenario_multiples.get("User Case", {}).get(metric) or current, peer, moat_score),
        }
        for scenario in scenario_labels:
            row[scenario] = scenario_multiples.get(scenario, {}).get(metric)
        rows.append(row)
    columns = ["Metric", "Current Company", *scenario_labels, "Peer Median", "Sector Median", "Premium / Discount vs Peer", "Interpretation"]
    return pd.DataFrame(rows)[columns]
