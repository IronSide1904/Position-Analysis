from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd


SOTP_SCENARIOS = ["Bear Case", "Base Case", "Bull Case", "User Case", "Market-Implied Case"]
VALUATION_METHODS = ["EV/Revenue", "EV/EBITDA", "EV/EBIT", "EV/NOPAT", "EV/OCF", "EV/FCF", "P/E", "Manual Value"]
NORMALIZED_BASIS_OPTIONS = ["Final forecast year", "Average of final 2 years", "Average of final 3 years", "Manual normalized year"]


SECTOR_MULTIPLE_FALLBACKS = {
    "technology": {"EV/Revenue": 5.0, "EV/EBITDA": 18.0, "EV/EBIT": 20.0, "EV/NOPAT": 24.0, "EV/OCF": 20.0, "EV/FCF": 22.0, "P/E": 24.0},
    "healthcare": {"EV/Revenue": 4.0, "EV/EBITDA": 15.0, "EV/EBIT": 18.0, "EV/NOPAT": 22.0, "EV/OCF": 18.0, "EV/FCF": 20.0, "P/E": 22.0},
    "industrials": {"EV/Revenue": 2.0, "EV/EBITDA": 11.0, "EV/EBIT": 14.0, "EV/NOPAT": 16.0, "EV/OCF": 14.0, "EV/FCF": 16.0, "P/E": 16.0},
    "consumer cyclical": {"EV/Revenue": 1.6, "EV/EBITDA": 10.0, "EV/EBIT": 13.0, "EV/NOPAT": 15.0, "EV/OCF": 13.0, "EV/FCF": 15.0, "P/E": 15.0},
    "default": {"EV/Revenue": 2.5, "EV/EBITDA": 12.0, "EV/EBIT": 15.0, "EV/NOPAT": 18.0, "EV/OCF": 16.0, "EV/FCF": 18.0, "P/E": 18.0},
}


PRODUCT_SEGMENT_FALLBACKS = {
    "AAPL": [
        ("iPhone", 0.50, "EV/Revenue", 4.5, "Reported product revenue category / manual allocation"),
        ("Mac", 0.08, "EV/Revenue", 3.0, "Reported product revenue category / manual allocation"),
        ("iPad", 0.07, "EV/Revenue", 2.5, "Reported product revenue category / manual allocation"),
        ("Wearables, Home and Accessories", 0.10, "EV/Revenue", 3.5, "Reported product revenue category / manual allocation"),
        ("Services", 0.25, "EV/OCF", 22.0, "Reported services category / manual allocation"),
    ],
    "AMZN": [
        ("AWS", 0.18, "EV/OCF", 20.0, "Segment/product-service fallback"),
        ("Online Stores", 0.38, "EV/Revenue", 1.2, "Revenue disaggregation fallback"),
        ("Third-Party Seller Services", 0.20, "EV/Revenue", 3.5, "Revenue disaggregation fallback"),
        ("Advertising", 0.12, "EV/Revenue", 6.0, "Revenue disaggregation fallback"),
        ("Subscriptions", 0.08, "EV/Revenue", 4.0, "Revenue disaggregation fallback"),
        ("Physical Stores / Other", 0.04, "EV/Revenue", 0.8, "Manual allocation fallback"),
    ],
    "MSFT": [
        ("Intelligent Cloud / Azure", 0.34, "EV/OCF", 22.0, "Segment/product-service fallback"),
        ("Productivity & Business Processes", 0.36, "EV/OCF", 24.0, "Segment fallback"),
        ("More Personal Computing", 0.18, "EV/NOPAT", 18.0, "Segment fallback"),
        ("LinkedIn", 0.06, "EV/Revenue", 7.0, "Product-service fallback"),
        ("Gaming", 0.06, "EV/Revenue", 4.0, "Product-service fallback"),
    ],
    "NBIS": [
        ("Blackwell capacity economics", 0.45, "EV/Revenue", 6.0, "Manual capacity economics fallback"),
        ("Rubin capacity economics", 0.20, "EV/Revenue", 5.5, "Manual capacity economics fallback"),
        ("Other compute / cloud services", 0.25, "EV/Revenue", 4.0, "Manual capacity economics fallback"),
        ("Customer prepayments / services", 0.10, "EV/OCF", 12.0, "Manual funding/service fallback"),
    ],
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


def _latest_model_year(historicals: pd.DataFrame | None) -> int | None:
    if historicals is None or historicals.empty or "Period" not in historicals:
        return None
    for period in reversed(historicals["Period"].dropna().astype(str).tolist()):
        match = re.search(r"(20\d{2}|19\d{2})", period)
        if match:
            return int(match.group(1))
    return None


def _forecast_label(historicals: pd.DataFrame | None, year: int) -> str:
    latest_year = _latest_model_year(historicals)
    if latest_year:
        return f"FY{latest_year + int(year)}{'E' if int(year) == 1 else 'F'}"
    return f"FY{int(year)}{'E' if int(year) == 1 else 'F'}"


def _actual_label(period: str) -> str:
    text = str(period or "").strip()
    if not text:
        return "Latest / LTM"
    if "ltm" in text.lower() or "latest" in text.lower():
        return "Latest / LTM"
    match = re.search(r"(20\d{2}|19\d{2})", text)
    if match:
        suffix = "A" if not re.search(r"[AEF]$", text.strip(), re.IGNORECASE) else text.strip()[-1].upper()
        return f"FY{match.group(1)}{suffix}"
    return text


def _dedupe(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def sotp_timeframe_options(historicals: pd.DataFrame | None = None, dcf_output: dict | None = None, assumptions: dict | None = None) -> list[str]:
    options = ["Latest / LTM"]
    if historicals is not None and not historicals.empty and "Period" in historicals:
        actuals = [_actual_label(period) for period in historicals["Period"].dropna().astype(str).tolist()]
        options.extend([label for label in actuals[-4:] if label != "Latest / LTM"])
    forecast = (dcf_output or {}).get("forecast_table", pd.DataFrame())
    if forecast is not None and not forecast.empty and "Year" in forecast:
        for year in pd.to_numeric(forecast["Year"], errors="coerce").dropna().astype(int).tolist():
            options.append(_forecast_label(historicals, year))
        options.append("Terminal Year")
    elif assumptions:
        for year in range(1, int(assumptions.get("forecast_years", 5) or 5) + 1):
            options.append(_forecast_label(historicals, year))
        options.append("Terminal Year")
    options.append("Normalized Year")
    return _dedupe(options)


def _forecast_lookup(historicals: pd.DataFrame | None, forecast: pd.DataFrame | None) -> dict[str, pd.Series]:
    lookup: dict[str, pd.Series] = {}
    if forecast is None or forecast.empty or "Year" not in forecast:
        return lookup
    for _, row in forecast.iterrows():
        year = _safe_float(row.get("Year"))
        if year is None:
            continue
        lookup[_forecast_label(historicals, int(year))] = row
    return lookup


def _latest_actual_row(historicals: pd.DataFrame | None) -> pd.Series | None:
    if historicals is None or historicals.empty:
        return None
    return historicals.iloc[-1]


def _actual_lookup(historicals: pd.DataFrame | None) -> dict[str, pd.Series]:
    lookup: dict[str, pd.Series] = {}
    if historicals is None or historicals.empty or "Period" not in historicals:
        return lookup
    for _, row in historicals.iterrows():
        lookup[_actual_label(str(row.get("Period") or ""))] = row
    latest = _latest_actual_row(historicals)
    if latest is not None:
        lookup["Latest / LTM"] = latest
    return lookup


def _mean_row(rows: list[pd.Series]) -> pd.Series | None:
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    averaged = numeric.mean(axis=0, skipna=True)
    for column in frame.columns:
        if column not in averaged or pd.isna(averaged.get(column)):
            averaged[column] = frame[column].dropna().iloc[-1] if not frame[column].dropna().empty else None
    return averaged


def _timeframe_row(
    historicals: pd.DataFrame | None,
    dcf_output: dict | None,
    timeframe: str | None,
    normalized_basis: str = "Final forecast year",
    normalized_timeframe: str | None = None,
) -> tuple[pd.Series | None, str, str]:
    forecast = (dcf_output or {}).get("forecast_table", pd.DataFrame())
    forecast_lookup = _forecast_lookup(historicals, forecast)
    actual_lookup = _actual_lookup(historicals)
    options = sotp_timeframe_options(historicals, dcf_output)
    selected = timeframe if timeframe in options else ("Normalized Year" if "Normalized Year" in options else options[-1])
    basis = f"Segment/product financials from {selected}"
    if selected == "Normalized Year":
        if normalized_basis == "Manual normalized year" and normalized_timeframe in forecast_lookup:
            return forecast_lookup[normalized_timeframe], selected, f"Manual normalized year: {normalized_timeframe}"
        forecast_rows = list(forecast_lookup.values())
        if not forecast_rows:
            return actual_lookup.get("Latest / LTM"), selected, "Normalized Year fallback: Latest / LTM"
        if normalized_basis == "Average of final 3 years":
            return _mean_row(forecast_rows[-3:]), selected, "Average of final 3 forecast years"
        if normalized_basis == "Average of final 2 years":
            return _mean_row(forecast_rows[-2:]), selected, "Average of final 2 forecast years"
        return forecast_rows[-1], selected, "Final forecast year"
    if selected == "Terminal Year":
        if forecast_lookup:
            return list(forecast_lookup.values())[-1], selected, "Terminal Year uses final forecast year operating metrics"
        return actual_lookup.get("Latest / LTM"), selected, "Terminal Year fallback: Latest / LTM"
    if selected in forecast_lookup:
        return forecast_lookup[selected], selected, basis
    if selected in actual_lookup:
        return actual_lookup[selected], selected, basis
    return actual_lookup.get("Latest / LTM"), selected, "Selected timeframe unavailable; using Latest / LTM"


def _row_financials(row: pd.Series | None, assumptions: dict | None = None) -> dict:
    assumptions = assumptions or {}
    if row is None:
        return {}
    revenue = _safe_float(row.get("Revenue"))
    gross_margin = _safe_float(row.get("Gross Margin"), assumptions.get("gross_margin", 0.45))
    opex_pct = _safe_float(row.get("OPEX % Revenue"), assumptions.get("opex_pct_revenue"))
    if opex_pct is None and revenue:
        opex = _safe_float(row.get("OPEX"))
        opex_pct = _safe_div(abs(opex), revenue) if opex is not None else None
    if opex_pct is None and gross_margin is not None:
        opex_pct = max(float(gross_margin) - float(assumptions.get("operating_margin", 0.15) or 0.15), 0.0)
    capex_pct = _safe_float(row.get("Total CAPEX % Revenue"), _safe_float(row.get("CAPEX % Revenue")))
    if capex_pct is None and revenue:
        capex_pct = _safe_div(abs(_safe_float(row.get("CAPEX")) or _safe_float(row.get("Total CAPEX")) or 0.0), revenue)
    ocf_margin = _safe_float(row.get("OCF Margin")) or _safe_div(row.get("OCF"), revenue) or assumptions.get("ocf_margin", 0.16)
    nopat_margin = _safe_float(row.get("NOPAT Margin")) or _safe_div(row.get("NOPAT"), revenue) or assumptions.get("nopat_margin", 0.12)
    return {
        "Revenue": revenue,
        "Gross Margin": gross_margin,
        "OPEX % Revenue": opex_pct,
        "OCF Margin": ocf_margin,
        "NOPAT Margin": nopat_margin,
        "CAPEX % Revenue": capex_pct,
        "D&A % Revenue": _safe_float(row.get("D&A % Revenue"), capex_pct),
    }


def _manual_period_value(row: pd.Series, timeframe: str, metric: str) -> float | None:
    candidates = [
        f"{timeframe} {metric}",
        f"{metric} {timeframe}",
        f"{timeframe} {metric} ($B)",
        f"{metric} {timeframe} ($B)",
    ]
    for column in candidates:
        if column in row and row.get(column) is not None:
            value = _safe_float(row.get(column))
            if value is not None:
                return value * 1e9 if "($B)" in column else value
    return None


def apply_sotp_timeframe(
    segment_data: pd.DataFrame | None,
    historicals: pd.DataFrame | None = None,
    dcf_output: dict | None = None,
    assumptions: dict | None = None,
    timeframe: str | None = "Normalized Year",
    normalized_basis: str = "Final forecast year",
    normalized_timeframe: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    assumptions = assumptions or {}
    segments = normalize_segment_table(segment_data, assumptions)
    row, selected, basis = _timeframe_row(historicals, dcf_output, timeframe, normalized_basis, normalized_timeframe)
    period_financials = _row_financials(row, assumptions)
    context = {
        "timeframe": selected,
        "basis": basis,
        "available_timeframes": sotp_timeframe_options(historicals, dcf_output, assumptions),
        "warnings": [],
    }
    if segments.empty:
        return segments, context
    base_total = pd.to_numeric(segments["Revenue"], errors="coerce").fillna(0).sum()
    period_revenue = period_financials.get("Revenue")
    if period_revenue is None:
        context["warnings"].append(f"SOTP timeframe unavailable for {selected}: consolidated revenue is missing. Using current segment revenue.")
        return segments, context
    latest_revenue = _row_financials(_latest_actual_row(historicals), assumptions).get("Revenue") or base_total
    latest_margin_context = _row_financials(_latest_actual_row(historicals), assumptions)
    for idx, raw in segments.iterrows():
        manual_revenue = _manual_period_value(raw, selected, "Revenue")
        if manual_revenue is not None:
            segments.at[idx, "Revenue"] = max(manual_revenue, 0.0)
        elif base_total:
            share = (_safe_float(raw.get("Revenue"), 0.0) or 0.0) / base_total
            segments.at[idx, "Revenue"] = max(period_revenue * share, 0.0)
        elif latest_revenue:
            scale = period_revenue / latest_revenue
            segments.at[idx, "Revenue"] = max((_safe_float(raw.get("Revenue"), 0.0) or 0.0) * scale, 0.0)
        for column in ["Gross Margin", "OPEX % Revenue", "OCF Margin", "NOPAT Margin", "CAPEX % Revenue"]:
            period_value = period_financials.get(column)
            latest_value = latest_margin_context.get(column)
            if period_value is None or latest_value is None:
                continue
            current = _safe_float(raw.get(column))
            if current is None:
                segments.at[idx, column] = period_value
            else:
                if column == "CAPEX % Revenue":
                    segments.at[idx, column] = max(current + (period_value - latest_value), 0.0)
                else:
                    segments.at[idx, column] = float(np.clip(current + (period_value - latest_value), -1.0, 1.0))
    return segments, context


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
    ticker = str(dataset.get("ticker") or "").upper()
    product_fallbacks = PRODUCT_SEGMENT_FALLBACKS.get(ticker)
    if product_fallbacks and revenue > 0:
        rows = []
        for segment, weight, segment_method, multiple, source in product_fallbacks:
            segment_revenue = revenue * weight
            segment_gross_margin = gross_margin
            segment_ocf_margin = ocf_margin
            segment_nopat_margin = nopat_margin
            if ticker == "AAPL" and "Services" in segment:
                segment_gross_margin = min(max(float(gross_margin or 0.45) + 0.18, 0.55), 0.85)
                segment_ocf_margin = min(max(float(ocf_margin or 0.16) + 0.08, 0.20), 0.60)
                segment_nopat_margin = min(max(float(nopat_margin or 0.12) + 0.08, 0.18), 0.55)
            elif ticker == "AAPL":
                segment_gross_margin = max(float(gross_margin or 0.45) - 0.04, 0.20)
                segment_ocf_margin = max(float(ocf_margin or 0.16) - 0.02, 0.03)
            rows.append(
                {
                    "Segment": segment,
                    "Revenue": segment_revenue,
                    "Revenue Growth": assumptions.get("revenue_cagr", 0.08),
                    "Gross Margin": segment_gross_margin,
                    "OPEX % Revenue": max(float(segment_gross_margin or 0.45) - float(assumptions.get("operating_margin", 0.15) or 0.15), 0.0),
                    "OCF Margin": segment_ocf_margin,
                    "NOPAT Margin": segment_nopat_margin,
                    "CAPEX % Revenue": capex_intensity,
                    "Reinvestment Need": assumptions.get("growth_capex_pct_revenue", 0.02),
                    "Valuation Method": segment_method,
                    "Selected Multiple": multiple,
                    "Peer Multiple": peer_multiple_for_method(None, segment_method, dataset.get("sector")),
                    "Market-Implied Multiple": None,
                    "Manual Segment Value": None,
                    "Discount / Premium": 0.0,
                    "Confidence": "Medium",
                    "Source": source,
                }
            )
        return pd.DataFrame(rows)
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
            "Manual Segment Value": None,
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
                "Manual Segment Value": None,
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
        "Manual Segment Value": None,
        "Discount / Premium": 0.0,
        "Confidence": "Manual Review",
        "Source": "Manual",
        "User Note": "",
    }
    for column, default in defaults.items():
        if column not in frame:
            frame[column] = default
        frame[column] = frame[column].fillna(default)
    frame["Valuation Method"] = frame["Valuation Method"].where(frame["Valuation Method"].isin(VALUATION_METHODS), "EV/NOPAT")
    extra_columns = [
        column
        for column in frame.columns
        if column not in defaults
        and (
            re.match(r"^(FY\d{4}[AEF]|Latest / LTM|Terminal Year|Normalized Year) Revenue$", str(column))
            or re.match(r"^(FY\d{4}[AEF]|Latest / LTM|Terminal Year|Normalized Year) Manual Segment Value$", str(column))
            or re.match(r"^FY\d+[AEF] Revenue$", str(column))
            or re.match(r"^FY\d+[AEF] Manual Segment Value$", str(column))
        )
    ]
    return frame[[*list(defaults.keys()), *extra_columns]]


def _segment_metric(row: pd.Series, method: str) -> tuple[str, float | None]:
    revenue = _safe_float(row.get("Revenue"), 0.0) or 0.0
    gross_margin = _safe_float(row.get("Gross Margin"), 0.45) or 0.45
    opex_ratio = _safe_float(row.get("OPEX % Revenue"), 0.30) or 0.30
    ocf_margin = _safe_float(row.get("OCF Margin"), 0.16) or 0.16
    nopat_margin = _safe_float(row.get("NOPAT Margin"), 0.12) or 0.12
    capex_ratio = abs(_safe_float(row.get("CAPEX % Revenue"), 0.03) or 0.03)
    ebit = revenue * max(gross_margin - opex_ratio, 0.0)
    if method == "EV/Revenue":
        return "Revenue", revenue
    if method == "EV/EBITDA":
        return "EBITDA", revenue * max(gross_margin - opex_ratio + capex_ratio, 0.0)
    if method == "EV/EBIT":
        return "EBIT", ebit
    if method == "EV/NOPAT":
        return "NOPAT", revenue * max(nopat_margin, 0.0)
    if method == "EV/OCF":
        return "OCF", revenue * max(ocf_margin, 0.0)
    if method == "EV/FCF":
        return "FCF", revenue * max(ocf_margin - capex_ratio, 0.0)
    if method == "P/E":
        return "Net Income / NOPAT", revenue * max(nopat_margin, 0.0)
    if method == "Manual Value":
        return "Manual Value", _safe_float(row.get("Manual Segment Value"), 0.0)
    return "NOPAT", revenue * max(nopat_margin, 0.0)


def _segment_basis(row: pd.Series, method: str) -> float:
    _metric, value = _segment_metric(row, method)
    return _safe_float(value, 0.0) or 0.0


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
    historicals: pd.DataFrame | None = None,
    timeframe: str | None = "Normalized Year",
    normalized_basis: str = "Final forecast year",
    normalized_timeframe: str | None = None,
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
    segment_data, timeframe_context = apply_sotp_timeframe(
        segment_data,
        historicals=historicals,
        dcf_output=dcf_output,
        assumptions=assumptions,
        timeframe=timeframe,
        normalized_basis=normalized_basis,
        normalized_timeframe=normalized_timeframe,
    )
    if segment_data.empty:
        return {
            "available": False,
            "scenario": scenario,
            "timeframe": timeframe_context.get("timeframe"),
            "timeframe_basis": timeframe_context.get("basis"),
            "normalized_basis": normalized_basis,
            "available_timeframes": timeframe_context.get("available_timeframes", []),
            "segments": pd.DataFrame(),
            "segment_table": pd.DataFrame(),
            "enterprise_value": None,
            "net_debt": _net_debt(market_data, assumptions),
            "equity_value": None,
            "fair_value_per_share": None,
            "upside_downside_pct": None,
            "sotp_vs_dcf_gap_pct": None,
            "whole_vs_sum_interpretation": "Segment data unavailable from filings. Use the manual segment builder.",
            "warnings": ["Segment data unavailable from filings. Manual segment builder is active.", *timeframe_context.get("warnings", [])],
            "summary": "Manual segment assumptions required; SEC segment data is unavailable.",
        }
    adjustments = _scenario_adjustments(scenario)
    rows = []
    total_ev = 0.0
    warnings = []
    reverse = None
    if scenario == "Market-Implied Case":
        reverse = run_reverse_sotp(
            market_data,
            segment_data,
            assumptions,
            peer_multiples,
            historicals=historicals,
            dcf_output=dcf_output,
            timeframe=timeframe_context.get("timeframe"),
            normalized_basis=normalized_basis,
            normalized_timeframe=normalized_timeframe,
        )
    for _, raw in segment_data.iterrows():
        row = raw.copy()
        method = str(row.get("Valuation Method") or "EV/NOPAT")
        revenue = (_safe_float(row.get("Revenue"), 0.0) or 0.0) * (1 + adjustments.get("growth", 0.0))
        row["Revenue"] = max(revenue, 0.0)
        row["Revenue Growth"] = (_safe_float(row.get("Revenue Growth"), 0.0) or 0.0) + adjustments.get("growth", 0.0)
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
        valuation_metric, metric_value = _segment_metric(row, method)
        basis = _safe_float(metric_value, 0.0) or 0.0
        if method == "Manual Value":
            segment_ev = _safe_float(row.get("Manual Segment Value"), 0.0) or 0.0
        else:
            segment_ev = basis * selected_multiple * (1 + discount)
            if basis <= 0:
                warnings.append(f"{row.get('Segment')}: {timeframe_context.get('timeframe')} {valuation_metric} is unavailable or non-positive. Choose another metric or add manual assumptions.")
        if method == "EV/Revenue" and (row.get("NOPAT Margin") is None or float(row.get("NOPAT Margin") or 0) <= 0):
            warnings.append(f"{row.get('Segment')}: EV/Revenue used because profit basis is unavailable or negative; review margin normalization.")
        gross_profit = row.get("Revenue") * (_safe_float(row.get("Gross Margin"), 0.45) or 0.45)
        ebit = row.get("Revenue") * max((_safe_float(row.get("Gross Margin"), 0.45) or 0.45) - (_safe_float(row.get("OPEX % Revenue"), 0.30) or 0.30), 0.0)
        capex = row.get("Revenue") * abs(_safe_float(row.get("CAPEX % Revenue"), 0.03) or 0.03)
        ebitda = row.get("Revenue") * max((_safe_float(row.get("Gross Margin"), 0.45) or 0.45) - (_safe_float(row.get("OPEX % Revenue"), 0.30) or 0.30) + abs(_safe_float(row.get("CAPEX % Revenue"), 0.03) or 0.03), 0.0)
        total_ev += segment_ev
        rows.append(
            {
                "Segment": row.get("Segment"),
                "SOTP Timeframe": timeframe_context.get("timeframe"),
                "Valuation Metric": valuation_metric,
                "Metric Value": metric_value,
                "Revenue": row.get("Revenue"),
                "Revenue Growth": row.get("Revenue Growth"),
                "Gross Margin": row.get("Gross Margin"),
                "Gross Profit": gross_profit,
                "OPEX % Revenue": row.get("OPEX % Revenue"),
                "EBITDA": ebitda,
                "OCF Margin": row.get("OCF Margin"),
                "NOPAT Margin": row.get("NOPAT Margin"),
                "CAPEX % Revenue": row.get("CAPEX % Revenue"),
                "CAPEX": capex,
                "Valuation Method": method,
                "Selected Multiple": selected_multiple,
                "Peer Multiple": peer_multiple,
                "Market-Implied Multiple": market_implied_multiple,
                "Manual Segment Value": row.get("Manual Segment Value"),
                "EBIT": ebit,
                "NOPAT": row.get("Revenue") * max((_safe_float(row.get("NOPAT Margin"), 0.12) or 0.12), 0.0),
                "OCF": row.get("Revenue") * max((_safe_float(row.get("OCF Margin"), 0.16) or 0.16), 0.0),
                "FCF": row.get("Revenue") * max((_safe_float(row.get("OCF Margin"), 0.16) or 0.16) - abs(_safe_float(row.get("CAPEX % Revenue"), 0.03) or 0.03), 0.0),
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
        "timeframe": timeframe_context.get("timeframe"),
        "timeframe_basis": timeframe_context.get("basis"),
        "normalized_basis": normalized_basis,
        "available_timeframes": timeframe_context.get("available_timeframes", []),
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
        "warnings": list(dict.fromkeys([*warnings, *timeframe_context.get("warnings", [])])),
        "summary": f"{scenario} {timeframe_context.get('timeframe')}: {conclusion}. {interpretation}",
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
    historicals: pd.DataFrame | None = None,
    timeframe: str | None = "Normalized Year",
    normalized_basis: str = "Final forecast year",
    normalized_timeframe: str | None = None,
) -> dict[str, dict]:
    return {
        scenario: run_sotp(
            segment_data,
            market_data,
            assumptions,
            scenario,
            dcf_output,
            peer_multiples,
            sector,
            historicals=historicals,
            timeframe=timeframe,
            normalized_basis=normalized_basis,
            normalized_timeframe=normalized_timeframe,
        )
        for scenario in SOTP_SCENARIOS
    }


def sotp_summary_table(scenarios: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for scenario, result in scenarios.items():
        rows.append(
            {
                "Scenario": scenario,
                "Timeframe": result.get("timeframe"),
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
    historicals: pd.DataFrame | None = None,
    dcf_output: dict | None = None,
    timeframe: str | None = "Normalized Year",
    normalized_basis: str = "Final forecast year",
    normalized_timeframe: str | None = None,
) -> dict:
    """
    Estimate segment values or multiples implied by the current enterprise value.

    This is an allocation model, not a reported fact.
    """
    segments, timeframe_context = apply_sotp_timeframe(
        segment_data,
        historicals=historicals,
        dcf_output=dcf_output,
        assumptions=base_segment_assumptions,
        timeframe=timeframe,
        normalized_basis=normalized_basis,
        normalized_timeframe=normalized_timeframe,
    )
    if segments.empty:
        return {
            "available": False,
            "timeframe": timeframe_context.get("timeframe"),
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
                "SOTP Timeframe": timeframe_context.get("timeframe"),
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
        "timeframe": timeframe_context.get("timeframe"),
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
