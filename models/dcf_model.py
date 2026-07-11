from __future__ import annotations

import numpy as np
import pandas as pd

from config import DCF_DEFAULTS
from analysis.clause_model_mapper import build_assumption_update_from_clause
from models.assumption_estimates import (
    AssumptionEstimate,
    estimate_da_pct_revenue,
    estimate_growth_capex,
    estimate_maintenance_capex,
    estimate_nopat_margin,
    estimate_ocf_margin,
    estimate_sbc_pct_revenue,
    estimate_working_capital_pct_revenue,
)


def _latest(historicals: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if historicals is None or historicals.empty or column not in historicals:
        return default
    value = historicals[column].dropna()
    return float(value.iloc[-1]) if not value.empty else default


def _latest_positive(historicals: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if historicals is None or historicals.empty or column not in historicals:
        return default
    values = pd.to_numeric(historicals[column], errors="coerce").dropna()
    values = values[values > 0]
    return float(values.iloc[-1]) if not values.empty else default


def _latest_optional(historicals: pd.DataFrame, column: str) -> float | None:
    if historicals is None or historicals.empty or column not in historicals:
        return None
    values = pd.to_numeric(historicals[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else None


def _ratio_history(historicals: pd.DataFrame, numerator: str, denominator: str = "Revenue") -> list[float]:
    if historicals is None or historicals.empty or numerator not in historicals or denominator not in historicals:
        return []
    nums = pd.to_numeric(historicals[numerator], errors="coerce")
    dens = pd.to_numeric(historicals[denominator], errors="coerce")
    ratios = []
    for num, den in zip(nums, dens):
        if pd.notna(num) and pd.notna(den) and abs(float(den)) > 1e-12:
            ratios.append(float(num) / float(den))
    return ratios


def _estimate_pct(estimate: AssumptionEstimate, revenue: float | None) -> float | None:
    if estimate.value is None or not revenue:
        return None
    return float(estimate.value) / float(revenue)


def _bounded(value: float | None, low: float, high: float, default: float) -> float:
    if value is None or pd.isna(value):
        return default
    return max(low, min(high, float(value)))


def _recent_revenue_cagr(historicals: pd.DataFrame, periods: int = 3) -> float | None:
    if historicals is None or historicals.empty or "Revenue" not in historicals:
        return None
    values = pd.to_numeric(historicals["Revenue"], errors="coerce").dropna()
    values = values[values > 0]
    if len(values) < 2:
        return None
    window = values.tail(periods + 1)
    start = float(window.iloc[0])
    end = float(window.iloc[-1])
    years = len(window) - 1
    if start <= 0 or years <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def _recent_average(values: list[float], default: float | None = None, periods: int = 3) -> float | None:
    clean = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not clean:
        return default
    return float(np.mean(clean[-periods:]))


def _yearly_assumption(assumptions: dict, year: int, key: str, default=None):
    yearly = assumptions.get("forecast_assumptions_by_year") or {}
    year_values = yearly.get(str(year)) or yearly.get(year) or {}
    value = year_values.get(key, assumptions.get(key, default))
    return default if value is None else value


def run_dcf(historicals: pd.DataFrame, market_data: dict, assumptions: dict) -> dict:
    """
    Forecast 5+ years and compute EV, equity value, and fair value per share.
    """
    years = int(assumptions.get("forecast_years", DCF_DEFAULTS["forecast_years"]))
    revenue = _latest_positive(historicals, "Revenue")
    if revenue <= 0:
        revenue = float(market_data.get("market_cap") or 0) * 0.25
    revenue_cagr = float(assumptions.get("revenue_cagr", 0.08))
    tax_rate = float(assumptions.get("tax_rate", DCF_DEFAULTS["tax_rate"]))
    gross_margin = float(assumptions.get("gross_margin", 0.45))
    opex_pct = float(assumptions.get("opex_pct_revenue", max(gross_margin - float(assumptions.get("operating_margin", 0.15)), 0.0)))
    operating_margin = float(assumptions.get("operating_margin", gross_margin - opex_pct))
    nopat_margin = float(assumptions.get("nopat_margin", operating_margin * (1 - tax_rate)))
    ocf_margin = float(assumptions.get("ocf_margin", 0.18))
    maintenance_capex_pct = float(assumptions.get("maintenance_capex_pct_revenue", 0.03))
    growth_capex_pct = float(assumptions.get("growth_capex_pct_revenue", 0.02))
    working_capital_pct = float(assumptions.get("working_capital_pct_revenue", 0.01))
    da_pct = float(assumptions.get("depreciation_amortization_pct_revenue", maintenance_capex_pct))
    if assumptions.get("use_da_as_maintenance_capex_proxy"):
        maintenance_capex_pct = da_pct
    capex_fade_year = int(assumptions.get("capex_fade_year", 3) or 3)
    sbc_pct = float(assumptions.get("sbc_pct_revenue", 0.0))
    share_growth = float(assumptions.get("diluted_share_growth", 0.0))
    dcf_mode = str(assumptions.get("dcf_mode", "FCFF")).upper()
    wacc = max(float(assumptions.get("wacc", DCF_DEFAULTS["wacc"])), 0.001)
    terminal_growth = float(assumptions.get("terminal_growth", DCF_DEFAULTS["terminal_growth"]))
    terminal_multiple = float(assumptions.get("terminal_multiple", DCF_DEFAULTS["terminal_multiple"]))
    shares = float(assumptions.get("diluted_shares") or _latest(historicals, "Diluted Shares") or market_data.get("shares_outstanding") or 0)
    net_debt = float(assumptions.get("net_debt") if assumptions.get("net_debt") is not None else _latest(historicals, "Net Debt"))
    mos = float(assumptions.get("margin_of_safety", DCF_DEFAULTS["margin_of_safety"]))
    current_price = float(market_data.get("price") or 0)

    rows = []
    discounted_fcfs = []
    current_revenue = revenue
    for year in range(1, years + 1):
        year_revenue_cagr = float(_yearly_assumption(assumptions, year, "revenue_cagr", revenue_cagr))
        year_tax_rate = float(_yearly_assumption(assumptions, year, "tax_rate", tax_rate))
        year_gross_margin = float(_yearly_assumption(assumptions, year, "gross_margin", gross_margin))
        year_opex_pct = float(_yearly_assumption(assumptions, year, "opex_pct_revenue", opex_pct))
        year_operating_margin = float(_yearly_assumption(assumptions, year, "operating_margin", year_gross_margin - year_opex_pct))
        year_nopat_margin = float(_yearly_assumption(assumptions, year, "nopat_margin", year_operating_margin * (1 - year_tax_rate)))
        year_ocf_margin = float(_yearly_assumption(assumptions, year, "ocf_margin", ocf_margin))
        year_da_pct = float(_yearly_assumption(assumptions, year, "depreciation_amortization_pct_revenue", da_pct))
        year_maintenance_capex_pct = float(_yearly_assumption(assumptions, year, "maintenance_capex_pct_revenue", maintenance_capex_pct))
        if assumptions.get("use_da_as_maintenance_capex_proxy"):
            year_maintenance_capex_pct = year_da_pct
        year_growth_capex_pct = float(_yearly_assumption(assumptions, year, "growth_capex_pct_revenue", growth_capex_pct))
        year_working_capital_pct = float(_yearly_assumption(assumptions, year, "working_capital_pct_revenue", working_capital_pct))
        year_sbc_pct = float(_yearly_assumption(assumptions, year, "sbc_pct_revenue", sbc_pct))
        year_share_growth = float(_yearly_assumption(assumptions, year, "diluted_share_growth", share_growth))

        current_revenue *= 1 + year_revenue_cagr
        gross_profit = current_revenue * year_gross_margin
        opex = current_revenue * year_opex_pct
        ebit = current_revenue * year_operating_margin
        nopat = current_revenue * year_nopat_margin
        da = current_revenue * year_da_pct
        ocf = current_revenue * year_ocf_margin
        effective_growth_capex_pct = year_growth_capex_pct
        if year > capex_fade_year and years > capex_fade_year:
            fade_progress = (year - capex_fade_year) / max(years - capex_fade_year, 1)
            effective_growth_capex_pct = year_growth_capex_pct * (1 - 0.5 * fade_progress)
        maintenance_capex = current_revenue * year_maintenance_capex_pct
        growth_capex = current_revenue * effective_growth_capex_pct
        capex = maintenance_capex + growth_capex
        working_capital = current_revenue * year_working_capital_pct
        fcff = nopat + da - maintenance_capex - working_capital
        normalized_cash_earnings = ocf - maintenance_capex
        if dcf_mode == "FCFF":
            fcf = fcff
        elif dcf_mode == "NOPAT":
            fcf = nopat
        else:
            fcf = ocf - capex
        pv = fcf / ((1 + wacc) ** year)
        discounted_fcfs.append(pv)
        rows.append(
            {
                "Year": year,
                "Revenue": current_revenue,
                "Revenue Growth": year_revenue_cagr,
                "Gross Margin": year_gross_margin,
                "COGS % Revenue": 1 - year_gross_margin,
                "Gross Profit": gross_profit,
                "OPEX % Revenue": year_opex_pct,
                "OPEX": opex,
                "EBIT": ebit,
                "EBIT Margin": year_operating_margin,
                "Tax Rate": year_tax_rate,
                "D&A": da,
                "D&A % Revenue": year_da_pct,
                "NOPAT": nopat,
                "NOPAT Margin": year_nopat_margin,
                "OCF": ocf,
                "OCF Margin": year_ocf_margin,
                "Maintenance CAPEX": maintenance_capex,
                "Maintenance CAPEX % Revenue": year_maintenance_capex_pct,
                "Growth CAPEX": growth_capex,
                "Growth CAPEX % Revenue": effective_growth_capex_pct,
                "CAPEX": capex,
                "Total CAPEX": capex,
                "Total CAPEX % Revenue": (year_maintenance_capex_pct + effective_growth_capex_pct),
                "Working Capital Investment": working_capital,
                "Working Capital % Revenue": year_working_capital_pct,
                "Normalized Cash Earnings": normalized_cash_earnings,
                "SBC": current_revenue * year_sbc_pct,
                "SBC % Revenue": year_sbc_pct,
                "FCF": fcf,
                "FCFF": fcff,
                "Diluted Shares": shares * ((1 + year_share_growth) ** year) if shares else None,
                "Diluted Share Growth": year_share_growth,
                "PV FCF": pv,
            }
        )

    final_fcf = rows[-1]["FCF"] if rows else 0
    terminal_gordon = final_fcf * (1 + terminal_growth) / max(wacc - terminal_growth, 0.001)
    terminal_exit = final_fcf * terminal_multiple
    terminal_value = (terminal_gordon + terminal_exit) / 2
    pv_terminal = terminal_value / ((1 + wacc) ** years)
    enterprise_value = sum(discounted_fcfs) + pv_terminal
    equity_value = enterprise_value - net_debt
    fair_value = equity_value / shares if shares else None
    buy_price = fair_value * (1 - mos) if fair_value is not None else None
    upside = (fair_value / current_price - 1) if fair_value and current_price else None
    tv_weight = pv_terminal / enterprise_value if enterprise_value else None

    warnings = []
    if tv_weight and tv_weight > 0.75:
        warnings.append("Valuation depends heavily on terminal value.")
    if shares <= 0:
        warnings.append("Diluted shares unavailable; per-share value cannot be calculated.")

    return {
        "forecast_table": pd.DataFrame(rows),
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "fair_value_per_share": fair_value,
        "buy_price_after_margin_of_safety": buy_price,
        "upside_downside_pct": upside,
        "terminal_value": terminal_value,
        "discounted_terminal_value": pv_terminal,
        "terminal_value_weight_pct": tv_weight,
        "dcf_mode": dcf_mode,
        "capex_method": "D&A proxy + explicit growth CAPEX" if assumptions.get("use_da_as_maintenance_capex_proxy") else "Explicit maintenance + growth CAPEX",
        "warnings": warnings,
    }


def build_dcf_sensitivity_table(base_assumptions: dict, wacc_range: list[float], terminal_growth_range: list[float]) -> pd.DataFrame:
    """
    Build WACC vs terminal-growth fair-value sensitivity table.
    """
    historicals = base_assumptions.get("historicals", pd.DataFrame())
    market_data = base_assumptions.get("market_data", {})
    rows = []
    for wacc in wacc_range:
        row = {"WACC": wacc}
        for growth in terminal_growth_range:
            assumptions = dict(base_assumptions)
            assumptions.update({"wacc": wacc, "terminal_growth": growth})
            row[f"{growth:.1%}"] = run_dcf(historicals, market_data, assumptions)["fair_value_per_share"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_dcf_output_table(dcf_output: dict, assumptions: dict, market_data: dict) -> pd.DataFrame:
    forecast = dcf_output.get("forecast_table", pd.DataFrame())
    rows = [
        {"Metric": "Forecast FCF / FCFF / NOPAT"},
        {"Metric": "Discount factor"},
        {"Metric": "Discounted cash flow"},
        {"Metric": "Terminal value"},
        {"Metric": "Discounted terminal value"},
    ]
    for _, row in forecast.iterrows() if forecast is not None and not forecast.empty else []:
        year = int(row.get("Year") or len(rows) + 1)
        column = f"Year {year}"
        rows[0][column] = row.get("FCF")
        rows[1][column] = 1 / ((1 + float(assumptions.get("wacc", DCF_DEFAULTS["wacc"]))) ** year)
        rows[2][column] = row.get("PV FCF")
    rows[3]["Terminal"] = dcf_output.get("terminal_value")
    rows[4]["Terminal"] = dcf_output.get("discounted_terminal_value")
    return pd.DataFrame(rows)


def build_reverse_dcf_table(reverse_output: dict, base_assumptions: dict, market_data: dict) -> pd.DataFrame:
    solves = reverse_output.get("solves") or {}
    labels = {
        "revenue_cagr": ("Solve for Revenue CAGR", "Holding Base Case margins, CAPEX, WACC, and terminal assumptions constant."),
        "nopat_margin": ("Solve for NOPAT Margin", "Holding Base Case growth, CAPEX, WACC, and terminal assumptions constant."),
        "ocf_margin": ("Solve for OCF Margin", "Holding Base Case growth, CAPEX, WACC, and terminal assumptions constant."),
        "terminal_growth": ("Solve for Terminal Growth", "Holding Base Case growth, margins, CAPEX, and WACC constant."),
        "terminal_multiple": ("Solve for Terminal Multiple", "Holding Base Case growth, margins, CAPEX, and WACC constant."),
    }
    rows = [
        {"Metric": "Current share price", "Value": market_data.get("price"), "Bounds": "", "Status": "", "Interpretation": "Market input"},
        {"Metric": "Current market cap", "Value": market_data.get("market_cap"), "Bounds": "", "Status": "", "Interpretation": "Market input"},
        {"Metric": "Current enterprise value", "Value": market_data.get("enterprise_value"), "Bounds": "", "Status": "", "Interpretation": "Market input"},
    ]
    for key, (label, method) in labels.items():
        item = solves.get(key) or {}
        low, high = item.get("bounds") or (None, None)
        is_multiple = key == "terminal_multiple"
        bounds = f"{low:.1f}x to {high:.1f}x" if is_multiple and low is not None else f"{low:.1%} to {high:.1%}" if low is not None else ""
        value = item.get("display_value") if item.get("implied") is None else item.get("implied")
        rows.append(
            {
                "Metric": label,
                "Value": value,
                "Bounds": bounds,
                "Status": item.get("status", "Unknown"),
                "Interpretation": method if item.get("status") == "Realistic" else item.get("conclusion"),
            }
        )
    rows.append({"Metric": "Market expectation conclusion", "Value": reverse_output.get("interpretation"), "Bounds": "", "Status": reverse_output.get("market_case"), "Interpretation": "Bounded reverse DCF conclusion"})
    return pd.DataFrame(rows)


def build_scenario_table(historicals: pd.DataFrame, market_data: dict, base_assumptions: dict) -> pd.DataFrame:
    scenarios = {
        "Bear": {"revenue_cagr": -0.03, "gross_margin": -0.03, "wacc": 0.015, "terminal_growth": -0.01, "terminal_multiple": -2.0},
        "Base": {},
        "Bull": {"revenue_cagr": 0.05, "gross_margin": 0.03, "wacc": -0.01, "terminal_growth": 0.01, "terminal_multiple": 2.0},
        "User": {},
        "Market-Implied": {},
    }
    values = {}
    for scenario, deltas in scenarios.items():
        assumptions = dict(base_assumptions)
        for key, delta in deltas.items():
            assumptions[key] = float(assumptions.get(key, 0) or 0) + delta
        assumptions["wacc"] = max(float(assumptions.get("wacc", 0.095)), 0.04)
        dcf = run_dcf(historicals, market_data, assumptions)
        final_revenue = dcf.get("forecast_table", pd.DataFrame()).iloc[-1].get("Revenue") if not dcf.get("forecast_table", pd.DataFrame()).empty else None
        final_fcf = dcf.get("forecast_table", pd.DataFrame()).iloc[-1].get("FCF") if not dcf.get("forecast_table", pd.DataFrame()).empty else None
        values[scenario] = {
            "Revenue CAGR": assumptions.get("revenue_cagr"),
            "Gross margin": assumptions.get("gross_margin"),
            "EBIT margin": assumptions.get("operating_margin"),
            "NOPAT margin": assumptions.get("nopat_margin"),
            "OCF margin": assumptions.get("ocf_margin"),
            "Maintenance CAPEX % revenue": assumptions.get("maintenance_capex_pct_revenue"),
            "Growth CAPEX % revenue": assumptions.get("growth_capex_pct_revenue"),
            "Total CAPEX % revenue": (float(assumptions.get("maintenance_capex_pct_revenue") or 0) + float(assumptions.get("growth_capex_pct_revenue") or 0)),
            "CAPEX Normalization Year": assumptions.get("capex_fade_year"),
            "Working Capital % revenue": assumptions.get("working_capital_pct_revenue"),
            "FCF margin": (final_fcf / final_revenue) if final_revenue else None,
            "WACC": assumptions.get("wacc"),
            "Terminal growth": assumptions.get("terminal_growth"),
            "Terminal multiple": assumptions.get("terminal_multiple"),
            "Fair value per share": dcf.get("fair_value_per_share"),
            "Upside / downside": dcf.get("upside_downside_pct"),
            "Margin-of-safety buy price": dcf.get("buy_price_after_margin_of_safety"),
        }
    rows = []
    for line_item in next(iter(values.values())).keys():
        row = {"Line Item": line_item}
        for scenario in ["Bear", "Base", "Bull", "User", "Market-Implied"]:
            row[scenario] = values[scenario][line_item]
        rows.append(row)
    return pd.DataFrame(rows)


def default_assumptions_from_historicals(historicals: pd.DataFrame, market_data: dict) -> dict:
    revenue = _latest_positive(historicals, "Revenue")
    gross_margin = _latest_optional(historicals, "Gross Margin")
    ebit = _latest_optional(historicals, "EBIT")
    gross_profit = _latest_optional(historicals, "Gross Profit")
    opex = _latest_optional(historicals, "OPEX")
    ocf = _latest_optional(historicals, "OCF")
    capex = _latest_optional(historicals, "Total CAPEX")
    ebitda = _latest_optional(historicals, "EBITDA")
    da = max(ebitda - ebit, 0) if ebitda is not None and ebit is not None else None
    business_profile = {
        "profile": market_data.get("business_profile") or market_data.get("stock_profile") or "General",
        "sector": market_data.get("sector"),
        "industry": market_data.get("industry"),
    }

    da_estimate = estimate_da_pct_revenue(
        revenue,
        depreciation_amortization=da,
        historical_da_pct=_ratio_history(historicals, "D&A"),
        business_profile=business_profile,
    )
    maintenance_estimate = estimate_maintenance_capex(
        revenue,
        capex,
        da,
        business_profile,
        historical_capex_pct_revenue=_ratio_history(historicals, "Total CAPEX"),
    )
    growth_estimate = estimate_growth_capex(
        revenue,
        capex,
        maintenance_estimate,
        historical_total_capex_pct_revenue=_ratio_history(historicals, "Total CAPEX"),
        business_profile=business_profile,
    )
    sbc = _latest_optional(historicals, "SBC")
    sbc_estimate = estimate_sbc_pct_revenue(
        revenue,
        sbc_raw=sbc,
        historical_sbc_pct=_ratio_history(historicals, "SBC"),
        business_profile=business_profile,
    )
    wc_estimate = estimate_working_capital_pct_revenue(
        revenue,
        historical_wc_pct=_ratio_history(historicals, "Working Capital Investment") or _ratio_history(historicals, "Working Capital"),
        business_profile=business_profile,
    )
    nopat_estimate = estimate_nopat_margin(
        revenue,
        ebit=ebit,
        gross_profit=gross_profit,
        opex=opex,
        tax_rate=DCF_DEFAULTS["tax_rate"],
        historical_nopat_margin=_ratio_history(historicals, "NOPAT"),
    )
    ocf_estimate = estimate_ocf_margin(
        revenue,
        operating_cash_flow=ocf,
        nopat=(nopat_estimate.value * revenue) if revenue and nopat_estimate.value is not None else None,
        depreciation_amortization=da,
        change_in_working_capital=(wc_estimate.value * revenue) if revenue and wc_estimate.value is not None else None,
        historical_ocf_margin=_ratio_history(historicals, "OCF"),
    )

    total_capex_pct = capex / revenue if revenue and capex is not None else None
    da_pct = da_estimate.value
    maintenance_capex_pct = _estimate_pct(maintenance_estimate, revenue)
    growth_capex_pct = _estimate_pct(growth_estimate, revenue)
    if maintenance_capex_pct is None:
        maintenance_capex_pct = 0.03
    if growth_capex_pct is None:
        growth_capex_pct = 0.02
    use_da_proxy = maintenance_estimate.evidence_grade == "Proxy-based"
    latest_shares = _latest(historicals, "Diluted Shares") or market_data.get("shares_outstanding")
    prior_shares = float(historicals["Diluted Shares"].dropna().iloc[-2]) if historicals is not None and len(historicals.get("Diluted Shares", pd.Series(dtype=float)).dropna()) >= 2 else latest_shares
    diluted_share_growth = (latest_shares / prior_shares - 1) if latest_shares and prior_shares else 0.0
    revenue_growth = _recent_revenue_cagr(historicals)
    if revenue_growth is None and historicals is not None and "Revenue" in historicals:
        revenue_changes = pd.to_numeric(historicals["Revenue"], errors="coerce").pct_change().dropna().tolist()
        revenue_growth = _recent_average(revenue_changes)
    revenue_growth = _bounded(revenue_growth, -0.10, 0.25, 0.08)
    opex_pct = _recent_average(_ratio_history(historicals, "OPEX"), None)
    if opex_pct is None and revenue and gross_margin is not None and ebit is not None:
        opex_pct = max(float(gross_margin) - (float(ebit) / float(revenue)), 0.0)
    opex_pct = _bounded(opex_pct, 0.0, 0.80, max((gross_margin if gross_margin is not None else 0.45) - 0.15, 0.0))
    normalized_gross_margin = _recent_average(_ratio_history(historicals, "Gross Profit"), gross_margin)
    normalized_gross_margin = _bounded(normalized_gross_margin, 0.0, 0.95, 0.45)
    assumptions = {
        "forecast_years": 5,
        "dcf_mode": "FCFF",
        "revenue_cagr": revenue_growth,
        "gross_margin": normalized_gross_margin,
        "operating_margin": normalized_gross_margin - opex_pct,
        "tax_rate": DCF_DEFAULTS["tax_rate"],
        "nopat_margin": nopat_estimate.value if nopat_estimate.value is not None else 0.12,
        "ocf_margin": ocf_estimate.value if ocf_estimate.value is not None else 0.16,
        "maintenance_capex_pct_revenue": maintenance_capex_pct,
        "growth_capex_pct_revenue": growth_capex_pct,
        "total_capex_pct_revenue": maintenance_capex_pct + growth_capex_pct,
        "depreciation_amortization_pct_revenue": da_pct,
        "use_da_as_maintenance_capex_proxy": use_da_proxy,
        "capex_fade_year": 3,
        "opex_pct_revenue": opex_pct,
        "working_capital_pct_revenue": wc_estimate.value if wc_estimate.value is not None else 0.01,
        "sbc_pct_revenue": sbc_estimate.value if sbc_estimate.value is not None else 0.02,
        "diluted_share_growth": diluted_share_growth,
        "sm_pct_revenue": 0.0,
        "rd_pct_revenue": 0.0,
        "ga_pct_revenue": 0.0,
        "wacc": DCF_DEFAULTS["wacc"],
        "terminal_growth": DCF_DEFAULTS["terminal_growth"],
        "terminal_multiple": DCF_DEFAULTS["terminal_multiple"],
        "diluted_shares": latest_shares,
        "net_debt": _latest(historicals, "Net Debt"),
        "margin_of_safety": DCF_DEFAULTS["margin_of_safety"],
    }
    assumptions["_sbc_real_zero"] = sbc_estimate.is_real_zero
    assumptions["_working_capital_real_zero"] = wc_estimate.is_real_zero
    assumptions["_da_real_zero"] = da_estimate.is_real_zero
    assumptions["assumption_estimates"] = {
        "revenue_cagr": {
            "value": revenue_growth,
            "method": "Recent revenue CAGR / YoY trend, bounded for a default base case.",
            "evidence_grade": "Calculated",
            "confidence": "Medium" if revenue_growth == 0.08 else "High",
            "warning": None,
            "source": "Historical revenue trend",
            "is_real_zero": revenue_growth == 0.0,
        },
        "gross_margin": {
            "value": normalized_gross_margin,
            "method": "Recent gross profit / revenue average.",
            "evidence_grade": "Calculated",
            "confidence": "High" if gross_margin is not None else "Medium",
            "warning": None,
            "source": "Income statement",
            "is_real_zero": normalized_gross_margin == 0.0,
        },
        "opex_pct_revenue": {
            "value": opex_pct,
            "method": "Recent OPEX / revenue average or gross margin less operating margin.",
            "evidence_grade": "Calculated",
            "confidence": "High" if opex is not None else "Medium",
            "warning": None,
            "source": "Income statement",
            "is_real_zero": opex_pct == 0.0,
        },
        "maintenance_capex_pct_revenue": {**maintenance_estimate.to_dict(), "value": maintenance_capex_pct},
        "growth_capex_pct_revenue": {**growth_estimate.to_dict(), "value": growth_capex_pct},
        "sbc_pct_revenue": sbc_estimate.to_dict(),
        "working_capital_pct_revenue": wc_estimate.to_dict(),
        "ocf_margin": ocf_estimate.to_dict(),
        "nopat_margin": nopat_estimate.to_dict(),
        "depreciation_amortization_pct_revenue": da_estimate.to_dict(),
    }
    if total_capex_pct is not None:
        assumptions["assumption_estimates"]["total_capex_pct_revenue"] = {
            "value": total_capex_pct,
            "method": "Total CAPEX divided by revenue.",
            "evidence_grade": "Calculated",
            "confidence": "High",
            "warning": None,
            "source": "Cash flow statement + revenue",
            "is_real_zero": total_capex_pct == 0.0,
        }
    return assumptions


def create_pending_assumption_update(clause_row: dict, old_value=None, new_value=None, scenario: str = "Base") -> dict:
    """
    Create a pending assumption update from a clause. Does not mutate DCF assumptions.
    """
    return build_assumption_update_from_clause(clause_row, old_value=old_value, new_value=new_value, scenario=scenario)
