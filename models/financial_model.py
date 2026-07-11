from __future__ import annotations

import re

import pandas as pd

from models.financial_derivations import derive_financial_rows

ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
HISTORICAL_COLUMNS = [
    "Period",
    "Revenue",
    "Gross Profit",
    "Gross Margin",
    "OPEX",
    "EBITDA",
    "EBIT",
    "NOPAT",
    "Net Income",
    "OCF",
    "Adjusted OCF",
    "Maintenance CAPEX",
    "Growth CAPEX",
    "Total CAPEX",
    "FCF",
    "Adjusted FCF",
    "SBC",
    "Diluted Shares",
    "Net Debt",
]

MODEL_ROW_ORDER = [
    "Revenue",
    "Revenue growth %",
    "COGS / Cost of sales",
    "COGS % revenue",
    "Gross profit",
    "Gross margin %",
    "S&M",
    "S&M % revenue",
    "R&D",
    "R&D % revenue",
    "G&A",
    "G&A % revenue",
    "Total OPEX",
    "OPEX % revenue",
    "EBIT",
    "EBIT margin %",
    "D&A",
    "D&A % revenue",
    "EBITDA",
    "EBITDA margin %",
    "Tax rate",
    "NOPAT",
    "NOPAT margin %",
    "Operating cash flow",
    "OCF margin %",
    "Adjusted OCF",
    "Adjusted OCF margin %",
    "Maintenance CAPEX",
    "Maintenance CAPEX % revenue",
    "Growth CAPEX",
    "Growth CAPEX % revenue",
    "Total CAPEX",
    "Total CAPEX % revenue",
    "FCF",
    "FCF margin %",
    "Adjusted FCF",
    "Adjusted FCF margin %",
    "SBC",
    "SBC % revenue",
    "SBC % gross profit",
    "SBC % OCF",
    "Diluted shares",
    "Diluted shares growth %",
]


def _value(financials: dict, key: str):
    item = financials.get("sec", {}).get(key, {})
    value = item.get("value") if isinstance(item, dict) else None
    return abs(value) if key == "capex" and value is not None else value


def _yf_value(yf_financials: dict, statement: str, candidates: list[str]):
    df = yf_financials.get(statement, pd.DataFrame())
    if df is None or df.empty:
        return None
    for row_name in candidates:
        if row_name in df.index:
            series = df.loc[row_name].dropna()
            if not series.empty:
                return float(series.iloc[0])
    return None


def _sec_annual_series(metrics: dict[str, pd.DataFrame], key: str, absolute: bool = False, flow: bool = True) -> pd.Series:
    df = metrics.get(key, pd.DataFrame())
    if df is None or df.empty or "val" not in df:
        return pd.Series(dtype="float64")
    annual = df[df["val"].notna()].copy()
    if "form" in annual:
        annual = annual[annual["form"].astype(str).str.upper().isin(ANNUAL_FORMS)]
    if "fp" in annual:
        annual = annual[annual["fp"].astype(str).str.upper().eq("FY")]
    if "fy" not in annual or annual.empty:
        return pd.Series(dtype="float64")
    if flow and {"start", "end"}.issubset(annual.columns):
        annual["_start_dt"] = pd.to_datetime(annual["start"], errors="coerce")
        annual["_end_dt"] = pd.to_datetime(annual["end"], errors="coerce")
        annual["_duration_days"] = (annual["_end_dt"] - annual["_start_dt"]).dt.days
        duration_rows = annual[annual["_duration_days"].notna()]
        if not duration_rows.empty and (duration_rows["_duration_days"] >= 300).any():
            annual = annual[annual["_duration_days"] >= 300]
    annual["_year"] = pd.to_numeric(annual["fy"], errors="coerce")
    annual = annual[annual["_year"].notna()]
    if annual.empty:
        return pd.Series(dtype="float64")
    annual["_year"] = annual["_year"].astype(int)
    annual["_filed_sort"] = annual.get("filed", pd.Series("", index=annual.index)).fillna("").astype(str)
    annual["_end_sort"] = annual.get("end", pd.Series("", index=annual.index)).fillna("").astype(str)
    tag_priority = annual["tag_priority"] if "tag_priority" in annual else pd.Series(0, index=annual.index)
    annual["_tag_priority"] = pd.to_numeric(tag_priority, errors="coerce").fillna(0)
    annual = annual.sort_values(
        ["_year", "_end_sort", "_filed_sort", "_tag_priority"],
        ascending=[True, True, True, False],
    )
    values = annual.drop_duplicates("_year", keep="last").set_index("_year")["val"].astype(float)
    return values.abs() if absolute else values


def _yf_annual_series(yf_financials: dict, statement: str, candidates: list[str], absolute: bool = False) -> pd.Series:
    df = yf_financials.get(statement, pd.DataFrame())
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    for row_name in candidates:
        if row_name not in df.index:
            continue
        series = df.loc[row_name].dropna()
        if series.empty:
            continue
        indexed = {}
        for column, value in series.items():
            try:
                year = pd.Timestamp(column).year
            except Exception:
                continue
            indexed[year] = float(value)
        out = pd.Series(indexed, dtype="float64").sort_index()
        return out.abs() if absolute else out
    return pd.Series(dtype="float64")


def _coalesce_series(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    if primary is None or primary.empty:
        return fallback if fallback is not None else pd.Series(dtype="float64")
    if fallback is None or fallback.empty:
        return primary
    return primary.combine_first(fallback)


def _latest_from_series(series: pd.Series):
    if series is None or series.empty:
        return None
    return series.dropna().iloc[-1] if not series.dropna().empty else None


def _float_or_none(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _row(
    period: str,
    revenue,
    gross_profit,
    operating_income,
    net_income,
    ocf,
    capex,
    da,
    sbc,
    shares,
    cash,
    debt,
) -> dict:
    revenue = _float_or_none(revenue)
    gross_profit = _float_or_none(gross_profit)
    operating_income = _float_or_none(operating_income)
    net_income = _float_or_none(net_income)
    ocf = _float_or_none(ocf)
    capex = abs(_float_or_none(capex)) if _float_or_none(capex) is not None else None
    da = abs(_float_or_none(da)) if _float_or_none(da) is not None else None
    sbc = _float_or_none(sbc)
    shares = _float_or_none(shares)
    cash = _float_or_none(cash)
    debt = _float_or_none(debt)

    maintenance_capex = min(capex, da) if capex is not None and da is not None else (capex * 0.6 if capex is not None else None)
    growth_capex = max(capex - maintenance_capex, 0) if capex is not None and maintenance_capex is not None else None
    tax_rate = 0.21
    nopat = operating_income * (1 - tax_rate) if operating_income is not None else None
    fcf = ocf - capex if ocf is not None and capex is not None else None
    adjusted_ocf = ocf
    adjusted_fcf = adjusted_ocf - capex if adjusted_ocf is not None and capex is not None else None
    opex = max(gross_profit - operating_income, 0) if gross_profit is not None and operating_income is not None else None

    return {
        "Period": period,
        "Revenue": revenue,
        "Gross Profit": gross_profit,
        "Gross Margin": gross_profit / revenue if revenue else None,
        "OPEX": opex,
        "EBITDA": operating_income + da if operating_income is not None and da is not None else None,
        "EBIT": operating_income,
        "NOPAT": nopat,
        "Net Income": net_income,
        "OCF": ocf,
        "Adjusted OCF": adjusted_ocf,
        "Maintenance CAPEX": maintenance_capex,
        "Growth CAPEX": growth_capex,
        "Total CAPEX": capex,
        "FCF": fcf,
        "Adjusted FCF": adjusted_fcf,
        "SBC": sbc,
        "Diluted Shares": shares,
        "Net Debt": debt - cash if debt is not None and cash is not None else None,
    }


def _fallback_latest_row(financials: dict, yf_financials: dict, market: dict) -> dict:
    revenue = _value(financials, "revenue") or _yf_value(yf_financials, "income_stmt", ["Total Revenue"])
    gross_profit = _value(financials, "gross_profit") or _yf_value(yf_financials, "income_stmt", ["Gross Profit"])
    operating_income = _value(financials, "operating_income") or _yf_value(
        yf_financials, "income_stmt", ["Operating Income", "Operating Income or Loss"]
    )
    net_income = _value(financials, "net_income") or _yf_value(yf_financials, "income_stmt", ["Net Income"])
    ocf = _value(financials, "operating_cash_flow") or _yf_value(
        yf_financials, "cashflow", ["Operating Cash Flow", "Total Cash From Operating Activities"]
    )
    capex = _value(financials, "capex") or _yf_value(yf_financials, "cashflow", ["Capital Expenditure"])
    da = _value(financials, "depreciation_amortization") or _yf_value(
        yf_financials, "cashflow", ["Depreciation And Amortization", "Depreciation"]
    )
    sbc = _value(financials, "sbc") or _yf_value(yf_financials, "cashflow", ["Stock Based Compensation"])
    shares = _value(financials, "shares") or market.get("shares_outstanding")
    cash = _value(financials, "cash") or market.get("cash") or 0
    debt = _value(financials, "debt") or market.get("debt") or 0
    return _row("Latest reported", revenue, gross_profit, operating_income, net_income, ocf, capex, da, sbc, shares, cash, debt)


def build_historical_financial_table(dataset: dict) -> pd.DataFrame:
    """
    Build annual financial rows for the cockpit.

    SEC companyfacts are preferred because they provide reported multi-year
    values without downloading full filings. yfinance annual statements are used
    as fallback for issuers without usable SEC facts.
    """
    financials = dataset.get("financials", {})
    yf_financials = financials.get("yfinance", {})
    market = dataset.get("market_data", {})
    sec_metrics = financials.get("sec_normalized", {}).get("metrics", {})

    revenue = _coalesce_series(_sec_annual_series(sec_metrics, "revenue"), _yf_annual_series(yf_financials, "income_stmt", ["Total Revenue"]))
    gross_profit = _coalesce_series(
        _sec_annual_series(sec_metrics, "gross_profit"), _yf_annual_series(yf_financials, "income_stmt", ["Gross Profit"])
    )
    operating_income = _coalesce_series(
        _sec_annual_series(sec_metrics, "operating_income"),
        _yf_annual_series(yf_financials, "income_stmt", ["Operating Income", "Operating Income or Loss"]),
    )
    net_income = _coalesce_series(_sec_annual_series(sec_metrics, "net_income"), _yf_annual_series(yf_financials, "income_stmt", ["Net Income"]))
    ocf = _coalesce_series(
        _sec_annual_series(sec_metrics, "operating_cash_flow"),
        _yf_annual_series(yf_financials, "cashflow", ["Operating Cash Flow", "Total Cash From Operating Activities"]),
    )
    capex = _coalesce_series(
        _sec_annual_series(sec_metrics, "capex", absolute=True),
        _yf_annual_series(yf_financials, "cashflow", ["Capital Expenditure"], absolute=True),
    )
    da = _coalesce_series(
        _sec_annual_series(sec_metrics, "depreciation_amortization", absolute=True),
        _yf_annual_series(yf_financials, "cashflow", ["Depreciation And Amortization", "Depreciation"], absolute=True),
    )
    sbc = _coalesce_series(
        _sec_annual_series(sec_metrics, "sbc"),
        _yf_annual_series(yf_financials, "cashflow", ["Stock Based Compensation"]),
    )
    shares = _sec_annual_series(sec_metrics, "shares_outstanding")
    cash = _coalesce_series(
        _sec_annual_series(sec_metrics, "cash", flow=False),
        _yf_annual_series(yf_financials, "balance_sheet", ["Cash And Cash Equivalents", "Cash"]),
    )
    debt_current = _sec_annual_series(sec_metrics, "debt_current", flow=False)
    debt_noncurrent = _sec_annual_series(sec_metrics, "debt_noncurrent", flow=False)
    sec_total_debt = _sec_annual_series(sec_metrics, "total_debt", flow=False)
    debt = _coalesce_series(debt_current.add(debt_noncurrent, fill_value=0), sec_total_debt)
    yf_debt = _yf_annual_series(yf_financials, "balance_sheet", ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"])
    debt = _coalesce_series(debt, yf_debt)

    years = sorted(set(revenue.dropna().index.tolist()) | set(ocf.dropna().index.tolist()) | set(operating_income.dropna().index.tolist()))
    if not years:
        return pd.DataFrame([_fallback_latest_row(financials, yf_financials, market)], columns=HISTORICAL_COLUMNS)

    rows = []
    latest_shares = _latest_from_series(shares) or market.get("shares_outstanding")
    latest_cash = _latest_from_series(cash) or market.get("cash") or 0
    latest_debt = _latest_from_series(debt) or market.get("debt") or 0
    for year in years[-8:]:
        rows.append(
            _row(
                f"FY {int(year)}",
                revenue.get(year),
                gross_profit.get(year),
                operating_income.get(year),
                net_income.get(year),
                ocf.get(year),
                capex.get(year),
                da.get(year),
                sbc.get(year),
                shares.get(year, latest_shares),
                cash.get(year, latest_cash),
                debt.get(year, latest_debt),
            )
        )

    return pd.DataFrame(rows, columns=HISTORICAL_COLUMNS)


def _year_from_period(period: str) -> int | None:
    match = re.search(r"(20\d{2}|19\d{2})", str(period or ""))
    return int(match.group(1)) if match else None


def _safe_div(numerator, denominator):
    if numerator is None or denominator is None:
        return None
    try:
        denominator = float(denominator)
        return float(numerator) / denominator if denominator else None
    except Exception:
        return None


def _safe_delta_pct(current, previous):
    current = _float_or_none(current)
    previous = _float_or_none(previous)
    if current is None or previous is None:
        return None
    return _safe_div(current - previous, previous)


def _actual_label(period: str) -> str:
    year = _year_from_period(period)
    return f"FY{year}A" if year else str(period or "Actual")


def _forecast_label(latest_year: int | None, year_index: int) -> str:
    year = (latest_year + year_index) if latest_year else year_index
    return f"FY{year}E" if year_index == 1 else f"FY{year}F"


def _model_rows_to_table(rows_by_period: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for line_item in MODEL_ROW_ORDER:
        row = {"Line Item": line_item}
        for period, values in rows_by_period.items():
            row[period] = values.get(line_item)
        rows.append(row)
    return pd.DataFrame(rows)


def _actual_model_values(row: pd.Series, prior_revenue=None, prior_shares=None) -> dict:
    revenue = row.get("Revenue")
    gross_profit = row.get("Gross Profit")
    ebit = row.get("EBIT")
    ebitda = row.get("EBITDA")
    ebit_num = _float_or_none(ebit)
    ebitda_num = _float_or_none(ebitda)
    da = max(ebitda_num - ebit_num, 0) if ebitda_num is not None and ebit_num is not None else None
    opex = row.get("OPEX")
    nopat = row.get("NOPAT")
    ocf = row.get("OCF")
    adjusted_ocf = row.get("Adjusted OCF")
    maintenance_capex = row.get("Maintenance CAPEX")
    growth_capex = row.get("Growth CAPEX")
    total_capex = row.get("Total CAPEX")
    fcf = row.get("FCF")
    adjusted_fcf = row.get("Adjusted FCF")
    sbc = row.get("SBC")
    shares = row.get("Diluted Shares")
    revenue_num = _float_or_none(revenue)
    gross_profit_num = _float_or_none(gross_profit)
    cogs = -(revenue_num - gross_profit_num) if revenue_num is not None and gross_profit_num is not None else None
    nopat_to_ebit = _safe_div(nopat, ebit)
    return {
        "Revenue": revenue,
        "Revenue growth %": _safe_delta_pct(revenue, prior_revenue),
        "COGS / Cost of sales": cogs,
        "COGS % revenue": _safe_div(abs(cogs), revenue) if cogs is not None else None,
        "Gross profit": gross_profit,
        "Gross margin %": row.get("Gross Margin"),
        "S&M": None,
        "S&M % revenue": None,
        "R&D": None,
        "R&D % revenue": None,
        "G&A": None,
        "G&A % revenue": None,
        "Total OPEX": opex,
        "OPEX % revenue": _safe_div(opex, revenue),
        "EBIT": ebit,
        "EBIT margin %": _safe_div(ebit, revenue),
        "D&A": da,
        "D&A % revenue": _safe_div(da, revenue),
        "EBITDA": ebitda,
        "EBITDA margin %": _safe_div(ebitda, revenue),
        "Tax rate": 1 - nopat_to_ebit if nopat_to_ebit is not None else None,
        "NOPAT": nopat,
        "NOPAT margin %": _safe_div(nopat, revenue),
        "Operating cash flow": ocf,
        "OCF margin %": _safe_div(ocf, revenue),
        "Adjusted OCF": adjusted_ocf,
        "Adjusted OCF margin %": _safe_div(adjusted_ocf, revenue),
        "Maintenance CAPEX": maintenance_capex,
        "Maintenance CAPEX % revenue": _safe_div(maintenance_capex, revenue),
        "Growth CAPEX": growth_capex,
        "Growth CAPEX % revenue": _safe_div(growth_capex, revenue),
        "Total CAPEX": total_capex,
        "Total CAPEX % revenue": _safe_div(total_capex, revenue),
        "FCF": fcf,
        "FCF margin %": _safe_div(fcf, revenue),
        "Adjusted FCF": adjusted_fcf,
        "Adjusted FCF margin %": _safe_div(adjusted_fcf, revenue),
        "SBC": sbc,
        "SBC % revenue": _safe_div(sbc, revenue),
        "SBC % gross profit": _safe_div(sbc, gross_profit),
        "SBC % OCF": _safe_div(sbc, ocf),
        "Diluted shares": shares,
        "Diluted shares growth %": _safe_delta_pct(shares, prior_shares),
    }


def _forecast_model_values(forecast_row: pd.Series, assumptions: dict, prior_revenue=None, prior_shares=None) -> dict:
    revenue = forecast_row.get("Revenue")
    gross_margin = forecast_row.get("Gross Margin", assumptions.get("gross_margin"))
    gross_profit = revenue * gross_margin if revenue is not None and gross_margin is not None else None
    sm = revenue * float(assumptions.get("sm_pct_revenue") or 0) if revenue is not None else None
    rd = revenue * float(assumptions.get("rd_pct_revenue") or 0) if revenue is not None else None
    ga = revenue * float(assumptions.get("ga_pct_revenue") or 0) if revenue is not None else None
    ebit = forecast_row.get("EBIT")
    if ebit is None and forecast_row.get("NOPAT") is not None:
        ebit = forecast_row.get("NOPAT") / max(1 - float(forecast_row.get("Tax Rate", assumptions.get("tax_rate", 0.21))), 0.01)
    opex = forecast_row.get("OPEX")
    if opex is None:
        opex = gross_profit - ebit if gross_profit is not None and ebit is not None else (sm or 0) + (rd or 0) + (ga or 0)
    da = forecast_row.get("D&A")
    ebitda = ebit + da if ebit is not None and da is not None else None
    ocf = forecast_row.get("OCF")
    maintenance_capex = forecast_row.get("Maintenance CAPEX")
    growth_capex = forecast_row.get("Growth CAPEX")
    total_capex = forecast_row.get("CAPEX")
    fcf = forecast_row.get("FCF")
    adjusted_ocf = ocf
    adjusted_fcf = adjusted_ocf - maintenance_capex if adjusted_ocf is not None and maintenance_capex is not None else None
    sbc = forecast_row.get("SBC")
    shares = forecast_row.get("Diluted Shares")
    cogs = -(revenue - gross_profit) if revenue is not None and gross_profit is not None else None
    return {
        "Revenue": revenue,
        "Revenue growth %": _safe_div(float(revenue or 0) - float(prior_revenue or 0), prior_revenue) if prior_revenue else None,
        "COGS / Cost of sales": cogs,
        "COGS % revenue": _safe_div(abs(cogs), revenue) if gross_profit is not None else None,
        "Gross profit": gross_profit,
        "Gross margin %": gross_margin,
        "S&M": sm,
        "S&M % revenue": assumptions.get("sm_pct_revenue"),
        "R&D": rd,
        "R&D % revenue": assumptions.get("rd_pct_revenue"),
        "G&A": ga,
        "G&A % revenue": assumptions.get("ga_pct_revenue"),
        "Total OPEX": opex,
        "OPEX % revenue": forecast_row.get("OPEX % Revenue", _safe_div(opex, revenue)),
        "EBIT": ebit,
        "EBIT margin %": forecast_row.get("EBIT Margin", _safe_div(ebit, revenue)),
        "D&A": da,
        "D&A % revenue": forecast_row.get("D&A % Revenue", _safe_div(da, revenue)),
        "EBITDA": ebitda,
        "EBITDA margin %": _safe_div(ebitda, revenue),
        "Tax rate": forecast_row.get("Tax Rate", assumptions.get("tax_rate")),
        "NOPAT": forecast_row.get("NOPAT"),
        "NOPAT margin %": forecast_row.get("NOPAT Margin", _safe_div(forecast_row.get("NOPAT"), revenue)),
        "Operating cash flow": ocf,
        "OCF margin %": forecast_row.get("OCF Margin", _safe_div(ocf, revenue)),
        "Adjusted OCF": adjusted_ocf,
        "Adjusted OCF margin %": _safe_div(adjusted_ocf, revenue),
        "Maintenance CAPEX": maintenance_capex,
        "Maintenance CAPEX % revenue": forecast_row.get("Maintenance CAPEX % Revenue", _safe_div(maintenance_capex, revenue)),
        "Growth CAPEX": growth_capex,
        "Growth CAPEX % revenue": forecast_row.get("Growth CAPEX % Revenue", _safe_div(growth_capex, revenue)),
        "Total CAPEX": total_capex,
        "Total CAPEX % revenue": forecast_row.get("Total CAPEX % Revenue", _safe_div(total_capex, revenue)),
        "FCF": fcf,
        "FCF margin %": _safe_div(fcf, revenue),
        "Adjusted FCF": adjusted_fcf,
        "Adjusted FCF margin %": _safe_div(adjusted_fcf, revenue),
        "SBC": sbc,
        "SBC % revenue": forecast_row.get("SBC % Revenue", _safe_div(sbc, revenue)),
        "SBC % gross profit": _safe_div(sbc, gross_profit),
        "SBC % OCF": _safe_div(sbc, ocf),
        "Diluted shares": shares,
        "Diluted shares growth %": forecast_row.get("Diluted Share Growth", _safe_div(float(shares or 0) - float(prior_shares or 0), prior_shares) if prior_shares else None),
    }


def build_time_axis_financial_model(historicals: pd.DataFrame, forecast_table: pd.DataFrame, assumptions: dict) -> pd.DataFrame:
    """
    Build Excel-style model table: line items down rows, periods across columns.
    """
    rows_by_period: dict[str, dict] = {}
    actuals = historicals.tail(5).copy() if historicals is not None and not historicals.empty else pd.DataFrame()
    hidden_prior_label = "__prior_period_for_change"
    prior_source = historicals.iloc[-6] if historicals is not None and len(historicals) > len(actuals) else None
    if prior_source is not None:
        rows_by_period[hidden_prior_label] = _actual_model_values(prior_source, prior_revenue=None, prior_shares=None)
    prior_revenue = prior_source.get("Revenue") if prior_source is not None else None
    prior_shares = prior_source.get("Diluted Shares") if prior_source is not None else None
    latest_year = None
    for _, row in actuals.iterrows():
        label = _actual_label(row.get("Period"))
        latest_year = _year_from_period(row.get("Period")) or latest_year
        rows_by_period[label] = _actual_model_values(row, prior_revenue=prior_revenue, prior_shares=prior_shares)
        prior_revenue = row.get("Revenue")
        prior_shares = row.get("Diluted Shares")
    if not actuals.empty:
        latest_row = actuals.iloc[-1]
        rows_by_period["LTM Latest"] = _actual_model_values(latest_row, prior_revenue=None, prior_shares=None)
    if forecast_table is not None and not forecast_table.empty:
        for _, row in forecast_table.iterrows():
            year_index = int(row.get("Year") or len(rows_by_period) + 1)
            label = _forecast_label(latest_year, year_index)
            rows_by_period[label] = _forecast_model_values(row, assumptions, prior_revenue=prior_revenue, prior_shares=prior_shares)
            prior_revenue = row.get("Revenue")
            prior_shares = row.get("Diluted Shares")
    table = _model_rows_to_table(rows_by_period)
    derived, log = derive_financial_rows(table)
    if hidden_prior_label in derived.columns:
        derived = derived.drop(columns=[hidden_prior_label])
    derived.attrs["derivation_log"] = log
    return derived


def build_financial_derivation_log(model_table: pd.DataFrame) -> pd.DataFrame:
    log = (model_table.attrs or {}).get("derivation_log") if model_table is not None else None
    if not log:
        _derived, log = derive_financial_rows(model_table)
    return pd.DataFrame(log)


def build_ev_to_equity_bridge(market_data: dict, dcf_output: dict, assumptions: dict) -> pd.DataFrame:
    shares = assumptions.get("diluted_shares") or market_data.get("shares_outstanding")
    fair_value = dcf_output.get("fair_value_per_share")
    rows = [
        {"Metric": "DCF enterprise value", "Value": dcf_output.get("enterprise_value"), "Source": "DCF forecast"},
        {"Metric": "Net debt / net cash", "Value": assumptions.get("net_debt"), "Source": "SEC / yfinance / model"},
        {"Metric": "DCF equity value", "Value": dcf_output.get("equity_value"), "Source": "Enterprise value - net debt"},
        {"Metric": "Diluted shares", "Value": shares, "Source": "SEC / Finviz / yfinance"},
        {"Metric": "Fair value per share", "Value": fair_value, "Source": "Equity value / diluted shares"},
        {"Metric": "Current share price", "Value": market_data.get("price"), "Source": "Finviz / yfinance"},
        {"Metric": "Upside / downside %", "Value": dcf_output.get("upside_downside_pct"), "Source": "Fair value vs current price"},
        {"Metric": "Margin of safety %", "Value": assumptions.get("margin_of_safety"), "Source": "User assumption"},
        {"Metric": "Buy price", "Value": dcf_output.get("buy_price_after_margin_of_safety"), "Source": "Fair value x margin of safety"},
        {"Metric": "Terminal value % of EV", "Value": dcf_output.get("terminal_value_weight_pct"), "Source": "DCF forecast"},
    ]
    return pd.DataFrame(rows)


def build_source_evidence_table(historicals: pd.DataFrame, dataset: dict) -> pd.DataFrame:
    if historicals is None or historicals.empty:
        return pd.DataFrame(columns=["Metric", "Period", "Value", "Source", "Evidence grade", "Filing date", "Confidence", "Notes"])
    rows = []
    latest = historicals.tail(5)
    filing_date = None
    filings = dataset.get("latest_filings") or []
    if filings:
        filing_date = filings[0].get("filing_date")
    for _, row in latest.iterrows():
        for metric in ["Revenue", "Gross Profit", "EBIT", "NOPAT", "OCF", "Total CAPEX", "FCF", "SBC", "Diluted Shares", "Net Debt"]:
            rows.append(
                {
                    "Metric": metric,
                    "Period": row.get("Period"),
                    "Value": row.get(metric),
                    "Source": "SEC companyfacts / yfinance fallback",
                    "Evidence grade": "Reported" if metric != "Maintenance CAPEX" else "Proxy-based",
                    "Filing date": filing_date,
                    "Confidence": "High" if row.get(metric) not in (None, 0) else "Low",
                    "Notes": "Reported line item. Maintenance/growth CAPEX split is proxy-based when undisclosed and should be reviewed through accounting interpretation.",
                }
            )
    return pd.DataFrame(rows)
