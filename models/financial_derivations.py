from __future__ import annotations

import math
from typing import Any

import pandas as pd


DEFAULT_TAX_RATE = 0.21

ELIGIBLE_PERCENT_CHANGE_ITEMS = {
    "revenue",
    "cogs / cost of sales",
    "cost of sales",
    "gross profit",
    "gross profit",
    "total opex",
    "opex",
    "ebit",
    "ebitda",
    "d&a",
    "depreciation and amortization",
    "nopat",
    "ocf",
    "operating cash flow",
    "maintenance capex",
    "growth capex",
    "capex",
    "total capex",
    "working capital change",
    "working capital investment",
    "fcf",
    "adjusted fcf",
    "net income",
    "sbc",
    "diluted shares",
    "net debt",
}

SOURCE_COLUMNS = {
    "Source",
    "source",
    "Confidence",
    "confidence",
    "Warning",
    "warning",
    "Notes",
    "notes",
}


def _canonical(value: Any) -> str:
    return " ".join(str(value or "").replace("_", " ").split()).strip().lower()


def _is_percent_like(label: str) -> bool:
    name = _canonical(label)
    return (
        "%" in name
        or "margin" in name
        or "rate" in name
        or "yield" in name
        or "ratio" in name
        or "multiple" in name
        or name.endswith("growth")
        or name.endswith("growth %")
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "unavailable", "none", "nan", "inf", "-inf", "n.m.", "nm"}:
            return None
        stripped = stripped.replace("$", "").replace(",", "").replace("x", "")
        is_percent = stripped.endswith("%")
        stripped = stripped.rstrip("%")
        multiplier = 0.01 if is_percent else 1.0
        try:
            number = float(stripped) * multiplier
        except ValueError:
            return None
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _period_columns(table: pd.DataFrame, line_item_col: str) -> list[str]:
    return [col for col in table.columns if col != line_item_col and col not in SOURCE_COLUMNS]


def _line_item_column(table: pd.DataFrame) -> str:
    for col in ["Line Item", "Metric", "metric", "line_item"]:
        if col in table.columns:
            return col
    return table.columns[0] if len(table.columns) else "Line Item"


def _row_index(table: pd.DataFrame, label: str, line_item_col: str) -> int | None:
    target = _canonical(label)
    aliases = {
        "gross profit": {"gross profit", "gross profit"},
        "operating cash flow": {"operating cash flow", "ocf"},
        "ocf": {"operating cash flow", "ocf"},
        "total capex": {"total capex", "capex"},
        "capex": {"total capex", "capex"},
        "total opex": {"total opex", "opex"},
        "cogs / cost of sales": {"cogs / cost of sales", "cost of sales", "cogs"},
    }
    candidates = aliases.get(target, {target})
    for idx, value in table[line_item_col].items():
        if _canonical(value) in candidates:
            return int(idx)
    return None


def _ensure_row(table: pd.DataFrame, label: str, line_item_col: str) -> int:
    idx = _row_index(table, label, line_item_col)
    if idx is not None:
        return idx
    row = {col: None for col in table.columns}
    row[line_item_col] = label
    table.loc[len(table)] = row
    return int(table.index[-1])


def _get(table: pd.DataFrame, label: str, period: str, line_item_col: str) -> float | None:
    idx = _row_index(table, label, line_item_col)
    if idx is None or period not in table.columns:
        return None
    return _as_float(table.at[idx, period])


def _set(
    table: pd.DataFrame,
    label: str,
    period: str,
    value: float | None,
    method: str,
    log: list[dict],
    line_item_col: str,
    confidence: str = "Calculated",
    warning: str = "",
    overwrite_zero: bool = False,
    overwrite: bool = False,
) -> None:
    if value is None or math.isnan(float(value)) or math.isinf(float(value)):
        return
    idx = _ensure_row(table, label, line_item_col)
    existing = _as_float(table.at[idx, period])
    if existing is not None and not overwrite and not (overwrite_zero and abs(existing) < 1e-12):
        return
    table.at[idx, period] = float(value)
    log.append(
        {
            "Line item": label,
            "Period": period,
            "Value": float(value),
            "Method": method,
            "Source confidence": confidence,
            "Warning": warning,
        }
    )


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def derive_revenue(gross_profit, cogs):
    """
    Derive revenue using sign-aware COGS logic.

    If cogs < 0:
        revenue = gross_profit - cogs
    If cogs > 0:
        revenue = gross_profit + cogs
    """
    gross_profit_value = _as_float(gross_profit)
    cogs_value = _as_float(cogs)
    if gross_profit_value is None or cogs_value is None or abs(cogs_value) < 1e-12:
        return None
    if cogs_value < 0:
        return gross_profit_value - cogs_value
    return gross_profit_value + cogs_value


def derive_financial_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Fill missing financial rows where they can be calculated reliably.

    Return:
    - updated dataframe
    - derivation log explaining what was calculated and how
    """
    if df is None or df.empty:
        return df, []
    table = df.copy()
    line_item_col = _line_item_column(table)
    periods = _period_columns(table, line_item_col)
    log: list[dict] = []

    for period in periods:
        revenue = _get(table, "Revenue", period, line_item_col)
        gross_profit = _get(table, "Gross profit", period, line_item_col)
        cogs = _get(table, "COGS / Cost of sales", period, line_item_col)

        if (revenue is None or abs(revenue) < 1e-12) and gross_profit is not None and cogs is not None:
            derived_revenue = derive_revenue(gross_profit, cogs)
            _set(
                table,
                "Revenue",
                period,
                derived_revenue,
                "Calculated from Gross Profit and sign-aware COGS because reported revenue was missing or zero.",
                log,
                line_item_col,
                confidence="Medium",
                overwrite_zero=True,
            )
            revenue = _get(table, "Revenue", period, line_item_col)

        if cogs is not None and cogs > 0:
            _set(
                table,
                "COGS / Cost of sales",
                period,
                -abs(cogs),
                "Normalized cost row to negative sign convention.",
                log,
                line_item_col,
                overwrite=True,
            )
            cogs = _get(table, "COGS / Cost of sales", period, line_item_col)

        if gross_profit is None and revenue is not None and cogs is not None:
            derived_gp = revenue + cogs if cogs < 0 else revenue - cogs
            _set(table, "Gross profit", period, derived_gp, "Calculated from Revenue and COGS.", log, line_item_col)
            gross_profit = _get(table, "Gross profit", period, line_item_col)

        if cogs is None and revenue is not None and gross_profit is not None:
            _set(table, "COGS / Cost of sales", period, -(revenue - gross_profit), "Calculated as negative cost from Revenue - Gross Profit.", log, line_item_col)
            cogs = _get(table, "COGS / Cost of sales", period, line_item_col)

        sm = _get(table, "S&M", period, line_item_col)
        rd = _get(table, "R&D", period, line_item_col)
        ga = _get(table, "G&A", period, line_item_col)
        other_opex = _get(table, "Other Operating Expenses", period, line_item_col)
        ebit = _get(table, "EBIT", period, line_item_col)
        opex = _get(table, "Total OPEX", period, line_item_col)

        reported_opex_lines = [value for value in [sm, rd, ga, other_opex] if value is not None]
        if opex is None and reported_opex_lines:
            _set(table, "Total OPEX", period, sum(abs(v) for v in reported_opex_lines), "Reported operating expense lines summed.", log, line_item_col)
            opex = _get(table, "Total OPEX", period, line_item_col)
        if opex is None and gross_profit is not None and ebit is not None:
            _set(table, "Total OPEX", period, gross_profit - ebit, "Calculated from Gross Profit - EBIT.", log, line_item_col)
            opex = _get(table, "Total OPEX", period, line_item_col)
        if opex is None and revenue is not None and cogs is not None and ebit is not None:
            _set(table, "Total OPEX", period, revenue - abs(cogs) - ebit, "Calculated from Revenue - COGS - EBIT.", log, line_item_col)
            opex = _get(table, "Total OPEX", period, line_item_col)

        if ebit is None and gross_profit is not None and opex is not None:
            _set(table, "EBIT", period, gross_profit - opex, "Calculated from Gross Profit - Total OPEX.", log, line_item_col)
            ebit = _get(table, "EBIT", period, line_item_col)

        nopat = _get(table, "NOPAT", period, line_item_col)
        tax_rate = _get(table, "Tax rate", period, line_item_col)
        if tax_rate is None and ebit is not None and nopat is not None and abs(ebit) > 1e-12:
            _set(table, "Tax rate", period, 1 - (nopat / ebit), "Calculated from NOPAT / EBIT.", log, line_item_col)
            tax_rate = _get(table, "Tax rate", period, line_item_col)
        if tax_rate is None:
            tax_rate = DEFAULT_TAX_RATE
            _set(
                table,
                "Tax rate",
                period,
                tax_rate,
                "Normalized tax rate fallback.",
                log,
                line_item_col,
                confidence="Low",
                warning="Reported effective tax rate unavailable; review manually.",
            )
        if nopat is None and ebit is not None:
            _set(
                table,
                "NOPAT",
                period,
                ebit * (1 - tax_rate),
                "Calculated from EBIT and normalized tax rate.",
                log,
                line_item_col,
                confidence="Medium",
            )

        da = _get(table, "D&A", period, line_item_col)
        maintenance_capex = _get(table, "Maintenance CAPEX", period, line_item_col)
        growth_capex = _get(table, "Growth CAPEX", period, line_item_col)
        total_capex = _get(table, "Total CAPEX", period, line_item_col)
        if total_capex is None:
            direct_capex = _get(table, "CAPEX", period, line_item_col)
            if direct_capex is not None:
                _set(table, "Total CAPEX", period, abs(direct_capex), "Reported capital expenditures.", log, line_item_col)
            elif maintenance_capex is not None and growth_capex is not None:
                _set(table, "Total CAPEX", period, abs(maintenance_capex) + abs(growth_capex), "Calculated from maintenance plus growth CAPEX.", log, line_item_col)
            total_capex = _get(table, "Total CAPEX", period, line_item_col)
        if maintenance_capex is None and da is not None:
            _set(
                table,
                "Maintenance CAPEX",
                period,
                abs(da),
                "Maintenance CAPEX uses D&A proxy.",
                log,
                line_item_col,
                confidence="Low",
                warning="Maintenance CAPEX uses D&A proxy - review reliability.",
            )
            maintenance_capex = _get(table, "Maintenance CAPEX", period, line_item_col)
        if growth_capex is None and total_capex is not None and maintenance_capex is not None:
            _set(table, "Growth CAPEX", period, max(abs(total_capex) - abs(maintenance_capex), 0), "Calculated from Total CAPEX - Maintenance CAPEX.", log, line_item_col)

        ocf = _get(table, "Operating cash flow", period, line_item_col)
        fcf = _get(table, "FCF", period, line_item_col)
        if fcf is None and ocf is not None and total_capex is not None:
            _set(table, "FCF", period, ocf - abs(total_capex), "Calculated from OCF - Total CAPEX.", log, line_item_col)

        adjusted_ocf = _get(table, "Adjusted OCF", period, line_item_col)
        adjusted_fcf = _get(table, "Adjusted FCF", period, line_item_col)
        if adjusted_fcf is None and adjusted_ocf is not None and maintenance_capex is not None:
            _set(table, "Adjusted FCF", period, adjusted_ocf - abs(maintenance_capex), "Calculated from Adjusted OCF - Maintenance CAPEX.", log, line_item_col)

        revenue = _get(table, "Revenue", period, line_item_col)
        gross_profit = _get(table, "Gross profit", period, line_item_col)
        cogs = _get(table, "COGS / Cost of sales", period, line_item_col)
        opex = _get(table, "Total OPEX", period, line_item_col)
        ebit = _get(table, "EBIT", period, line_item_col)
        nopat = _get(table, "NOPAT", period, line_item_col)
        ocf = _get(table, "Operating cash flow", period, line_item_col)
        da = _get(table, "D&A", period, line_item_col)
        total_capex = _get(table, "Total CAPEX", period, line_item_col)
        fcf = _get(table, "FCF", period, line_item_col)
        adjusted_fcf = _get(table, "Adjusted FCF", period, line_item_col)

        margin_inputs = [
            ("Revenue growth %", None),
            ("COGS % revenue", _safe_div(abs(cogs) if cogs is not None else None, revenue)),
            ("Gross margin %", _safe_div(gross_profit, revenue)),
            ("OPEX % revenue", _safe_div(opex, revenue)),
            ("EBIT margin %", _safe_div(ebit, revenue)),
            ("NOPAT margin %", _safe_div(nopat, revenue)),
            ("OCF margin %", _safe_div(ocf, revenue)),
            ("D&A % revenue", _safe_div(da, revenue)),
            ("Total CAPEX % revenue", _safe_div(abs(total_capex) if total_capex is not None else None, revenue)),
            ("FCF margin %", _safe_div(fcf, revenue)),
            ("Adjusted FCF margin %", _safe_div(adjusted_fcf, revenue)),
        ]
        for label, value in margin_inputs[1:]:
            _set(table, label, period, value, f"Calculated as {label.replace(' % revenue', '').replace(' margin %', '')} / Revenue.", log, line_item_col)

    table = add_percentage_change_rows(table, line_item_col=line_item_col, period_cols=periods)
    return table, log


def add_percentage_change_rows(
    table: pd.DataFrame,
    line_item_col: str = "Line Item",
    period_cols: list[str] | None = None,
    eligible_line_items: list[str] | None = None,
) -> pd.DataFrame:
    """
    For every eligible numeric line item, add a new row directly below it:
    '<Line Item> % change'

    % change = current period / prior period - 1
    """
    if table is None or table.empty or line_item_col not in table.columns:
        return table
    periods = period_cols or _period_columns(table, line_item_col)
    eligible = {_canonical(item) for item in (eligible_line_items or ELIGIBLE_PERCENT_CHANGE_ITEMS)}
    out_rows: list[dict] = []
    existing = {_canonical(value) for value in table[line_item_col].tolist()}

    for _, row in table.iterrows():
        row_dict = row.to_dict()
        out_rows.append(row_dict)
        label = str(row.get(line_item_col) or "")
        canonical = _canonical(label)
        pct_label = f"{label} % change"
        if canonical not in eligible or _is_percent_like(label) or _canonical(pct_label) in existing:
            continue
        pct_row = {col: None for col in table.columns}
        pct_row[line_item_col] = pct_label
        previous = None
        for period in periods:
            current = _as_float(row.get(period))
            if previous is None:
                pct_row[period] = 0.0 if current is not None else "n.m."
            elif current is None or abs(previous) < 1e-12:
                pct_row[period] = "n.m."
            elif previous < 0 < current or current < 0 < previous:
                pct_row[period] = "n.m."
            else:
                pct_row[period] = current / previous - 1
            previous = current
        out_rows.append(pct_row)
    return pd.DataFrame(out_rows, columns=table.columns)
