from __future__ import annotations

import math

import pandas as pd


UNAVAILABLE = "Unavailable"

DOLLAR_METRICS = {
    "Market Cap",
    "Enterprise Value",
    "EV",
    "Income",
    "Sales",
    "Revenue",
    "Net Income",
    "Cash",
    "Debt",
    "Net Debt",
    "Target Price",
    "Price",
    "Prev Close",
}

PER_SHARE_DOLLAR_METRICS = {
    "Book/sh",
    "Cash/sh",
    "Fair Value",
    "MOS Buy Price",
    "Current Price",
    "Price",
    "Target Price",
    "Prev Close",
    "Buy Price",
    "Buy Zone",
}

PERCENT_METRICS = {
    "Dividend Yield",
    "Dividend Est.",
    "Dividend TTM",
    "Dividend Gr. 3/5Y",
    "Payout",
    "EPS this Y",
    "EPS next Y",
    "EPS next 5Y",
    "EPS past 3/5Y",
    "Sales past 3/5Y",
    "EPS Y/Y TTM",
    "Sales Y/Y TTM",
    "EPS Q/Q",
    "Sales Q/Q",
    "ROA",
    "ROE",
    "ROIC",
    "ROI",
    "Gross Margin",
    "Oper. Margin",
    "Operating Margin",
    "Profit Margin",
    "SMA20",
    "SMA50",
    "SMA200",
    "Perf Week",
    "Perf Month",
    "Perf Quarter",
    "Perf Half Y",
    "Perf YTD",
    "Perf Year",
    "Perf 3Y",
    "Perf 5Y",
    "Perf 10Y",
    "Insider Own",
    "Insider Trans",
    "Inst Own",
    "Inst Trans",
    "Short Float",
    "Float / Outstanding",
    "Upside",
    "Upside / Downside",
    "DCF Upside",
    "Terminal Weight",
    "Terminal Value Weight",
    "Terminal Value % EV",
    "Terminal value % of EV",
    "Margin of Safety",
    "Margin of safety %",
    "WACC",
    "Revenue CAGR",
    "Growth",
}

MULTIPLE_METRICS = {
    "P/E",
    "Forward P/E",
    "EV/EBITDA",
    "EV/Sales",
    "EV/Revenue",
    "EV/FCF",
    "EV/NOPAT",
}

RATIO_METRICS_2_DECIMALS = {
    "Beta",
    "Rolling Beta",
    "Current Ratio",
    "Quick Ratio",
    "Debt/Eq",
    "Debt / Eq",
    "Debt / Equity",
    "LT Debt/Eq",
    "LT Debt / Eq",
    "LT Debt / Equity",
    "Short Ratio",
    "ATR",
    "ATR (14)",
    "Volatility",
    "Correlation",
    "Rolling Correlation",
    "Relative Volume",
    "Rel Volume",
    "PEG",
    "P/B",
    "P/S",
    "P/C",
    "P/FCF",
    "Recom",
    "Analyst Recommendation",
}

VOLUME_METRICS = {
    "Volume",
    "Avg Volume",
    "Average Volume",
    "Trades",
    "Shs Outstand",
    "Shares Outstanding",
    "Shs Float",
    "Shares Float",
    "Float",
    "Short Interest",
    "Employees",
    "Diluted Shares",
    "Diluted shares",
}


def _as_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        if pd.isna(value):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def fmt_dollar(value, scale: str = "auto") -> str:
    """
    Format dollar values.

    Examples:
    1000000 -> "$1,000,000"
    2450000000 with scale="M" -> "$2,450M"
    2836180000000 with scale="B" -> "$2,836.18B"
    """
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    scale = str(scale or "auto")
    if scale == "raw":
        return fmt_dollar_no_decimals(number)
    if scale == "auto":
        if abs(number) >= 1e9:
            return fmt_dollar_billions(number)
        return fmt_dollar_no_decimals(number)
    divisor = {"K": 1e3, "M": 1e6, "B": 1e9}.get(scale, 1)
    decimals = 2 if scale == "B" else 0
    return f"${number / divisor:,.{decimals}f}{scale}"


def fmt_dollar_no_decimals(value) -> str:
    """
    1000000 -> $1,000,000
    """
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    return f"${number:,.0f}"


def fmt_dollar_millions(value) -> str:
    """
    2450000000 -> $2,450M
    """
    return fmt_dollar(value, scale="M")


def fmt_dollar_billions(value, decimals: int = 2) -> str:
    """
    Format dollar values in billions.
    Example: 2836180000000 -> $2,836.18B
    """
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    return f"${number / 1e9:,.{decimals}f}B"


def fmt_per_share(value) -> str:
    """
    Format per-share values with two decimals.
    Example: 15.423 -> $15.42
    """
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    return f"${number:,.2f}"


def fmt_percent(value, decimals: int = 1) -> str:
    """
    Format decimal percentages with one decimal by default.
    Example: 0.124 -> 12.4%; 22.4 -> 22.4%
    """
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    if abs(number) > 1.5:
        return f"{number:,.{decimals}f}%"
    return f"{number:,.{decimals}%}"


def fmt_margin(value) -> str:
    """
    0.4523 -> 45.2%
    """
    return fmt_percent(value)


def fmt_multiple(value, decimals: int = 1) -> str:
    """
    Format valuation multiples with one decimal.
    Example: 15.234 -> 15.2x
    """
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    return f"{number:,.{decimals}f}x"


def fmt_ratio(value, decimals: int = 2) -> str:
    import math

    try:
        if value is None:
            return "Unavailable"

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return "Unavailable"

        return f"{value:,.{decimals}f}"

    except Exception:
        return "Unavailable"


def fmt_volume(value) -> str:
    """
    Format volume and share-count metrics.
    Examples:
    12013668 -> 12,013,668
    40250000 -> 40.25M
    7430000000 -> 7.43B
    """
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    abs_number = abs(number)
    if abs_number >= 1e9:
        return f"{number / 1e9:,.2f}B"
    if abs_number >= 2e7:
        return f"{number / 1e6:,.2f}M"
    return f"{number:,.0f}"


def fmt_score(value) -> str:
    """
    Format scores.
    Example: 74.2 -> 74/100
    """
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    return f"{number:,.0f}/100"


def fmt_shares(value) -> str:
    """
    Format share count.
    Example: 205000000 -> 205M
    """
    return fmt_volume(value)


def fmt_number(value, decimals: int = 0) -> str:
    """
    General number formatter.
    """
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    return f"{number:,.{decimals}f}"


def _canonical_metric(metric_name: str) -> str:
    return " ".join(str(metric_name or "").replace("_", " ").split()).strip().lower()


def _matches(metric_name: str, choices: set[str]) -> bool:
    canonical = _canonical_metric(metric_name)
    return any(canonical == _canonical_metric(choice) for choice in choices)


def format_market_summary_value(metric_name: str, value) -> str:
    """
    Format a market/fundamental metric based on the metric name.
    Used by Snapshot bottom Market / Fundamentals Summary strip.
    """
    name = str(metric_name or "")
    canonical = _canonical_metric(name)
    if _matches(name, PER_SHARE_DOLLAR_METRICS):
        return fmt_per_share(value)
    if _matches(name, DOLLAR_METRICS):
        if any(token in canonical for token in ["price", "book/sh", "cash/sh"]):
            return fmt_per_share(value)
        if any(token in canonical for token in ["market cap", "enterprise value", "revenue", "sales", "income", "cash", "debt", "ev"]):
            return fmt_dollar_billions(value)
        return fmt_dollar(value)
    if _matches(name, PERCENT_METRICS) or "%" in name:
        return fmt_percent(value, 1)
    if _matches(name, MULTIPLE_METRICS):
        return fmt_multiple(value, 1)
    if _matches(name, RATIO_METRICS_2_DECIMALS):
        return fmt_ratio(value, 2)
    if _matches(name, VOLUME_METRICS):
        return fmt_volume(value)
    if "score" in canonical:
        return fmt_score(value)
    return fmt_number(value, 0)
