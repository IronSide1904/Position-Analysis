from __future__ import annotations

import math

import pandas as pd


UNAVAILABLE = "Unavailable"


def _as_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def fmt_dollar(value, scale: str = "auto") -> str:
    """
    Format dollar values with commas.
    Examples:
    1000000 -> $1M
    2450000000 with scale='M' -> $2,450M
    2450000000 with scale='B' -> $2.5B
    """
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    scale = str(scale or "auto")
    if scale == "raw":
        return fmt_dollar_no_decimals(number)
    if scale == "auto":
        for suffix, divisor, decimals in [("T", 1e12, 1), ("B", 1e9, 1), ("M", 1e6, 0), ("K", 1e3, 0)]:
            if abs(number) >= divisor:
                return f"${number / divisor:,.{decimals}f}{suffix}"
        return f"${number:,.0f}"
    divisor = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(scale, 1)
    decimals = 1 if scale in {"B", "T"} else 0
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


def fmt_per_share(value) -> str:
    """
    Format per-share values with two decimals.
    Example: 15.423 -> $15.42
    """
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    return f"${number:,.2f}"


def fmt_percent(value) -> str:
    """
    Format decimal percentages with one decimal.
    Example: 0.124 -> 12.4%
    """
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    return f"{number:.1%}"


def fmt_margin(value) -> str:
    """
    0.4523 -> 45.2%
    """
    return fmt_percent(value)


def fmt_multiple(value) -> str:
    """
    Format valuation multiples with one decimal.
    Example: 15.234 -> 15.2x
    """
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    return f"{number:,.1f}x"


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
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    return f"{number / 1e6:,.0f}M"


def fmt_number(value, decimals: int = 0) -> str:
    """
    General number formatter.
    """
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    return f"{number:,.{decimals}f}"
