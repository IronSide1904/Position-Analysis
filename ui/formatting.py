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


def fmt_dollar(value, scale: str = "M") -> str:
    """
    Format dollar values with no decimals.
    Example: 1250000000 -> $1,250M
    """
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    divisor = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12, "raw": 1}.get(str(scale), 1e6)
    suffix = "" if scale == "raw" else str(scale)
    return f"${number / divisor:,.0f}{suffix}"


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


def fmt_number(value, decimals: int = 1) -> str:
    """
    General number formatter.
    """
    number = _as_float(value)
    if number is None:
        return UNAVAILABLE
    return f"{number:,.{decimals}f}"
