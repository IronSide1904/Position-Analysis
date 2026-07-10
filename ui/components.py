from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from ui.formatting import (
    UNAVAILABLE,
    format_market_summary_value,
    fmt_money as fmt_money_adaptive,
    fmt_multiple,
    fmt_number,
    fmt_percent,
    fmt_per_share,
    fmt_ratio,
    fmt_score,
    fmt_shares,
    fmt_volume,
)


MONEY_HINTS = {
    "revenue",
    "sales",
    "profit",
    "opex",
    "ebitda",
    "ebit",
    "nopat",
    "income",
    "ocf",
    "capex",
    "fcf",
    "sbc",
    "debt",
    "cash",
    "d&a",
    "depreciation",
    "amortization",
    "price",
    "fair value",
    "buy zone",
    "market cap",
    "enterprise",
    "equity",
    "pv",
}
PCT_HINTS = {
    "margin",
    "cagr",
    "wacc",
    "growth",
    "upside",
    "downside",
    "weight",
    "pct",
    "%",
    "yield",
    "rate",
}


def fmt_money(value):
    return fmt_money_adaptive(value)


def fmt_pct(value):
    return fmt_percent(value)


def fmt_number_display(value):
    return fmt_number(value, decimals=0)


def _is_missing(value) -> bool:
    try:
        if value is None:
            return True
        if isinstance(value, str) and value.strip().lower() in {"", "none", "nan", "inf", "-inf"}:
            return True
        if isinstance(value, float) and (pd.isna(value) or not math.isfinite(value)):
            return True
        return bool(pd.isna(value)) if not isinstance(value, (list, tuple, dict, set, pd.Series, pd.DataFrame)) else False
    except Exception:
        return False


def metric_row(items: list[tuple[str, object, str]]):
    cols = st.columns(len(items))
    for col, (label, value, kind) in zip(cols, items):
        if kind == "money":
            display = fmt_money(value)
        elif kind == "per_share":
            display = fmt_per_share(value)
        elif kind == "pct":
            display = fmt_pct(value)
        elif kind == "multiple":
            display = fmt_multiple(value)
        elif kind == "score":
            display = fmt_score(value)
        elif kind == "shares":
            display = fmt_shares(value)
        elif kind == "volume":
            display = fmt_volume(value)
        elif kind == "ratio":
            display = fmt_ratio(value)
        else:
            display = UNAVAILABLE if _is_missing(value) else value
        col.metric(label, display)


def _format_cell(value, column_name: str):
    if _is_missing(value):
        return UNAVAILABLE
    name = str(column_name).replace("_", " ").lower()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "per share" in name or "share price" in name or name in {"price", "fair value", "buy price"}:
            return fmt_per_share(value)
        if "multiple" in name or name.endswith(" pe") or name.endswith(" p/e"):
            return fmt_multiple(value)
        if "score" in name:
            return fmt_score(value)
        if "shares" in name or "share count" in name or "volume" in name:
            return fmt_shares(value)
        if "ratio" in name or name in {"beta", "atr", "peg", "p/b", "p/s", "p/c", "p/fcf"}:
            return fmt_ratio(value)
        if any(hint in name for hint in PCT_HINTS) and abs(float(value)) <= 5:
            return fmt_pct(value)
        if any(hint in name for hint in MONEY_HINTS):
            return fmt_money(value)
        market_summary = format_market_summary_value(column_name, value)
        if market_summary != UNAVAILABLE:
            return market_summary
        return fmt_number(value, decimals=0)
    return value


def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    display_df = df.copy()
    label_col = next((col for col in ["Line Item", "Metric", "metric", "assumption", "field", "driver"] if col in display_df.columns), None)
    if "value" in display_df.columns and label_col:
        display_df["value"] = [
            _format_cell(value, label)
            for value, label in zip(display_df["value"], display_df[label_col])
        ]
    elif label_col:
        for col in display_df.columns:
            if col == label_col:
                continue
            display_df[col] = [
                _format_cell(value, label)
                for value, label in zip(display_df[col], display_df[label_col])
            ]
    for col in display_df.columns:
        if label_col and (col == "value" or col != label_col):
            continue
        if pd.api.types.is_numeric_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda value, name=col: _format_cell(value, name))
    for col in display_df.select_dtypes(include=["object"]).columns:
        non_null = display_df[col].dropna()
        if len({type(value) for value in non_null}) > 1:
            display_df[col] = display_df[col].map(lambda value: UNAVAILABLE if _is_missing(value) else str(value))
        else:
            display_df[col] = display_df[col].map(lambda value: UNAVAILABLE if _is_missing(value) else value)
    return display_df


def show_warnings(warnings: list[str]):
    for warning in warnings or []:
        st.warning(warning)


def show_table(df: pd.DataFrame, empty_message: str = "Not enough data available."):
    if df is None or df.empty:
        st.info(empty_message)
    else:
        display_df = format_dataframe_for_display(df)
        st.dataframe(display_df, width="stretch", hide_index=True)
