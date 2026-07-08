from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.formatting import UNAVAILABLE, fmt_dollar_millions, fmt_multiple, fmt_number, fmt_percent, fmt_per_share, fmt_score, fmt_shares


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
    return fmt_dollar_millions(value)


def fmt_pct(value):
    return fmt_percent(value)


def fmt_number_display(value):
    return fmt_number(value, decimals=0)


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
        else:
            display = UNAVAILABLE if value is None or (isinstance(value, float) and pd.isna(value)) else value
        col.metric(label, display)


def _format_cell(value, column_name: str):
    if value is None or pd.isna(value):
        return UNAVAILABLE
    name = str(column_name).replace("_", " ").lower()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "per share" in name or "share price" in name or name in {"price", "fair value", "buy price"}:
            return fmt_per_share(value)
        if "multiple" in name or name.endswith(" pe") or name.endswith(" p/e"):
            return fmt_multiple(value)
        if "score" in name:
            return fmt_score(value)
        if "shares" in name or "share count" in name:
            return fmt_shares(value)
        if any(hint in name for hint in PCT_HINTS) and abs(float(value)) <= 5:
            return fmt_pct(value)
        if any(hint in name for hint in MONEY_HINTS):
            return fmt_money(value)
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
            display_df[col] = display_df[col].map(lambda value: UNAVAILABLE if pd.isna(value) else str(value))
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
