from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DEFAULT_ITEMS = ["Revenue", "Gross profit", "Total OPEX", "Operating cash flow", "NOPAT", "FCF"]


def _canonical(value: object) -> str:
    return " ".join(str(value or "").replace("_", " ").split()).strip().lower()


def _period_columns(table: pd.DataFrame, line_item_col: str) -> list[str]:
    ignored = {line_item_col, "Source", "source", "Confidence", "confidence", "Warning", "warning", "Notes", "notes"}
    return [col for col in table.columns if col not in ignored]


def _is_percent_row(label: str) -> bool:
    text = _canonical(label)
    return "% change" in text or "%" in text or "margin" in text or "yield" in text or "rate" in text


def _is_ratio_row(label: str) -> bool:
    text = _canonical(label)
    return "ratio" in text or "multiple" in text or text.endswith(" x")


def _is_absolute_row(label: str) -> bool:
    return not _is_percent_row(label) and not _is_ratio_row(label)


def _key(title: str, suffix: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60]
    return f"financial_chart_{slug}_{suffix}"


def _widget_key(title: str, key_prefix: str | None, suffix: str) -> str:
    return f"{key_prefix}_{suffix}" if key_prefix else _key(title, suffix)


def _format_hover(label: str) -> str:
    if _is_percent_row(label):
        return "%{y:.1%}"
    if _is_ratio_row(label):
        return "%{y:.1f}x"
    if "share" in _canonical(label):
        return "%{y:,.1f}"
    return "$%{y:,.0f}"


def render_financial_line_chart(
    table: pd.DataFrame,
    title: str,
    line_item_col: str = "Line Item",
    period_cols: list[str] | None = None,
    default_items: list[str] | None = None,
    key_prefix: str | None = None,
):
    """
    Render an interactive line chart from a financial table.

    User can select/toggle line items.
    X-axis = years/periods
    Y-axis = values
    """
    if table is None or table.empty or line_item_col not in table.columns:
        st.info("Financial chart unavailable because the table has no line items.")
        return

    periods = period_cols or _period_columns(table, line_item_col)
    if not periods:
        st.info("Financial chart unavailable because period columns are missing.")
        return

    chart_mode = st.segmented_control(
        "Chart mode",
        ["Absolute values", "% change", "Margins / ratios"],
        default="Absolute values",
        key=_widget_key(title, key_prefix, "mode"),
    )

    frame = table.copy()
    frame[line_item_col] = frame[line_item_col].astype(str)
    if chart_mode == "% change":
        available = [item for item in frame[line_item_col] if "% change" in _canonical(item)]
    elif chart_mode == "Margins / ratios":
        available = [item for item in frame[line_item_col] if _is_percent_row(item) and "% change" not in _canonical(item) or _is_ratio_row(item)]
    else:
        available = [item for item in frame[line_item_col] if _is_absolute_row(item)]

    if not available:
        st.info(f"No {chart_mode.lower()} rows are available for this table.")
        return

    defaults = [item for item in (default_items or DEFAULT_ITEMS) if item in available]
    if not defaults:
        defaults = available[: min(5, len(available))]

    selected_items = st.multiselect(
        "Line items to chart",
        options=available,
        default=defaults,
        key=_widget_key(title, key_prefix, "items"),
    )
    if not selected_items:
        st.info("Select at least one line item to display.")
        return

    fig = go.Figure()
    for item in selected_items:
        row = frame[frame[line_item_col] == item]
        if row.empty:
            continue
        values = pd.to_numeric(row.iloc[0][periods], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=values,
                mode="lines+markers",
                name=item,
                hovertemplate=f"{item}<br>%{{x}}: {_format_hover(item)}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=430,
        margin=dict(l=20, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title="Years / periods")
    if chart_mode == "Absolute values":
        fig.update_yaxes(title="Value", tickprefix="$", separatethousands=True)
    elif chart_mode == "% change":
        fig.update_yaxes(title="% change", tickformat=".1%")
    else:
        fig.update_yaxes(title="Margin / ratio", tickformat=".1%")

    st.plotly_chart(fig, width="stretch", key=_widget_key(title, key_prefix, "plot"))
    st.caption(
        "What this shows: selected financial line items across reported and forecast periods. "
        "Current interpretation: compare growth, margin, cash conversion, and reinvestment direction before adjusting assumptions."
    )
