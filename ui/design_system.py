from __future__ import annotations

import html
from typing import Iterable

import pandas as pd
import streamlit as st

from ui.formatting import UNAVAILABLE, fmt_number, fmt_percent, fmt_per_share, fmt_score


STATUS_CLASS = {
    "positive": "pa-positive",
    "supportive": "pa-positive",
    "buy": "pa-positive",
    "undervalued": "pa-positive",
    "tradable": "pa-positive",
    "caution": "pa-caution",
    "watchlist": "pa-caution",
    "unknown": "pa-caution",
    "manual review": "pa-caution",
    "partial": "pa-caution",
    "warning": "pa-warning",
    "expensive": "pa-warning",
    "risk": "pa-warning",
    "negative": "pa-negative",
    "avoid": "pa-negative",
    "broken": "pa-negative",
    "dangerous": "pa-negative",
    "info": "pa-info",
    "neutral": "pa-neutral",
    "unavailable": "pa-neutral",
}


def _render_html(markup: str) -> None:
    if hasattr(st, "html"):
        st.html(markup)
    else:
        st.markdown(markup, unsafe_allow_html=True)


def _safe_text(value) -> str:
    if value is None:
        return UNAVAILABLE
    if isinstance(value, float) and pd.isna(value):
        return UNAVAILABLE
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "inf", "-inf"}:
        return UNAVAILABLE
    return text


def status_class(status: str | None, value=None) -> str:
    joined = f"{status or ''} {_safe_text(value)}".lower()
    for key, css_class in STATUS_CLASS.items():
        if key in joined:
            return css_class
    return "pa-neutral"


def apply_design_system() -> None:
    _render_html(
        """
        <style>
        :root {
            --pa-bg-dark: #080D18;
            --pa-panel-dark: #0F172A;
            --pa-card-dark: #0B1220;
            --pa-card-border-dark: rgba(148, 163, 184, 0.28);
            --pa-text-dark: #F8FAFC;
            --pa-muted-dark: #CBD5E1;
            --pa-soft-dark: #94A3B8;
            --pa-green: #22C55E;
            --pa-yellow: #F59E0B;
            --pa-orange: #F97316;
            --pa-red: #F87171;
            --pa-blue: #38BDF8;
            --pa-teal: #14B8A6;
            --pa-cyan: #22D3EE;
            --pa-gray: #64748B;
        }
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: var(--pa-bg-dark) !important;
            color: var(--pa-text-dark) !important;
        }
        [data-testid="stHeader"] {
            background: rgba(8, 13, 24, 0.78) !important;
        }
        [data-testid="stSidebar"] {
            background: #070B14 !important;
            border-right: 1px solid rgba(148, 163, 184, 0.18) !important;
        }
        [data-testid="stSidebar"] * {
            color: #E5E7EB !important;
        }
        .block-container { max-width: 1520px; padding-top: 1rem; }
        h1, h2, h3, h4, h5, h6, p, label, span {
            color: inherit;
        }
        .pa-cockpit-hero {
            border: 1px solid rgba(148, 163, 184, 0.30);
            border-radius: 8px;
            padding: 18px 20px;
            margin-bottom: 16px;
            background:
                radial-gradient(circle at top right, rgba(20, 184, 166, 0.16), transparent 34%),
                linear-gradient(145deg, rgba(15, 23, 42, 0.96), rgba(8, 13, 24, 0.98));
            color: var(--pa-text-dark);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06), 0 14px 30px rgba(2, 6, 23, 0.18);
        }
        .pa-cockpit-kicker {
            color: #99F6E4;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .pa-cockpit-title {
            color: #F8FAFC;
            font-size: clamp(1.45rem, 2.3vw, 2.45rem);
            font-weight: 850;
            line-height: 1.08;
            margin: 0;
        }
        .pa-cockpit-subtitle {
            color: #CBD5E1;
            margin-top: 7px;
            font-size: 0.94rem;
            max-width: 1000px;
        }
        .pa-card-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 14px 0 18px 0;
        }
        .pa-card-grid.numeric {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }
        .pa-card {
            position: relative;
            border-radius: 8px;
            padding: 18px 18px 16px;
            min-height: 132px;
            height: auto;
            overflow: hidden;
            border: 1px solid var(--pa-card-border-dark);
            background:
                linear-gradient(145deg, rgba(15, 23, 42, 0.94), rgba(8, 13, 24, 0.96)),
                radial-gradient(circle at top right, rgba(34, 211, 238, 0.12), transparent 34%);
            color: var(--pa-text-dark);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 12px 28px rgba(2, 6, 23, 0.18);
        }
        .pa-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 3px;
            background: var(--pa-gray);
        }
        .pa-card.pa-positive {
            border-color: rgba(34, 197, 94, 0.38);
            background:
                linear-gradient(145deg, rgba(10, 30, 24, 0.94), rgba(8, 13, 24, 0.96)),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.16), transparent 36%);
        }
        .pa-card.pa-caution,
        .pa-card.pa-warning {
            border-color: rgba(245, 158, 11, 0.38);
            background:
                linear-gradient(145deg, rgba(35, 26, 9, 0.94), rgba(8, 13, 24, 0.96)),
                radial-gradient(circle at top right, rgba(245, 158, 11, 0.16), transparent 36%);
        }
        .pa-card.pa-negative {
            border-color: rgba(248, 113, 113, 0.40);
            background:
                linear-gradient(145deg, rgba(34, 15, 20, 0.94), rgba(8, 13, 24, 0.96)),
                radial-gradient(circle at top right, rgba(248, 113, 113, 0.16), transparent 36%);
        }
        .pa-card.pa-info {
            border-color: rgba(56, 189, 248, 0.38);
            background:
                linear-gradient(145deg, rgba(8, 25, 38, 0.94), rgba(8, 13, 24, 0.96)),
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.16), transparent 36%);
        }
        .pa-card.pa-positive::before { background: linear-gradient(90deg, var(--pa-green), rgba(34, 197, 94, 0.25)); }
        .pa-card.pa-caution::before,
        .pa-card.pa-warning::before { background: linear-gradient(90deg, var(--pa-yellow), rgba(245, 158, 11, 0.25)); }
        .pa-card.pa-negative::before { background: linear-gradient(90deg, var(--pa-red), rgba(248, 113, 113, 0.25)); }
        .pa-card.pa-info::before { background: linear-gradient(90deg, var(--pa-blue), rgba(56, 189, 248, 0.25)); }
        .pa-card-title {
            color: #CBD5E1 !important;
            font-size: 0.9rem;
            font-weight: 800;
            opacity: 1;
            margin-bottom: 10px;
        }
        .pa-card-value {
            color: #FFFFFF !important;
            font-size: clamp(1.55rem, 2.6vw, 2.7rem);
            font-weight: 850;
            line-height: 1.08;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere;
        }
        .pa-card-subtitle {
            color: #AEBCCB !important;
            margin-top: 10px;
            font-size: 0.86rem;
            line-height: 1.35;
            opacity: 1;
            white-space: normal !important;
        }
        .pa-card-meta {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 10px;
        }
        .pa-badge {
            display: inline-block;
            padding: 4px 9px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 800;
            margin-top: 8px;
            color: #E5E7EB !important;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: rgba(15, 23, 42, 0.72);
        }
        .pa-card *, .pa-card div, .pa-card span, .pa-card p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        .pa-section-shell {
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 8px;
            padding: 16px 18px;
            background:
                linear-gradient(145deg, rgba(15, 23, 42, 0.92), rgba(8, 13, 24, 0.95)),
                radial-gradient(circle at top right, rgba(14, 116, 144, 0.16), transparent 34%);
            color: #F8FAFC;
            margin: 12px 0 16px 0;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 12px 28px rgba(2, 6, 23, 0.16);
        }
        .pa-section-kicker {
            color: #99F6E4;
            font-size: 0.76rem;
            font-weight: 850;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .pa-section-heading {
            color: #F8FAFC;
            font-size: 1.25rem;
            font-weight: 850;
            margin-bottom: 4px;
        }
        .pa-section-copy {
            color: #CBD5E1;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        .pa-summary-panel {
            border-radius: 8px;
            border: 1px solid rgba(148, 163, 184, 0.24);
            background:
                linear-gradient(145deg, rgba(15, 23, 42, 0.92), rgba(8, 13, 24, 0.95)),
                radial-gradient(circle at top right, rgba(20, 184, 166, 0.12), transparent 34%);
            padding: 16px 18px;
            margin: 14px 0;
            color: #CBD5E1;
        }
        .pa-summary-panel h4 {
            color: #F8FAFC;
            margin: 0 0 10px 0;
            font-size: 1rem;
        }
        .pa-summary-panel li { margin-bottom: 6px; }
        div[data-testid="stTabs"] div[role="tablist"],
        div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 0.75rem !important;
            padding: 0.25rem 0 0.55rem !important;
            margin: 0.65rem 0 1.15rem !important;
            border: 0 !important;
            border-radius: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
            overflow-x: visible !important;
        }
        div[data-testid="stTabs"] div[data-baseweb="tab-border"],
        div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
            display: none !important;
        }
        div[data-testid="stTabs"] button[role="tab"],
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            position: relative !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            flex: 0 0 auto !important;
            min-height: 3rem !important;
            width: auto !important;
            color: #B6C2D1 !important;
            background:
                linear-gradient(180deg, rgba(30, 41, 59, 0.96), rgba(8, 13, 24, 0.98)) !important;
            border: 1px solid rgba(148, 163, 184, 0.42) !important;
            border-radius: 12px !important;
            padding: 0.66rem 1.08rem !important;
            margin: 0 !important;
            isolation: isolate;
            overflow: hidden;
            box-shadow:
                0 10px 24px rgba(2, 6, 23, 0.30),
                inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
            transition: background 170ms ease, border-color 170ms ease, box-shadow 170ms ease, transform 170ms ease, color 170ms ease;
        }
        div[data-testid="stTabs"] button[role="tab"]::before,
        div[data-testid="stTabs"] button[data-baseweb="tab"]::before {
            content: "";
            position: absolute;
            inset: 0;
            z-index: -1;
            background:
                radial-gradient(circle at top left, rgba(103, 232, 249, 0.28), transparent 42%),
                linear-gradient(135deg, rgba(34, 211, 238, 0.18), rgba(20, 184, 166, 0.05));
            opacity: 0;
            transition: opacity 170ms ease;
        }
        div[data-testid="stTabs"] button[role="tab"]:hover,
        div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
            color: #F8FAFC !important;
            border-color: rgba(103, 232, 249, 0.48) !important;
            box-shadow: 0 12px 26px rgba(8, 47, 73, 0.30) !important;
            transform: translateY(-1px);
        }
        div[data-testid="stTabs"] button[role="tab"]:hover::before,
        div[data-testid="stTabs"] button[data-baseweb="tab"]:hover::before {
            opacity: 1;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
            color: #FFFFFF !important;
            background: linear-gradient(135deg, #0891B2, #0F766E) !important;
            border-color: rgba(165, 243, 252, 0.72) !important;
            box-shadow:
                0 16px 32px rgba(8, 145, 178, 0.34),
                inset 0 1px 0 rgba(255, 255, 255, 0.24) !important;
            transform: translateY(-1px);
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]::after,
        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"]::after {
            content: "";
            position: absolute;
            left: 18%;
            right: 18%;
            bottom: 0.32rem;
            height: 3px;
            border-radius: 999px;
            background: rgba(240, 253, 250, 0.9);
            box-shadow: 0 0 14px rgba(240, 253, 250, 0.62);
        }
        div[data-testid="stTabs"] button p {
            color: inherit !important;
            font-size: 0.95rem;
            font-weight: 800 !important;
            white-space: normal !important;
        }
        [data-testid="stRadio"] div[role="radiogroup"] {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 0.75rem !important;
        }
        [data-testid="stRadio"] div[role="radiogroup"] label {
            position: relative !important;
            min-height: 3.05rem !important;
            color: #B6C2D1 !important;
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.96), rgba(8, 13, 24, 0.98)) !important;
            border: 1px solid rgba(148, 163, 184, 0.42) !important;
            border-radius: 12px !important;
            padding: 0.68rem 1.12rem !important;
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.30), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        }
        [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
            color: #FFFFFF !important;
            background: linear-gradient(135deg, #0891B2, #0F766E) !important;
            border-color: rgba(165, 243, 252, 0.78) !important;
        }
        [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }
        [data-testid="stRadio"] div[role="radiogroup"] label p,
        [data-testid="stRadio"] div[role="radiogroup"] label span {
            color: inherit !important;
            font-weight: 850 !important;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(148, 163, 184, 0.24) !important;
            border-radius: 8px !important;
            padding: 0.75rem !important;
            background: rgba(15, 23, 42, 0.58) !important;
            color: #F8FAFC !important;
        }
        div[data-testid="stMetric"] * { color: #F8FAFC !important; }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricDelta"] { color: #CBD5E1 !important; }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 8px;
            overflow: hidden;
            background: rgba(15, 23, 42, 0.42);
            box-shadow: 0 10px 26px rgba(2, 6, 23, 0.16);
        }
        div[data-testid="stDataFrame"] div[role="columnheader"] {
            color: #F8FAFC !important;
            background: rgba(15, 23, 42, 0.92) !important;
            font-weight: 800 !important;
        }
        div[data-testid="stAlert"] {
            color: #E5E7EB !important;
            border: 1px solid rgba(148, 163, 184, 0.24) !important;
            border-radius: 8px !important;
            background: rgba(15, 23, 42, 0.88) !important;
        }
        div[data-testid="stAlert"] * {
            color: #E5E7EB !important;
        }
        @media (max-width: 1050px) {
            .pa-card-grid, .pa-card-grid.numeric { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 720px) {
            .pa-card-grid, .pa-card-grid.numeric { grid-template-columns: 1fr; }
            .pa-cockpit-hero { padding: 16px; }
        }
        </style>
        """
    )


def render_cockpit_header(title: str, subtitle: str, kicker: str = "Decision Cockpit") -> None:
    _render_html(
        f"""
        <div class="pa-cockpit-hero">
            <div class="pa-cockpit-kicker">{html.escape(_safe_text(kicker))}</div>
            <p class="pa-cockpit-title">{html.escape(_safe_text(title))}</p>
            <div class="pa-cockpit-subtitle">{html.escape(_safe_text(subtitle))}</div>
        </div>
        """
    )


def render_section(title: str, subtitle: str | None = None, kicker: str | None = None) -> None:
    _render_html(
        f"""
        <div class="pa-section-shell">
            <div class="pa-section-kicker">{html.escape(_safe_text(kicker or "Cockpit Section"))}</div>
            <div class="pa-section-heading">{html.escape(_safe_text(title))}</div>
            <div class="pa-section-copy">{html.escape(_safe_text(subtitle or ""))}</div>
        </div>
        """
    )


def render_status_card(
    title: str,
    value,
    subtitle: str | None = None,
    status: str | None = None,
    score: float | None = None,
    confidence: str | None = None,
    source_status: str | None = None,
    help_text: str | None = None,
) -> str:
    display = _safe_text(value)
    css_class = status_class(status, display)
    badges = []
    if score is not None:
        badges.append(f"Score {fmt_score(score)}")
    if confidence:
        badges.append(f"Confidence {confidence}")
    if source_status:
        badges.append(source_status)
    badge_html = "".join(f'<span class="pa-badge">{html.escape(_safe_text(badge))}</span>' for badge in badges)
    help_html = f'<div class="pa-card-subtitle">{html.escape(_safe_text(help_text))}</div>' if help_text else ""
    return f"""
    <div class="pa-card {css_class}">
        <div class="pa-card-title">{html.escape(_safe_text(title))}</div>
        <div class="pa-card-value">{html.escape(display)}</div>
        <div class="pa-card-subtitle">{html.escape(_safe_text(subtitle)) if subtitle else ""}</div>
        <div class="pa-card-meta">{badge_html}</div>
        {help_html}
    </div>
    """


def render_status_grid(cards: Iterable[dict], numeric: bool = False) -> None:
    html_cards = [render_status_card(**card) for card in cards]
    cls = "pa-card-grid numeric" if numeric else "pa-card-grid"
    _render_html(f'<div class="{cls}">{"".join(html_cards)}</div>')


def render_decision_summary(summary: dict) -> None:
    sections = [
        ("What matters now", summary.get("what_matters")),
        ("What supports the view", summary.get("supporting")),
        ("What contradicts the view", summary.get("contradicting")),
        ("What needs manual review", summary.get("manual_review")),
        ("Next action", summary.get("next_action")),
    ]
    rows = []
    for label, value in sections:
        items = value if isinstance(value, list) else [value]
        clean_items = [html.escape(_safe_text(item)) for item in items if _safe_text(item) != UNAVAILABLE]
        if clean_items:
            rows.append(f"<h4>{html.escape(label)}</h4><ul>{''.join(f'<li>{item}</li>' for item in clean_items)}</ul>")
    _render_html(f'<div class="pa-summary-panel">{"".join(rows) or "Summary unavailable."}</div>')


def render_score_explanation(score_components: list[dict]) -> None:
    if not score_components:
        st.info("Score explanation unavailable.")
        return
    frame = pd.DataFrame(score_components)
    st.dataframe(frame, width="stretch", hide_index=True)


def render_tearsheet(summary: dict) -> None:
    text = "\n".join(
        [
            f"Decision: {_safe_text(summary.get('decision'))}",
            f"Valuation: {_safe_text(summary.get('valuation'))}",
            f"Swing: {_safe_text(summary.get('swing'))}",
            f"Data confidence: {_safe_text(summary.get('confidence'))}",
            f"Next action: {_safe_text(summary.get('next_action'))}",
        ]
    )
    st.download_button("Export one-page tear sheet", text, file_name="pa11r-tearsheet.txt", mime="text/plain")
    st.code(text, language="text")


def render_copy_summary(summary: dict) -> None:
    text = " ".join(_safe_text(value) for value in summary.values() if _safe_text(value) != UNAVAILABLE)
    st.text_area("Copy final summary as text", value=text, height=130)


def format_short_score(value, max_score: int = 10) -> str:
    try:
        if value is None or pd.isna(value):
            return UNAVAILABLE
        number = float(value)
        if max_score == 10:
            return f"{number:.1f}/10"
        return fmt_score(number)
    except Exception:
        return UNAVAILABLE


def render_numeric_fact(label: str, value, kind: str = "number") -> dict:
    if kind == "percent":
        display = fmt_percent(value)
    elif kind == "per_share":
        display = fmt_per_share(value)
    else:
        display = fmt_number(value)
    return {"title": label, "value": display, "status": "info"}
