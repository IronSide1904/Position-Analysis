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
    st.markdown(
        """
        <style>
        :root {
            --pa-bg-dark: #0B1020;
            --pa-card-dark: #111827;
            --pa-card-border-dark: #334155;
            --pa-text-dark: #F9FAFB;
            --pa-muted-dark: #9CA3AF;
            --pa-bg-light: #F8FAFC;
            --pa-card-light: #FFFFFF;
            --pa-card-border-light: #CBD5E1;
            --pa-text-light: #0F172A;
            --pa-muted-light: #475569;
            --pa-green: #22C55E;
            --pa-yellow: #EAB308;
            --pa-orange: #F97316;
            --pa-red: #EF4444;
            --pa-blue: #3B82F6;
            --pa-gray: #64748B;
        }
        .block-container { max-width: 1520px; padding-top: 1rem; }
        .pa-cockpit-hero {
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 18px;
            padding: 18px 22px;
            margin-bottom: 16px;
            background:
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.22), transparent 32%),
                linear-gradient(135deg, #0B1020 0%, #111827 58%, #172554 100%);
            color: #F9FAFB;
        }
        .pa-cockpit-kicker {
            color: #93C5FD;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .pa-cockpit-title {
            color: #F9FAFB;
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
            gap: 14px;
            margin: 14px 0 18px 0;
        }
        .pa-card-grid.numeric {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }
        .pa-card {
            border-radius: 16px;
            padding: 18px 20px;
            min-height: 140px;
            height: auto;
            overflow: visible;
            border: 1px solid var(--pa-card-border-dark);
            background: var(--pa-card-dark);
            color: var(--pa-text-dark);
            box-shadow: 0 12px 30px rgba(2, 6, 23, 0.18);
        }
        .pa-card-title {
            font-size: 0.92rem;
            font-weight: 800;
            opacity: 0.9;
            margin-bottom: 10px;
        }
        .pa-card-value {
            font-size: clamp(1.55rem, 2.6vw, 2.7rem);
            font-weight: 850;
            line-height: 1.08;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere;
        }
        .pa-card-subtitle {
            margin-top: 10px;
            font-size: 0.86rem;
            line-height: 1.35;
            opacity: 0.82;
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
            border: 1px solid rgba(255,255,255,0.16);
            background: rgba(255,255,255,0.08);
        }
        .pa-positive { border-left: 5px solid var(--pa-green); }
        .pa-caution { border-left: 5px solid var(--pa-yellow); }
        .pa-warning { border-left: 5px solid var(--pa-orange); }
        .pa-negative { border-left: 5px solid var(--pa-red); }
        .pa-info { border-left: 5px solid var(--pa-blue); }
        .pa-neutral { border-left: 5px solid var(--pa-gray); }
        .pa-card *, .pa-card div, .pa-card span, .pa-card p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        .pa-section-shell {
            border: 1px solid #CBD5E1;
            border-radius: 16px;
            padding: 16px 18px;
            background: #FFFFFF;
            color: #0F172A;
            margin: 12px 0 16px 0;
        }
        .pa-section-kicker {
            color: #2563EB;
            font-size: 0.76rem;
            font-weight: 850;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .pa-section-heading {
            color: #0F172A;
            font-size: 1.25rem;
            font-weight: 850;
            margin-bottom: 4px;
        }
        .pa-section-copy {
            color: #475569;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        .pa-summary-panel {
            border-radius: 16px;
            border: 1px solid #CBD5E1;
            background: #F8FAFC;
            padding: 16px 18px;
            margin: 14px 0;
            color: #0F172A;
        }
        .pa-summary-panel h4 {
            margin: 0 0 10px 0;
            font-size: 1rem;
        }
        .pa-summary-panel li { margin-bottom: 6px; }
        div[data-testid="stTabs"] button p {
            font-size: 0.95rem;
            font-weight: 800;
            white-space: normal !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #334155;
            border-radius: 10px;
            overflow: hidden;
        }
        @media (prefers-color-scheme: light) {
            .pa-card {
                background: var(--pa-card-light);
                color: var(--pa-text-light);
                border-color: var(--pa-card-border-light);
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
            }
            .pa-card-subtitle { color: var(--pa-muted-light); opacity: 1; }
            .pa-badge {
                color: #0F172A;
                background: #F1F5F9;
                border-color: #CBD5E1;
            }
        }
        @media (max-width: 1050px) {
            .pa-card-grid, .pa-card-grid.numeric { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 720px) {
            .pa-card-grid, .pa-card-grid.numeric { grid-template-columns: 1fr; }
            .pa-cockpit-hero { padding: 16px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_cockpit_header(title: str, subtitle: str, kicker: str = "Decision Cockpit") -> None:
    st.markdown(
        f"""
        <div class="pa-cockpit-hero">
            <div class="pa-cockpit-kicker">{html.escape(_safe_text(kicker))}</div>
            <p class="pa-cockpit-title">{html.escape(_safe_text(title))}</p>
            <div class="pa-cockpit-subtitle">{html.escape(_safe_text(subtitle))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title: str, subtitle: str | None = None, kicker: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="pa-section-shell">
            <div class="pa-section-kicker">{html.escape(_safe_text(kicker or "Cockpit Section"))}</div>
            <div class="pa-section-heading">{html.escape(_safe_text(title))}</div>
            <div class="pa-section-copy">{html.escape(_safe_text(subtitle or ""))}</div>
        </div>
        """,
        unsafe_allow_html=True,
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
    st.markdown(f'<div class="{cls}">{"".join(html_cards)}</div>', unsafe_allow_html=True)


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
    st.markdown(f'<div class="pa-summary-panel">{"".join(rows) or "Summary unavailable."}</div>', unsafe_allow_html=True)


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
