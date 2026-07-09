from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from analysis.accounting_interpreter import build_accounting_interpretation, build_accounting_interpretation_table
from analysis.capex_ocf_nopat import analyze_capex_ocf_nopat_quality
from analysis.clauses import extract_relevant_clauses
from analysis.clause_pipeline import run_clause_extraction_pipeline
from analysis.compensation import analyze_compensation_alignment
from analysis.guidance import analyze_guidance_accuracy
from analysis.ma_strategy import analyze_ma_strategy
from analysis.management import analyze_management_and_board
from analysis.moat import analyze_moat
from analysis.operating_leverage import analyze_operating_leverage
from analysis.peers import build_peer_comparison, select_peer_candidates
from analysis.risks import analyze_risks_and_thesis_breakers
from analysis.thesis import build_thesis_summary
from data_sources.loader import load_company_dataset
from models.dcf_model import (
    build_dcf_output_table,
    build_dcf_sensitivity_table,
    build_reverse_dcf_table,
    build_scenario_table,
    create_pending_assumption_update,
    default_assumptions_from_historicals,
    run_dcf,
)
from models.financial_model import build_ev_to_equity_bridge, build_historical_financial_table, build_source_evidence_table, build_time_axis_financial_model
from models.reverse_dcf import compare_clause_to_reverse_dcf, run_reverse_dcf
from models.scoring import score_investment
from models.sotp_model import build_default_segment_data, run_sotp
from models.multiples_model import calculate_current_multiples, peer_median_multiples
from ui.charts import (
    dcf_sensitivity_heatmap,
    fcf_projection_chart,
    financial_cash_flow_chart,
    financial_profitability_chart,
    financial_revenue_margin_chart,
    ma_timeline_chart,
    moat_score_bar,
    peer_multiple_chart,
    peer_scatter,
    price_action_chart,
    reverse_dcf_chart,
    sbc_vs_buybacks_chart,
    scenario_valuation_bar,
)
from ui.components import fmt_money, fmt_pct, format_dataframe_for_display, metric_row, show_table, show_warnings
from ui.design_system import (
    apply_design_system,
    format_short_score,
    render_cockpit_header,
    render_copy_summary,
    render_decision_summary,
    render_section,
    render_status_grid,
    render_tearsheet,
)
from ui.formatting import (
    UNAVAILABLE,
    format_market_summary_value,
    fmt_dollar,
    fmt_multiple,
    fmt_percent,
    fmt_per_share,
    fmt_ratio,
    fmt_score,
    fmt_shares,
    fmt_volume,
)
from ui.multiples import render_multiples_tab
from ui.sotp import get_active_sotp, render_sotp_tab


@st.cache_data(show_spinner=False, ttl=3600)
def cached_dataset(ticker: str, include_deep_sec: bool = False):
    return load_company_dataset(ticker, include_deep_sec=include_deep_sec)


def _css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; max-width: 1480px; }
        div[data-testid="stMetric"] {
            border: 1px solid #343b46 !important;
            border-radius: 8px !important;
            padding: 0.7rem 0.8rem !important;
            background: #151922 !important;
            color: #f8fafc !important;
        }
        div[data-testid="stMetric"] > div { background: transparent !important; }
        div[data-testid="stMetric"] * { color: #f8fafc !important; }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricDelta"] { color: #cbd5e1 !important; }
        .pa-header {
            border-bottom: 1px solid #d9e0e8;
            padding-bottom: 0.8rem;
            margin-bottom: 0.9rem;
        }
        .pa-title { font-size: 1.55rem; font-weight: 700; margin: 0; }
        .pa-subtle { color: #5d6978; font-size: 0.9rem; }
        .pa-build {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border: 1px solid #93c5fd;
            border-radius: 999px;
            padding: 0.22rem 0.65rem;
            margin: 0 0 0.6rem 0;
            background: #eff6ff;
            color: #1e3a8a;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .pa-tab-note {
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 8px;
            padding: 0.45rem 0.65rem;
            margin: 0.25rem 0 0.45rem 0;
            background: rgba(15, 23, 42, 0.72);
            color: #cbd5e1;
            font-size: 0.85rem;
            font-weight: 700;
        }
        .pa-pill {
            display: inline-block;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 999px;
            padding: 0.16rem 0.55rem;
            margin: 0 0.25rem 0.25rem 0;
            font-size: 0.78rem;
            background: rgba(15, 23, 42, 0.58);
            color: #cbd5e1;
        }
        .pa-pill-ok { border-color: rgba(34, 197, 94, 0.38); background: rgba(34, 197, 94, 0.12); color: #bbf7d0; }
        .pa-pill-warn { border-color: rgba(245, 158, 11, 0.38); background: rgba(245, 158, 11, 0.12); color: #fde68a; }
        .pa-band {
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            background: rgba(15, 23, 42, 0.55);
            color: #e5e7eb;
            margin-bottom: 0.8rem;
        }
        .pa-section-title {
            font-size: 0.86rem;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            color: #5d6978;
            margin-bottom: 0.35rem;
        }
        div[data-testid="stAlert"] {
            background: #f8fafc !important;
            color: #182230 !important;
            border: 1px solid #c8d3df !important;
            border-radius: 8px !important;
        }
        div[data-testid="stAlert"] * {
            color: #182230 !important;
        }
        .pa-summary-bar {
            display: grid;
            grid-template-columns: repeat(8, minmax(110px, 1fr));
            gap: 0.45rem;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 8px;
            padding: 0.55rem;
            margin: 0.75rem 0 0.85rem 0;
            background: rgba(15, 23, 42, 0.55);
        }
        .pa-summary-item {
            border-left: 3px solid #9db4d4;
            padding: 0.2rem 0.45rem;
            min-width: 0;
        }
        .pa-summary-label {
            color: #64748b;
            font-size: 0.72rem;
            line-height: 1.1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .pa-summary-value {
            color: #f8fafc;
            font-size: 0.95rem;
            font-weight: 700;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .pa-box {
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            background: rgba(15, 23, 42, 0.58);
            color: #e5e7eb;
            margin-bottom: 0.75rem;
        }
        .pa-box * { color: #e5e7eb !important; }
        .pa-box-title {
            color: #99f6e4;
            font-size: 0.86rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .pa-card-light {
            background-color: rgba(15, 23, 42, 0.58);
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 8px;
            padding: 16px;
            color: #e5e7eb;
            margin-bottom: 0.75rem;
        }
        .pa-card-light * { color: #e5e7eb !important; }
        .pa-muted { color: #9CA3AF !important; }
        .pa-notice-warning {
            background-color: rgba(245, 158, 11, 0.12);
            color: #fde68a !important;
            border-left: 4px solid #F59E0B;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .pa-notice-risk {
            background-color: rgba(248, 113, 113, 0.12);
            color: #fecaca !important;
            border-left: 4px solid #EF4444;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .pa-notice-success {
            background-color: rgba(34, 197, 94, 0.12);
            color: #bbf7d0 !important;
            border-left: 4px solid #22C55E;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .pa-notice-warning *, .pa-notice-risk *, .pa-notice-success * { color: inherit !important; }
        @media (max-width: 1100px) {
            .pa-summary-bar { grid-template-columns: repeat(4, minmax(110px, 1fr)); }
        }
        @media (max-width: 720px) {
            .pa-summary-bar { grid-template-columns: repeat(2, minmax(110px, 1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _pill(label: str, ok: bool = True) -> str:
    cls = "pa-pill pa-pill-ok" if ok else "pa-pill pa-pill-warn"
    return f'<span class="{cls}">{label}</span>'


def _source_status(dataset: dict) -> None:
    sources = set(dataset.get("sources", []))
    pills = [
        _pill("SEC/EDGAR", "SEC/EDGAR" in sources),
        _pill("Finviz Elite", "Finviz Elite" in sources),
        _pill("yfinance", "yfinance" in sources),
    ]
    st.markdown("".join(pills), unsafe_allow_html=True)


def _data_coverage(dataset: dict, historicals: pd.DataFrame) -> pd.DataFrame:
    market = dataset.get("market_data", {})
    sec = dataset.get("financials", {}).get("sec", {})
    debt = sec.get("debt", {}) if isinstance(sec, dict) else {}
    rows = [
        {"area": "Price history", "status": "Loaded" if not dataset.get("price_history", pd.DataFrame()).empty else "Missing", "source": "yfinance"},
        {"area": "Company profile", "status": "Loaded" if dataset.get("company") else "Missing", "source": ", ".join(dataset.get("sources", []))},
        {"area": "Market cap", "status": "Loaded" if market.get("market_cap") else "Missing", "source": "Finviz / yfinance"},
        {"area": "Enterprise value", "status": "Loaded" if market.get("enterprise_value") else "Missing", "source": "yfinance"},
        {"area": "Shares outstanding", "status": "Loaded" if market.get("shares_outstanding") or sec.get("shares", {}).get("value") else "Missing", "source": "Finviz / yfinance / SEC"},
        {"area": "SEC companyfacts", "status": "Loaded" if sec.get("revenue", {}).get("value") else "Missing", "source": "SEC"},
        {"area": "Full filing text", "status": "Loaded" if dataset.get("filing_texts") else "Not loaded", "source": "SEC evidence mode"},
        {"area": "Historical model table", "status": "Loaded" if historicals is not None and not historicals.empty else "Missing", "source": "SEC / yfinance"},
        {"area": "Debt detail", "status": "Loaded" if debt.get("confidence") == "reported" else "Manual review required", "source": debt.get("source") or "SEC / yfinance / manual review"},
        {"area": "Segment data", "status": "Manual review required", "source": "10-K segment note / company IR"},
        {"area": "Maintenance CAPEX split", "status": "Proxy-based", "source": "Accounting interpretation / manual review"},
    ]
    return pd.DataFrame(rows)


def _filing_metadata(dataset: dict) -> dict:
    rows = dataset.get("deep_filings") or dataset.get("latest_filings") or []
    by_form = {}
    for item in rows:
        form = item.get("form")
        if not form or form in by_form:
            continue
        by_form[form] = {
            "form": form,
            "filing_date": item.get("filing_date"),
            "accession_number": item.get("accession_number"),
            "source_url": item.get("document_url") or item.get("filing_url"),
        }
    return {"latest_filings": rows, **by_form}


def _fmt_number(value, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return UNAVAILABLE
    return f"{float(value):,.0f}{suffix}"


def _fmt_plain(value) -> str:
    if value is None or pd.isna(value):
        return UNAVAILABLE
    return f"{float(value):,.0f}"


def _fmt_ratio(value) -> str:
    return fmt_ratio(value)


def _fmt_summary_text(value) -> str:
    if value is None:
        return UNAVAILABLE
    text = str(value)
    if text.lower() in {"none", "nan", "inf", "-inf", ""}:
        return UNAVAILABLE
    return text


def _summary_bar(ctx: dict) -> None:
    dataset = ctx["dataset"]
    market = dataset.get("market_data", {})
    dcf = ctx["base_dcf"]
    reverse = ctx["reverse"]
    scoring = ctx["scoring"]
    cards = ctx.get("accounting_interpretation", {}).get("cards", {})
    sources = set(dataset.get("sources", []))
    data_bits = [
        f"SEC {'OK' if 'SEC/EDGAR' in sources else 'Missing'}",
        f"Finviz {'OK' if 'Finviz Elite' in sources else 'Missing'}",
        f"Yahoo {'OK' if 'yfinance' in sources else 'Missing'}",
        f"Clauses {'OK' if dataset.get('evidence_loaded') or not ctx.get('clauses', pd.DataFrame()).empty else 'Fast'}",
    ]
    items = [
        ("Ticker", dataset.get("ticker")),
        ("Price", fmt_per_share(market.get("price"))),
        ("Market Cap", fmt_dollar(market.get("market_cap"))),
        ("EV", fmt_dollar(market.get("enterprise_value"))),
        ("PA-11R Score", fmt_score(scoring.get("total_score"))),
        ("Rating", scoring.get("recommendation")),
        ("DCF Upside", fmt_percent(dcf.get("upside_downside_pct"))),
        ("Fair Value", fmt_per_share(dcf.get("fair_value_per_share"))),
        ("MOS Buy Price", fmt_per_share(dcf.get("buy_price_after_margin_of_safety"))),
        ("Reverse DCF", reverse.get("market_case")),
        ("Accounting", cards.get("Main Accounting Distortion")),
        ("Data", " / ".join(data_bits)),
    ]
    html = ['<div class="pa-summary-bar">']
    for label, value in items:
        html.append(
            f'<div class="pa-summary-item"><div class="pa-summary-label">{label}</div><div class="pa-summary-value">{_fmt_summary_text(value)}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _notice(message: str, kind: str = "warning") -> None:
    cls = {
        "warning": "pa-notice-warning",
        "risk": "pa-notice-risk",
        "success": "pa-notice-success",
    }.get(kind, "pa-notice-warning")
    st.markdown(f'<div class="{cls}">{message}</div>', unsafe_allow_html=True)


def _manual_review_items(ctx: dict) -> list[dict]:
    dataset = ctx.get("dataset", {})
    filings = dataset.get("latest_filings") or []
    latest = filings[0] if filings else {}
    source_url = latest.get("document_url") or latest.get("filing_url") or "https://www.sec.gov/edgar/search/"
    sec_fin = dataset.get("financials", {}).get("sec", {})
    debt = sec_fin.get("debt", {}) if isinstance(sec_fin, dict) else {}
    items = []
    if debt.get("confidence") != "reported":
        items.append(
            {
                "Data Needed": "Debt detail",
                "Reason": "SEC companyfacts did not include reliable current/noncurrent or total debt tags.",
                "Primary Source": "Latest 10-K or 10-Q",
                "Section to Review": "Balance Sheet; Debt Note; Borrowings / Credit Facility Note; Liquidity and Capital Resources",
                "Keywords": "debt, borrowings, credit facility, term loan, revolving credit, maturity",
                "Source URL": source_url,
                "Fallback Sources": "SEC filing; company investor relations; annual report; earnings release",
                "Dashboard Action": "Manual Review Required",
            }
        )
    if ctx.get("accounting_interpretation", {}).get("capex", {}).get("confidence") != "High":
        items.append(
            {
                "Data Needed": "Maintenance vs growth CAPEX split",
                "Reason": "The split is usually undisclosed and may be proxy-based.",
                "Primary Source": "Latest 10-K or 10-Q",
                "Section to Review": "Capital expenditures; Liquidity and Capital Resources; PP&E note",
                "Keywords": "maintenance capital, growth capital, capacity expansion, facility, equipment, infrastructure",
                "Source URL": source_url,
                "Fallback Sources": "SEC filing; earnings call transcript; investor presentation",
                "Dashboard Action": "Manual Review Required",
            }
        )
    return items


def _data_quality_table(ctx: dict) -> pd.DataFrame:
    dataset = ctx.get("dataset", {})
    sec_fin = dataset.get("financials", {}).get("sec", {})
    debt = sec_fin.get("debt", {}) if isinstance(sec_fin, dict) else {}
    issues = []
    if debt.get("confidence") != "reported":
        issues.append(
            {
                "Issue": "Debt detail unavailable from SEC companyfacts",
                "Meaning": "Specific XBRL debt tags were not found. This does not mean debt is zero.",
                "Impact": "Net debt and EV bridge may be lower confidence.",
                "Handled By": "Trying alternate SEC debt tags, total debt tag, yfinance totalDebt, and manual review guidance.",
                "Where to Verify": "10-K / 10-Q balance sheet, Debt note, Liquidity and Capital Resources.",
            }
        )
    if not dataset.get("evidence_loaded"):
        issues.append(
            {
                "Issue": "Full filing text not loaded",
                "Meaning": "Fast mode uses metadata and structured facts only.",
                "Impact": "Clause, management, M&A, and risk context is lower confidence.",
                "Handled By": "Load SEC evidence when deeper text review is needed.",
                "Where to Verify": "Latest 10-K / 10-Q / DEF 14A.",
            }
        )
    if not issues:
        issues.append(
            {
                "Issue": "No major data-quality issues detected",
                "Meaning": "Core providers returned enough data for the cockpit.",
                "Impact": "Normal model sensitivity still applies.",
                "Handled By": "SEC / Finviz / yfinance source coverage.",
                "Where to Verify": "Source evidence table.",
            }
        )
    return pd.DataFrame(issues)


def _critical_warnings(warnings: list[str]) -> list[str]:
    quiet_patterns = ["SEC companyfacts missing debt_current", "SEC companyfacts missing debt_noncurrent", "SEC companyfacts missing total_debt"]
    return [warning for warning in warnings or [] if not any(pattern in str(warning) for pattern in quiet_patterns)]


def _data_coverage_expander(ctx: dict) -> None:
    dataset = ctx["dataset"]
    with st.expander("Data Coverage / Source Quality", expanded=False):
        show_table(_data_coverage(dataset, ctx["historicals"]), "Data coverage unavailable.")
        show_table(_data_quality_table(ctx), "No data-quality notes.")
        manual_items = _manual_review_items(ctx)
        if manual_items:
            st.subheader("Manual Review Required")
            show_table(pd.DataFrame(manual_items), "No manual review items.")
            c1, c2, c3, c4 = st.columns(4)
            c1.button("Open SEC filing", disabled=True)
            c2.button("Search filing text", disabled=True)
            c3.button("Search company IR", disabled=True)
            c4.button("Search earnings release", disabled=True)


def _finviz_decision_snapshot(market: dict) -> pd.DataFrame:
    rows = [
        ("Share / Float", "Shares Outstanding", format_market_summary_value("Shares Outstanding", market.get("shares_outstanding"))),
        ("Share / Float", "Shares Float", format_market_summary_value("Shares Float", market.get("shares_float"))),
        ("Share / Float", "Float / Outstanding", format_market_summary_value("Float / Outstanding", market.get("float_outstanding_pct"))),
        ("Short / Liquidity", "Short Float", format_market_summary_value("Short Float", market.get("short_float"))),
        ("Short / Liquidity", "Short Ratio", format_market_summary_value("Short Ratio", market.get("short_ratio"))),
        ("Short / Liquidity", "Average Volume", format_market_summary_value("Average Volume", market.get("average_volume"))),
        ("Short / Liquidity", "Volume", format_market_summary_value("Volume", market.get("volume"))),
        ("Short / Liquidity", "Relative Volume", format_market_summary_value("Relative Volume", market.get("relative_volume"))),
        ("Volatility / Technical", "Beta", format_market_summary_value("Beta", market.get("beta"))),
        ("Volatility / Technical", "ATR", format_market_summary_value("ATR", market.get("atr"))),
        ("Volatility / Technical", "Perf Week", format_market_summary_value("Perf Week", market.get("volatility_week"))),
        ("Volatility / Technical", "Perf Month", format_market_summary_value("Perf Month", market.get("volatility_month"))),
        ("Volatility / Technical", "Gap", fmt_pct(market.get("gap"))),
        ("Volatility / Technical", "Change", fmt_pct(market.get("change"))),
        ("Volatility / Technical", "SMA20", format_market_summary_value("SMA20", market.get("sma20"))),
        ("Volatility / Technical", "SMA50", format_market_summary_value("SMA50", market.get("sma50"))),
        ("Volatility / Technical", "SMA200", format_market_summary_value("SMA200", market.get("sma200"))),
        ("Valuation", "P/E", format_market_summary_value("P/E", market.get("pe"))),
        ("Valuation", "Forward P/E", format_market_summary_value("Forward P/E", market.get("forward_pe"))),
        ("Valuation", "PEG", format_market_summary_value("PEG", market.get("peg"))),
        ("Valuation", "P/S", format_market_summary_value("P/S", market.get("ps"))),
        ("Valuation", "P/B", format_market_summary_value("P/B", market.get("pb"))),
        ("Valuation", "P/FCF", format_market_summary_value("P/FCF", market.get("pfcf"))),
        ("Profitability", "ROA", format_market_summary_value("ROA", market.get("roa"))),
        ("Profitability", "ROE", format_market_summary_value("ROE", market.get("roe"))),
        ("Profitability", "ROIC", format_market_summary_value("ROIC", market.get("roi"))),
        ("Profitability", "Gross Margin", format_market_summary_value("Gross Margin", market.get("gross_margin"))),
        ("Profitability", "Operating Margin", format_market_summary_value("Operating Margin", market.get("operating_margin"))),
        ("Profitability", "Profit Margin", format_market_summary_value("Profit Margin", market.get("profit_margin"))),
        ("Balance Sheet", "Current Ratio", format_market_summary_value("Current Ratio", market.get("current_ratio"))),
        ("Balance Sheet", "Quick Ratio", format_market_summary_value("Quick Ratio", market.get("quick_ratio"))),
        ("Balance Sheet", "LT Debt / Equity", format_market_summary_value("LT Debt / Equity", market.get("lt_debt_to_equity"))),
        ("Balance Sheet", "Debt / Equity", format_market_summary_value("Debt / Equity", market.get("debt_to_equity"))),
        ("Calendar", "Earnings date", market.get("earnings_date") or UNAVAILABLE),
    ]
    return pd.DataFrame(rows, columns=["category", "field", "value"])


def _top_assumption_drivers(assumptions: dict) -> pd.DataFrame:
    rows = [
        {"driver": "Revenue CAGR", "value": assumptions.get("revenue_cagr"), "why it matters": "Compounds every forecast year."},
        {"driver": "NOPAT margin", "value": assumptions.get("nopat_margin"), "why it matters": "Turns revenue into normalized after-tax profit."},
        {"driver": "WACC", "value": assumptions.get("wacc"), "why it matters": "Discounts future cash flows back to today."},
    ]
    return pd.DataFrame(rows)


def _top_three_drivers(ctx: dict) -> list[str]:
    assumptions = ctx.get("base_assumptions", {})
    capex = ctx.get("accounting_interpretation", {}).get("capex", {})
    return [
        f"Revenue growth must track near {fmt_percent(assumptions.get('revenue_cagr'))}.",
        f"OCF margin must hold near {fmt_percent(assumptions.get('ocf_margin'))}.",
        f"CAPEX split needs review: {capex.get('classification', 'Unclear')} classification.",
    ]


def _top_three_risks(ctx: dict) -> list[str]:
    risks = list(ctx.get("risks", {}).get("top_risks", []) or [])
    accounting = ctx.get("accounting_interpretation", {}).get("warnings", []) or []
    dcf_warnings = ctx.get("base_dcf", {}).get("warnings", []) or []
    combined = risks + accounting + dcf_warnings
    return (combined or ["Manual review required where data is unavailable."])[:3]


def _risk_review_table(ctx: dict, limit: int = 4) -> pd.DataFrame:
    risk_rows = ctx.get("risks", {}).get("risk_rows", []) or []
    rows = []
    if risk_rows:
        for row in risk_rows[:limit]:
            rows.append(
                {
                    "Risk": row.get("risk"),
                    "Why it matters": row.get("explanation"),
                    "Model impact": row.get("model_line"),
                    "Review action": row.get("review_action"),
                }
            )
    else:
        for risk in _top_three_risks(ctx)[:limit]:
            rows.append(
                {
                    "Risk": risk,
                    "Why it matters": "Manual review required; the dashboard did not extract a clean risk explanation.",
                    "Model impact": "Scenario probability / WACC / margin of safety",
                    "Review action": "Review source filing language before sizing.",
                }
            )
    return pd.DataFrame(rows)


def _clean_classification(value) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nan", "inf", "-inf"}:
        return "Unknown"
    if "unknown" in text.lower() or "insufficient" in text.lower():
        return "Unknown"
    return text


def _data_confidence(ctx: dict) -> tuple[str, str, str]:
    dataset = ctx.get("dataset", {})
    sources = set(dataset.get("sources", []))
    missing = _manual_review_items(ctx)
    if {"SEC/EDGAR", "Finviz Elite", "yfinance"}.issubset(sources) and not missing:
        return "High", "Core market, financial, and source coverage are loaded.", "supportive"
    if "SEC/EDGAR" in sources and "yfinance" in sources:
        return "Medium", "Core providers are loaded; selected items still require manual review.", "caution"
    if sources:
        return "Partial", "Some providers are loaded, but source coverage is incomplete.", "warning"
    return "Low", "Provider data is unavailable or blocked.", "negative"


def _valuation_view(ctx: dict) -> tuple[str, str, str]:
    dcf = ctx.get("base_dcf", {})
    upside = dcf.get("upside_downside_pct")
    terminal = dcf.get("terminal_value_weight_pct")
    if upside is None:
        return "Unknown", "Fair value cannot be calculated from current data.", "caution"
    if upside >= 0.2:
        view = "Undervalued"
        status = "supportive"
    elif upside >= -0.1:
        view = "Fair"
        status = "neutral"
    else:
        view = "Expensive"
        status = "warning"
    subtitle = f"Fair value gap is {fmt_percent(upside)}."
    if terminal and terminal > 0.65:
        subtitle += f" Terminal value is high at {fmt_percent(terminal)} of EV."
    return view, subtitle, status


def _snapshot_valuation_cards(ctx: dict) -> list[dict]:
    market = ctx.get("dataset", {}).get("market_data", {})
    dcf = ctx.get("base_dcf", {})
    try:
        sotp = get_active_sotp(ctx, "Base Case")
    except Exception:
        segments = build_default_segment_data(ctx.get("historicals"), ctx.get("dataset", {}), ctx.get("base_assumptions", {}))
        sotp = run_sotp(
            segments,
            market,
            ctx.get("base_assumptions", {}),
            scenario="Base Case",
            dcf_output=dcf,
            peer_multiples=ctx.get("peer_df"),
            sector=ctx.get("dataset", {}).get("sector"),
        )
    current_multiples = calculate_current_multiples(ctx.get("historicals"), market)
    peer_medians, _warnings = peer_median_multiples(ctx.get("peer_df"), ctx.get("dataset", {}).get("sector"), ctx.get("dataset", {}).get("industry"))
    current_ev_ocf = current_multiples.get("EV/OCF")
    peer_ev_ocf = peer_medians.get("EV/OCF")
    premium = current_ev_ocf / peer_ev_ocf - 1 if current_ev_ocf is not None and peer_ev_ocf else None
    multiple_risk = "High" if premium is not None and premium > 0.35 else "Medium" if premium is not None and premium > 0.15 else "Normal"
    premium_text = fmt_percent(premium) if premium is not None else UNAVAILABLE
    whole_status = "supportive" if ">" in str(sotp.get("whole_vs_sum")) else "warning" if "Overvalued" in str(sotp.get("whole_vs_sum")) else "neutral"
    return [
        {"title": "DCF Fair Value", "value": fmt_per_share(dcf.get("fair_value_per_share")), "subtitle": "Intrinsic value anchor.", "status": "info"},
        {"title": "SOTP Fair Value", "value": fmt_per_share(sotp.get("fair_value_per_share")), "subtitle": "Base-case sum-of-the-parts read.", "status": "info"},
        {"title": "Current Price", "value": fmt_per_share(market.get("price")), "subtitle": "Provider market price.", "status": "neutral"},
        {"title": "Whole vs Parts", "value": sotp.get("whole_vs_sum") or "Unavailable", "subtitle": sotp.get("whole_vs_sum_interpretation"), "status": whole_status},
        {"title": "Scenario Multiple Risk", "value": multiple_risk, "subtitle": f"Current EV/OCF premium vs peer: {premium_text}.", "status": "warning" if multiple_risk == "High" else "caution" if multiple_risk == "Medium" else "supportive"},
        {"title": "Peer Premium / Discount", "value": premium_text, "subtitle": "Current EV/OCF versus peer/sector reference.", "status": "warning" if premium is not None and premium > 0.25 else "supportive" if premium is not None and premium < -0.15 else "neutral"},
    ]


def _risk_level(ctx: dict) -> tuple[str, str, str]:
    risk_score = ctx.get("risks", {}).get("risk_score")
    warnings = len(ctx.get("accounting_interpretation", {}).get("warnings", []) or [])
    if risk_score is None and warnings:
        return "Manual Review", "Accounting and evidence warnings need review before sizing.", "caution"
    try:
        score = float(risk_score or 50)
    except Exception:
        score = 50
    if score >= 70:
        return "Elevated", "Risk score is high; require stronger margin of safety.", "warning"
    if warnings:
        return "Medium", "Some accounting or source issues need review.", "caution"
    return "Normal", "No major extracted risk spike in fast mode.", "neutral"


def _market_regime(ctx: dict) -> tuple[str, str, str]:
    market = ctx.get("dataset", {}).get("market_data", {})
    sma20 = market.get("sma20")
    sma50 = market.get("sma50")
    change = market.get("change")
    positives = sum(1 for value in [sma20, sma50, change] if value is not None and value > 0)
    if positives >= 2:
        return "Supportive", "Price action is above key short-term references.", "supportive"
    if positives == 1:
        return "Neutral", "Market tape is mixed; wait for confirmation.", "neutral"
    return "Risk-Off", "Momentum context is weak or unavailable.", "warning"


def _swing_view(ctx: dict) -> tuple[str, str, str]:
    market = ctx.get("dataset", {}).get("market_data", {})
    rel_vol = market.get("relative_volume")
    short_float = market.get("short_float")
    change = market.get("change")
    sma20 = market.get("sma20")
    if change is None and sma20 is None:
        return "Unknown", "Swing context requires live market and technical data.", "caution"
    if change and change > 0.08:
        return "Tradable Pullback", "Move may be extended; prefer pullback or reduced size.", "caution"
    if (sma20 or 0) > 0 and (rel_vol or 0) >= 0.8:
        return "Tradable", "Short-term trend and volume context are supportive.", "supportive"
    if short_float and short_float > 0.15:
        return "Warning", "Short interest is high; volatility can cut both ways.", "warning"
    return "Neutral", "Setup is not broken, but confirmation is incomplete.", "neutral"


def _volume_context(ctx: dict) -> tuple[str, str, str]:
    market = ctx.get("dataset", {}).get("market_data", {})
    rel_vol = market.get("relative_volume")
    change = market.get("change")
    if rel_vol is None:
        return "Unavailable", "Relative volume is missing.", "neutral"
    if rel_vol >= 1.5 and (change or 0) > 0:
        return "Accumulation", f"Relative volume is {fmt_ratio(rel_vol)} with positive price action.", "supportive"
    if rel_vol >= 1.5 and (change or 0) < 0:
        return "Distribution Warning", f"Relative volume is {fmt_ratio(rel_vol)} while price is down.", "warning"
    if rel_vol >= 0.7:
        return "Neutral", f"Relative volume is {fmt_ratio(rel_vol)}.", "neutral"
    return "Quiet", f"Relative volume is low at {fmt_ratio(rel_vol)}.", "neutral"


def _swing_volatility(ctx: dict) -> tuple[str, str, str]:
    market = ctx.get("dataset", {}).get("market_data", {})
    atr = market.get("atr")
    beta = market.get("beta")
    if atr is None and beta is None:
        return "Unavailable", "ATR and beta are missing.", "neutral"
    if (beta or 0) >= 1.8:
        return "Dangerous", f"Beta is {fmt_ratio(beta)}; sizing should be reduced.", "negative"
    if (beta or 0) >= 1.25:
        return "Elevated", f"Beta is {fmt_ratio(beta)} and ATR is {fmt_ratio(atr)}.", "warning"
    return "Tradable", f"Beta is {fmt_ratio(beta)} and ATR is {fmt_ratio(atr)}.", "supportive"


def infer_stock_profile(dataset: dict) -> str:
    sector = str(dataset.get("sector") or "").lower()
    industry = str(dataset.get("industry") or "").lower()
    text = f"{sector} {industry}"
    if any(token in text for token in ["software", "saas", "application", "cloud"]):
        return "SaaS / Software"
    if any(token in text for token in ["hardware", "industrial", "semiconductor", "electronics", "equipment"]):
        return "Industrial / Hardware"
    if any(token in text for token in ["marketplace", "internet retail", "platform"]):
        return "Marketplace / Platform"
    if any(token in text for token in ["bank", "financial", "insurance", "asset management"]):
        return "Financial"
    if any(token in text for token in ["consumer", "retail", "apparel", "restaurant"]):
        return "Consumer"
    if any(token in text for token in ["energy", "oil", "gas", "commodity", "mining"]):
        return "Energy / Commodity"
    return "General"


def _manual_review_plan_table(ctx: dict) -> pd.DataFrame:
    items = _manual_review_items(ctx)
    rows = []
    for item in items:
        rows.append(
            {
                "Data needed": item.get("Data Needed"),
                "Why missing": item.get("Reason"),
                "Model impact": item.get("Dashboard Action"),
                "Where to verify": item.get("Section to Review"),
                "Keywords": item.get("Keywords"),
                "Dashboard plan": "1. Try structured provider tags. 2. Check fallback market data. 3. Review latest filing note. 4. Mark confidence low if unresolved.",
                "Source link": item.get("Source URL"),
            }
        )
    if not rows:
        rows.append(
            {
                "Data needed": "No critical missing item",
                "Why missing": "Core providers returned enough data for cockpit use.",
                "Model impact": "Normal sensitivity review still applies.",
                "Where to verify": "Source evidence table",
                "Keywords": "revenue, cash flow, debt, capex, risk",
                "Dashboard plan": "Keep provider cache fresh and review filings for thesis-changing evidence.",
                "Source link": "Provider / SEC source tables",
            }
        )
    return pd.DataFrame(rows)


def _decision_summary(ctx: dict) -> dict:
    valuation_view, valuation_subtitle, _ = _valuation_view(ctx)
    swing_view, swing_subtitle, _ = _swing_view(ctx)
    confidence, confidence_subtitle, _ = _data_confidence(ctx)
    risk_rows = _risk_review_table(ctx, limit=3)
    contradicting = [
        f"{row.get('Risk')}: {row.get('Why it matters')}"
        for row in risk_rows.to_dict("records")
        if row.get("Risk")
    ]
    return {
        "what_matters": [
            f"Investment view is {ctx['scoring'].get('recommendation') or 'Unknown'} while valuation reads {valuation_view}.",
            f"Swing view is {swing_view}. {swing_subtitle}",
            ctx.get("reverse", {}).get("interpretation") or "Reverse DCF benchmark unavailable.",
        ],
        "supporting": _top_three_drivers(ctx)[:2],
        "contradicting": contradicting or _top_three_risks(ctx)[:3],
        "manual_review": [row.get("Data Needed") for row in _manual_review_items(ctx)] or [confidence_subtitle],
        "next_action": "Use a starter/watchlist posture until valuation, moat evidence, and manual-review items support higher conviction.",
    }


def _tearsheet_summary(ctx: dict) -> dict:
    valuation_view, valuation_subtitle, _ = _valuation_view(ctx)
    swing_view, swing_subtitle, _ = _swing_view(ctx)
    confidence, confidence_subtitle, _ = _data_confidence(ctx)
    return {
        "decision": ctx.get("scoring", {}).get("recommendation") or "Unknown",
        "valuation": f"{valuation_view}: {valuation_subtitle}",
        "swing": f"{swing_view}: {swing_subtitle}",
        "confidence": f"{confidence}: {confidence_subtitle}",
        "next_action": "Review assumptions, reverse DCF, moat evidence, and data-quality plan before sizing.",
    }


def _top_clause_impacts(clauses: pd.DataFrame) -> pd.DataFrame:
    columns = ["topic", "model_line_affected", "direction", "confidence", "suggested_assumption_change"]
    if clauses is None or clauses.empty:
        return pd.DataFrame(columns=["Topic", "Model Impact", "Direction", "Confidence", "Action"])
    available = [column for column in columns if column in clauses]
    frame = clauses[available].head(3).copy()
    rename = {
        "topic": "Topic",
        "model_line_affected": "Model Impact",
        "direction": "Direction",
        "confidence": "Confidence",
        "suggested_assumption_change": "Action",
    }
    return frame.rename(columns=rename)


def _first_sentences(text: str | None, limit: int = 3) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return UNAVAILABLE
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    return " ".join(sentences[:limit])


def _company_story(ctx: dict) -> None:
    dataset = ctx["dataset"]
    profile = ctx.get("accounting_interpretation", {}).get("business_profile", {})
    management = ctx.get("management", {})
    moat = ctx.get("moat", {})
    company = dataset.get("company") or dataset.get("ticker")
    description = dataset.get("company_description")
    sector = dataset.get("sector") or UNAVAILABLE
    industry = dataset.get("industry") or UNAVAILABLE

    st.caption("Company Story: what the company does, how the business model works, product/economic profile, and management or founder evidence.")
    metric_row(
        [
            ("Company", company, "text"),
            ("Sector", sector, "text"),
            ("Industry", industry, "text"),
            ("Business Model", profile.get("business_model"), "text"),
            ("Asset Intensity", profile.get("asset_intensity"), "text"),
        ]
    )

    c1, c2 = st.columns([0.58, 0.42])
    with c1:
        st.subheader("Business and Products")
        st.write(_first_sentences(description, limit=4))
        story_rows = pd.DataFrame(
            [
                {"Topic": "Business model", "Read": profile.get("business_model", UNAVAILABLE), "Evidence": "; ".join(profile.get("evidence", [])[:2]) or "Manual review required"},
                {"Topic": "Products / offering", "Read": industry, "Evidence": "Derived from company profile and industry classification."},
                {"Topic": "Economic profile", "Read": profile.get("margin_profile", UNAVAILABLE), "Evidence": f"CAPEX profile: {profile.get('capex_profile', UNAVAILABLE)}"},
                {"Topic": "Working capital", "Read": profile.get("working_capital_profile", UNAVAILABLE), "Evidence": "Based on industry and filing-clause signals."},
                {"Topic": "Moat", "Read": moat.get("classification", UNAVAILABLE), "Evidence": moat.get("terminal_value_implication", UNAVAILABLE)},
            ]
        )
        show_table(story_rows, "Company story unavailable.")

    with c2:
        st.subheader("Management / Founder Story")
        st.write(management.get("summary") or "Management story unavailable. Load SEC evidence for deeper founder, board, and governance context.")
        metric_row(
            [
                ("Style", management.get("style"), "text"),
                ("Score", management.get("management_score"), "score"),
                ("Confidence", profile.get("confidence"), "text"),
            ]
        )
        strengths = management.get("strengths") or []
        if strengths:
            _mini_list("Founder / management evidence", strengths)
        else:
            _mini_list("Founder / management evidence", ["No founder-specific evidence detected in loaded data.", "Load SEC evidence for fuller management context."])
        show_warnings(management.get("red_flags", []))

    with st.expander("Source Context"):
        show_table(
            pd.DataFrame(
                [
                    {"Field": "Ticker", "Value": dataset.get("ticker"), "Source": "User input / providers"},
                    {"Field": "Company name", "Value": dataset.get("company"), "Source": "SEC / Finviz / yfinance"},
                    {"Field": "Description", "Value": _first_sentences(description, limit=2), "Source": "yfinance profile"},
                    {"Field": "CIK", "Value": dataset.get("cik"), "Source": "SEC"},
                    {"Field": "Evidence mode", "Value": "Loaded" if dataset.get("evidence_loaded") else "Fast mode", "Source": "Dashboard state"},
                ]
            ),
            "Source context unavailable.",
        )


def _mini_list(title: str, items: list[str]) -> None:
    lines = [f"{index}. {item}" for index, item in enumerate(items[:3], start=1)]
    body = "<br>".join(lines) if lines else UNAVAILABLE
    st.markdown(f'<div class="pa-box"><div class="pa-box-title">{title}</div>{body}</div>', unsafe_allow_html=True)


def _assumption_evidence_table(assumptions: dict) -> pd.DataFrame:
    important = [
        "revenue_cagr",
        "gross_margin",
        "operating_margin",
        "nopat_margin",
        "tax_rate",
        "ocf_margin",
        "maintenance_capex_pct_revenue",
        "growth_capex_pct_revenue",
        "working_capital_pct_revenue",
        "sbc_pct_revenue",
        "diluted_share_growth",
        "wacc",
        "terminal_growth",
        "terminal_multiple",
        "margin_of_safety",
    ]
    return pd.DataFrame(
        [
            {
                "assumption": key,
                "base value": assumptions.get(key),
                "user value": assumptions.get(key),
                "source / evidence": "Historical financials, market data, or user slider",
                "confidence": "Medium",
                "linked clause": "",
            }
            for key in important
        ]
    )


def _accounting_assumption_flags(interpretation: dict, assumptions: dict) -> pd.DataFrame:
    if not interpretation:
        return pd.DataFrame()
    capex = interpretation.get("capex", {})
    ocf = interpretation.get("ocf", {})
    nopat = interpretation.get("nopat", {})
    da = interpretation.get("depreciation_amortization", {})
    rows = [
        {
            "assumption": "maintenance_capex_pct_revenue",
            "current value": assumptions.get("maintenance_capex_pct_revenue"),
            "interpretation": da.get("reason"),
            "suggested review": f"{da.get('recommended_maintenance_capex_method')} before using D&A as proxy",
            "confidence": da.get("confidence"),
        },
        {
            "assumption": "growth_capex_pct_revenue",
            "current value": assumptions.get("growth_capex_pct_revenue"),
            "interpretation": capex.get("classification"),
            "suggested review": "; ".join(capex.get("dcf_implications", [])[:2]),
            "confidence": capex.get("confidence"),
        },
        {
            "assumption": "ocf_margin",
            "current value": assumptions.get("ocf_margin"),
            "interpretation": f"{ocf.get('quality')} OCF quality",
            "suggested review": "; ".join(ocf.get("adjusted_ocf_suggestions", [])[:2]) or "Keep user-confirmed assumption.",
            "confidence": ocf.get("confidence"),
        },
        {
            "assumption": "nopat_margin",
            "current value": assumptions.get("nopat_margin"),
            "interpretation": f"{nopat.get('quality')} NOPAT quality",
            "suggested review": "; ".join(nopat.get("adjustments", [])[:2]) or "Keep normalized margin under sensitivity review.",
            "confidence": nopat.get("confidence"),
        },
    ]
    return pd.DataFrame(rows)


def _accounting_reality_check(ctx: dict, expanded: bool = True) -> None:
    interpretation = ctx.get("accounting_interpretation") or {}
    cards = interpretation.get("cards", {})
    with st.expander("Accounting Interpretation / Economic Reality Check", expanded=expanded):
        st.caption(
            "Reported D&A, OCF, CAPEX, NOPAT, and FCF are treated as inputs that need business-model, industry, and clause interpretation before valuation changes."
        )
        metric_row(
            [
                ("D&A Reliability", cards.get("D&A Reliability as Maintenance CAPEX Proxy"), "text"),
                ("OCF Quality", cards.get("OCF Quality"), "text"),
                ("CAPEX Classification", cards.get("CAPEX Classification"), "text"),
                ("NOPAT Quality", cards.get("NOPAT Quality"), "text"),
                ("Main Distortion", cards.get("Main Accounting Distortion"), "text"),
            ]
        )
        st.caption(f"Valuation confidence from accounting quality: {interpretation.get('valuation_confidence', 'Medium')}")
        show_warnings(interpretation.get("warnings", []))
        show_table(
            build_accounting_interpretation_table(interpretation, ctx.get("historicals")),
            "Accounting interpretation unavailable.",
        )


def _accounting_reality_compact(ctx: dict) -> None:
    interpretation = ctx.get("accounting_interpretation") or {}
    cards = interpretation.get("cards", {})
    st.markdown('<div class="pa-section-title">Accounting Reality Check</div>', unsafe_allow_html=True)
    st.caption(
        "Reported D&A, OCF, CAPEX, NOPAT, and FCF are inputs that need business-model, industry, and clause interpretation before valuation changes."
    )
    metric_row(
        [
            ("D&A Proxy", cards.get("D&A Reliability as Maintenance CAPEX Proxy"), "text"),
            ("OCF Quality", cards.get("OCF Quality"), "text"),
            ("CAPEX", cards.get("CAPEX Classification"), "text"),
        ]
    )
    metric_row(
        [
            ("NOPAT Quality", cards.get("NOPAT Quality"), "text"),
            ("Distortion", cards.get("Main Accounting Distortion"), "text"),
            ("Confidence Impact", interpretation.get("valuation_confidence"), "text"),
        ]
    )
    for warning in (interpretation.get("warnings") or [])[:3]:
        _notice(str(warning), "warning")
    with st.expander("Show full accounting interpretation"):
        show_table(
            build_accounting_interpretation_table(interpretation, ctx.get("historicals")),
            "Accounting interpretation unavailable.",
        )


def _style_financial_model_table(df: pd.DataFrame):
    if df is None or df.empty:
        return df

    def cell_style(_value, column: str):
        text = str(column)
        if text == "Line Item":
            return "font-weight: 700; color: #0f172a;"
        if "YTD" in text or "LTM" in text:
            return "background-color: #fff7d6;"
        if text.endswith("E"):
            return "background-color: #fff7d6;"
        if text.endswith("F"):
            return "background-color: #eaf4ff;"
        if text.endswith("A"):
            return "background-color: #f8fafc;"
        return ""

    return df.style.apply(lambda row: [cell_style(value, column) for column, value in row.items()], axis=1)


def _reverse_dcf_comparison_table(reverse: dict, assumptions: dict) -> pd.DataFrame:
    pairs = [
        ("Revenue CAGR", reverse.get("implied_revenue_cagr"), assumptions.get("revenue_cagr"), "pct"),
        ("NOPAT Margin", reverse.get("implied_nopat_margin"), assumptions.get("nopat_margin"), "pct"),
        ("OCF Margin", reverse.get("implied_ocf_margin"), assumptions.get("ocf_margin"), "pct"),
        ("Terminal Growth", reverse.get("implied_terminal_growth"), assumptions.get("terminal_growth"), "pct"),
        ("Terminal Multiple", reverse.get("implied_terminal_multiple"), assumptions.get("terminal_multiple"), "multiple"),
        ("WACC", reverse.get("implied_wacc"), assumptions.get("wacc"), "pct"),
    ]
    rows = []
    for metric, market_value, user_value, kind in pairs:
        gap = None
        if isinstance(market_value, (int, float)) and isinstance(user_value, (int, float)):
            gap = market_value - user_value
        rows.append(
            {
                "Metric": metric,
                "Market Implied": market_value,
                "Your Base Case": user_value,
                "Gap": "Market higher" if gap and gap > 0 else "Market lower" if gap and gap < 0 else "In line / unavailable",
                "Format": kind,
            }
        )
    return pd.DataFrame(rows).drop(columns=["Format"])


def _peer_summary_table(peer_df: pd.DataFrame) -> pd.DataFrame:
    if peer_df is None or peer_df.empty:
        return pd.DataFrame()
    frame = peer_df.copy()
    wanted = [
        "ticker",
        "ev_sales",
        "ev_ebitda",
        "ev_fcf",
        "fcf_yield",
        "revenue_growth",
        "gross_margin",
        "operating_margin",
        "ocf_margin",
    ]
    available = [column for column in wanted if column in frame]
    summary = frame[available].copy()
    numeric_cols = summary.select_dtypes(include=["number"]).columns
    if len(numeric_cols):
        median = {column: summary[column].median() if column in numeric_cols else "Peer median" for column in summary.columns}
        if "ticker" in summary:
            median["ticker"] = "Peer median"
        summary = pd.concat([summary, pd.DataFrame([median])], ignore_index=True)
    return summary


def _dcf_forecast_output_table(dcf_output: dict, assumptions: dict) -> pd.DataFrame:
    forecast = dcf_output.get("forecast_table", pd.DataFrame())
    if forecast is None or forecast.empty:
        return pd.DataFrame()
    rows = [
        {"Metric": "Revenue"},
        {"Metric": "Revenue Growth %"},
        {"Metric": "EBIT Margin %"},
        {"Metric": "NOPAT"},
        {"Metric": "OCF"},
        {"Metric": "Maintenance CAPEX"},
        {"Metric": "Growth CAPEX"},
        {"Metric": "FCFF / FCF"},
        {"Metric": "Discount Factor"},
        {"Metric": "Discounted FCF"},
        {"Metric": "Terminal Value"},
        {"Metric": "Discounted Terminal Value"},
    ]
    prior_revenue = None
    for _, row in forecast.iterrows():
        year = int(row.get("Year") or 0)
        suffix = "E" if year == 1 else "F"
        column = f"FY{year}{suffix}"
        revenue = row.get("Revenue")
        revenue_growth = (revenue / prior_revenue - 1) if prior_revenue else assumptions.get("revenue_cagr")
        prior_revenue = revenue
        discount_factor = 1 / ((1 + float(assumptions.get("wacc", 0.095))) ** year)
        rows[0][column] = revenue
        rows[1][column] = revenue_growth
        rows[2][column] = assumptions.get("operating_margin")
        rows[3][column] = row.get("NOPAT")
        rows[4][column] = row.get("OCF")
        rows[5][column] = row.get("Maintenance CAPEX")
        rows[6][column] = row.get("Growth CAPEX")
        rows[7][column] = row.get("FCF")
        rows[8][column] = discount_factor
        rows[9][column] = row.get("PV FCF")
    rows[10]["Terminal"] = dcf_output.get("terminal_value")
    rows[11]["Terminal"] = dcf_output.get("discounted_terminal_value")
    return pd.DataFrame(rows)


ASSUMPTION_GROUPS = {
    "Growth": "These assumptions determine the size of the forecast revenue base.",
    "Margins & OPEX": "These assumptions determine how much revenue converts into operating profit and NOPAT.",
    "Cash Conversion": "These assumptions determine whether accounting profit converts into operating cash flow.",
    "Reinvestment": "These assumptions determine how much cash must be reinvested before owners get FCF.",
    "Terminal Value": "These assumptions drive the discount rate and the value after the explicit forecast period.",
    "Dilution": "These assumptions determine how enterprise value converts into per-share value.",
}


ASSUMPTION_METADATA = {
    "forecast_years": {
        "label": "Forecast Years",
        "unit": "years",
        "group": "Growth",
        "description": "Number of explicit years forecast before terminal value is calculated.",
        "model_line": "Forecast Period",
        "affects": ["Revenue", "FCF", "Terminal Value", "Fair Value"],
        "default_source": "Model framework default.",
        "reasonable_range": "Usually 5-10 years. Longer periods need unusually durable visibility.",
        "warning": "Long forecast periods can hide aggressive terminal assumptions.",
        "min": 5,
        "max": 10,
        "step": 1,
        "source": "Scenario-based",
    },
    "revenue_cagr": {
        "label": "Revenue CAGR",
        "unit": "percent",
        "group": "Growth",
        "description": "Annualized revenue growth during the explicit forecast period.",
        "model_line": "Revenue",
        "affects": ["Revenue", "Gross Profit", "OPEX", "OCF", "NOPAT", "FCF", "Terminal Value"],
        "default_source": "Historical growth, guidance, backlog, and analyst scenario.",
        "reasonable_range": "Low-growth mature firms: 0%-5%; quality growth: 5%-15%; high-growth/small-cap: 15%+ only with evidence.",
        "warning": "Do not raise revenue growth without evidence such as backlog, pricing, volume, customer growth, capacity, or market expansion.",
        "min": -0.20,
        "max": 0.60,
        "step": 0.005,
        "source": "Scenario-based",
    },
    "gross_margin": {
        "label": "Gross Margin",
        "unit": "percent",
        "group": "Margins & OPEX",
        "description": "Gross profit as a percentage of revenue.",
        "model_line": "Gross Profit",
        "affects": ["Gross Profit", "EBIT", "NOPAT", "FCF"],
        "default_source": "Historical gross margin and peer margin comparison.",
        "reasonable_range": "Anchor to history, segment mix, pricing, input costs, and peers.",
        "warning": "Margin expansion requires evidence from mix shift, pricing, automation, cost control, or scale.",
        "min": -0.20,
        "max": 0.90,
        "step": 0.005,
        "source": "Calculated",
    },
    "opex_pct_revenue": {
        "label": "OPEX % Revenue",
        "unit": "percent",
        "group": "Margins & OPEX",
        "description": "Operating expenses as a percentage of revenue. Lower OPEX % usually means better operating leverage.",
        "model_line": "OPEX",
        "affects": ["EBIT", "NOPAT", "FCF", "Operating Leverage"],
        "default_source": "Reported OPEX lines or calculated as Gross Profit minus EBIT.",
        "reasonable_range": "Compare with history and peer OPEX ratios.",
        "warning": "Do not lower OPEX % unless evidence supports sales efficiency, G&A leverage, lower R&D intensity, restructuring, or scale benefits.",
        "min": 0.0,
        "max": 0.90,
        "step": 0.005,
        "source": "Calculated",
    },
    "tax_rate": {
        "label": "Tax Rate",
        "unit": "percent",
        "group": "Margins & OPEX",
        "description": "Cash tax rate applied to operating profit in the NOPAT bridge.",
        "model_line": "NOPAT",
        "affects": ["NOPAT", "FCF", "Fair Value"],
        "default_source": "Model default and reported effective tax rate context.",
        "reasonable_range": "Usually 15%-30% unless loss carryforwards, geography, or credits justify a different rate.",
        "warning": "Low tax rates should be tied to tax assets or jurisdiction mix.",
        "min": 0.0,
        "max": 0.40,
        "step": 0.005,
        "source": "Estimated",
    },
    "nopat_margin": {
        "label": "Direct NOPAT Margin",
        "unit": "percent",
        "group": "Margins & OPEX",
        "description": "Direct override for normalized after-tax operating profit as a percentage of revenue.",
        "model_line": "NOPAT",
        "affects": ["NOPAT", "FCF", "Fair Value"],
        "default_source": "EBIT margin after tax unless direct override is active.",
        "reasonable_range": "Should reconcile to gross margin, OPEX intensity, tax rate, and peers.",
        "warning": "Direct NOPAT overrides can obscure whether the change came from gross margin, OPEX, or tax rate.",
        "min": -0.20,
        "max": 0.60,
        "step": 0.005,
        "source": "Calculated",
    },
    "ocf_margin": {
        "label": "OCF Margin",
        "unit": "percent",
        "group": "Cash Conversion",
        "description": "Operating cash flow as a percentage of revenue. This captures cash conversion quality.",
        "model_line": "OCF",
        "affects": ["OCF", "FCF", "DCF Fair Value"],
        "default_source": "Reported OCF divided by revenue, adjusted for working-capital distortions if available.",
        "reasonable_range": "Check against historical OCF margin, working capital behavior, and peers.",
        "warning": "OCF can be temporarily distorted by receivables, inventory, payables, deferred revenue, or one-time cash items.",
        "min": -0.20,
        "max": 0.60,
        "step": 0.005,
        "source": "Calculated",
    },
    "working_capital_pct_revenue": {
        "label": "Working Capital % Revenue",
        "unit": "percent",
        "group": "Cash Conversion",
        "description": "Cash tied up or released through receivables, inventory, payables, deferred revenue, and contract assets.",
        "model_line": "Working Capital",
        "affects": ["OCF", "FCF"],
        "default_source": "Historical working capital changes and business model needs.",
        "reasonable_range": "Negative working capital can support OCF; inventory-heavy growth can reduce OCF.",
        "warning": "Backlog growth may require inventory or receivables investment before cash is collected.",
        "min": -0.10,
        "max": 0.20,
        "step": 0.005,
        "source": "Estimated",
    },
    "maintenance_capex_pct_revenue": {
        "label": "Maintenance CAPEX % Revenue",
        "unit": "percent",
        "group": "Reinvestment",
        "description": "CAPEX required to maintain current operations.",
        "model_line": "Maintenance CAPEX",
        "affects": ["FCF", "Normalized Cash Earnings"],
        "default_source": "Company disclosure or D&A proxy if undisclosed.",
        "reasonable_range": "Asset-light software may be low; manufacturing/industrial businesses may be much higher.",
        "warning": "If undisclosed, a D&A proxy may be wrong. Review asset intensity, industry, and CAPEX notes.",
        "min": 0.0,
        "max": 0.25,
        "step": 0.005,
        "source": "Estimated",
    },
    "growth_capex_pct_revenue": {
        "label": "Growth CAPEX % Revenue",
        "unit": "percent",
        "group": "Reinvestment",
        "description": "Reinvestment intended to create future capacity, revenue, or efficiency.",
        "model_line": "Growth CAPEX",
        "affects": ["FCF", "Future Revenue", "Margins"],
        "default_source": "CAPEX notes, capacity expansion, backlog, new facilities, or technology investment.",
        "reasonable_range": "Can be temporarily elevated during expansion cycles.",
        "warning": "Growth CAPEX may reduce near-term FCF but improve future revenue or margins if execution succeeds.",
        "min": 0.0,
        "max": 0.35,
        "step": 0.005,
        "source": "Estimated",
    },
    "wacc": {
        "label": "WACC",
        "unit": "percent",
        "group": "Terminal Value",
        "description": "Discount rate used to convert future cash flows into present value.",
        "model_line": "Discount Rate",
        "affects": ["Enterprise Value", "Fair Value Per Share"],
        "default_source": "Risk-free rate + equity risk premium + company-specific risk.",
        "reasonable_range": "Usually 8%-12%; higher for small caps, cyclicals, leverage, weak visibility, or governance risk.",
        "warning": "A small WACC change can have a large valuation impact.",
        "min": 0.04,
        "max": 0.20,
        "step": 0.005,
        "source": "Estimated",
    },
    "terminal_growth": {
        "label": "Terminal Growth",
        "unit": "percent",
        "group": "Terminal Value",
        "description": "Long-term growth rate after the explicit forecast period.",
        "model_line": "Terminal Value",
        "affects": ["Terminal Value", "Fair Value Per Share"],
        "default_source": "Long-run industry growth, moat, inflation, and maturity profile.",
        "reasonable_range": "Usually 0%-3%. Higher requires strong moat and durable growth evidence.",
        "warning": "Aggressive terminal growth can overstate valuation.",
        "min": -0.02,
        "max": 0.06,
        "step": 0.005,
        "source": "Scenario-based",
    },
    "terminal_multiple": {
        "label": "Terminal Multiple",
        "unit": "multiple",
        "group": "Terminal Value",
        "description": "Exit multiple applied to final-year cash flow.",
        "model_line": "Terminal Value",
        "affects": ["Terminal Value", "Fair Value Per Share"],
        "default_source": "Peer multiples, moat, growth durability, profitability, and capital intensity.",
        "reasonable_range": "Weak/no-growth: <=10x; low-growth: ~12x; stronger quality/growth: ~15x; higher only with evidence.",
        "warning": "If terminal value is a large share of EV, this assumption is critical.",
        "min": 4.0,
        "max": 35.0,
        "step": 0.5,
        "source": "Scenario-based",
    },
    "sbc_pct_revenue": {
        "label": "SBC % Revenue",
        "unit": "percent",
        "group": "Dilution",
        "description": "Stock-based compensation as a percentage of revenue.",
        "model_line": "SBC",
        "affects": ["Dilution", "Owner Earnings Quality"],
        "default_source": "Reported SBC divided by revenue.",
        "reasonable_range": "High-growth software may be higher; mature firms should trend lower.",
        "warning": "High SBC can transfer value from shareholders to employees even when FCF looks strong.",
        "min": 0.0,
        "max": 0.30,
        "step": 0.005,
        "source": "Calculated",
    },
    "diluted_share_growth": {
        "label": "Diluted Share Growth",
        "unit": "percent",
        "group": "Dilution",
        "description": "Annual increase or decrease in diluted shares during the forecast period.",
        "model_line": "Diluted Shares",
        "affects": ["Fair Value Per Share", "Owner Dilution"],
        "default_source": "Recent diluted share count trend.",
        "reasonable_range": "Buybacks can be negative; heavy SBC can make this positive.",
        "warning": "Per-share value can fall even if enterprise value rises when dilution is high.",
        "min": -0.10,
        "max": 0.20,
        "step": 0.005,
        "source": "Calculated",
    },
    "diluted_shares": {
        "label": "Diluted Shares",
        "unit": "shares",
        "group": "Dilution",
        "description": "Share count used to convert equity value into fair value per share.",
        "model_line": "Fair Value Per Share",
        "affects": ["Fair Value Per Share"],
        "default_source": "Reported diluted shares or market provider shares outstanding.",
        "reasonable_range": "Should tie to latest diluted share count, not basic shares, when available.",
        "warning": "Wrong share count directly distorts per-share value.",
        "source": "Reported",
    },
    "margin_of_safety": {
        "label": "Margin of Safety",
        "unit": "percent",
        "group": "Terminal Value",
        "description": "Discount applied to fair value to calculate the buy-zone price.",
        "model_line": "Buy Zone",
        "affects": ["Buy Price", "Decision Readout"],
        "default_source": "Model default based on uncertainty.",
        "reasonable_range": "Usually 20%-40%; higher for weak confidence or high terminal-value dependence.",
        "warning": "A low margin of safety assumes high confidence in the model.",
        "min": 0.0,
        "max": 0.60,
        "step": 0.05,
        "source": "User-edited",
    },
}


ASSUMPTION_KEYS = list(ASSUMPTION_METADATA.keys())


VALUATION_BASIS_OPTIONS = {
    "OCF-based FCF": {
        "mode": "FCF",
        "description": "Uses operating cash flow minus maintenance and growth CAPEX. Best when OCF quality is reliable.",
    },
    "NOPAT bridge": {
        "mode": "FCFF",
        "description": "Uses normalized operating profit after tax plus non-cash items minus reinvestment. Best for normalized economic profitability.",
    },
    "Adjusted OCF-based FCF": {
        "mode": "FCF",
        "description": "Uses the OCF-based method while documenting adjustments for one-time or timing distortions. Best when reported OCF is noisy.",
    },
}


def _assumption_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded_value(value, minimum, maximum):
    number = _assumption_float(value, minimum)
    return min(max(number, minimum), maximum)


def _assumption_unit(key: str) -> str:
    return ASSUMPTION_METADATA.get(key, {}).get("unit", "decimal")


def format_assumption_value(value, unit: str) -> str:
    if value is None:
        return UNAVAILABLE
    if unit == "percent":
        return fmt_percent(value, 1)
    if unit == "multiple":
        return fmt_multiple(value, 1)
    if unit == "per_share":
        return fmt_per_share(value)
    if unit == "money":
        return fmt_money(value)
    if unit == "years":
        try:
            return f"{int(float(value))} years"
        except (TypeError, ValueError):
            return UNAVAILABLE
    if unit == "shares":
        return fmt_shares(value)
    if unit == "decimal":
        return fmt_ratio(value, 2)
    return str(value)


def _parse_assumption_value(value, unit: str):
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number / 100 if unit == "percent" and abs(number) > 1.5 else number
    text = str(value).strip().replace(",", "").replace("$", "")
    multiplier = 1.0
    if text.endswith("%"):
        text = text[:-1]
        multiplier = 0.01
    elif text.lower().endswith("x"):
        text = text[:-1]
    elif text.lower().endswith("b"):
        text = text[:-1]
        multiplier = 1e9
    elif text.lower().endswith("m"):
        text = text[:-1]
        multiplier = 1e6
    elif text.lower().endswith("k"):
        text = text[:-1]
        multiplier = 1e3
    elif unit == "percent":
        multiplier = 0.01
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def _assumption_source(key: str, scenario_scope: str, current_value, base_value) -> str:
    if abs(_assumption_float(current_value) - _assumption_float(base_value)) > 0.000001:
        return "User-edited" if scenario_scope == "User Case" else "Scenario-based"
    return ASSUMPTION_METADATA.get(key, {}).get("source", "Scenario-based")


def _derive_opex_pct(assumptions: dict) -> float:
    if assumptions.get("opex_pct_revenue") is not None:
        return _assumption_float(assumptions.get("opex_pct_revenue"))
    gross = _assumption_float(assumptions.get("gross_margin"), 0.45)
    operating = _assumption_float(assumptions.get("operating_margin"), 0.15)
    return max(gross - operating, 0.0)


def _normalize_assumption_bridge(assumptions: dict, direct_nopat_override: bool = False) -> dict:
    normalized = dict(assumptions)
    gross = _assumption_float(normalized.get("gross_margin"), 0.45)
    opex = _assumption_float(normalized.get("opex_pct_revenue"), _derive_opex_pct(normalized))
    tax_rate = _assumption_float(normalized.get("tax_rate"), 0.21)
    operating_margin = gross - opex
    normalized["opex_pct_revenue"] = opex
    normalized["operating_margin"] = operating_margin
    if not direct_nopat_override:
        normalized["nopat_margin"] = operating_margin * (1 - tax_rate)
    return normalized


def _build_assumption_scenarios(base: dict, user: dict | None = None) -> dict:
    base_case = _normalize_assumption_bridge(base)
    bear = dict(base_case)
    bull = dict(base_case)
    for key, delta in {
        "revenue_cagr": -0.03,
        "gross_margin": -0.03,
        "ocf_margin": -0.03,
        "growth_capex_pct_revenue": 0.02,
        "wacc": 0.015,
        "terminal_growth": -0.01,
        "terminal_multiple": -2.0,
    }.items():
        bear[key] = _assumption_float(bear.get(key)) + delta
    for key, delta in {
        "revenue_cagr": 0.05,
        "gross_margin": 0.03,
        "ocf_margin": 0.03,
        "growth_capex_pct_revenue": -0.01,
        "wacc": -0.01,
        "terminal_growth": 0.01,
        "terminal_multiple": 2.0,
    }.items():
        bull[key] = _assumption_float(bull.get(key)) + delta
    return {
        "Base Case": base_case,
        "Bear Case": _normalize_assumption_bridge(bear),
        "Bull Case": _normalize_assumption_bridge(bull),
        "User Case": _normalize_assumption_bridge(user or base_case),
    }


def _market_implied_assumptions(reverse: dict, base: dict) -> dict:
    return {
        "revenue_cagr": reverse.get("implied_revenue_cagr"),
        "nopat_margin": reverse.get("implied_nopat_margin"),
        "ocf_margin": reverse.get("implied_ocf_margin"),
        "terminal_growth": reverse.get("implied_terminal_growth"),
        "terminal_multiple": reverse.get("implied_terminal_multiple"),
        "wacc": reverse.get("implied_wacc"),
        "gross_margin": base.get("gross_margin"),
        "opex_pct_revenue": _derive_opex_pct(base),
        "maintenance_capex_pct_revenue": base.get("maintenance_capex_pct_revenue"),
        "growth_capex_pct_revenue": base.get("growth_capex_pct_revenue"),
        "working_capital_pct_revenue": base.get("working_capital_pct_revenue"),
        "diluted_shares": base.get("diluted_shares"),
    }


def calculate_assumption_impact(base_assumptions: dict, edited_assumptions: dict, changed_key: str, historicals: pd.DataFrame, market_data: dict) -> dict:
    base_dcf = run_dcf(historicals, market_data, _normalize_assumption_bridge(base_assumptions))
    single_change = _normalize_assumption_bridge({**base_assumptions, changed_key: edited_assumptions.get(changed_key)})
    changed_dcf = run_dcf(historicals, market_data, single_change)
    base_fv = base_dcf.get("fair_value_per_share")
    new_fv = changed_dcf.get("fair_value_per_share")
    delta = (new_fv - base_fv) if base_fv is not None and new_fv is not None else None
    pct = (delta / base_fv) if delta is not None and base_fv else None
    return {"fair_value_delta": delta, "fair_value_delta_pct": pct, "new_fair_value": new_fv}


def _scenario_comparison_table(scenarios: dict, reverse: dict, user_assumptions: dict) -> pd.DataFrame:
    market_case = _market_implied_assumptions(reverse or {}, scenarios["Base Case"])
    rows = []
    for key in [
        "revenue_cagr",
        "gross_margin",
        "opex_pct_revenue",
        "nopat_margin",
        "ocf_margin",
        "maintenance_capex_pct_revenue",
        "growth_capex_pct_revenue",
        "working_capital_pct_revenue",
        "wacc",
        "terminal_growth",
        "terminal_multiple",
        "diluted_shares",
    ]:
        meta = ASSUMPTION_METADATA[key]
        unit = meta["unit"]
        rows.append(
            {
                "Assumption": meta["label"],
                "Bear": format_assumption_value(scenarios["Bear Case"].get(key), unit),
                "Base": format_assumption_value(scenarios["Base Case"].get(key), unit),
                "Bull": format_assumption_value(scenarios["Bull Case"].get(key), unit),
                "User": format_assumption_value(user_assumptions.get(key), unit),
                "Market-Implied": format_assumption_value(market_case.get(key), unit),
            }
        )
    return pd.DataFrame(rows)


def _profile_assumption_note(profile: str, key: str) -> str:
    if profile == "SaaS / Software":
        notes = {
            "opex_pct_revenue": "For software, OPEX % revenue mainly reflects sales efficiency, R&D scale, and G&A leverage.",
            "ocf_margin": "For software, OCF margin can be supported by upfront billing and deferred revenue. Check timing distortions.",
        }
    elif profile == "Industrial / Hardware":
        notes = {
            "opex_pct_revenue": "For industrial/hardware firms, OPEX may improve with scale, but gross margin and CAPEX efficiency often matter more.",
            "ocf_margin": "For industrial/hardware firms, OCF can be pressured by inventory build, receivables, and backlog conversion timing.",
            "growth_capex_pct_revenue": "Growth CAPEX may be needed for capacity expansion, equipment, automation, or backlog conversion.",
        }
    elif profile == "Marketplace / Platform":
        notes = {
            "opex_pct_revenue": "For platforms, OPEX depends on marketing intensity, platform investment, and trust/safety costs.",
            "ocf_margin": "For platforms, OCF depends on take rate, payment timing, and working capital structure.",
        }
    else:
        notes = {}
    return notes.get(key, "Use company history, peer economics, and filing evidence before changing this assumption.")


def validate_assumption_ranges(assumptions: dict, historicals: pd.DataFrame | None = None, peer_data=None, moat_analysis=None) -> list[dict]:
    warnings = []
    revenue_cagr = _assumption_float(assumptions.get("revenue_cagr"))
    terminal_growth = _assumption_float(assumptions.get("terminal_growth"))
    opex = _assumption_float(assumptions.get("opex_pct_revenue"), _derive_opex_pct(assumptions))
    ocf_margin = _assumption_float(assumptions.get("ocf_margin"))
    growth_capex = _assumption_float(assumptions.get("growth_capex_pct_revenue"))
    wacc = _assumption_float(assumptions.get("wacc"))
    if revenue_cagr > 0.25:
        warnings.append({"Assumption": "Revenue CAGR", "Current Value": format_assumption_value(revenue_cagr, "percent"), "Severity": "High", "Reason": "Revenue CAGR above 25% requires strong evidence.", "Suggested Review": "Check backlog, pricing, volume, capacity, and market expansion evidence."})
    if terminal_growth > 0.03:
        warnings.append({"Assumption": "Terminal Growth", "Current Value": format_assumption_value(terminal_growth, "percent"), "Severity": "High", "Reason": "Terminal growth above 3% requires explicit durable moat evidence.", "Suggested Review": "Review moat score, industry maturity, and long-run GDP/inflation anchor."})
    if opex < 0:
        warnings.append({"Assumption": "OPEX % Revenue", "Current Value": format_assumption_value(opex, "percent"), "Severity": "High", "Reason": "OPEX cannot be negative in a normal operating model.", "Suggested Review": "Review gross margin and operating margin bridge."})
    if wacc < 0.06:
        warnings.append({"Assumption": "WACC", "Current Value": format_assumption_value(wacc, "percent"), "Severity": "Medium", "Reason": "Very low WACC can overstate fair value.", "Suggested Review": "Check company size, leverage, cyclicality, and visibility."})
    if wacc > 0.15:
        warnings.append({"Assumption": "WACC", "Current Value": format_assumption_value(wacc, "percent"), "Severity": "Medium", "Reason": "Very high WACC may reflect an elevated-risk case.", "Suggested Review": "Confirm risk premium and scenario intent."})
    if historicals is not None and not historicals.empty:
        if "OCF Margin" in historicals and not historicals["OCF Margin"].dropna().empty:
            historical_max = historicals["OCF Margin"].dropna().max()
            if ocf_margin > historical_max + 0.05:
                warnings.append({"Assumption": "OCF Margin", "Current Value": format_assumption_value(ocf_margin, "percent"), "Severity": "Medium", "Reason": "OCF margin is materially above recent history.", "Suggested Review": "Check receivables, inventory, deferred revenue, and one-time cash items."})
        if "Total CAPEX" in historicals and "Revenue" in historicals:
            capex_pct = (historicals["Total CAPEX"] / historicals["Revenue"].replace(0, pd.NA)).dropna()
            if not capex_pct.empty and growth_capex < max(float(capex_pct.median()) * 0.25, 0.005):
                warnings.append({"Assumption": "Growth CAPEX % Revenue", "Current Value": format_assumption_value(growth_capex, "percent"), "Severity": "Low", "Reason": "Growth CAPEX is well below historical CAPEX intensity.", "Suggested Review": "Confirm reinvestment needs and whether maintenance CAPEX already captures the spend."})
    return warnings


def _render_assumption_explanation(key: str, profile: str, scope: str, source: str, fair_value_impact: dict | None = None) -> None:
    meta = ASSUMPTION_METADATA[key]
    impact = " -> ".join(meta.get("affects", []))
    impact_text = ""
    if fair_value_impact and fair_value_impact.get("fair_value_delta") is not None:
        impact_text = f"Fair Value Impact: {fmt_per_share(fair_value_impact.get('fair_value_delta'))} / share ({fmt_percent(fair_value_impact.get('fair_value_delta_pct'))})."
    st.markdown(
        f"""
        <div class="pa-box">
            <div class="pa-box-title">Selected Assumption Explanation</div>
            <strong>{meta["label"]}</strong><br/>
            <span class="pa-pill">{scope}</span><span class="pa-pill">{source}</span><br/>
            <strong>What it means:</strong> {meta["description"]}<br/>
            <strong>Model impact:</strong> {impact}<br/>
            <strong>Default source:</strong> {meta["default_source"]}<br/>
            <strong>Reasonable range:</strong> {meta["reasonable_range"]}<br/>
            <strong>Profile note:</strong> {_profile_assumption_note(profile, key)}<br/>
            <strong>Warning:</strong> {meta["warning"]}<br/>
            <strong>{impact_text}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assumption_slider(
    key: str,
    current_value: float,
    base_value: float,
    min_value: float,
    max_value: float,
    step: float,
    scenario_scope: str,
):
    meta = ASSUMPTION_METADATA[key]
    unit = meta["unit"]
    delta = _assumption_float(current_value) - _assumption_float(base_value)
    delta_text = format_assumption_value(delta, unit) if unit != "percent" else f"{delta * 100:+.1f} pts"
    source = _assumption_source(key, scenario_scope, current_value, base_value)
    label = f"{meta['label']} | {scenario_scope}: {format_assumption_value(current_value, unit)} | Base: {format_assumption_value(base_value, unit)} | Delta {delta_text}"
    st.caption(f"Scope: {scenario_scope} only | Source: {source} | Affects: {' -> '.join(meta['affects'])}")
    if unit == "percent":
        slider_value = st.slider(
            label,
            min_value=min_value * 100,
            max_value=max_value * 100,
            value=_bounded_value(current_value, min_value, max_value) * 100,
            step=step * 100,
            format="%.1f%%",
            help=f"{meta['description']} Reasonable range: {meta['reasonable_range']}",
            key=f"assumption_slider_{scenario_scope}_{key}",
        )
        return slider_value / 100
    if unit == "multiple":
        return st.slider(
            label,
            min_value=float(min_value),
            max_value=float(max_value),
            value=_bounded_value(current_value, min_value, max_value),
            step=float(step),
            format="%.1fx",
            help=f"{meta['description']} Reasonable range: {meta['reasonable_range']}",
            key=f"assumption_slider_{scenario_scope}_{key}",
        )
    if unit == "years":
        return st.slider(
            label,
            min_value=int(min_value),
            max_value=int(max_value),
            value=int(_bounded_value(current_value, min_value, max_value)),
            step=int(step),
            help=f"{meta['description']} Reasonable range: {meta['reasonable_range']}",
            key=f"assumption_slider_{scenario_scope}_{key}",
        )
    if unit == "shares":
        return st.number_input(
            label,
            min_value=0.0,
            value=max(float(current_value or 0), 0.0),
            step=1_000_000.0,
            format="%.0f",
            help=f"{meta['description']} Reasonable range: {meta['reasonable_range']}",
            key=f"assumption_number_{scenario_scope}_{key}",
        )
    return st.slider(
        label,
        min_value=float(min_value),
        max_value=float(max_value),
        value=_bounded_value(current_value, min_value, max_value),
        step=float(step),
        help=f"{meta['description']} Reasonable range: {meta['reasonable_range']}",
        key=f"assumption_slider_{scenario_scope}_{key}",
    )


def _assumption_change_rows(base: dict, edited: dict, scenario_scope: str, historicals: pd.DataFrame, market: dict) -> list[dict]:
    rows = []
    for key in ASSUMPTION_KEYS:
        old_value = base.get(key)
        new_value = edited.get(key)
        if abs(_assumption_float(new_value) - _assumption_float(old_value)) <= 0.000001:
            continue
        meta = ASSUMPTION_METADATA[key]
        impact = calculate_assumption_impact(base, edited, key, historicals, market)
        rows.append(
            {
                "Active": True,
                "Scenario": scenario_scope,
                "Assumption": meta["label"],
                "Old Value": format_assumption_value(old_value, meta["unit"]),
                "New Value": format_assumption_value(new_value, meta["unit"]),
                "Delta": format_assumption_value(_assumption_float(new_value) - _assumption_float(old_value), meta["unit"]) if meta["unit"] != "percent" else f"{(_assumption_float(new_value) - _assumption_float(old_value)) * 100:+.1f} pts",
                "Source": _assumption_source(key, scenario_scope, new_value, old_value),
                "User Note": "",
                "Fair Value Impact": fmt_per_share(impact.get("fair_value_delta")),
                "Confidence": "Medium",
            }
        )
    return rows


def _apply_active_change_log(base: dict, edited: dict, log_rows: list[dict]) -> dict:
    assumptions = dict(edited)
    label_to_key = {meta["label"]: key for key, meta in ASSUMPTION_METADATA.items()}
    for row in log_rows:
        if not row.get("Active", True):
            key = row.get("Key") or label_to_key.get(row.get("Assumption"))
            if key:
                assumptions[key] = base.get(key)
            continue
        key = row.get("Key") or label_to_key.get(row.get("Assumption"))
        if not key:
            continue
        parsed = _parse_assumption_value(row.get("New Value"), _assumption_unit(key))
        if parsed is not None:
            assumptions[key] = parsed
    return _normalize_assumption_bridge(assumptions, bool(assumptions.get("use_direct_nopat_override")))


def _assumption_editor(ctx: dict) -> dict:
    base = _normalize_assumption_bridge(ctx["base_assumptions"])
    market = ctx["dataset"].get("market_data", {})
    historicals = ctx["historicals"]
    reverse = ctx.get("reverse", {})
    profile = infer_stock_profile(ctx["dataset"])
    user_state_key = f"assumption_user_case_{ctx['dataset'].get('ticker', 'default')}"
    st.session_state.setdefault(user_state_key, dict(base))

    st.subheader("DCF Assumption Workbench")
    st.caption("Adjust assumptions with clear scenario scope, source, model impact, fair-value impact, and notes. The data-fetching layer is unchanged.")

    scope_col, compare_col = st.columns([0.58, 0.42])
    with scope_col:
        scenario_scope = st.segmented_control(
            "Which case are you editing?",
            ["User Case", "Base Case", "Bull Case", "Bear Case"],
            default="User Case",
            help="Choose which valuation case your assumption changes apply to. User Case is recommended for personal adjustments. Base/Bull/Bear should only be changed if you want to redefine the scenario framework.",
        )
        scenario_scope = scenario_scope or "User Case"
    with compare_col:
        compare_to = st.selectbox("Compare assumption changes against", ["Base Case", "Market-Implied Case", "Prior User Case"], index=0)

    st.markdown(f'<span class="pa-pill pa-pill-ok">You are editing: {scenario_scope}</span> <span class="pa-pill">Compare: Current {scenario_scope} vs {compare_to}</span>', unsafe_allow_html=True)
    if scenario_scope != "User Case":
        st.warning("You are editing a core scenario. Consider using User Case unless you intentionally want to redefine the model framework.")

    scenarios = _build_assumption_scenarios(base, st.session_state[user_state_key])
    working = dict(scenarios[scenario_scope])
    prior_user_case = dict(st.session_state[user_state_key])

    preset_cols = st.columns(4)
    if preset_cols[0].button("Reset User Case to Base", key="reset_user_case_to_base"):
        st.session_state[user_state_key] = dict(scenarios["Base Case"])
        working = dict(st.session_state[user_state_key])
        st.success("User Case reset to Base Case.")
    if preset_cols[1].button("Copy Bull Case to User Case", key="copy_bull_to_user_case"):
        st.session_state[user_state_key] = dict(scenarios["Bull Case"])
        working = dict(st.session_state[user_state_key])
        st.success("Bull Case copied to User Case.")
    if preset_cols[2].button("Copy Bear Case to User Case", key="copy_bear_to_user_case"):
        st.session_state[user_state_key] = dict(scenarios["Bear Case"])
        working = dict(st.session_state[user_state_key])
        st.success("Bear Case copied to User Case.")
    expanded = preset_cols[3].toggle("Expanded Assumption Workbench", value=True, help="Use a wider, always-visible workbench layout for assumption review.")

    st.markdown('<div class="pa-section-title">Valuation Basis</div>', unsafe_allow_html=True)
    basis_default = next((label for label, item in VALUATION_BASIS_OPTIONS.items() if item["mode"] == str(working.get("dcf_mode", "FCFF")).upper()), "NOPAT bridge")
    basis = st.segmented_control("Current valuation basis", list(VALUATION_BASIS_OPTIONS.keys()), default=basis_default)
    basis = basis or basis_default
    st.caption(VALUATION_BASIS_OPTIONS[basis]["description"])
    working["dcf_mode"] = VALUATION_BASIS_OPTIONS[basis]["mode"]

    direct_nopat_override = st.toggle(
        "Use direct NOPAT margin override instead of OPEX-derived EBIT bridge",
        value=bool(working.get("use_direct_nopat_override", False)),
        help="Off: NOPAT is derived from Gross Margin minus OPEX % Revenue, then tax. On: the Direct NOPAT Margin slider controls NOPAT directly.",
    )
    working["use_direct_nopat_override"] = direct_nopat_override

    comparison = _scenario_comparison_table(scenarios, reverse, working)
    st.markdown('<div class="pa-section-title">Scenario Comparison Mini Table</div>', unsafe_allow_html=True)
    show_table(comparison, "Scenario comparison unavailable.")

    selected_key = st.selectbox(
        "Selected assumption explanation",
        ASSUMPTION_KEYS,
        format_func=lambda key: ASSUMPTION_METADATA[key]["label"],
        help="Pick an assumption to see definition, scope, source, reasonable range, and model impact.",
    )

    edited = dict(working)
    group_keys = {
        group: [key for key in ASSUMPTION_KEYS if ASSUMPTION_METADATA[key]["group"] == group]
        for group in ASSUMPTION_GROUPS
    }
    selected_group = st.segmented_control(
        "Assumption Groups",
        list(ASSUMPTION_GROUPS.keys()),
        default="Growth",
        help="Move between assumption groups without losing scenario scope.",
    )
    selected_group = selected_group or "Growth"
    st.caption(ASSUMPTION_GROUPS[selected_group])
    for key in group_keys[selected_group]:
        if key == "nopat_margin" and not direct_nopat_override:
            st.info("Direct NOPAT Margin is inactive because the OPEX-derived EBIT bridge is active.")
            continue
        meta = ASSUMPTION_METADATA[key]
        if "min" in meta:
            edited[key] = render_assumption_slider(
                key,
                edited.get(key),
                base.get(key),
                meta["min"],
                meta["max"],
                meta["step"],
                scenario_scope,
            )
        else:
            edited[key] = render_assumption_slider(
                key,
                edited.get(key),
                base.get(key),
                0,
                max(float(edited.get(key) or base.get(key) or 0) * 2, 1.0),
                1.0,
                scenario_scope,
            )
        with st.expander(f"What does {meta['label']} affect?"):
            _render_assumption_explanation(
                key,
                profile,
                f"{scenario_scope} only",
                _assumption_source(key, scenario_scope, edited.get(key), base.get(key)),
                calculate_assumption_impact(base, edited, key, historicals, market),
            )
    group_changed = [
        key for key in group_keys[selected_group]
        if abs(_assumption_float(edited.get(key)) - _assumption_float(base.get(key))) > 0.000001
    ]
    st.caption(f"Mini impact summary: {len(group_changed)} changed assumptions in {selected_group}.")

    edited = _normalize_assumption_bridge(edited, direct_nopat_override)
    if scenario_scope == "User Case":
        st.session_state[user_state_key] = dict(edited)

    selected_impact = calculate_assumption_impact(base, edited, selected_key, historicals, market)
    _render_assumption_explanation(
        selected_key,
        profile,
        f"{scenario_scope} only",
        _assumption_source(selected_key, scenario_scope, edited.get(selected_key), base.get(selected_key)),
        selected_impact,
    )

    active_warnings = validate_assumption_ranges(edited, historicals, ctx.get("peer_df"), ctx.get("moat"))
    if active_warnings:
        st.markdown('<div class="pa-section-title">Range Validation Warnings</div>', unsafe_allow_html=True)
        show_table(pd.DataFrame(active_warnings), "No range warnings.")

    st.markdown('<div class="pa-section-title">Fair Value Impact</div>', unsafe_allow_html=True)
    base_fv = run_dcf(historicals, market, base).get("fair_value_per_share")
    edited_fv = run_dcf(historicals, market, edited).get("fair_value_per_share")
    fv_delta = (edited_fv - base_fv) if edited_fv is not None and base_fv is not None else None
    metric_row(
        [
            ("Base Fair Value", base_fv, "per_share"),
            ("Edited Fair Value", edited_fv, "per_share"),
            ("Change vs Base", fv_delta, "per_share"),
        ]
    )

    user_note = st.text_area(
        "Why are you changing this assumption?",
        placeholder="Example: Backlog disclosure suggests higher revenue conversion, but new equipment raises growth CAPEX near term.",
        key=f"assumption_note_{scenario_scope}",
    )
    change_rows = _assumption_change_rows(base, edited, scenario_scope, historicals, market)
    for row in change_rows:
        if not row.get("User Note"):
            row["User Note"] = user_note
    st.markdown('<div class="pa-section-title">Assumption Change Log</div>', unsafe_allow_html=True)
    if change_rows:
        edited_log = st.data_editor(
            pd.DataFrame(change_rows),
            width="stretch",
            hide_index=True,
            num_rows="dynamic",
            disabled=["Old Value", "Delta", "Fair Value Impact", "Source"],
            key=f"assumption_change_log_{scenario_scope}",
        )
        edited = _apply_active_change_log(base, edited, edited_log.to_dict("records"))
        st.session_state["assumption_update_log"] = edited_log.to_dict("records")
    else:
        st.info("No active assumption changes versus Base Case.")

    st.caption("Evidence-applied changes default to User Case only. Use the Evidence & Assumptions tab to explicitly send clauses into the editable log.")
    if not expanded:
        st.info("Expanded workbench is off; detailed explanations remain available inside each group expander.")
    return edited


def _build_context(ticker: str, peer_override: str, fetch_peers: bool, include_deep_sec: bool = False) -> dict:
    dataset = cached_dataset(ticker, include_deep_sec)
    historicals = build_historical_financial_table(dataset)
    clauses = extract_relevant_clauses(dataset.get("filing_texts", {}))
    accounting_interpretation = build_accounting_interpretation(dataset, historicals, clauses)
    base_assumptions = default_assumptions_from_historicals(historicals, dataset.get("market_data", {}))
    base_dcf = run_dcf(historicals, dataset.get("market_data", {}), base_assumptions)
    reverse = run_reverse_dcf(dataset.get("market_data", {}), historicals, base_assumptions)
    default_peers = select_peer_candidates(ticker, dataset.get("sector"), dataset.get("industry"))
    peers = [p.strip().upper() for p in peer_override.split(",") if p.strip()] or default_peers
    can_fetch_peers = fetch_peers and peers and ("yfinance" in dataset.get("sources", []) or bool(peer_override.strip()))
    peer_df = build_peer_comparison(ticker, peers) if can_fetch_peers else pd.DataFrame()
    capex_quality = analyze_capex_ocf_nopat_quality(historicals, clauses)
    leverage = analyze_operating_leverage(historicals, peer_df)
    ma = analyze_ma_strategy(dataset.get("filing_texts", {}), historicals)
    management = analyze_management_and_board(dataset.get("filing_texts", {}), dataset.get("submissions", {}))
    guidance = analyze_guidance_accuracy(dataset.get("filing_texts", {}), historicals)
    alignment = analyze_compensation_alignment(dataset.get("filing_texts", {}), historicals)
    moat = analyze_moat(dataset, historicals, dataset.get("filing_texts", {}), peer_df, clauses)
    risks = analyze_risks_and_thesis_breakers(dataset.get("filing_texts", {}), clauses, historicals)
    scoring = score_investment(
        {
            "dcf": base_dcf,
            "reverse_dcf": reverse,
            "moat": moat,
            "management": management,
            "ma": ma,
            "alignment": alignment,
            "quality": capex_quality,
            "operating_leverage": leverage,
        }
    )
    thesis = build_thesis_summary(dataset, base_dcf, reverse, moat, scoring)
    return {
        "dataset": dataset,
        "historicals": historicals,
        "clauses": clauses,
        "accounting_interpretation": accounting_interpretation,
        "base_assumptions": base_assumptions,
        "base_dcf": base_dcf,
        "reverse": reverse,
        "peer_df": peer_df,
        "capex_quality": capex_quality,
        "leverage": leverage,
        "ma": ma,
        "management": management,
        "guidance": guidance,
        "alignment": alignment,
        "moat": moat,
        "risks": risks,
        "scoring": scoring,
        "thesis": thesis,
    }


def _overview(ctx: dict) -> None:
    dataset = ctx["dataset"]
    market = dataset.get("market_data", {})
    scoring = ctx["scoring"]
    dcf = ctx["base_dcf"]
    moat = ctx["moat"]

    reverse = ctx["reverse"]
    historicals = ctx["historicals"]
    accounting = ctx.get("accounting_interpretation", {})
    accounting_cards = accounting.get("cards", {})

    st.caption("Snapshot: decision, why it matters, evidence, model impact, and the adjustments that deserve attention.")
    left, right = st.columns([1.15, 0.85])
    with left:
        with st.container(border=True):
            st.subheader("Investment View")
            metric_row(
                [
                    ("Decision", scoring.get("recommendation"), "text"),
                    ("Conviction", scoring.get("conviction"), "text"),
                    ("PA-11R Score", scoring.get("total_score"), "score"),
                    ("Moat", moat.get("classification"), "text"),
                ]
            )
            st.write(ctx["thesis"].get("valuation_view") or "Thesis unavailable.")
            st.write(scoring.get("position_size_guidance") or "Position sizing guidance unavailable.")
            _mini_list("Top 3 valuation drivers", _top_three_drivers(ctx))
            _mini_list("Top 3 risks", _top_three_risks(ctx))

            st.subheader("Top Clause-Driven Model Impacts")
            show_table(_top_clause_impacts(ctx["clauses"]), "No clause impacts available in fast mode.")

            with st.expander("Show scorecard and assumption drivers"):
                show_table(ctx["scoring"]["scorecard"], "Scorecard unavailable.")
                show_table(_top_assumption_drivers(ctx["base_assumptions"]))

    with right:
        with st.container(border=True):
            st.subheader("Valuation Snapshot")
            metric_row(
                [
                    ("Price", market.get("price"), "per_share"),
                    ("Fair Value", dcf.get("fair_value_per_share"), "per_share"),
                    ("Upside", dcf.get("upside_downside_pct"), "pct"),
                ]
            )
            metric_row(
                [
                    ("MOS Buy Price", dcf.get("buy_price_after_margin_of_safety"), "per_share"),
                    ("DCF Confidence", accounting.get("valuation_confidence"), "text"),
                    ("Terminal Weight", dcf.get("terminal_value_weight_pct"), "pct"),
                ]
            )
            metric_row(
                [
                    ("Reverse DCF", reverse.get("market_case"), "text"),
                    ("Accounting Quality", accounting_cards.get("OCF Quality"), "text"),
                    ("Main Distortion", accounting_cards.get("Main Accounting Distortion"), "text"),
                ]
            )
            st.write(reverse.get("interpretation") or "Reverse DCF conclusion unavailable.")
            with st.expander("Show data coverage"):
                show_table(_data_coverage(dataset, historicals))

    c1, c2 = st.columns([0.62, 0.38])
    with c1:
        st.plotly_chart(price_action_chart(dataset.get("price_history")), width="stretch", key="v2_price_action")
    with c2:
        _accounting_reality_compact(ctx)

    with st.expander("Finviz Decision Snapshot"):
        show_table(_finviz_decision_snapshot(market), "No Finviz decision fields available.")


def _valuation(ctx: dict) -> None:
    market = ctx["dataset"].get("market_data", {})
    st.caption("DCF Assumption Workbench: choose scenario scope, edit assumptions with context, see fair-value impact, then review the detailed model.")
    left, right = st.columns([0.52, 0.48])
    with left:
        assumptions = _assumption_editor(ctx)
    user_dcf = run_dcf(ctx["historicals"], market, assumptions)
    reverse = run_reverse_dcf(market, ctx["historicals"], assumptions)
    model_table = build_time_axis_financial_model(ctx["historicals"], user_dcf.get("forecast_table"), assumptions)
    dcf_output_table = _dcf_forecast_output_table(user_dcf, assumptions)
    dcf_bridge_table = build_dcf_output_table(user_dcf, assumptions, market)
    reverse_table = build_reverse_dcf_table(reverse, assumptions, market)
    ev_bridge = build_ev_to_equity_bridge(market, user_dcf, assumptions)
    scenario_table = build_scenario_table(ctx["historicals"], market, assumptions)

    valuation_summary = pd.DataFrame(
        [
            {"metric": "Enterprise value", "value": user_dcf.get("enterprise_value")},
            {"metric": "Equity value", "value": user_dcf.get("equity_value")},
            {"metric": "Fair value per share", "value": user_dcf.get("fair_value_per_share")},
            {"metric": "Buy zone after margin of safety", "value": user_dcf.get("buy_price_after_margin_of_safety")},
            {"metric": "Upside / Downside", "value": user_dcf.get("upside_downside_pct")},
            {"metric": "Terminal value weight", "value": user_dcf.get("terminal_value_weight_pct")},
        ]
    )
    assumptions_table = _assumption_evidence_table(assumptions)
    accounting_flags = _accounting_assumption_flags(ctx.get("accounting_interpretation"), assumptions)

    with right:
        st.markdown('<div class="pa-section-title">Step 3: Valuation Impact</div>', unsafe_allow_html=True)
        metric_row(
            [
                ("Fair Value / Share", user_dcf.get("fair_value_per_share"), "per_share"),
                ("Current Price", market.get("price"), "per_share"),
                ("Upside / Downside", user_dcf.get("upside_downside_pct"), "pct"),
            ]
        )
        metric_row(
            [
                ("Enterprise Value", user_dcf.get("enterprise_value"), "money"),
                ("Equity Value", user_dcf.get("equity_value"), "money"),
                ("Diluted Shares", assumptions.get("diluted_shares"), "shares"),
            ]
        )
        metric_row(
            [
                ("MOS Buy Price", user_dcf.get("buy_price_after_margin_of_safety"), "per_share"),
                ("Terminal Value % EV", user_dcf.get("terminal_value_weight_pct"), "pct"),
                ("DCF Confidence", ctx.get("accounting_interpretation", {}).get("valuation_confidence"), "text"),
            ]
        )
        if user_dcf.get("upside_downside_pct") is not None:
            if user_dcf.get("upside_downside_pct") < 0:
                _notice("Overvalued under current assumptions.", "risk")
            else:
                _notice("Potentially undervalued under current assumptions.", "success")
        if user_dcf.get("terminal_value_weight_pct") and user_dcf.get("terminal_value_weight_pct") > 0.65:
            _notice(
                f"Terminal Value Warning: terminal value represents {fmt_percent(user_dcf.get('terminal_value_weight_pct'))} of enterprise value. The valuation is highly sensitive to terminal assumptions.",
                "warning",
            )
        show_warnings(user_dcf.get("warnings", []))
        show_warnings(ctx.get("accounting_interpretation", {}).get("warnings", []))
        show_table(valuation_summary, "Valuation summary unavailable.")

    st.markdown('<div class="pa-section-title">Step 4: Review Detailed Model</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fcf_projection_chart(ctx["historicals"], user_dcf["forecast_table"]), width="stretch", key="v2_fcf_projection")
        st.caption("The line chart compares reported FCF with the forecast generated from the current sliders.")
    with c2:
        st.plotly_chart(reverse_dcf_chart(reverse, assumptions), width="stretch", key="v2_reverse_dcf")
        st.caption("Reverse DCF compares your revenue CAGR assumption with the growth implied by the current market price.")

    sensitivity = build_dcf_sensitivity_table(
        {**assumptions, "historicals": ctx["historicals"], "market_data": market},
        [0.075, 0.085, 0.095, 0.105, 0.115],
        [0.01, 0.02, 0.03, 0.04],
    )
    st.plotly_chart(dcf_sensitivity_heatmap(sensitivity), width="stretch", key="v2_sensitivity")
    st.caption("Y-axis is WACC / discount rate. X-axis is terminal growth. Each cell is the resulting fair value per share.")
    c3, c4 = st.columns([0.45, 0.55])
    with c3:
        st.subheader("DCF Valuation Summary")
        show_table(valuation_summary, "Valuation summary unavailable.")
    with c4:
        st.subheader("Forecast Table")
        show_table(user_dcf["forecast_table"], "Forecast unavailable.")

    with st.expander("1. Historical Financials / Operating Model", expanded=True):
        show_table(model_table[model_table["Line Item"].isin([
            "Revenue",
            "Revenue growth %",
            "COGS / Cost of sales",
            "COGS % revenue",
            "Gross profit",
            "Gross margin %",
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
        ])])
    with st.expander("2. Cash Flow / CAPEX / NOPAT", expanded=True):
        show_table(model_table[model_table["Line Item"].isin([
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
        ])])
    with st.expander("3. Forecast Assumptions"):
        show_table(assumptions_table, "Assumptions unavailable.")
        st.subheader("Accounting-Driven Assumption Flags")
        st.caption("These are suggested reviews only. The dashboard does not override your assumptions without confirmation.")
        show_table(accounting_flags, "No accounting-driven assumption flags available.")
    with st.expander("4. DCF Output"):
        show_table(dcf_output_table, "DCF output unavailable.")
        st.subheader("EV to Equity Bridge")
        show_table(dcf_bridge_table, "DCF bridge unavailable.")
    with st.expander("5. Reverse DCF"):
        show_table(reverse_table, "Reverse DCF unavailable.")
        st.write(reverse.get("interpretation"))
    with st.expander("6. EV to Equity Bridge"):
        show_table(ev_bridge, "EV bridge unavailable.")
    with st.expander("7. Scenario Table"):
        show_table(scenario_table, "Scenario table unavailable.")
    with st.expander("8. Source / Evidence Table"):
        show_table(build_source_evidence_table(ctx["historicals"], ctx["dataset"]), "Source evidence unavailable.")


def _financial_reports(ctx: dict) -> None:
    historicals = ctx["historicals"]
    st.caption("Financials: Excel-style model layout with financial line items down rows and time periods across columns.")
    if historicals is None or historicals.empty:
        st.info("No reported financial table is available.")
        return
    dcf = run_dcf(historicals, ctx["dataset"].get("market_data", {}), ctx["base_assumptions"])
    model_table = build_time_axis_financial_model(historicals, dcf.get("forecast_table"), ctx["base_assumptions"])
    metric_row(
        [
            ("Latest Revenue", historicals["Revenue"].iloc[-1] if "Revenue" in historicals else None, "money"),
            ("Latest FCF", historicals["FCF"].iloc[-1] if "FCF" in historicals else None, "money"),
            ("Gross Margin", historicals["Gross Margin"].iloc[-1] if "Gross Margin" in historicals else None, "pct"),
            ("Net Debt", historicals["Net Debt"].iloc[-1] if "Net Debt" in historicals else None, "money"),
        ]
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(financial_revenue_margin_chart(historicals), width="stretch", key="v2_financial_revenue_margin")
        st.plotly_chart(financial_profitability_chart(historicals), width="stretch", key="v2_financial_profitability")
    with c2:
        st.plotly_chart(financial_cash_flow_chart(historicals), width="stretch", key="v2_financial_cash_flow")
        with st.expander("Reported Provider Table"):
            show_table(historicals, "No financial table available.")

    st.subheader("Financial Model: Actuals + Forecast")
    st.caption("Neutral = actuals, yellow = latest/YTD/estimate, blue = forecast. Values are display-formatted only; model calculations use raw floats.")
    if model_table is None or model_table.empty:
        st.info("Financial model table unavailable.")
    else:
        st.dataframe(_style_financial_model_table(format_dataframe_for_display(model_table)), width="stretch", hide_index=True)

    with st.expander("Row Groups: Operating Model"):
        show_table(model_table[model_table["Line Item"].isin([
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
        ])])
    with st.expander("Row Groups: Cash Flow / CAPEX / SBC"):
        show_table(model_table[model_table["Line Item"].isin([
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
            "Diluted shares",
            "Diluted shares growth %",
        ])])

    with st.expander("SBC / Dilution"):
        show_table(model_table[model_table["Line Item"].isin(["SBC", "SBC % revenue", "SBC % gross profit", "SBC % OCF", "Diluted shares", "Diluted shares growth %"])])


def _evidence(ctx: dict) -> None:
    dataset = ctx["dataset"]
    st.caption("Evidence: filing metadata, guidance extraction, and lightweight clause context. Use the Clause / Annotation Map tab for structured model-line impacts.")
    if not dataset.get("evidence_loaded"):
        st.info("SEC filing metadata is loaded. Use 'Load SEC evidence' in the sidebar to download and analyze full filing text.")
    filings = dataset.get("filings", {})
    filing_rows = []
    for label in ["latest_10k", "latest_10q", "latest_proxy"]:
        item = filings.get(label)
        if item:
            filing_rows.append({"type": item.get("form"), "filing_date": item.get("filing_date"), "report_date": item.get("report_date"), "document": item.get("primary_document")})
    for item in filings.get("latest_8ks", []):
        filing_rows.append({"type": item.get("form"), "filing_date": item.get("filing_date"), "report_date": item.get("report_date"), "document": item.get("primary_document")})

    c1, c2 = st.columns([0.4, 0.6])
    with c1:
        st.markdown('<div class="pa-section-title">Filings</div>', unsafe_allow_html=True)
        show_table(pd.DataFrame(filing_rows), "No filings available.")
        st.markdown('<div class="pa-section-title">Guidance</div>', unsafe_allow_html=True)
        st.write(ctx["guidance"].get("summary"))
        show_table(ctx["guidance"].get("table"), "No guidance table available.")
    with c2:
        st.markdown('<div class="pa-section-title">Clause Map</div>', unsafe_allow_html=True)
        show_table(ctx["clauses"], "No valuation-relevant clauses extracted.")


def _reverse_dcf_tab(ctx: dict) -> None:
    st.caption("Reverse DCF: market-implied expectations compared with your base assumptions.")
    reverse = ctx["reverse"]
    assumptions = ctx["base_assumptions"]
    metric_row(
        [
            ("Market Case", reverse.get("market_case"), "text"),
            ("Implied Revenue CAGR", reverse.get("implied_revenue_cagr"), "pct"),
            ("Base Revenue CAGR", assumptions.get("revenue_cagr"), "pct"),
            ("Terminal Multiple", assumptions.get("terminal_multiple"), "multiple"),
        ]
    )
    st.write(reverse.get("interpretation") or "Reverse DCF interpretation unavailable.")
    comparison = _reverse_dcf_comparison_table(reverse, assumptions)
    show_table(comparison, "Reverse DCF comparison unavailable.")
    positive_clauses = ctx["clauses"]
    if positive_clauses is not None and not positive_clauses.empty and reverse.get("market_case") in {"Bull", "Optimistic"}:
        st.warning("Positive filing clauses may already be reflected in the market-implied case. Require execution upside before raising assumptions.")
    st.plotly_chart(reverse_dcf_chart(reverse, assumptions), width="stretch", key="v2_reverse_tab_chart")
    with st.expander("Full Reverse DCF Detail"):
        show_table(build_reverse_dcf_table(reverse, assumptions, ctx["dataset"].get("market_data", {})), "Reverse DCF detail unavailable.")


def _multiples_peers(ctx: dict) -> None:
    st.caption("Multiples / Peers: clean relative valuation view first, full provider table behind the expander.")
    peer_df = ctx["peer_df"]
    clean = _peer_summary_table(peer_df)
    show_table(clean, "No peer data available. Enable peer fetch or add peer overrides.")
    if clean is not None and not clean.empty:
        st.write("Valuation view: compare premium/discount only after checking growth, margin, and OCF conversion versus the peer median.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(peer_scatter(peer_df), width="stretch", key="v2_peer_scatter_clean")
    with c2:
        st.plotly_chart(peer_multiple_chart(peer_df), width="stretch", key="v2_peer_multiple_clean")
    with st.expander("Show full peer table"):
        show_table(peer_df, "No full peer table available.")


def _accounting_quality(ctx: dict) -> None:
    st.caption("Accounting Quality: reported rows separated from economic interpretation.")
    _accounting_reality_check(ctx, expanded=True)
    interpretation = ctx.get("accounting_interpretation", {})
    capex = interpretation.get("capex", {})
    capex_rows = pd.DataFrame(
        [
            {"Metric": "Reported CAPEX", "Value": capex.get("maintenance_capex_estimate", 0) + capex.get("growth_capex_estimate", 0) + capex.get("uncertain_capex", 0), "Read": "Reported total CAPEX / reconstructed split"},
            {"Metric": "Interpreted Maintenance CAPEX", "Value": capex.get("maintenance_capex_estimate"), "Read": capex.get("method")},
            {"Metric": "Interpreted Growth CAPEX", "Value": capex.get("growth_capex_estimate"), "Read": capex.get("classification")},
            {"Metric": "Uncertain CAPEX", "Value": capex.get("uncertain_capex"), "Read": "Manual review required" if capex.get("uncertain_capex") else "Low / none"},
        ]
    )
    st.subheader("CAPEX Split")
    show_table(capex_rows, "CAPEX interpretation unavailable.")
    with st.expander("Source / Evidence Table"):
        show_table(build_source_evidence_table(ctx["historicals"], ctx["dataset"]), "Source evidence unavailable.")


def _ma_summary_rows(ctx: dict) -> pd.DataFrame:
    dataset = ctx["dataset"]
    sec_fin = dataset.get("financials", {}).get("sec", {})
    profile = ctx.get("accounting_interpretation", {}).get("business_profile", {})
    ma = ctx["ma"]
    assets = sec_fin.get("assets", {}).get("value") if isinstance(sec_fin, dict) else None
    goodwill = sec_fin.get("goodwill", {}).get("value") if isinstance(sec_fin, dict) else None
    intangibles = sec_fin.get("intangibles", {}).get("value") if isinstance(sec_fin, dict) else None
    goodwill_ratio = goodwill / assets if goodwill is not None and assets else None
    intangibles_ratio = intangibles / assets if intangibles is not None and assets else None
    integration_risk = "High" if ma.get("red_flags") else "Medium" if profile.get("acquisition_intensity") in {"Medium", "High"} else "Low"
    return pd.DataFrame(
        [
            {"Metric": "Acquisition Intensity", "Value": profile.get("acquisition_intensity", UNAVAILABLE), "Read": "Business profile and M&A clauses"},
            {"Metric": "Goodwill / Assets", "Value": goodwill_ratio, "Read": "SEC companyfacts when available"},
            {"Metric": "Intangibles / Assets", "Value": intangibles_ratio, "Read": "SEC companyfacts when available"},
            {"Metric": "Acquired Revenue Evidence", "Value": "Available" if not ma.get("timeline", pd.DataFrame()).empty else "Unavailable", "Read": "Extracted filing language"},
            {"Metric": "Integration Risk", "Value": integration_risk, "Read": "Impairment, integration, goodwill, and acquisition language"},
            {"Metric": "M&A Quality", "Value": ma.get("classification", "Unknown"), "Read": "Does M&A appear value-creating or revenue-padding?"},
        ]
    )


def _ma_manual_review_table(ctx: dict) -> pd.DataFrame:
    dataset = ctx["dataset"]
    filings = dataset.get("latest_filings") or []
    source_url = (filings[0].get("document_url") or filings[0].get("filing_url")) if filings else "https://www.sec.gov/edgar/search/"
    return pd.DataFrame(
        [
            {
                "Data Needed": "Acquisition timeline",
                "Where to Review": "Business Combinations note; Goodwill and Intangibles note; MD&A; Investing cash flow; 8-K acquisition filings",
                "Keywords": "acquisition, purchase price, goodwill, intangible, integration, impairment, business combination",
                "Source URL": source_url,
                "Model Impact": "Revenue growth, margins, amortization, debt/WACC, diluted shares, terminal multiple",
            }
        ]
    )


def _ma_management_sbc(ctx: dict) -> None:
    st.caption("M&A / Management / SBC: capital allocation, governance, and dilution signals.")
    management = ctx["management"]
    alignment = ctx["alignment"]
    ma = ctx["ma"]
    ma_summary = _ma_summary_rows(ctx)
    metric_row(
        [
            ("Management", management.get("management_score"), "score"),
            ("Alignment", alignment.get("alignment_score"), "score"),
            ("M&A Quality", ma.get("classification"), "text"),
            ("SBC Signal", alignment.get("sbc_risk", "Manual review required"), "text"),
        ]
    )
    st.subheader("M&A Summary")
    show_table(ma_summary, "M&A summary unavailable.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Management")
        st.write(management.get("summary") or "Management read unavailable.")
        st.write("Style:", management.get("style") or UNAVAILABLE)
        show_warnings(management.get("red_flags", []))
        st.subheader("SBC / Dilution")
        show_table(alignment.get("sbc_table"), "No SBC table available.")
        st.plotly_chart(sbc_vs_buybacks_chart(alignment), width="stretch", key="v2_sbc_management")
        st.subheader("Capital Allocation Review")
        show_table(
            pd.DataFrame(
                [
                    {"Review Area": "SBC dilution", "Question": "Is SBC creating material per-share dilution?", "Model Impact": "Diluted shares / fair value per share"},
                    {"Review Area": "Buybacks", "Question": "Are repurchases offsetting dilution or merely masking it?", "Model Impact": "Share count trend / owner earnings"},
                    {"Review Area": "Compensation alignment", "Question": "Does pay link to FCF, ROIC, or per-share value?", "Model Impact": "Confidence / margin of safety"},
                    {"Review Area": "Governance", "Question": "Are board independence and controls sufficient?", "Model Impact": "WACC / position sizing"},
                    {"Review Area": "M&A discipline", "Question": "Does capital allocation improve organic economics?", "Model Impact": "Revenue quality / terminal multiple"},
                ]
            ),
            "Capital allocation review unavailable.",
        )
    with c2:
        st.subheader("M&A")
        st.write(ma.get("summary") or "M&A read unavailable.")
        for flag in ma.get("red_flags", [])[:3]:
            _notice(flag, "risk")
        st.plotly_chart(ma_timeline_chart(ma), width="stretch", key="v2_ma_management")
        timeline = ma.get("timeline")
        if timeline is None or timeline.empty:
            _notice("M&A timeline unavailable from extracted filings. Use the manual review guide below.", "warning")
            show_table(_ma_manual_review_table(ctx), "No M&A manual review guide available.")
        else:
            show_table(timeline, "No M&A timeline available.")
        with st.expander("M&A Model Implications"):
            show_table(
                pd.DataFrame(
                    [
                        {"Model Line": "Revenue growth", "Question": "Is growth bought or organic?", "Action": "Separate acquired revenue from organic growth."},
                        {"Model Line": "NOPAT margin", "Question": "Are integration/amortization costs recurring?", "Action": "Review normalized operating margin."},
                        {"Model Line": "D&A / amortization", "Question": "Are acquired intangibles depressing EBIT?", "Action": "Use accounting interpretation before changing assumptions."},
                        {"Model Line": "Debt / WACC", "Question": "Was the deal cash/debt funded?", "Action": "Review financing and risk premium."},
                        {"Model Line": "Diluted shares", "Question": "Was the deal stock-funded?", "Action": "Review dilution and per-share value."},
                    ]
                )
            )


def _moat_risks(ctx: dict) -> None:
    st.caption("Moat / Risks: competitive quality and what can break the thesis.")
    moat = ctx["moat"]
    risks = ctx["risks"]
    metric_row(
        [
            ("Moat Score", f"{fmt_ratio(moat.get('moat_score'), 1)}/10", "text"),
            ("Moat Class", moat.get("classification"), "text"),
            ("Moat Confidence", moat.get("confidence"), "text"),
            ("Risk Score", f"{fmt_ratio(risks.get('risk_score'), 1)}/10", "text"),
        ]
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(moat_score_bar(moat.get("moat_sources")), width="stretch", key="v2_moat_risks_bar")
        st.write(moat.get("terminal_value_implication") or "Moat implication unavailable.")
    with c2:
        entrant = moat.get("new_entrant_test", {})
        st.subheader("New Entrant Test")
        show_table(
            pd.DataFrame(
                [
                    {"Question": "How much capital would a new entrant need?", "Read": "Review capital intensity and scale evidence.", "Evidence": "; ".join(entrant.get("incumbent_advantages", [])[:2]) or "Manual review required"},
                    {"Question": "Could a new entrant underprice?", "Read": "Review pricing power and peer margins.", "Evidence": "Peer and margin context required."},
                    {"Question": "Could customers switch easily?", "Read": "Review switching cost / integration evidence.", "Evidence": "; ".join(entrant.get("entrant_risks", [])[:2]) or "Manual review required"},
                    {"Question": "Does incumbent have distribution, data, compliance, or integration advantage?", "Read": "Review moat source scorecard.", "Evidence": moat.get("peer_context", UNAVAILABLE)},
                    {"Question": "Would a new entrant reach breakeven quickly?", "Read": "Review unit economics, scale, and customer acquisition costs.", "Evidence": "Manual review required"},
                ]
            ),
            "New entrant test unavailable.",
        )
        st.subheader("Top Risks Explained")
        show_table(_risk_review_table(ctx, limit=4), "No extracted risks available.")
        st.write("Bear case:", risks.get("bear_case_implications") or UNAVAILABLE)

    st.subheader("Moat Scorecard")
    scorecard = moat.get("moat_sources")
    if scorecard is not None and not scorecard.empty:
        show_table(
            scorecard.rename(
                columns={
                    "moat_source": "Moat Source",
                    "score_1_to_10": "Score (1-10)",
                    "evidence": "Evidence",
                    "confidence": "Confidence",
                    "model_implication": "Model Impact",
                }
            )[["Moat Source", "Score (1-10)", "Evidence", "Confidence", "Model Impact"]],
            "Moat scorecard unavailable.",
        )

    st.subheader("Thesis Breakers")
    assumptions = ctx.get("base_assumptions", {})
    show_table(
        pd.DataFrame(
            [
                {"Thesis Breaker": "Revenue growth misses plan", "Metric to Watch": "Revenue growth", "Threshold": fmt_percent(assumptions.get("revenue_cagr", 0.08)), "Source": "SEC filings / earnings release", "Current Status": "Monitor", "Model Impact": "Revenue CAGR / scenario probability"},
                {"Thesis Breaker": "OCF conversion weakens", "Metric to Watch": "OCF margin", "Threshold": fmt_percent(assumptions.get("ocf_margin", 0.15)), "Source": "Cash flow statement", "Current Status": "Monitor", "Model Impact": "OCF margin / FCF"},
                {"Thesis Breaker": "SBC remains high", "Metric to Watch": "SBC % revenue", "Threshold": "10.0%", "Source": "Cash flow statement / compensation note", "Current Status": "Monitor", "Model Impact": "Dilution / per-share value"},
                {"Thesis Breaker": "Terminal assumptions unsupported", "Metric to Watch": "Moat score / terminal value weight", "Threshold": "Terminal value > 65.0% EV", "Source": "DCF sensitivity", "Current Status": "Monitor", "Model Impact": "Terminal multiple / WACC"},
            ]
        ),
        "Thesis breaker table unavailable.",
    )
    st.subheader("Risk Factor Translation")
    risk_rows = []
    for risk in (risks.get("top_risks", []) or [])[:5]:
        risk_text = str(risk)
        lower = risk_text.lower()
        model_line = "revenue_growth" if "customer" in lower or "demand" in lower else "wacc" if "risk" in lower or "litigation" in lower else "terminal_multiple"
        risk_rows.append(
            {
                "Legal / Filing Language": risk_text[:180],
                "Business Risk": "Revenue volatility" if model_line == "revenue_growth" else "Higher risk premium" if model_line == "wacc" else "Weaker durability",
                "Model Line Affected": model_line,
                "Scenario Impact": "Increase bear-case probability / manual review",
            }
        )
    show_table(pd.DataFrame(risk_rows), "Risk factor translation unavailable.")


def _final_decision(ctx: dict) -> None:
    st.caption("Final Decision: concise readout after reviewing valuation, reverse DCF, clauses, quality, and risks.")
    scoring = ctx["scoring"]
    dcf = ctx["base_dcf"]
    market = ctx["dataset"].get("market_data", {})
    metric_row(
        [
            ("Decision", scoring.get("recommendation"), "text"),
            ("PA-11R Score", scoring.get("total_score"), "score"),
            ("Fair Value", dcf.get("fair_value_per_share"), "per_share"),
            ("Current Price", market.get("price"), "per_share"),
            ("Upside / Downside", dcf.get("upside_downside_pct"), "pct"),
        ]
    )
    st.subheader("Why")
    st.write(ctx["thesis"].get("valuation_view") or "Decision thesis unavailable.")
    st.write(scoring.get("position_size_guidance") or "Position size guidance unavailable.")
    c1, c2 = st.columns(2)
    with c1:
        _mini_list("Top 3 things that matter", _top_three_drivers(ctx))
    with c2:
        _mini_list("Top 3 risks", _top_three_risks(ctx))
    with st.expander("Show full scorecard"):
        show_table(scoring.get("scorecard"), "Scorecard unavailable.")


def _quality(ctx: dict) -> None:
    st.caption("Quality & Moat: cash conversion, operating leverage, management signals, SBC, M&A language, and moat indicators.")
    if not ctx["dataset"].get("evidence_loaded"):
        st.info("Scores below use fast financial data only. Load SEC evidence for filing-text moat, management, M&A, compensation, and risk signals.")
    quality = ctx["capex_quality"]
    leverage = ctx["leverage"]
    management = ctx["management"]
    alignment = ctx["alignment"]
    moat = ctx["moat"]
    ma = ctx["ma"]

    metric_row(
        [
            ("Quality", quality.get("quality_score"), "text"),
            ("Leverage", leverage.get("classification"), "text"),
            ("Management", management.get("management_score"), "text"),
            ("Alignment", alignment.get("alignment_score"), "text"),
        ]
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cash Quality")
        st.write(quality.get("summary"))
        metric_row(
            [
                ("CAPEX / Revenue", quality.get("metrics", {}).get("capex_pct_revenue"), "pct"),
                ("OCF Conversion", quality.get("metrics", {}).get("ocf_conversion"), "text"),
            ]
        )
        show_warnings(quality.get("red_flags", []))
        st.subheader("Management")
        st.write(management.get("summary"))
        st.write("Style:", management.get("style"))
        show_warnings(management.get("red_flags", []))
    with c2:
        st.subheader("Moat")
        metric_row([("Score", moat.get("moat_score"), "text"), ("Class", moat.get("classification"), "text"), ("Confidence", moat.get("confidence"), "text")])
        st.plotly_chart(moat_score_bar(moat.get("moat_sources")), width="stretch", key="v2_moat_bar")
        st.write(moat.get("terminal_value_implication"))
        st.subheader("SBC and M&A")
        show_table(alignment.get("sbc_table"), "No SBC table available.")
        st.plotly_chart(sbc_vs_buybacks_chart(alignment), width="stretch", key="v2_sbc")
        st.write(ma.get("summary"))
        st.plotly_chart(ma_timeline_chart(ma), width="stretch", key="v2_ma")


def _peers_risks(ctx: dict) -> None:
    st.caption("Peers & Risks: relative market data, beta, enterprise value, extracted risks, and thesis breakers.")
    if not ctx["dataset"].get("evidence_loaded"):
        st.info("Risk text is unavailable until SEC evidence is loaded. Peer data and financial risk signals are still available.")
    peer_df = ctx["peer_df"]
    risks = ctx["risks"]
    st.subheader("Peers")
    show_table(peer_df, "No peer data available. Enable peer fetch or add peer overrides.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(peer_scatter(peer_df), width="stretch", key="v2_peer_scatter")
    with c2:
        st.plotly_chart(peer_multiple_chart(peer_df), width="stretch", key="v2_peer_multiple")

    st.subheader("Risks and Breakers")
    metric_row([("Risk Score", risks.get("risk_score"), "text")])
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("Top risks")
        show_table(_risk_review_table(ctx, limit=5), "No extracted risks available.")
    with c4:
        st.markdown("Thesis breakers")
        for breaker in risks.get("thesis_breakers", []):
            st.write("-", breaker)
        st.write("Bear case:", risks.get("bear_case_implications"))


def _filter_options(df: pd.DataFrame, column: str) -> list[str]:
    if df is None or df.empty or column not in df:
        return []
    label_map = {
        "review_dcf": "Review DCF",
        "flag_risk": "Flag Risk",
        "manual_review": "Manual Review",
        "update_scenario": "Update Scenario",
        "Review DCF": "Review DCF",
        "Flag risk": "Flag Risk",
        "Manual review": "Manual Review",
        "Update scenario": "Update Scenario",
    }
    values = df[column].dropna().astype(str).str.strip()
    values = values[values.ne("")]
    if column == "dashboard_action":
        values = values.map(lambda value: label_map.get(value, value))
    return sorted(set(values.tolist()))


def _filtered_clauses(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    filters = st.columns(4)
    selected = {}
    filter_columns = [
        ("form", "Form"),
        ("topic", "Topic"),
        ("subtopic", "Subtopic"),
        ("model_line_affected", "Model line"),
        ("direction", "Direction"),
        ("confidence", "Confidence"),
        ("evidence_grade", "Evidence grade"),
        ("review_status", "Review status"),
    ]
    for index, (column, label) in enumerate(filter_columns):
        options = _filter_options(df, column)
        with filters[index % 4]:
            selected[column] = st.multiselect(label, options, key=f"clause_filter_{column}")
    out = df.copy()
    for column, values in selected.items():
        if values:
            normalized = out[column].astype(str).str.strip()
            if column == "dashboard_action":
                normalized = normalized.map(
                    lambda value: {
                        "review_dcf": "Review DCF",
                        "flag_risk": "Flag Risk",
                        "manual_review": "Manual Review",
                        "update_scenario": "Update Scenario",
                        "Review DCF": "Review DCF",
                        "Flag risk": "Flag Risk",
                        "Manual review": "Manual Review",
                        "Update scenario": "Update Scenario",
                    }.get(value, value)
                )
            out = out[normalized.isin(values)]
    return out


def _clause_annotation_map(ctx: dict) -> None:
    ticker = ctx["dataset"].get("ticker")
    request_key = f"clause_annotations_requested_{ticker}"
    st.subheader("Clause Map")
    st.caption("Default view shows model impact first. Full filing text and assumption review details sit behind expanders.")
    if not st.session_state.get(request_key):
        st.info("Fast cockpit data is loaded. Click below to fetch SEC filing text and extract valuation-relevant clauses.")
        if st.button("Extract clause annotations", type="primary"):
            st.session_state[request_key] = True
            st.rerun()
        return

    with st.spinner("Fetching SEC filing text and extracting clauses..."):
        deep_dataset = cached_dataset(ticker, include_deep_sec=True)
        clause_df = run_clause_extraction_pipeline(
            deep_dataset.get("filing_texts", {}),
            _filing_metadata(deep_dataset),
            ticker=deep_dataset.get("ticker"),
            cik=deep_dataset.get("cik"),
        )

    for warning in clause_df.attrs.get("warnings", []):
        st.warning(warning)
    if clause_df.empty:
        st.info("No relevant clauses found. Manual review may still be required.")
        return

    filtered = _filtered_clauses(clause_df)
    full_cols = [
        "topic",
        "subtopic",
        "section",
        "clause_text",
        "model_line_affected",
        "direction",
        "timeframe",
        "confidence",
        "evidence_grade",
        "suggested_assumption_change",
        "dashboard_action",
        "review_status",
    ]
    compact_cols = [column for column in ["topic", "model_line_affected", "direction", "confidence", "dashboard_action"] if column in filtered]
    compact = filtered[compact_cols].copy()
    if "dashboard_action" in compact:
        compact["dashboard_action"] = compact["dashboard_action"].astype(str).str.strip().map(
            lambda value: {
                "review_dcf": "Review DCF",
                "flag_risk": "Flag Risk",
                "manual_review": "Manual Review",
                "update_scenario": "Update Scenario",
                "Flag risk": "Flag Risk",
                "Manual review": "Manual Review",
                "Update scenario": "Update Scenario",
            }.get(value, value)
        )
    show_table(
        compact.rename(
            columns={
                "topic": "Topic",
                "model_line_affected": "Model Impact",
                "direction": "Direction",
                "confidence": "Confidence",
                "dashboard_action": "Action",
            }
        ),
        "No clauses match the selected filters.",
    )

    with st.expander("Show full clause table"):
        show_table(filtered[[column for column in full_cols if column in filtered]], "No clauses match the selected filters.")

    st.subheader("Review Actions")
    row_options = [f"{idx}: {row['topic']} -> {row['model_line_affected']}" for idx, row in filtered.reset_index().iterrows()]
    if not row_options:
        return
    selected_label = st.selectbox("Selected clause", row_options)
    selected_pos = int(selected_label.split(":", 1)[0])
    selected_row = filtered.reset_index(drop=True).iloc[selected_pos].to_dict()
    note = st.text_input("User note", key=f"clause_note_{ticker}")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Mark reviewed"):
            st.success("Marked reviewed placeholder. Persistent review workflow will be added after fetch quality is approved.")
    with c2:
        if st.button("Ignore"):
            st.info("Ignore placeholder. Clause was not removed from source evidence.")
    with c3:
        if st.button("Send to DCF assumption log"):
            update = create_pending_assumption_update({**selected_row, "user_note": note})
            st.session_state.setdefault("assumption_update_log", []).append(update)
            st.json(update)
    comparison = compare_clause_to_reverse_dcf(selected_row, ctx.get("reverse", {}))
    st.write("Reverse DCF check:", comparison.get("interpretation"))

    log = st.session_state.get("assumption_update_log", [])
    if log:
        st.subheader("Pending Assumption Update Log")
        show_table(pd.DataFrame(log), "No pending assumption updates.")

    debug = clause_df.attrs.get("debug", {})
    if debug:
        with st.expander("Debug details"):
            st.json(debug)


def _data_lab(ctx: dict, key_prefix: str = "data_lab") -> None:
    dataset = ctx["dataset"]
    historicals = ctx["historicals"]
    market = dataset.get("market_data", {})
    st.subheader("Reported Financials")
    show_table(historicals, "No financial table available.")

    st.subheader("SOTP Sandbox")
    manual_segments = pd.DataFrame(
        [
            {
                "segment": "Core business",
                "revenue": historicals["Revenue"].iloc[-1] if not historicals.empty else 0,
                "margin": ctx["base_assumptions"].get("nopat_margin", 0.12),
                "multiple": ctx["base_assumptions"].get("terminal_multiple", 15.0),
            }
        ]
    )
    sotp = run_sotp(manual_segments, {"default_margin": ctx["base_assumptions"].get("nopat_margin", 0.12), "default_multiple": ctx["base_assumptions"].get("terminal_multiple", 15.0)})
    metric_row([("SOTP EV", sotp.get("enterprise_value"), "money"), ("Net Debt", historicals["Net Debt"].iloc[-1] if not historicals.empty else None, "money")])
    show_table(sotp.get("segment_table"), "No SOTP table available.")
    st.plotly_chart(scenario_valuation_bar(ctx["base_dcf"]), width="stretch", key=f"{key_prefix}_v2_scenario")

    with st.expander("Raw provider snapshot"):
        st.json(
            {
                "ticker": dataset.get("ticker"),
                "company": dataset.get("company"),
                "sector": dataset.get("sector"),
                "industry": dataset.get("industry"),
                "cik": dataset.get("cik"),
                "sources": dataset.get("sources"),
                "warnings": dataset.get("warnings"),
                "market_data": market,
                "finviz_available": dataset.get("finviz", {}).get("available"),
                "yfinance_available": dataset.get("yfinance", {}).get("available"),
            }
        )


def _pa11r_snapshot(ctx: dict) -> None:
    dataset = ctx["dataset"]
    market = dataset.get("market_data", {})
    scoring = ctx["scoring"]
    dcf = ctx["base_dcf"]
    reverse = ctx["reverse"]
    moat = ctx["moat"]
    investment_status = "supportive" if scoring.get("recommendation") == "Buy" else "warning" if scoring.get("recommendation") == "Avoid" else "caution"
    swing_view, swing_subtitle, swing_status = _swing_view(ctx)
    regime, regime_subtitle, regime_status = _market_regime(ctx)
    valuation_view, valuation_subtitle, valuation_status = _valuation_view(ctx)
    risk_level, risk_subtitle, risk_status = _risk_level(ctx)
    confidence, confidence_subtitle, confidence_status = _data_confidence(ctx)
    moat_class = _clean_classification(moat.get("classification"))
    moat_score = moat.get("moat_score")

    render_section(
        "Decision Picture",
        "Investment, swing, market regime, valuation, moat, risk, and data confidence are deliberately separated so a stock can be fundamentally expensive but still technically tradable.",
        "Snapshot",
    )
    render_status_grid(
        [
            {
                "title": "Investment View",
                "value": scoring.get("recommendation") or "Unknown",
                "subtitle": scoring.get("position_size_guidance"),
                "status": investment_status,
                "score": scoring.get("total_score"),
                "confidence": scoring.get("conviction"),
            },
            {"title": "Swing View", "value": swing_view, "subtitle": swing_subtitle, "status": swing_status},
            {"title": "Market Regime", "value": regime, "subtitle": regime_subtitle, "status": regime_status},
        ]
    )
    render_status_grid(
        [
            {"title": "Valuation View", "value": valuation_view, "subtitle": valuation_subtitle, "status": valuation_status},
            {"title": "Fair Value Gap", "value": fmt_percent(dcf.get("upside_downside_pct")), "subtitle": f"Fair value {fmt_per_share(dcf.get('fair_value_per_share'))} vs price {fmt_per_share(market.get('price'))}.", "status": valuation_status},
            {"title": "Market-Implied Case", "value": reverse.get("market_case") or "Unknown", "subtitle": reverse.get("interpretation"), "status": "info"},
            {"title": "Moat", "value": moat_class, "subtitle": moat.get("terminal_value_implication") or "Review Business Quality tab for moat evidence.", "status": "caution" if moat_class == "Unknown" else "supportive" if "wide" in moat_class.lower() or "narrow" in moat_class.lower() else "warning", "confidence": moat.get("confidence"), "help_text": f"Moat score: {format_short_score(moat_score)}"},
            {"title": "Risk Level", "value": risk_level, "subtitle": risk_subtitle, "status": risk_status},
            {"title": "Data Confidence", "value": confidence, "subtitle": confidence_subtitle, "status": confidence_status},
        ]
    )
    render_section(
        "Valuation Method Reconciliation",
        "DCF, SOTP, and multiples are separate lenses. The snapshot only accepts the valuation read when they can be reconciled.",
        "DCF / SOTP / Multiples",
    )
    render_status_grid(_snapshot_valuation_cards(ctx))

    c1, c2 = st.columns([0.58, 0.42])
    with c1:
        render_section("Price Action", "Current market context is secondary to the investment view, but it helps with timing and exposure.", "Trading Context")
        st.plotly_chart(price_action_chart(dataset.get("price_history")), width="stretch", key="pa11r_snapshot_price")
    with c2:
        render_section("Accounting Reality Check", "Accounting warnings affect confidence and assumptions, not just score.", "Quality")
        _accounting_reality_compact(ctx)

    render_decision_summary(_decision_summary(ctx))
    with st.expander("Top 4 risks and manual-review plan", expanded=False):
        show_table(_risk_review_table(ctx, limit=4), "No top risks available.")
        show_table(_manual_review_plan_table(ctx), "No manual-review plan available.")
    with st.expander("Market / Fundamentals Summary", expanded=False):
        show_table(_finviz_decision_snapshot(market), "No market summary available.")
    with st.expander("One-page tear sheet", expanded=False):
        render_tearsheet(_tearsheet_summary(ctx))
        render_copy_summary(_tearsheet_summary(ctx))


def _assumption_update_log_editor(ctx: dict, key_prefix: str = "evidence") -> None:
    log = st.session_state.setdefault("assumption_update_log", [])
    if not log:
        log.extend(
            [
                {
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "case": "User Case",
                    "model_line": "revenue_cagr",
                    "old_value": ctx.get("base_assumptions", {}).get("revenue_cagr"),
                    "new_value": ctx.get("base_assumptions", {}).get("revenue_cagr"),
                    "evidence_source": "Manual",
                    "evidence_summary": "Starter log entry for user-reviewed assumption changes.",
                    "user_note": "",
                    "confidence": "Manual Review",
                    "fair_value_impact": UNAVAILABLE,
                    "status": "inactive",
                }
            ]
        )
    edited = st.data_editor(
        pd.DataFrame(log),
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        key=f"{key_prefix}_assumption_update_editor",
    )
    st.session_state["assumption_update_log"] = edited.to_dict("records")


def _assumption_workbench(ctx: dict, key_prefix: str = "evidence") -> None:
    render_section(
        "Assumption Workbench",
        "Map clauses, news, events, or manual evidence into DCF cases. Applied changes are tracked in an editable log before they affect the model.",
        "Evidence to Model",
    )
    clauses = ctx.get("clauses")
    if clauses is None or clauses.empty:
        st.info("No extracted clause evidence in fast mode. Use Clause Map to fetch SEC filing evidence, or add manual entries below.")
    else:
        compact_cols = [column for column in ["topic", "model_line_affected", "direction", "confidence", "suggested_assumption_change"] if column in clauses]
        show_table(clauses[compact_cols].head(8), "No clause evidence available.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        apply_case = st.selectbox(
            "Apply to case",
            ["Base Case", "Bull Case", "Bear Case", "User Case"],
            key=f"{key_prefix}_workbench_case",
        )
    with c2:
        model_line = st.selectbox(
            "Model line",
            ["revenue_cagr", "nopat_margin", "ocf_margin", "maintenance_capex_pct_revenue", "growth_capex_pct_revenue", "wacc", "terminal_multiple"],
            key=f"{key_prefix}_workbench_line",
        )
    with c3:
        new_value = st.text_input("New value", value="", key=f"{key_prefix}_workbench_value")
    with c4:
        confidence = st.selectbox(
            "Confidence",
            ["Manual Review", "Low", "Medium", "High"],
            key=f"{key_prefix}_workbench_confidence",
        )
    note = st.text_area("User note / evidence summary", value="", key=f"{key_prefix}_workbench_note", height=90)
    if st.button("Add assumption update", key=f"{key_prefix}_add_assumption_update"):
        st.session_state.setdefault("assumption_update_log", []).append(
            {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "case": apply_case,
                "model_line": model_line,
                "old_value": ctx.get("base_assumptions", {}).get(model_line),
                "new_value": new_value,
                "evidence_source": "User / Workbench",
                "evidence_summary": note or "Manual evidence item",
                "user_note": note,
                "confidence": confidence,
                "fair_value_impact": "Recalculate after activating",
                "status": "inactive",
            }
        )
        st.success("Assumption update added to the editable log.")
    _assumption_update_log_editor(ctx, key_prefix=key_prefix)


def _pa11r_valuation_tab(ctx: dict, analyst_details: bool) -> None:
    render_section(
        "Valuation Result",
        "Reverse DCF is treated as the market benchmark. Compare your base/user case against what the current price already implies.",
        "Valuation",
    )
    market = ctx["dataset"].get("market_data", {})
    dcf = ctx["base_dcf"]
    reverse = ctx["reverse"]
    assumptions = ctx["base_assumptions"]
    render_status_grid(
        [
            {"title": "Fair Value / Share", "value": fmt_per_share(dcf.get("fair_value_per_share")), "subtitle": "Base-case DCF output.", "status": "info"},
            {"title": "Current Price", "value": fmt_per_share(market.get("price")), "subtitle": "Market price from provider snapshot.", "status": "neutral"},
            {"title": "Upside / Downside", "value": fmt_percent(dcf.get("upside_downside_pct")), "subtitle": "Fair value versus market price.", "status": "supportive" if (dcf.get("upside_downside_pct") or 0) > 0 else "warning"},
            {"title": "MOS Buy Price", "value": fmt_per_share(dcf.get("buy_price_after_margin_of_safety")), "subtitle": "Buy zone after margin of safety.", "status": "info"},
            {"title": "Market-Implied Case", "value": reverse.get("market_case") or "Unknown", "subtitle": reverse.get("interpretation"), "status": "info"},
            {"title": "Terminal Value Weight", "value": fmt_percent(dcf.get("terminal_value_weight_pct")), "subtitle": "Higher weight means more terminal sensitivity.", "status": "warning" if (dcf.get("terminal_value_weight_pct") or 0) > 0.65 else "neutral"},
        ],
        numeric=True,
    )
    show_table(
        pd.DataFrame(
            [
                {"Case": "Bear Case", "Revenue CAGR": max((assumptions.get("revenue_cagr") or 0) - 0.03, -0.2), "NOPAT Margin": assumptions.get("nopat_margin"), "Terminal Multiple": max((assumptions.get("terminal_multiple") or 0) - 2, 0), "Read": "Stress lower growth / multiple"},
                {"Case": "Base Case", "Revenue CAGR": assumptions.get("revenue_cagr"), "NOPAT Margin": assumptions.get("nopat_margin"), "Terminal Multiple": assumptions.get("terminal_multiple"), "Read": "Dashboard base case"},
                {"Case": "Bull Case", "Revenue CAGR": (assumptions.get("revenue_cagr") or 0) + 0.05, "NOPAT Margin": assumptions.get("nopat_margin"), "Terminal Multiple": (assumptions.get("terminal_multiple") or 0) + 2, "Read": "Evidence-supported upside case"},
                {"Case": "User Case", "Revenue CAGR": assumptions.get("revenue_cagr"), "NOPAT Margin": assumptions.get("nopat_margin"), "Terminal Multiple": assumptions.get("terminal_multiple"), "Read": "Editable through assumption controls"},
                {"Case": "Market-Implied Case", "Revenue CAGR": reverse.get("implied_revenue_cagr"), "NOPAT Margin": reverse.get("implied_nopat_margin"), "Terminal Multiple": reverse.get("implied_terminal_multiple"), "Read": reverse.get("market_case")},
            ]
        ),
        "Scenario comparison unavailable.",
    )
    profile = infer_stock_profile(ctx["dataset"])
    st.info(f"Stock-profile assumption group: {profile}. The control panel below uses existing assumptions until profile-specific operating KPIs are available.")
    _valuation(ctx)
    if analyst_details:
        with st.expander("Assumption Workbench / Update Log", expanded=True):
            _assumption_workbench(ctx, key_prefix="valuation")


def _pa11r_evidence_assumptions_tab(ctx: dict, analyst_details: bool) -> None:
    render_section(
        "Evidence & Assumptions",
        "Use this tab to connect filing clauses and manual evidence to DCF assumptions. Tables stay behind expanders unless analyst details are enabled.",
        "Evidence",
    )
    _assumption_workbench(ctx, key_prefix="evidence")
    with st.expander("Clause / News / Event Map", expanded=analyst_details):
        _clause_annotation_map(ctx)
    with st.expander("Filing Metadata and Guidance", expanded=analyst_details):
        _evidence(ctx)


def _pa11r_business_quality_tab(ctx: dict, analyst_details: bool) -> None:
    render_section(
        "Business Quality",
        "Moat, accounting quality, new entrant risk, and thesis breakers are separated from pure valuation.",
        "Business Quality",
    )
    moat = ctx["moat"]
    render_status_grid(
        [
            {"title": "Moat", "value": _clean_classification(moat.get("classification")), "subtitle": moat.get("terminal_value_implication"), "status": "caution" if "unknown" in _clean_classification(moat.get("classification")).lower() else "supportive", "confidence": moat.get("confidence"), "help_text": f"Moat score: {format_short_score(moat.get('moat_score'))}"},
            {"title": "Accounting Quality", "value": ctx.get("accounting_interpretation", {}).get("cards", {}).get("OCF Quality") or "Unknown", "subtitle": "OCF / CAPEX / NOPAT interpretation affects DCF confidence.", "status": "caution"},
            {"title": "Risk Level", "value": _risk_level(ctx)[0], "subtitle": _risk_level(ctx)[1], "status": _risk_level(ctx)[2]},
        ]
    )
    _company_story(ctx)
    _accounting_quality(ctx)
    _moat_risks(ctx)


def _pa11r_management_tab(ctx: dict, analyst_details: bool) -> None:
    render_section(
        "Management & Capital Allocation",
        "Management credibility, SBC, M&A quality, compensation alignment, and peer context live here.",
        "Capital Allocation",
    )
    _ma_management_sbc(ctx)
    with st.expander("Multiples / Peer Context", expanded=analyst_details):
        _multiples_peers(ctx)


def _sources_data_quality_tab(ctx: dict, analyst_details: bool) -> None:
    confidence, subtitle, status = _data_confidence(ctx)
    render_section(
        "Sources & Data Quality",
        "Missing data is a controlled state. It creates a review/fetch plan instead of showing scary top-level errors.",
        "Data Quality",
    )
    render_status_grid(
        [
            {"title": "Data Coverage", "value": confidence, "subtitle": subtitle, "status": status},
            {"title": "Manual Review Items", "value": len(_manual_review_items(ctx)), "subtitle": "Debt, segment, moat, and CAPEX split items are tracked here.", "status": "caution" if _manual_review_items(ctx) else "supportive"},
            {"title": "Provider Sources", "value": ", ".join(ctx["dataset"].get("sources", [])) or "Unavailable", "subtitle": "SEC / Finviz / yfinance availability.", "status": "info"},
        ]
    )
    show_table(_manual_review_plan_table(ctx), "No manual-review plan available.")
    with st.expander("Data Coverage Table", expanded=analyst_details):
        show_table(_data_coverage(ctx["dataset"], ctx["historicals"]), "Data coverage unavailable.")
        show_table(_data_quality_table(ctx), "No data-quality notes.")
    with st.expander("Financial Reports", expanded=False):
        _financial_reports(ctx)
    if analyst_details:
        _data_lab(ctx, key_prefix="sources")


def _render_pa11r_hybrid(ctx: dict, analyst_details: bool) -> None:
    dataset = ctx["dataset"]
    render_cockpit_header(
        f"{dataset.get('ticker')} - {dataset.get('company') or 'Company unavailable'}",
        f"{dataset.get('sector') or 'Sector unavailable'} / {dataset.get('industry') or 'Industry unavailable'}",
        "PA-11R Hybrid Investment Cockpit",
    )
    _source_status(dataset)
    show_warnings(_critical_warnings(dataset.get("warnings", [])))
    tabs = st.tabs(
        [
            "Snapshot",
            "Valuation",
            "SOTP",
            "Multiples & Peers",
            "Evidence & Assumptions",
            "Business Quality",
            "Management & Capital Allocation",
            "Sources & Data Quality",
        ]
    )
    with tabs[0]:
        _pa11r_snapshot(ctx)
    with tabs[1]:
        _pa11r_valuation_tab(ctx, analyst_details)
    with tabs[2]:
        render_sotp_tab(ctx, analyst_details, key_prefix="pa11r_sotp")
    with tabs[3]:
        render_multiples_tab(ctx, key_prefix="pa11r_multiples")
    with tabs[4]:
        _pa11r_evidence_assumptions_tab(ctx, analyst_details)
    with tabs[5]:
        _pa11r_business_quality_tab(ctx, analyst_details)
    with tabs[6]:
        _pa11r_management_tab(ctx, analyst_details)
    with tabs[7]:
        _sources_data_quality_tab(ctx, analyst_details)


def _render_mr1_lite(ctx: dict, analyst_details: bool) -> None:
    dataset = ctx["dataset"]
    market = dataset.get("market_data", {})
    swing, swing_subtitle, swing_status = _swing_view(ctx)
    regime, regime_subtitle, regime_status = _market_regime(ctx)
    volume, volume_subtitle, volume_status = _volume_context(ctx)
    vol, vol_subtitle, vol_status = _swing_volatility(ctx)
    confidence, confidence_subtitle, confidence_status = _data_confidence(ctx)
    render_cockpit_header(
        f"{dataset.get('ticker')} MR-1 Lite",
        "Market regime, volume, relative context, volatility, and suggested swing exposure. This is separate from the PA-11R investment view.",
        "MR-1 Lite Trading Cockpit",
    )
    tabs = st.tabs(["Snapshot", "Trading Setup", "Regime & Relative Context", "Volume & Volatility", "Sources & Data Quality"])
    with tabs[0]:
        render_status_grid(
            [
                {"title": "Swing View", "value": swing, "subtitle": swing_subtitle, "status": swing_status},
                {"title": "Market Regime", "value": regime, "subtitle": regime_subtitle, "status": regime_status},
                {"title": "Suggested Exposure", "value": "Starter / Reduced" if swing_status != "supportive" else "Normal Swing", "subtitle": "Exposure is based on setup quality and data confidence.", "status": swing_status},
                {"title": "Volume Context", "value": volume, "subtitle": volume_subtitle, "status": volume_status},
                {"title": "Relative Context", "value": "Supportive" if (market.get("sma20") or 0) > 0 else "Neutral", "subtitle": f"SMA20 {fmt_percent(market.get('sma20'))}; SMA50 {fmt_percent(market.get('sma50'))}.", "status": "supportive" if (market.get("sma20") or 0) > 0 else "neutral"},
                {"title": "Data Confidence", "value": confidence, "subtitle": confidence_subtitle, "status": confidence_status},
            ]
        )
        render_decision_summary(
            {
                "what_matters": [f"Swing view is {swing}.", f"Market regime is {regime}.", f"Volume context is {volume}."],
                "supporting": [swing_subtitle, regime_subtitle],
                "contradicting": [volume_subtitle if volume_status in {"warning", "negative"} else vol_subtitle],
                "manual_review": [confidence_subtitle],
                "next_action": "Trade only when price action, volume, and regime align; otherwise wait or use reduced exposure.",
            }
        )
    with tabs[1]:
        render_section("Trading Setup", "Price action and exposure controls for swing decisions.", "MR-1")
        render_status_grid(
            [
                {"title": "Current Price", "value": fmt_per_share(market.get("price")), "subtitle": "Latest provider price.", "status": "info"},
                {"title": "Change", "value": fmt_percent(market.get("change")), "subtitle": "Current session / provider change.", "status": "supportive" if (market.get("change") or 0) > 0 else "warning"},
                {"title": "ATR", "value": fmt_ratio(market.get("atr")), "subtitle": "Volatility sizing input.", "status": "neutral"},
                {"title": "Beta", "value": fmt_ratio(market.get("beta")), "subtitle": "Market sensitivity.", "status": "warning" if (market.get("beta") or 0) > 1.25 else "neutral"},
            ],
            numeric=True,
        )
        st.plotly_chart(price_action_chart(dataset.get("price_history")), width="stretch", key="mr1_price")
    with tabs[2]:
        render_section("Regime & Relative Context", "Use trend and benchmark-like fields to avoid fighting the tape.", "MR-1")
        show_table(_finviz_decision_snapshot(market), "No relative context available.")
    with tabs[3]:
        render_section("Volume & Volatility", "Volume confirms or contradicts the move; volatility controls position size.", "MR-1")
        render_status_grid(
            [
                {"title": "Volume Context", "value": volume, "subtitle": volume_subtitle, "status": volume_status},
                {"title": "Swing Volatility", "value": vol, "subtitle": vol_subtitle, "status": vol_status},
                {"title": "Short Float", "value": fmt_percent(market.get("short_float")), "subtitle": "High short interest can increase volatility.", "status": "warning" if (market.get("short_float") or 0) > 0.15 else "neutral"},
            ]
        )
    with tabs[4]:
        _sources_data_quality_tab(ctx, analyst_details)


def render_dashboard():
    st.set_page_config(page_title="PA-11R Hybrid", layout="wide")
    _css()
    apply_design_system()

    with st.sidebar:
        st.header("Research Setup")
        dashboard_mode = st.radio("Dashboard", ["PA-11R Hybrid", "MR-1 Lite"], horizontal=False)
        ticker = st.text_input("Ticker", value="AAPL").upper().strip()
        peer_override = st.text_input("Peer override", value="", help="Comma-separated tickers")
        fetch_peers = st.toggle("Fetch peers", value=True)
        analyst_details = st.toggle("Show analyst details", value=False)
        debug = st.toggle("Show data lab", value=False)
        evidence_key = f"evidence_loaded_{ticker or 'EMPTY'}"
        st.caption("Fast mode loads SEC JSON metadata only. Evidence mode downloads full filing text once and reuses cache.")
        if st.button("Load SEC evidence", disabled=st.session_state.get(evidence_key, False)):
            st.session_state[evidence_key] = True
            cached_dataset.clear()
            st.rerun()
        if st.session_state.get(evidence_key, False) and st.button("Use fast mode"):
            st.session_state[evidence_key] = False
            cached_dataset.clear()
            st.rerun()
        if st.button("Refresh data", type="primary"):
            cached_dataset.clear()
            st.rerun()

    if not ticker:
        st.info("Enter a ticker to begin.")
        return

    include_deep_sec = bool(st.session_state.get(f"evidence_loaded_{ticker}", False))
    spinner_text = "Loading SEC filing evidence..." if include_deep_sec else "Building fast research snapshot..."
    with st.spinner(spinner_text):
        ctx = _build_context(ticker, peer_override, fetch_peers, include_deep_sec=include_deep_sec)

    st.caption("Mode: SEC evidence loaded" if include_deep_sec else "Mode: fast SEC JSON snapshot")
    if dashboard_mode == "MR-1 Lite":
        _render_mr1_lite(ctx, analyst_details or debug)
    else:
        _render_pa11r_hybrid(ctx, analyst_details or debug)
    if debug:
        with st.expander("Debug Data Lab", expanded=False):
            _data_lab(ctx, key_prefix="debug")
