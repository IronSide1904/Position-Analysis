from __future__ import annotations

import html
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
from persistence.analysis_store import (
    build_analysis_payload,
    compare_analyses,
    compute_state_hash,
    delete_analysis,
    duplicate_analysis,
    export_analysis_json,
    import_analysis_json,
    list_saved_analyses,
    load_analysis,
    save_analysis,
    update_analysis,
)
from models.dcf_model import (
    build_dcf_output_table,
    build_dcf_sensitivity_table,
    build_reverse_dcf_table,
    build_scenario_table,
    create_pending_assumption_update,
    default_assumptions_from_historicals,
    run_dcf,
)
from models.company_story import build_company_story_summary
from models.financial_derivations import add_percentage_change_rows
from models.financial_model import (
    build_ev_to_equity_bridge,
    build_financial_derivation_log,
    build_historical_financial_table,
    build_source_evidence_table,
    build_time_axis_financial_model,
)
from models.reverse_dcf import compare_clause_to_reverse_dcf, run_reverse_dcf
from models.scoring import score_investment
from models.sotp_model import build_default_segment_data, run_sotp, run_sotp_scenarios, sotp_summary_table
from models.multiples_model import calculate_current_multiples, calculate_scenario_implied_multiples, peer_median_multiples, sector_median_multiples
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
from ui.financial_charts import render_financial_line_chart
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
        .pa-dcf-hero {
            border: 1px solid rgba(45, 212, 191, 0.45);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin: 0.45rem 0 0.8rem 0;
            background: linear-gradient(135deg, rgba(8, 47, 73, 0.85), rgba(15, 23, 42, 0.92));
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 0 0 1px rgba(14, 165, 233, 0.08);
        }
        .pa-dcf-kicker {
            color: #99f6e4;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.18rem;
        }
        .pa-dcf-title {
            color: #f8fafc;
            font-size: 1.45rem;
            line-height: 1.15;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .pa-dcf-subtitle {
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.35;
            margin-bottom: 0.45rem;
        }
        .pa-dcf-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
        }
        .pa-dcf-chip {
            color: #ccfbf1;
            border: 1px solid rgba(45, 212, 191, 0.3);
            background: rgba(20, 184, 166, 0.12);
            border-radius: 999px;
            padding: 0.16rem 0.55rem;
            font-size: 0.76rem;
            font-weight: 700;
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
    return add_percentage_change_rows(pd.DataFrame(rows), line_item_col="Metric")


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


def _capex_view(ctx: dict) -> dict:
    assumptions = _normalize_assumption_bridge(ctx.get("base_assumptions", {}))
    maintenance = assumptions.get("maintenance_capex_pct_revenue")
    growth = assumptions.get("growth_capex_pct_revenue")
    total = assumptions.get("total_capex_pct_revenue")
    use_da = bool(assumptions.get("use_da_as_maintenance_capex_proxy"))
    capex = (ctx.get("accounting_interpretation") or {}).get("capex", {})
    if total is None:
        view = "Unclear"
    elif (growth or 0) > (maintenance or 0) * 1.25:
        view = "Growth-heavy"
    elif (maintenance or 0) > (growth or 0) * 1.25:
        view = "Maintenance-heavy"
    else:
        view = "Mixed"
    evidence_grade = "Proxy-based" if use_da else capex.get("confidence") or "Calculated"
    if capex.get("classification") and evidence_grade != "Proxy-based":
        evidence_grade = f"{evidence_grade} / clause-adjusted"
    dcf_impact = "Near-term FCF pressure" if (growth or 0) >= 0.03 else "Normal reinvestment drag"
    if use_da:
        dcf_impact = f"{dcf_impact}; maintenance CAPEX uses D&A proxy"
    return {
        "view": view,
        "maintenance": maintenance,
        "growth": growth,
        "total": total,
        "evidence_grade": evidence_grade,
        "method": "D&A proxy + explicit growth CAPEX" if use_da else "Explicit maintenance + growth CAPEX",
        "dcf_impact": dcf_impact,
    }


def _capex_snapshot_table(ctx: dict) -> pd.DataFrame:
    capex = _capex_view(ctx)
    return pd.DataFrame(
        [
            {"Metric": "CAPEX View", "Read": capex["view"]},
            {"Metric": "Maintenance CAPEX", "Read": fmt_percent(capex["maintenance"])},
            {"Metric": "Growth CAPEX", "Read": fmt_percent(capex["growth"])},
            {"Metric": "Total CAPEX", "Read": fmt_percent(capex["total"])},
            {"Metric": "CAPEX Evidence", "Read": capex["evidence_grade"]},
            {"Metric": "DCF impact", "Read": capex["dcf_impact"]},
        ]
    )


def _capex_bridge_table(ctx: dict) -> pd.DataFrame:
    historicals = ctx.get("historicals")
    assumptions = _normalize_assumption_bridge(ctx.get("base_assumptions", {}))
    if historicals is None or historicals.empty:
        return pd.DataFrame()
    latest = historicals.iloc[-1]
    revenue = latest.get("Revenue")
    reported_capex = latest.get("Total CAPEX")
    da = latest.get("D&A")
    if da is None and latest.get("EBITDA") is not None and latest.get("EBIT") is not None:
        da = max(float(latest.get("EBITDA") or 0) - float(latest.get("EBIT") or 0), 0)
    maintenance = float(revenue or 0) * float(assumptions.get("maintenance_capex_pct_revenue") or 0)
    growth = max(float(reported_capex or 0) - maintenance, 0)
    total = float(reported_capex or 0)
    return pd.DataFrame(
        [
            {"Metric": "Reported Total CAPEX", "Value": total, "Source Badge": "Reported", "Interpretation": "Cash flow statement CAPEX."},
            {"Metric": "D&A", "Value": da, "Source Badge": "Reported / Calculated", "Interpretation": "Potential maintenance CAPEX proxy when no better split is disclosed."},
            {"Metric": "Estimated Maintenance CAPEX", "Value": maintenance, "Source Badge": "Proxy-based" if assumptions.get("use_da_as_maintenance_capex_proxy") else "Calculated", "Interpretation": "Reinvestment needed to sustain current operations."},
            {"Metric": "Estimated Growth CAPEX", "Value": growth, "Source Badge": "Calculated", "Interpretation": "Reported CAPEX above estimated maintenance CAPEX."},
            {"Metric": "Growth CAPEX % Total CAPEX", "Value": growth / total if total else None, "Source Badge": "Calculated", "Interpretation": "Higher means near-term FCF is being used for expansion."},
            {"Metric": "CAPEX / Revenue", "Value": total / revenue if revenue else None, "Source Badge": "Calculated", "Interpretation": "Reinvestment intensity versus sales base."},
            {"Metric": "D&A / Revenue", "Value": da / revenue if revenue and da is not None else None, "Source Badge": "Calculated", "Interpretation": "D&A intensity versus sales base."},
            {"Metric": "CAPEX vs D&A ratio", "Value": total / da if da else None, "Source Badge": "Calculated", "Interpretation": "Above 1.0x can indicate growth investment, inflation, or under-depreciated assets."},
        ]
    )


def _clip_text(text: object, max_chars: int = 360) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean or clean == UNAVAILABLE:
        return UNAVAILABLE
    if len(clean) <= max_chars:
        return clean
    clipped = clean[: max_chars + 1].rsplit(" ", 1)[0].rstrip(" .,;:")
    return f"{clipped}..."


def _first_sentences(text: str | None, limit: int = 2, max_chars: int = 360) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return UNAVAILABLE
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    return _clip_text(" ".join(sentences[:limit]), max_chars=max_chars)


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
        st.write(_first_sentences(description, limit=2, max_chars=420))
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
        st.write(_clip_text(management.get("summary") or "Management story unavailable. Load SEC evidence for deeper founder, board, and governance context.", 360))
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
        "total_capex_pct_revenue",
        "depreciation_amortization_pct_revenue",
        "use_da_as_maintenance_capex_proxy",
        "capex_fade_year",
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
                "base value": format_assumption_value(assumptions.get(key), _assumption_unit(key)),
                "user value": format_assumption_value(assumptions.get(key), _assumption_unit(key)),
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
            return "font-weight: 700; color: #f8fafc; background-color: #0b1220;"
        if "YTD" in text or "LTM" in text:
            return "background-color: rgba(245, 158, 11, 0.16); color: #f8fafc;"
        if text.endswith("E"):
            return "background-color: rgba(245, 158, 11, 0.16); color: #f8fafc;"
        if text.endswith("F"):
            return "background-color: rgba(96, 165, 250, 0.14); color: #f8fafc;"
        if text.endswith("A"):
            return "background-color: rgba(15, 23, 42, 0.72); color: #f8fafc;"
        return "color: #f8fafc;"

    return df.style.apply(lambda row: [cell_style(value, column) for column, value in row.items()], axis=1)


def _format_financial_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    display = format_dataframe_for_display(df)
    if display is None or display.empty:
        return display
    return display.replace({UNAVAILABLE: "n.m.", "Unavailable": "n.m."}).fillna("n.m.")


def _show_financial_table(df: pd.DataFrame, empty_message: str = "Financial table unavailable.") -> None:
    if df is None or df.empty:
        st.info(empty_message)
    else:
        st.dataframe(_format_financial_table_for_display(df), width="stretch", hide_index=True)


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


def _ratio_or_none(numerator, denominator):
    try:
        if numerator is None or denominator in (None, 0) or pd.isna(numerator) or pd.isna(denominator):
            return None
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _latest_actual_column_label(historicals: pd.DataFrame | None) -> str:
    if historicals is None or historicals.empty or "Period" not in historicals:
        return "Latest Actual"
    period = str(historicals.iloc[-1].get("Period") or "").strip()
    if not period or period.lower() in {"latest", "nan", "none"}:
        return "Latest Actual"
    return period if period.endswith("A") else f"{period}A"


def _add_latest_actual_dcf_column(rows_by_metric: dict, historicals: pd.DataFrame | None, assumptions: dict) -> float | None:
    if historicals is None or historicals.empty:
        return None
    latest = historicals.iloc[-1]
    prior = historicals.iloc[-2] if len(historicals) >= 2 else None
    column = _latest_actual_column_label(historicals)

    def value(name: str):
        return latest.get(name) if name in latest else None

    revenue = value("Revenue")
    prior_revenue = prior.get("Revenue") if prior is not None and "Revenue" in prior else None
    revenue_growth = _ratio_or_none(float(revenue or 0) - float(prior_revenue or 0), prior_revenue) if prior_revenue else None
    ebit = value("EBIT")
    tax_rate = value("Tax Rate")
    if tax_rate is None:
        tax_rate = assumptions.get("tax_rate")
    nopat = value("NOPAT")
    if nopat is None and ebit is not None and tax_rate is not None:
        nopat = float(ebit) * (1 - float(tax_rate))
    da = value("D&A")
    if da is None and value("EBITDA") is not None and ebit is not None:
        da = max(float(value("EBITDA") or 0) - float(ebit or 0), 0)
    ocf = value("OCF")
    maintenance_capex = value("Maintenance CAPEX")
    total_capex = value("Total CAPEX")
    if maintenance_capex is None and revenue is not None:
        maintenance_capex = float(revenue) * float(assumptions.get("maintenance_capex_pct_revenue") or 0)
    growth_capex = value("Growth CAPEX")
    if growth_capex is None and total_capex is not None and maintenance_capex is not None:
        growth_capex = max(float(total_capex or 0) - float(maintenance_capex or 0), 0)
    if total_capex is None and maintenance_capex is not None and growth_capex is not None:
        total_capex = float(maintenance_capex or 0) + float(growth_capex or 0)
    working_capital = value("Working Capital Investment")
    fcf = value("FCF")
    if fcf is None and ocf is not None and total_capex is not None:
        fcf = float(ocf or 0) - float(total_capex or 0)
    fcff = value("FCFF")
    if fcff is None and nopat is not None and da is not None and maintenance_capex is not None:
        fcff = float(nopat or 0) + float(da or 0) - float(maintenance_capex or 0) - float(working_capital or 0)

    rows_by_metric["Revenue"][column] = revenue
    rows_by_metric["Revenue Growth %"][column] = revenue_growth
    rows_by_metric["EBIT"][column] = ebit
    rows_by_metric["EBIT Margin %"][column] = _ratio_or_none(ebit, revenue)
    rows_by_metric["Tax Rate"][column] = tax_rate
    rows_by_metric["NOPAT"][column] = nopat
    rows_by_metric["D&A"][column] = da
    rows_by_metric["D&A % Revenue"][column] = _ratio_or_none(da, revenue)
    rows_by_metric["OCF"][column] = ocf
    rows_by_metric["OCF Margin %"][column] = _ratio_or_none(ocf, revenue)
    rows_by_metric["Maintenance CAPEX"][column] = maintenance_capex
    rows_by_metric["Maintenance CAPEX % Revenue"][column] = _ratio_or_none(maintenance_capex, revenue)
    rows_by_metric["Growth CAPEX"][column] = growth_capex
    rows_by_metric["Growth CAPEX % Revenue"][column] = _ratio_or_none(growth_capex, revenue)
    rows_by_metric["Total CAPEX"][column] = total_capex
    rows_by_metric["Total CAPEX % Revenue"][column] = _ratio_or_none(total_capex, revenue)
    rows_by_metric["Working Capital"][column] = working_capital
    rows_by_metric["Working Capital % Revenue"][column] = _ratio_or_none(working_capital, revenue)
    rows_by_metric["FCF"][column] = fcf
    rows_by_metric["FCF Margin %"][column] = _ratio_or_none(fcf, revenue)
    rows_by_metric["FCFF"][column] = fcff
    return revenue


def _dcf_forecast_output_table(dcf_output: dict, assumptions: dict, historicals: pd.DataFrame | None = None) -> pd.DataFrame:
    forecast = dcf_output.get("forecast_table", pd.DataFrame())
    if forecast is None or forecast.empty:
        return pd.DataFrame()
    metrics = [
        "Revenue",
        "Revenue Growth %",
        "EBIT",
        "EBIT Margin %",
        "Tax Rate",
        "NOPAT",
        "D&A",
        "D&A % Revenue",
        "OCF",
        "OCF Margin %",
        "Maintenance CAPEX",
        "Maintenance CAPEX % Revenue",
        "Growth CAPEX",
        "Growth CAPEX % Revenue",
        "Total CAPEX",
        "Total CAPEX % Revenue",
        "Working Capital",
        "Working Capital % Revenue",
        "FCF",
        "FCF Margin %",
        "FCFF",
        "Discount Factor",
        "Discounted FCF / FCFF",
        "Terminal Value",
        "Discounted Terminal Value",
    ]
    rows = [{"Metric": metric} for metric in metrics]
    row_by_metric = {row["Metric"]: row for row in rows}
    prior_revenue = _add_latest_actual_dcf_column(row_by_metric, historicals, assumptions)
    for _, row in forecast.iterrows():
        year = int(row.get("Year") or 0)
        suffix = "E" if year == 1 else "F"
        column = f"FY{year}{suffix}"
        revenue = row.get("Revenue")
        revenue_growth = (revenue / prior_revenue - 1) if prior_revenue else assumptions.get("revenue_cagr")
        prior_revenue = revenue
        discount_factor = 1 / ((1 + float(assumptions.get("wacc", 0.095))) ** year)
        tax_rate = row.get("Tax Rate", float(assumptions.get("tax_rate", 0.21) or 0.21))
        nopat = row.get("NOPAT")
        ebit = row.get("EBIT")
        if ebit is None:
            ebit = nopat / max(1 - tax_rate, 0.01) if nopat is not None else None
        da = row.get("D&A")
        ocf = row.get("OCF")
        maintenance_capex = row.get("Maintenance CAPEX")
        growth_capex = row.get("Growth CAPEX")
        total_capex = row.get("Total CAPEX", row.get("CAPEX"))
        working_capital = row.get("Working Capital Investment")
        fcf = row.get("FCF")
        fcff = row.get("FCFF")
        row_by_metric["Revenue"][column] = revenue
        row_by_metric["Revenue Growth %"][column] = revenue_growth
        row_by_metric["EBIT"][column] = ebit
        row_by_metric["EBIT Margin %"][column] = row.get("EBIT Margin", ebit / revenue if revenue else None)
        row_by_metric["Tax Rate"][column] = tax_rate
        row_by_metric["NOPAT"][column] = nopat
        row_by_metric["D&A"][column] = da
        row_by_metric["D&A % Revenue"][column] = row.get("D&A % Revenue", da / revenue if revenue else None)
        row_by_metric["OCF"][column] = ocf
        row_by_metric["OCF Margin %"][column] = row.get("OCF Margin", ocf / revenue if revenue else None)
        row_by_metric["Maintenance CAPEX"][column] = maintenance_capex
        row_by_metric["Maintenance CAPEX % Revenue"][column] = row.get("Maintenance CAPEX % Revenue", maintenance_capex / revenue if revenue else None)
        row_by_metric["Growth CAPEX"][column] = growth_capex
        row_by_metric["Growth CAPEX % Revenue"][column] = row.get("Growth CAPEX % Revenue", growth_capex / revenue if revenue else None)
        row_by_metric["Total CAPEX"][column] = total_capex
        row_by_metric["Total CAPEX % Revenue"][column] = row.get("Total CAPEX % Revenue", total_capex / revenue if revenue else None)
        row_by_metric["Working Capital"][column] = working_capital
        row_by_metric["Working Capital % Revenue"][column] = row.get("Working Capital % Revenue", working_capital / revenue if revenue else None)
        row_by_metric["FCF"][column] = fcf
        row_by_metric["FCF Margin %"][column] = fcf / revenue if revenue else None
        row_by_metric["FCFF"][column] = fcff
        row_by_metric["Discount Factor"][column] = discount_factor
        row_by_metric["Discounted FCF / FCFF"][column] = row.get("PV FCF")
    row_by_metric["Terminal Value"]["Terminal"] = dcf_output.get("terminal_value")
    row_by_metric["Discounted Terminal Value"]["Terminal"] = dcf_output.get("discounted_terminal_value")
    return pd.DataFrame(rows)


ASSUMPTION_GROUPS = {
    "Growth": "These assumptions determine the size of the forecast revenue base.",
    "Margins": "These assumptions determine how much revenue converts into operating profit and NOPAT.",
    "Cash Conversion": "These assumptions determine whether accounting profit converts into operating cash flow.",
    "Reinvestment / CAPEX": "These assumptions determine how much cash must be reinvested before owners get FCF.",
    "Dilution": "These assumptions determine how enterprise value converts into per-share value.",
    "Terminal Value": "These assumptions drive the discount rate and the value after the explicit forecast period.",
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
        "group": "Margins",
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
        "group": "Margins",
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
        "group": "Margins",
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
        "group": "Margins",
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
        "group": "Reinvestment / CAPEX",
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
        "group": "Reinvestment / CAPEX",
        "description": "Capital expenditure required to maintain current operations.",
        "model_line": "Maintenance CAPEX",
        "affects": ["Normalized Cash Earnings", "FCFF", "FCF", "Fair Value"],
        "default_source": "Company disclosure or D&A proxy if undisclosed.",
        "reasonable_range": "Asset-light software may be low; asset-heavy industrial or infrastructure businesses may be much higher.",
        "warning": "D&A is only a proxy. It may be misleading for software, acquisition-heavy, data-center, infrastructure, or high-growth capacity-expansion companies.",
        "min": 0.0,
        "max": 0.25,
        "step": 0.005,
        "source": "Estimated",
    },
    "growth_capex_pct_revenue": {
        "label": "Growth CAPEX % Revenue",
        "unit": "percent",
        "group": "Reinvestment / CAPEX",
        "description": "Capital expenditure intended to create future revenue, capacity, automation, or efficiency.",
        "model_line": "Growth CAPEX",
        "affects": ["Near-term FCF", "Future Revenue", "Future Margin", "Fair Value"],
        "default_source": "Total CAPEX minus maintenance CAPEX, adjusted using filing clauses and business logic.",
        "reasonable_range": "Can be temporarily elevated during investment cycles, capacity expansion, infrastructure build-out, or automation projects.",
        "warning": "Do not treat all growth CAPEX as recurring maintenance cost, but do not ignore it if growth requires continuous reinvestment.",
        "min": 0.0,
        "max": 0.35,
        "step": 0.005,
        "source": "Estimated",
    },
    "total_capex_pct_revenue": {
        "label": "Total CAPEX % Revenue",
        "unit": "percent",
        "group": "Reinvestment / CAPEX",
        "description": "Total capital expenditures as a percentage of revenue, calculated as maintenance CAPEX plus growth CAPEX.",
        "model_line": "Total CAPEX",
        "affects": ["FCF", "Cash Conversion", "Reinvestment Intensity"],
        "default_source": "Reported cash flow statement CAPEX.",
        "reasonable_range": "Should be checked against company history and peers.",
        "warning": "High total CAPEX may reflect either growth investment or maintenance burden. Classify before using in normalized valuation.",
        "source": "Calculated",
        "derived": True,
    },
    "depreciation_amortization_pct_revenue": {
        "label": "D&A % Revenue",
        "unit": "percent",
        "group": "Reinvestment / CAPEX",
        "description": "Depreciation and amortization as a percentage of revenue, used in the FCFF bridge and optional maintenance CAPEX proxy.",
        "model_line": "D&A",
        "affects": ["FCFF", "Maintenance CAPEX Proxy", "Accounting Quality"],
        "default_source": "Reported D&A divided by revenue or financial-model estimate.",
        "reasonable_range": "Compare with asset intensity, accounting policy, and CAPEX history.",
        "warning": "D&A can understate maintenance needs for growth-heavy companies and overstate them for acquisition-heavy amortization.",
        "min": 0.0,
        "max": 0.30,
        "step": 0.005,
        "source": "Calculated",
    },
    "use_da_as_maintenance_capex_proxy": {
        "label": "Use D&A as Maintenance CAPEX Proxy",
        "unit": "bool",
        "group": "Reinvestment / CAPEX",
        "description": "When enabled, maintenance CAPEX follows D&A % revenue because no better maintenance-growth CAPEX split is disclosed.",
        "model_line": "Maintenance CAPEX",
        "affects": ["Maintenance CAPEX", "Normalized Cash Earnings", "FCFF", "Fair Value"],
        "default_source": "Enabled when D&A is available and no disclosed CAPEX split is present.",
        "reasonable_range": "Use only as a transparent proxy; disable if filings disclose maintenance CAPEX or D&A is unreliable.",
        "warning": "D&A proxy is never silent. It must be reviewed when the company is acquisition-heavy, asset-light, or investing for capacity.",
        "source": "Proxy-based",
    },
    "capex_fade_year": {
        "label": "CAPEX Normalization Year",
        "unit": "years",
        "group": "Reinvestment / CAPEX",
        "description": "Year when elevated growth CAPEX is expected to normalize.",
        "model_line": "Growth CAPEX Forecast",
        "affects": ["FCF Forecast", "Terminal Value"],
        "default_source": "Scenario assumption based on investment cycle and filing commentary.",
        "reasonable_range": "Usually 2-5 years depending on project duration and business model.",
        "warning": "Do not assume CAPEX normalizes without evidence from management commentary, project timing, or historical cycle.",
        "min": 2,
        "max": 5,
        "step": 1,
        "source": "Scenario-based",
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


TAB_CONTENT_MAP = {
    "DCF Model": [
        "assumption_workbench",
        "dcf_output",
        "ev_to_equity_bridge",
        "reverse_dcf",
        "scenario_table",
        "sensitivity_heatmap",
        "assumption_log",
    ],
    "Snapshot": [
        "decision_summary",
        "valuation_snapshot",
        "top_drivers",
        "top_risks",
        "top_evidence_impacts",
        "compact_market_summary",
        "compact_capex_view",
    ],
    "Valuation & DCF": [
        "assumption_workbench",
        "dcf_output",
        "ev_to_equity_bridge",
        "reverse_dcf",
        "scenario_table",
        "sensitivity_heatmap",
        "assumption_log",
    ],
    "Financials & Reinvestment": [
        "historical_financials",
        "margin_trends",
        "ocf_quality",
        "capex_bridge",
        "da_interpretation",
        "sbc_dilution",
    ],
    "Evidence & Assumptions": [
        "clause_map",
        "evidence_to_model_mapping",
        "guidance_implications",
        "manual_notes",
    ],
    "Business Quality & Risks": [
        "moat",
        "operating_leverage",
        "peer_quality",
        "management",
        "mna",
        "risks",
        "thesis_breakers",
    ],
    "Sources & Review": [
        "data_coverage",
        "manual_review_plan",
        "source_table",
        "debug_if_enabled",
    ],
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


ASSUMPTION_RANGE_DEFAULTS = {
    "revenue_cagr": {
        "mature": {"min": -0.05, "max": 0.15, "step": 0.005},
        "quality_growth": {"min": -0.05, "max": 0.25, "step": 0.005},
        "high_growth": {"min": -0.10, "max": 0.40, "step": 0.005},
        "turnaround": {"min": -0.20, "max": 0.30, "step": 0.005},
        "default": {"min": -0.05, "max": 0.25, "step": 0.005},
    },
    "gross_margin": {"default": {"min": 0.00, "max": 0.90, "step": 0.005}},
    "opex_pct_revenue": {"default": {"min": 0.00, "max": 1.00, "step": 0.005}},
    "tax_rate": {"default": {"min": 0.00, "max": 0.40, "step": 0.005}},
    "nopat_margin": {"default": {"min": -0.20, "max": 0.60, "step": 0.005}},
    "ocf_margin": {"default": {"min": -0.20, "max": 0.60, "step": 0.005}},
    "working_capital_pct_revenue": {"default": {"min": -0.10, "max": 0.20, "step": 0.005}},
    "maintenance_capex_pct_revenue": {"default": {"min": 0.00, "max": 0.30, "step": 0.0025}},
    "growth_capex_pct_revenue": {"default": {"min": 0.00, "max": 0.50, "step": 0.0025}},
    "depreciation_amortization_pct_revenue": {"default": {"min": 0.00, "max": 0.30, "step": 0.0025}},
    "capex_fade_year": {"default": {"min": 2, "max": 5, "step": 1}},
    "wacc": {"default": {"min": 0.06, "max": 0.18, "step": 0.0025}},
    "terminal_growth": {"default": {"min": -0.02, "max": 0.05, "step": 0.001}},
    "terminal_multiple": {"default": {"min": 5.0, "max": 35.0, "step": 0.5}},
    "sbc_pct_revenue": {"default": {"min": 0.00, "max": 0.30, "step": 0.005}},
    "diluted_share_growth": {"default": {"min": -0.10, "max": 0.20, "step": 0.005}},
    "margin_of_safety": {"default": {"min": 0.00, "max": 0.60, "step": 0.025}},
}


def _profile_range_key(stock_profile: str, assumption_key: str, base_value: float | None = None) -> str:
    profile = str(stock_profile or "").lower()
    if assumption_key != "revenue_cagr":
        return "default"
    if "software" in profile or "platform" in profile:
        return "high_growth" if (base_value or 0) >= 0.15 else "quality_growth"
    if "energy" in profile or "commodity" in profile or "financial" in profile:
        return "mature"
    if "consumer" in profile or "industrial" in profile or "hardware" in profile:
        return "quality_growth" if (base_value or 0) >= 0.08 else "mature"
    if base_value is not None and base_value < 0:
        return "turnaround"
    return "default"


def get_assumption_range(
    assumption_key: str,
    stock_profile: str,
    historical_value: float | None = None,
    base_value: float | None = None,
    bear_value: float | None = None,
    bull_value: float | None = None,
    market_implied_value: float | None = None,
) -> dict:
    """
    Return min, max, step, warning_level, and explanation for assumption control.
    """
    meta = ASSUMPTION_METADATA.get(assumption_key, {})
    unit = meta.get("unit")
    defaults = ASSUMPTION_RANGE_DEFAULTS.get(assumption_key)
    if defaults:
        range_key = _profile_range_key(stock_profile, assumption_key, base_value)
        default_config = dict(defaults.get(range_key) or defaults.get("default"))
    else:
        default_config = {
            "min": meta.get("min", 0.0),
            "max": meta.get("max", max(_assumption_float(base_value, 0), 1.0)),
            "step": meta.get("step", 1.0),
        }
    config = dict(default_config)
    values = [
        value
        for value in [historical_value, base_value, bear_value, bull_value, market_implied_value]
        if value is not None
    ]
    if values:
        buffer = 2.0 if unit == "multiple" else 0.03 if unit == "percent" else max(abs(max(values, key=abs)) * 0.05, 1.0)
        config["min"] = min(float(config["min"]), min(values) - buffer)
        config["max"] = max(float(config["max"]), max(values) + buffer)
    if unit == "percent":
        config["min"] = max(float(config["min"]), -1.0)
        config["max"] = min(float(config["max"]), 1.5)
    if unit == "multiple":
        config["min"] = max(float(config["min"]), 0.0)
        config["max"] = min(float(config["max"]), 80.0)
    expanded = bool(values and (min(values) < float(default_config["min"]) or max(values) > float(default_config["max"])))
    if assumption_key == "revenue_cagr" and not expanded:
        explanation = "Profile-aware range: normal companies use a tighter growth range; wider ranges appear only when evidence requires it."
    else:
        explanation = "Range expands automatically when base, bear, bull, historical, or market-implied values sit outside the default."
    config.update({"warning_level": "expanded" if expanded else "normal", "explanation": explanation})
    return config


def _assumption_unit(key: str) -> str:
    return ASSUMPTION_METADATA.get(key, {}).get("unit", "decimal")


def format_assumption_value(value, unit: str) -> str:
    if value is None:
        return UNAVAILABLE
    if unit == "bool":
        return "Yes" if bool(value) else "No"
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
    if unit == "bool":
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
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
    normalized.setdefault("use_da_as_maintenance_capex_proxy", bool(normalized.get("depreciation_amortization_pct_revenue")))
    if normalized.get("use_da_as_maintenance_capex_proxy"):
        normalized["maintenance_capex_pct_revenue"] = _assumption_float(
            normalized.get("depreciation_amortization_pct_revenue"),
            _assumption_float(normalized.get("maintenance_capex_pct_revenue"), 0.03),
        )
    normalized["maintenance_capex_pct_revenue"] = max(_assumption_float(normalized.get("maintenance_capex_pct_revenue")), 0.0)
    normalized["growth_capex_pct_revenue"] = max(_assumption_float(normalized.get("growth_capex_pct_revenue")), 0.0)
    normalized["total_capex_pct_revenue"] = normalized["maintenance_capex_pct_revenue"] + normalized["growth_capex_pct_revenue"]
    normalized["capex_fade_year"] = int(_assumption_float(normalized.get("capex_fade_year"), 3))
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
        "total_capex_pct_revenue": base.get("total_capex_pct_revenue"),
        "depreciation_amortization_pct_revenue": base.get("depreciation_amortization_pct_revenue"),
        "use_da_as_maintenance_capex_proxy": base.get("use_da_as_maintenance_capex_proxy"),
        "capex_fade_year": base.get("capex_fade_year"),
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
        "total_capex_pct_revenue",
        "depreciation_amortization_pct_revenue",
        "use_da_as_maintenance_capex_proxy",
        "working_capital_pct_revenue",
        "capex_fade_year",
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


def _historical_assumption_value(historicals: pd.DataFrame | None, key: str):
    if historicals is None or historicals.empty:
        return None
    if key in {"revenue_cagr", "diluted_share_growth"}:
        return None
    direct_column = {
        "gross_margin": "Gross Margin",
        "ocf_margin": "OCF Margin",
        "nopat_margin": "NOPAT Margin",
        "diluted_shares": "Diluted Shares",
        "net_debt": "Net Debt",
    }.get(key)
    if direct_column and direct_column in historicals:
        series = pd.to_numeric(historicals[direct_column], errors="coerce").dropna()
        return float(series.iloc[-1]) if not series.empty else None
    if "Revenue" not in historicals:
        return None
    revenue = pd.to_numeric(historicals["Revenue"], errors="coerce").replace(0, pd.NA)
    ratio_column = {
        "maintenance_capex_pct_revenue": "Maintenance CAPEX",
        "growth_capex_pct_revenue": "Growth CAPEX",
        "total_capex_pct_revenue": "Total CAPEX",
        "depreciation_amortization_pct_revenue": "D&A",
        "working_capital_pct_revenue": "Working Capital Investment",
        "sbc_pct_revenue": "SBC",
    }.get(key)
    if ratio_column and ratio_column in historicals:
        ratio = (pd.to_numeric(historicals[ratio_column], errors="coerce").abs() / revenue).dropna()
        return float(ratio.iloc[-1]) if not ratio.empty else None
    if key == "opex_pct_revenue" and {"Gross Margin", "EBIT"}.issubset(historicals.columns):
        ebit_margin = (pd.to_numeric(historicals["EBIT"], errors="coerce") / revenue).dropna()
        gross = pd.to_numeric(historicals["Gross Margin"], errors="coerce").dropna()
        if not ebit_margin.empty and not gross.empty:
            return max(float(gross.iloc[-1]) - float(ebit_margin.iloc[-1]), 0.0)
    return None


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


def _assumption_delta_text(delta: float | None, unit: str) -> str:
    if delta is None:
        return UNAVAILABLE
    if unit == "percent":
        return f"{delta * 100:+.1f} pts"
    if unit == "multiple":
        return f"{delta:+.1f}x"
    if unit == "money":
        return fmt_money(delta)
    if unit == "shares":
        return fmt_shares(delta)
    return fmt_ratio(delta, 2)


def _assumption_input_display(value: float | None, unit: str) -> float:
    if unit == "percent":
        return _assumption_float(value) * 100
    if unit == "years":
        return int(_assumption_float(value, 5))
    return _assumption_float(value)


def _assumption_input_model_value(value: float, unit: str):
    if unit == "percent":
        return value / 100
    if unit == "years":
        return int(value)
    return value


def _scenario_marker_pills(markers: list[tuple[str, object, str, str]]) -> str:
    parts = []
    for label, value, unit, status in markers:
        text = format_assumption_value(value, unit)
        css = "pa-pill-ok" if status == "user" else "pa-pill-warn" if status == "market" else ""
        parts.append(f'<span class="pa-pill {css}">{html.escape(label)}: {html.escape(str(text))}</span>')
    return " ".join(parts)


def _control_warning(key: str, value: float | None, range_info: dict, historicals: pd.DataFrame | None = None) -> str | None:
    value = _assumption_float(value)
    if range_info.get("warning_level") == "expanded":
        return "Range was expanded because a scenario, history, or market-implied value sits outside the normal profile range."
    if key == "revenue_cagr" and value > 0.25:
        return "Revenue CAGR above 25% requires strong evidence such as backlog, pricing power, capacity expansion, or secular growth."
    if key == "terminal_growth" and value > 0.03:
        return "Terminal growth above 3% requires durable moat evidence."
    if key == "wacc" and value < 0.06:
        return "WACC below 6% can overstate valuation unless risk is unusually low."
    if key == "ocf_margin" and historicals is not None and not historicals.empty and "OCF Margin" in historicals:
        historical_max = pd.to_numeric(historicals["OCF Margin"], errors="coerce").dropna()
        if not historical_max.empty and value > float(historical_max.max()) + 0.05:
            return "OCF margin is above recent history; require cash conversion evidence."
    return None


DCF_ROW_METADATA = {
    "revenue_cagr": {"label": "Revenue Growth %", "unit": "%", "assumption_key": "revenue_cagr", "source": "Scenario-based", "explanation_key": "revenue_cagr"},
    "cogs_pct_revenue": {"label": "COGS % Revenue", "unit": "%", "assumption_key": "gross_margin", "source": "Calculated", "explanation_key": "gross_margin"},
    "opex_pct_revenue": {"label": "OPEX % Revenue", "unit": "%", "assumption_key": "opex_pct_revenue", "source": "Calculated", "explanation_key": "opex_pct_revenue"},
    "tax_rate": {"label": "Tax Rate", "unit": "%", "assumption_key": "tax_rate", "source": "Estimated", "explanation_key": "tax_rate"},
    "nopat_margin": {"label": "NOPAT Margin Override %", "unit": "%", "assumption_key": "nopat_margin", "source": "Calculated", "explanation_key": "nopat_margin"},
    "ocf_margin": {"label": "OCF Margin %", "unit": "%", "assumption_key": "ocf_margin", "source": "Calculated", "explanation_key": "ocf_margin"},
    "depreciation_amortization_pct_revenue": {"label": "D&A % Revenue", "unit": "%", "assumption_key": "depreciation_amortization_pct_revenue", "source": "Calculated", "explanation_key": "depreciation_amortization_pct_revenue"},
    "maintenance_capex_pct_revenue": {"label": "Maintenance CAPEX % Revenue", "unit": "%", "assumption_key": "maintenance_capex_pct_revenue", "source": "Proxy-based", "explanation_key": "maintenance_capex_pct_revenue"},
    "growth_capex_pct_revenue": {"label": "Growth CAPEX % Revenue", "unit": "%", "assumption_key": "growth_capex_pct_revenue", "source": "Estimated", "explanation_key": "growth_capex_pct_revenue"},
    "working_capital_pct_revenue": {"label": "Working Capital % Revenue", "unit": "%", "assumption_key": "working_capital_pct_revenue", "source": "Estimated", "explanation_key": "working_capital_pct_revenue"},
    "sbc_pct_revenue": {"label": "SBC % Revenue", "unit": "%", "assumption_key": "sbc_pct_revenue", "source": "Calculated", "explanation_key": "sbc_pct_revenue"},
    "diluted_share_growth": {"label": "Diluted Share Growth %", "unit": "%", "assumption_key": "diluted_share_growth", "source": "Calculated", "explanation_key": "diluted_share_growth"},
}


VALUATION_ROW_METADATA = {
    "wacc": {"label": "WACC", "unit": "%", "assumption_key": "wacc", "source": "Estimated", "explanation_key": "wacc"},
    "terminal_growth": {"label": "Terminal Growth %", "unit": "%", "assumption_key": "terminal_growth", "source": "Scenario-based", "explanation_key": "terminal_growth"},
    "terminal_multiple": {"label": "Terminal Multiple", "unit": "x", "assumption_key": "terminal_multiple", "source": "Scenario-based", "explanation_key": "terminal_multiple"},
    "margin_of_safety": {"label": "Margin of Safety %", "unit": "%", "assumption_key": "margin_of_safety", "source": "User-edited", "explanation_key": "margin_of_safety"},
    "capex_fade_year": {"label": "CAPEX Normalization Year", "unit": "years", "assumption_key": "capex_fade_year", "source": "Scenario-based", "explanation_key": "capex_fade_year"},
}


ASSUMPTION_MODEL_LINE_MAP = {
    "revenue_cagr": "Revenue growth %",
    "cogs_pct_revenue": "COGS % revenue",
    "opex_pct_revenue": "OPEX % revenue",
    "tax_rate": "Tax rate",
    "nopat_margin": "NOPAT margin %",
    "ocf_margin": "OCF margin %",
    "depreciation_amortization_pct_revenue": "D&A % revenue",
    "maintenance_capex_pct_revenue": "Maintenance CAPEX % revenue",
    "growth_capex_pct_revenue": "Growth CAPEX % revenue",
    "working_capital_pct_revenue": "Working Capital % Revenue",
    "sbc_pct_revenue": "SBC % revenue",
    "diluted_share_growth": "Diluted shares growth %",
}


ASSUMPTION_FALLBACK_LINES = {
    "revenue_cagr": ["Revenue growth %", "Revenue % change"],
    "maintenance_capex_pct_revenue": ["Maintenance CAPEX % revenue", "D&A % revenue"],
    "sbc_pct_revenue": ["SBC % revenue"],
    "diluted_share_growth": ["Diluted shares growth %"],
}


def _latest_model_year(historicals: pd.DataFrame | None) -> int | None:
    if historicals is None or historicals.empty or "Period" not in historicals:
        return None
    for period in reversed(historicals["Period"].dropna().astype(str).tolist()):
        match = re.search(r"(20\d{2}|19\d{2})", period)
        if match:
            return int(match.group(1))
    return None


def _forecast_period_specs(historicals: pd.DataFrame | None, years: int) -> list[tuple[int, str]]:
    latest_year = _latest_model_year(historicals)
    specs = []
    for year in range(1, int(years or 5) + 1):
        if latest_year:
            label = f"FY{latest_year + year}{'E' if year == 1 else 'F'}"
        else:
            label = f"FY{year}{'E' if year == 1 else 'F'}"
        specs.append((year, label))
    return specs


def _model_period_columns(model_table: pd.DataFrame | None) -> list[str]:
    if model_table is None or model_table.empty:
        return []
    return [col for col in model_table.columns if col != "Line Item"]


def _model_line_value(model_table: pd.DataFrame | None, line_item: str, period: str):
    if model_table is None or model_table.empty or "Line Item" not in model_table or period not in model_table:
        return None
    rows = model_table[model_table["Line Item"].astype(str).str.lower() == str(line_item).lower()]
    if rows.empty:
        return None
    value = rows.iloc[0].get(period)
    if pd.isna(value):
        return None
    return value


def _first_model_line_value(model_table: pd.DataFrame | None, line_items: list[str], period: str):
    for line_item in line_items:
        value = _model_line_value(model_table, line_item, period)
        if value is not None:
            return value
    return None


def _numeric_or_none(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _assumption_actual_value(row_key: str, model_table: pd.DataFrame | None, period: str, assumptions: dict):
    meta = DCF_ROW_METADATA.get(row_key, {})
    direct_lines = ASSUMPTION_FALLBACK_LINES.get(row_key) or [ASSUMPTION_MODEL_LINE_MAP.get(row_key)]
    direct_value = _numeric_or_none(_first_model_line_value(model_table, [line for line in direct_lines if line], period))

    if direct_value is not None:
        if row_key == "depreciation_amortization_pct_revenue" and abs(direct_value) < 1e-12:
            return _numeric_or_none(assumptions.get(meta.get("assumption_key")))
        return direct_value

    if row_key == "growth_capex_pct_revenue":
        total_capex_pct = _numeric_or_none(_model_line_value(model_table, "Total CAPEX % revenue", period))
        maintenance_pct = _assumption_actual_value("maintenance_capex_pct_revenue", model_table, period, assumptions)
        if total_capex_pct is not None and maintenance_pct is not None:
            return max(total_capex_pct - maintenance_pct, 0.0)
        return _numeric_or_none(assumptions.get("growth_capex_pct_revenue")) or 0.0

    if row_key == "maintenance_capex_pct_revenue":
        da_pct = _numeric_or_none(_model_line_value(model_table, "D&A % revenue", period))
        if da_pct is not None and abs(da_pct) > 1e-12:
            return da_pct
        return _numeric_or_none(assumptions.get("maintenance_capex_pct_revenue")) or 0.0

    if row_key in {"working_capital_pct_revenue", "sbc_pct_revenue", "diluted_share_growth"}:
        return 0.0

    if row_key == "cogs_pct_revenue":
        gross_margin = _numeric_or_none(assumptions.get("gross_margin"))
        return 1 - gross_margin if gross_margin is not None else None

    return _numeric_or_none(assumptions.get(meta.get("assumption_key")))


def _period_change(current, previous):
    try:
        current = float(current)
        previous = float(previous)
    except (TypeError, ValueError):
        return None
    if abs(previous) < 1e-12:
        return None
    return current / previous - 1


def _display_assumption_number(value, unit: str):
    if value is None:
        return None
    if unit == "%":
        return round(float(value) * 100, 1)
    if unit == "years":
        return int(float(value))
    return round(float(value), 1)


def _internal_assumption_number(value, unit: str):
    if value is None or value == "":
        return None
    if unit == "%":
        return float(value) / 100
    if unit == "years":
        return int(float(value))
    return float(value)


def _matrix_value_for_key(assumptions: dict, year: int, row_key: str):
    yearly = assumptions.get("forecast_assumptions_by_year") or {}
    year_values = yearly.get(str(year)) or yearly.get(year) or {}
    if row_key == "cogs_pct_revenue":
        gross = year_values.get("gross_margin", assumptions.get("gross_margin"))
        return 1 - _assumption_float(gross, 0.45)
    meta = DCF_ROW_METADATA[row_key]
    return year_values.get(meta["assumption_key"], assumptions.get(meta["assumption_key"]))


def _build_assumption_matrix(
    assumptions: dict,
    historicals: pd.DataFrame | None,
    model_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[tuple[int, str]], list[str]]:
    specs = _forecast_period_specs(historicals, int(assumptions.get("forecast_years", 5) or 5))
    forecast_labels = [label for _, label in specs]
    actual_labels = [label for label in _model_period_columns(model_table) if label not in forecast_labels and label != "Terminal"]
    rows = []
    for row_key, meta in DCF_ROW_METADATA.items():
        row = {"Row Key": row_key, "Assumption": meta["label"], "Unit": meta["unit"], "Evidence": meta["source"]}
        for label in actual_labels:
            actual_value = _display_assumption_number(_assumption_actual_value(row_key, model_table, label, assumptions), meta["unit"])
            row[label] = 0.0 if actual_value is None and meta["unit"] == "%" else actual_value
        for year, label in specs:
            row[label] = _display_assumption_number(_matrix_value_for_key(assumptions, year, row_key), meta["unit"])
        rows.append(row)
    ordered_cols = ["Row Key", "Assumption", "Unit", "Evidence", *actual_labels, *forecast_labels]
    return pd.DataFrame(rows)[ordered_cols], specs, actual_labels


def _build_assumption_change_table(assumption_matrix: pd.DataFrame, period_columns: list[str]) -> pd.DataFrame:
    if assumption_matrix is None or assumption_matrix.empty:
        return pd.DataFrame()
    rows = []
    for _, source in assumption_matrix.iterrows():
        row = {
            "Assumption": f"{source.get('Assumption')} % change",
            "Unit": "%",
            "Evidence": "Calculated",
        }
        previous = None
        for period in period_columns:
            current = source.get(period)
            row[period] = _period_change(current, previous) if previous is not None else None
            previous = current
        rows.append(row)
    return pd.DataFrame(rows)


def _build_valuation_assumption_table(assumptions: dict) -> pd.DataFrame:
    rows = []
    for row_key, meta in VALUATION_ROW_METADATA.items():
        value = assumptions.get(meta["assumption_key"])
        rows.append(
            {
                "Row Key": row_key,
                "Assumption": meta["label"],
                "Unit": meta["unit"],
                "Value": _display_assumption_number(value, meta["unit"]),
                "Evidence": meta["source"],
            }
        )
    return pd.DataFrame(rows)


def handle_assumption_table_edit(edited_df, original_df, row_metadata, scenario_name, period_columns=None) -> list[dict]:
    changes = []
    original = original_df.set_index("Row Key")
    edited = edited_df.set_index("Row Key")
    columns = period_columns or ["Value"]
    for row_key, meta in row_metadata.items():
        if row_key not in edited.index or row_key not in original.index:
            continue
        for period in columns:
            if period not in edited.columns or period not in original.columns:
                continue
            old_value = original.at[row_key, period]
            new_value = edited.at[row_key, period]
            try:
                old_float = float(old_value)
                new_float = float(new_value)
            except (TypeError, ValueError):
                continue
            if abs(old_float - new_float) <= 0.000001:
                continue
            changes.append(
                {
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "scenario": scenario_name,
                    "row_key": row_key,
                    "label": meta["label"],
                    "period": period,
                    "old_value": old_float,
                    "new_value": new_float,
                    "unit": meta["unit"],
                    "source": "User-edited",
                    "reason / user note": "",
                    "fair_value_impact": "Recalculate after activation",
                    "status": "Active",
                }
            )
    return changes


def _apply_assumption_matrix(assumptions: dict, edited_matrix: pd.DataFrame, specs: list[tuple[int, str]]) -> dict:
    out = dict(assumptions)
    yearly = {str(key): dict(value) for key, value in (out.get("forecast_assumptions_by_year") or {}).items()}
    for _, row in edited_matrix.iterrows():
        row_key = row.get("Row Key")
        meta = DCF_ROW_METADATA.get(row_key)
        if not meta:
            continue
        for year, label in specs:
            value = _internal_assumption_number(row.get(label), meta["unit"])
            if value is None:
                continue
            year_values = yearly.setdefault(str(year), {})
            if row_key == "cogs_pct_revenue":
                year_values["gross_margin"] = max(0.0, min(1.0, 1 - value))
            else:
                year_values[meta["assumption_key"]] = value
    out["forecast_assumptions_by_year"] = yearly
    return _normalize_assumption_bridge(out, bool(out.get("use_direct_nopat_override")))


def _apply_valuation_assumption_table(assumptions: dict, edited_table: pd.DataFrame) -> dict:
    out = dict(assumptions)
    for _, row in edited_table.iterrows():
        row_key = row.get("Row Key")
        meta = VALUATION_ROW_METADATA.get(row_key)
        if not meta:
            continue
        value = _internal_assumption_number(row.get("Value"), meta["unit"])
        if value is not None:
            out[meta["assumption_key"]] = value
    return _normalize_assumption_bridge(out, bool(out.get("use_direct_nopat_override")))


def _append_unique_assumption_changes(ticker: str, changes: list[dict]) -> None:
    if not changes:
        return
    signature = "|".join(f"{item['scenario']}:{item['row_key']}:{item['period']}:{item['old_value']}->{item['new_value']}" for item in changes)
    state_key = f"last_assumption_matrix_signature_{ticker}"
    if st.session_state.get(state_key) == signature:
        return
    st.session_state[state_key] = signature
    st.session_state.setdefault("assumption_update_log", []).extend(changes)


def _render_matrix_validation_warnings(assumptions: dict, historicals: pd.DataFrame | None) -> None:
    warnings = validate_assumption_ranges(assumptions, historicals)
    if (assumptions.get("sbc_pct_revenue") or 0) > 0.10:
        warnings.append({"Assumption": "SBC % Revenue", "Current Value": format_assumption_value(assumptions.get("sbc_pct_revenue"), "percent"), "Severity": "Medium", "Reason": "SBC above 10% should flag dilution risk.", "Suggested Review": "Check diluted share growth and SBC quality."})
    if (assumptions.get("terminal_multiple") or 0) > 15:
        warnings.append({"Assumption": "Terminal Multiple", "Current Value": format_assumption_value(assumptions.get("terminal_multiple"), "multiple"), "Severity": "Medium", "Reason": "Terminal multiple above 15x requires durable growth or moat evidence.", "Suggested Review": "Review moat score, peer multiples, and terminal value weight."})
    if warnings:
        st.markdown('<div class="pa-section-title">Validation Warnings</div>', unsafe_allow_html=True)
        show_table(pd.DataFrame(warnings), "No validation warnings.")


def _render_assumption_matrix_workbench(ctx: dict, base: dict, working: dict, scenario_scope: str, profile: str) -> dict:
    ticker = ctx["dataset"].get("ticker", "default")
    market = ctx["dataset"].get("market_data", {})
    preview_dcf = run_dcf(ctx["historicals"], market, working)
    preview_model_table = build_time_axis_financial_model(ctx["historicals"], preview_dcf.get("forecast_table"), working)
    st.markdown('<div class="pa-section-title">Excel-Style DCF Assumption Model</div>', unsafe_allow_html=True)
    st.caption("Edit the forecast assumption rows directly. Percentage rows use human units: enter 8.0 for 8.0%. Actual and calculated output rows stay locked in the model output table.")
    original_matrix, specs, locked_period_columns = _build_assumption_matrix(working, ctx.get("historicals"), preview_model_table)
    period_columns = [label for _, label in specs]
    all_period_columns = [*locked_period_columns, *period_columns]
    read_only = scenario_scope == "Market-Implied Case"
    edited_matrix = st.data_editor(
        original_matrix,
        width="stretch",
        height=520,
        hide_index=True,
        column_config={
            "Row Key": None,
            **{label: st.column_config.NumberColumn(label, format="%.1f") for label in period_columns},
        },
        disabled=["Row Key", "Assumption", "Unit", "Evidence", *all_period_columns] if read_only else ["Row Key", "Assumption", "Unit", "Evidence", *locked_period_columns],
        key=f"dcf_assumption_matrix_{ticker}_{scenario_scope}",
    )
    changes = handle_assumption_table_edit(edited_matrix, original_matrix, DCF_ROW_METADATA, scenario_scope, period_columns)
    edited = _apply_assumption_matrix(working, edited_matrix, specs)

    assumption_change_table = _build_assumption_change_table(edited_matrix, all_period_columns)
    if not assumption_change_table.empty:
        st.markdown("**Assumption % Change by Period (Locked)**")
        st.caption("This mirrors the model time axis and shows each assumption's period-to-period change. It is calculated, not editable.")
        _show_financial_table(assumption_change_table, "Assumption change table unavailable.")

    val_col, explain_col = st.columns([0.48, 0.52])
    with val_col:
        st.markdown("**Terminal / Valuation Assumptions**")
        original_valuation = _build_valuation_assumption_table(edited)
        edited_valuation = st.data_editor(
            original_valuation,
            width="stretch",
            hide_index=True,
            column_config={"Row Key": None},
            disabled=["Row Key", "Assumption", "Unit", "Evidence", "Value"] if read_only else ["Row Key", "Assumption", "Unit", "Evidence"],
            key=f"dcf_valuation_matrix_{ticker}_{scenario_scope}",
        )
        changes.extend(handle_assumption_table_edit(edited_valuation, original_valuation, VALUATION_ROW_METADATA, scenario_scope, ["Value"]))
        edited = _apply_valuation_assumption_table(edited, edited_valuation)
    _append_unique_assumption_changes(ticker, changes)

    with explain_col:
        row_options = list(DCF_ROW_METADATA.keys()) + list(VALUATION_ROW_METADATA.keys())
        selected_row = st.selectbox(
            "Selected row explanation",
            row_options,
            format_func=lambda key: (DCF_ROW_METADATA.get(key) or VALUATION_ROW_METADATA.get(key))["label"],
            key=f"selected_dcf_matrix_row_{ticker}",
        )
        explanation_key = (DCF_ROW_METADATA.get(selected_row) or VALUATION_ROW_METADATA.get(selected_row))["explanation_key"]
        _render_assumption_explanation(explanation_key, profile, f"{scenario_scope} table edit", "User-edited" if changes else _assumption_source(explanation_key, scenario_scope, edited.get(explanation_key), base.get(explanation_key)), calculate_assumption_impact(base, edited, explanation_key, ctx["historicals"], market))

    edited_dcf = run_dcf(ctx["historicals"], market, edited)
    base_dcf = run_dcf(ctx["historicals"], market, base)
    st.markdown('<div class="pa-section-title">Live Valuation Recalculation</div>', unsafe_allow_html=True)
    metric_row(
        [
            ("Edited Fair Value", edited_dcf.get("fair_value_per_share"), "per_share"),
            ("Base Fair Value", base_dcf.get("fair_value_per_share"), "per_share"),
            ("Current Price", market.get("price"), "per_share"),
            ("Upside / Downside", edited_dcf.get("upside_downside_pct"), "pct"),
        ]
    )
    st.markdown('<div class="pa-section-title">Calculated DCF Output (Locked)</div>', unsafe_allow_html=True)
    st.caption("These rows recalculate from the editable assumption table above. They are display-only formula outputs, like the locked rows in an Excel model.")
    _show_financial_table(_dcf_forecast_output_table(edited_dcf, edited, ctx.get("historicals")), "DCF output unavailable.")
    _render_matrix_validation_warnings(edited, ctx.get("historicals"))
    if st.session_state.get("assumption_update_log"):
        st.markdown('<div class="pa-section-title">Assumption Change Log</div>', unsafe_allow_html=True)
        log_df = st.data_editor(
            pd.DataFrame(st.session_state.get("assumption_update_log", [])),
            width="stretch",
            hide_index=True,
            num_rows="dynamic",
            key=f"assumption_matrix_log_{ticker}",
        )
        st.session_state["assumption_update_log"] = log_df.to_dict("records")
    return edited


def render_assumption_slider(
    key: str,
    current_value: float,
    base_value: float,
    min_value: float,
    max_value: float,
    step: float,
    scenario_scope: str,
    *,
    stock_profile: str = "General",
    bear_value: float | None = None,
    bull_value: float | None = None,
    market_implied_value: float | None = None,
    historical_value: float | None = None,
    fair_value_impact: dict | None = None,
    historicals: pd.DataFrame | None = None,
):
    meta = ASSUMPTION_METADATA[key]
    unit = meta["unit"]
    range_info = get_assumption_range(
        key,
        stock_profile,
        historical_value=historical_value,
        base_value=base_value,
        bear_value=bear_value,
        bull_value=bull_value,
        market_implied_value=market_implied_value,
    )
    min_value = range_info.get("min", min_value)
    max_value = range_info.get("max", max_value)
    step = range_info.get("step", step)
    current_value = _bounded_value(current_value, min_value, max_value)
    delta = _assumption_float(current_value) - _assumption_float(base_value)
    delta_text = _assumption_delta_text(delta, unit)
    source = _assumption_source(key, scenario_scope, current_value, base_value)
    impact = fair_value_impact or {}
    impact_text = UNAVAILABLE
    if impact.get("fair_value_delta") is not None:
        impact_text = f"{fmt_per_share(impact.get('fair_value_delta'))} / share ({fmt_percent(impact.get('fair_value_delta_pct'))})"
    market_text = format_assumption_value(market_implied_value, unit)
    markers = [
        ("Bear", bear_value, unit, "bear"),
        ("Base", base_value, unit, "base"),
        ("User", current_value, unit, "user"),
        ("Bull", bull_value, unit, "bull"),
        ("Market", market_implied_value, unit, "market"),
    ]
    st.markdown(
        f"""
        <div class="pa-box">
            <div class="pa-box-title">{html.escape(meta["label"])}</div>
            <strong>User Case:</strong> {html.escape(format_assumption_value(current_value, unit))}
            &nbsp; <strong>Base:</strong> {html.escape(format_assumption_value(base_value, unit))}
            &nbsp; <strong>Market-Implied:</strong> {html.escape(str(market_text))}
            &nbsp; <strong>Delta vs Base:</strong> {html.escape(delta_text)}<br/>
            <span class="pa-pill">Scope: {html.escape(scenario_scope)} only</span>
            <span class="pa-pill">Source: {html.escape(source)}</span>
            <span class="pa-pill">Range: {html.escape(format_assumption_value(min_value, unit))} to {html.escape(format_assumption_value(max_value, unit))}</span><br/>
            <strong>Direction:</strong> Lower assumption &lt;- enter a smaller value | enter a larger value -&gt; Higher assumption<br/>
            <strong>Purpose:</strong> {html.escape(meta["description"])}<br/>
            <strong>Model impact:</strong> {html.escape(" -> ".join(meta.get("affects", [])))}<br/>
            <strong>Fair-value impact:</strong> {html.escape(impact_text)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    input_min = _assumption_input_display(min_value, unit)
    input_max = _assumption_input_display(max_value, unit)
    input_value = _assumption_input_display(current_value, unit)
    input_step = _assumption_input_display(step, unit) or 1.0
    input_label = "User value (%)" if unit == "percent" else "User value"
    if unit == "percent":
        edited_display = st.number_input(
            input_label,
            min_value=float(input_min),
            max_value=float(input_max),
            value=float(input_value),
            step=float(input_step),
            format="%.1f",
            help=f"Enter the exact assumption value. Difference versus Base Case is {delta_text}.",
            key=f"assumption_number_{scenario_scope}_{key}",
        )
    elif unit == "multiple":
        edited_display = st.number_input(
            "User value (x)",
            min_value=float(min_value),
            max_value=float(max_value),
            value=float(input_value),
            step=float(step),
            format="%.1f",
            help=f"Enter the exact multiple. Difference versus Base Case is {delta_text}.",
            key=f"assumption_number_{scenario_scope}_{key}",
        )
    elif unit == "years":
        edited_display = st.number_input(
            input_label,
            min_value=int(min_value),
            max_value=int(max_value),
            value=int(input_value),
            step=int(step),
            help="Enter the explicit forecast period.",
            key=f"assumption_number_{scenario_scope}_{key}",
        )
    elif unit == "shares":
        edited_display = st.number_input(
            input_label,
            min_value=max(0.0, float(min_value or 0)),
            max_value=float(max_value),
            value=max(float(input_value or 0), 0.0),
            step=1_000_000.0,
            format="%.0f",
            help="Enter diluted shares used for per-share valuation.",
            key=f"assumption_number_{scenario_scope}_{key}",
        )
    else:
        edited_display = st.number_input(
            input_label,
            min_value=float(min_value),
            max_value=float(max_value),
            value=float(input_value),
            step=float(step),
            help=f"Enter the exact assumption value. Difference versus Base Case is {delta_text}.",
            key=f"assumption_number_{scenario_scope}_{key}",
        )
    st.markdown(
        f"""
        <div class="pa-box">
            <strong>Scenario markers:</strong><br/>
            {_scenario_marker_pills(markers)}<br/>
            <strong>Range logic:</strong> {html.escape(range_info.get("explanation", ""))}
        </div>
        """,
        unsafe_allow_html=True,
    )
    warning = _control_warning(key, _assumption_input_model_value(edited_display, unit), range_info, historicals)
    if warning:
        st.warning(warning)
    return _assumption_input_model_value(edited_display, unit)


def _assumption_change_rows(base: dict, edited: dict, scenario_scope: str, historicals: pd.DataFrame, market: dict) -> list[dict]:
    rows = []
    for key in ASSUMPTION_KEYS:
        if ASSUMPTION_METADATA[key].get("derived"):
            continue
        old_value = base.get(key)
        new_value = edited.get(key)
        if ASSUMPTION_METADATA[key].get("unit") == "bool":
            if bool(new_value) == bool(old_value):
                continue
        elif abs(_assumption_float(new_value) - _assumption_float(old_value)) <= 0.000001:
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
                "Delta vs Base": _assumption_delta_text(_assumption_float(new_value) - _assumption_float(old_value), meta["unit"]),
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

    st.caption("Table-first workflow: choose a case, then edit the forecast rows below.")

    scope_col, compare_col = st.columns([0.58, 0.42])
    with scope_col:
        scenario_scope = st.segmented_control(
            "Which case are you editing?",
            ["User Case", "Base Case", "Bull Case", "Bear Case", "Market-Implied Case"],
            default="User Case",
            help="Choose which valuation case your assumption changes apply to. User Case is recommended. Market-Implied is read-only.",
        )
        scenario_scope = scenario_scope or "User Case"
    with compare_col:
        compare_to = st.selectbox("Compare assumption changes against", ["Base Case", "Market-Implied Case", "Prior User Case"], index=0)

    st.markdown(f'<span class="pa-pill pa-pill-ok">You are editing: {scenario_scope}</span> <span class="pa-pill">Compare: Current {scenario_scope} vs {compare_to}</span>', unsafe_allow_html=True)
    if scenario_scope not in {"User Case", "Market-Implied Case"}:
        st.warning("You are editing a core scenario. Consider using User Case unless you intentionally want to redefine the model framework.")
    if scenario_scope == "Market-Implied Case":
        st.info("Market-Implied Case is read-only. Use it as a benchmark, then copy a scenario to User Case for edits.")

    scenarios = _build_assumption_scenarios(base, st.session_state[user_state_key])
    prior_user_case = dict(st.session_state[user_state_key])
    market_case_assumptions = _market_implied_assumptions(reverse or {}, scenarios["Base Case"])
    working = dict(market_case_assumptions if scenario_scope == "Market-Implied Case" else scenarios[scenario_scope])

    with st.expander("Scenario reset / copy actions", expanded=False):
        preset_cols = st.columns(3)
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

    basis_default = next((label for label, item in VALUATION_BASIS_OPTIONS.items() if item["mode"] == str(working.get("dcf_mode", "FCFF")).upper()), "NOPAT bridge")
    working["dcf_mode"] = VALUATION_BASIS_OPTIONS[basis_default]["mode"]
    working["use_direct_nopat_override"] = bool(working.get("use_direct_nopat_override", False))
    working["use_da_as_maintenance_capex_proxy"] = bool(working.get("use_da_as_maintenance_capex_proxy", False))
    with st.expander("Model method controls", expanded=False):
        basis = st.segmented_control("Current valuation basis", list(VALUATION_BASIS_OPTIONS.keys()), default=basis_default)
        basis = basis or basis_default
        st.caption(VALUATION_BASIS_OPTIONS[basis]["description"])
        working["dcf_mode"] = VALUATION_BASIS_OPTIONS[basis]["mode"]

        direct_nopat_override = st.toggle(
            "Use direct NOPAT margin override instead of OPEX-derived EBIT bridge",
            value=bool(working.get("use_direct_nopat_override", False)),
            help="Off: NOPAT is derived from Gross Margin minus OPEX % Revenue, then tax. On: direct NOPAT margin controls NOPAT.",
        )
        working["use_direct_nopat_override"] = direct_nopat_override

        use_da_proxy = st.toggle(
            "Use D&A proxy for Maintenance CAPEX",
            value=bool(working.get("use_da_as_maintenance_capex_proxy", False)),
            help="When enabled, maintenance CAPEX follows D&A % revenue. Use this only when maintenance/growth CAPEX is not disclosed.",
        )
        working["use_da_as_maintenance_capex_proxy"] = use_da_proxy

    edited = _render_assumption_matrix_workbench(ctx, base, working, scenario_scope, profile)
    if scenario_scope == "User Case":
        st.session_state[user_state_key] = dict(edited)

    bottom_col, impact_col = st.columns([0.58, 0.42])
    with bottom_col:
        comparison = _scenario_comparison_table(scenarios, reverse, edited)
        st.markdown('<div class="pa-section-title">Scenario Comparison</div>', unsafe_allow_html=True)
        show_table(comparison, "Scenario comparison unavailable.")

    with impact_col:
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

    show_legacy_controls = False
    with st.expander("Optional legacy slider controls", expanded=False):
        show_legacy_controls = st.toggle("Enable legacy sliders for quick one-line adjustments", value=False, help="The DCF table above is the primary modeling interface.")
    if not show_legacy_controls:
        return edited

    selected_key = st.selectbox(
        "Selected assumption explanation",
        ASSUMPTION_KEYS,
        format_func=lambda key: ASSUMPTION_METADATA[key]["label"],
        help="Pick an assumption to see definition, scope, source, reasonable range, and model impact.",
    )
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
        if meta.get("derived"):
            edited = _normalize_assumption_bridge(edited, direct_nopat_override)
            st.markdown(
                f"""
                <div class="pa-box">
                    <div class="pa-box-title">{html.escape(meta["label"])}</div>
                    <strong>Calculated value:</strong> {html.escape(format_assumption_value(edited.get(key), meta["unit"]))}<br/>
                    <strong>Formula:</strong> Maintenance CAPEX % Revenue + Growth CAPEX % Revenue<br/>
                    <strong>Why it matters:</strong> {html.escape(meta["description"])}
                </div>
                """,
                unsafe_allow_html=True,
            )
            continue
        if meta.get("unit") == "bool":
            edited[key] = st.checkbox(
                meta["label"],
                value=bool(edited.get(key)),
                help=f"{meta['description']} Warning: {meta['warning']}",
                key=f"assumption_bool_{scenario_scope}_{key}",
            )
            _render_assumption_explanation(
                key,
                profile,
                f"{scenario_scope} only",
                _assumption_source(key, scenario_scope, edited.get(key), base.get(key)),
                None,
            )
            continue
        fair_value_impact = calculate_assumption_impact(base, edited, key, historicals, market)
        if "min" in meta:
            edited[key] = render_assumption_slider(
                key,
                edited.get(key),
                base.get(key),
                meta["min"],
                meta["max"],
                meta["step"],
                scenario_scope,
                stock_profile=profile,
                bear_value=scenarios["Bear Case"].get(key),
                bull_value=scenarios["Bull Case"].get(key),
                market_implied_value=market_case_assumptions.get(key),
                historical_value=_historical_assumption_value(historicals, key),
                fair_value_impact=fair_value_impact,
                historicals=historicals,
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
                stock_profile=profile,
                bear_value=scenarios["Bear Case"].get(key),
                bull_value=scenarios["Bull Case"].get(key),
                market_implied_value=market_case_assumptions.get(key),
                historical_value=_historical_assumption_value(historicals, key),
                fair_value_impact=fair_value_impact,
                historicals=historicals,
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
            disabled=["Old Value", "Delta vs Base", "Fair Value Impact", "Source"],
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
    company_story = build_company_story_summary(
        dataset,
        filing_texts=dataset.get("filing_texts", {}),
        peers=peer_df,
        news_items=dataset.get("news_items") or dataset.get("news"),
        social_buzz=dataset.get("social_buzz"),
        web_context=dataset.get("web_context"),
    )
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
        "company_story": company_story,
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


def _session_key_for_ticker(prefix: str, ticker: str) -> str:
    return f"{prefix}_{ticker or 'default'}"


def _current_user_assumptions(ctx: dict) -> dict:
    ticker = ctx.get("dataset", {}).get("ticker", "default")
    return dict(st.session_state.get(_session_key_for_ticker("assumption_user_case", ticker), ctx.get("base_assumptions", {})))


def _current_sotp_segments(ctx: dict):
    ticker = ctx.get("dataset", {}).get("ticker", "default")
    return st.session_state.get(f"sotp_{ticker}_segments", build_default_segment_data(ctx.get("historicals"), ctx.get("dataset", {}), ctx.get("base_assumptions", {})))


def _source_metadata_for_save(dataset: dict) -> dict:
    filings = dataset.get("filings", {}) or {}
    return {
        "sec": {
            "available": "SEC/EDGAR" in set(dataset.get("sources", [])),
            "cik": dataset.get("cik"),
            "latest_10k": filings.get("latest_10k"),
            "latest_10q": filings.get("latest_10q"),
            "latest_proxy": filings.get("latest_proxy"),
            "warnings": dataset.get("warnings", []),
        },
        "finviz": {
            "available": bool(dataset.get("finviz", {}).get("available")),
            "fields_used": sorted([key for key, value in (dataset.get("market_data", {}) or {}).items() if value is not None]),
            "warnings": [dataset.get("finviz", {}).get("error")] if dataset.get("finviz", {}).get("error") else [],
        },
        "yfinance": {
            "available": bool(dataset.get("yfinance", {}).get("available")) or "yfinance" in set(dataset.get("sources", [])),
            "warnings": [dataset.get("yfinance", {}).get("error")] if dataset.get("yfinance", {}).get("error") else [],
        },
    }


def _dcf_scenario_outputs(ctx: dict, user_assumptions: dict) -> tuple[dict, dict]:
    market = ctx.get("dataset", {}).get("market_data", {})
    scenarios = _build_assumption_scenarios(ctx.get("base_assumptions", {}), user_assumptions)
    market_case = _market_implied_assumptions(ctx.get("reverse", {}), scenarios["Base Case"])
    scenarios["Market-Implied Case"] = _normalize_assumption_bridge({**scenarios["Base Case"], **{k: v for k, v in market_case.items() if v is not None}})
    outputs = {case: run_dcf(ctx.get("historicals"), market, assumptions) for case, assumptions in scenarios.items()}
    return scenarios, outputs


def collect_dashboard_state(ctx: dict) -> dict:
    dataset = ctx.get("dataset", {})
    ticker = dataset.get("ticker", "default")
    market = dataset.get("market_data", {})
    user_assumptions = _current_user_assumptions(ctx)
    dcf_scenarios, dcf_outputs = _dcf_scenario_outputs(ctx, user_assumptions)
    sotp_segments = _current_sotp_segments(ctx)
    sotp_outputs = run_sotp_scenarios(sotp_segments, market, user_assumptions, dcf_outputs.get("User Case"), ctx.get("peer_df"), dataset.get("sector"))
    multiples_basis = st.session_state.get("pa11r_multiples_basis", "Normalized Year")
    scenario_multiples = calculate_scenario_implied_multiples(dcf_outputs, ctx.get("historicals"), market, multiples_basis)
    peer_medians, _warnings = peer_median_multiples(ctx.get("peer_df"), dataset.get("sector"), dataset.get("industry"))
    sector_medians = sector_median_multiples(dataset.get("sector"), dataset.get("industry"))
    swing_view, _swing_subtitle, _swing_status = _swing_view(ctx)
    regime, _regime_subtitle, _regime_status = _market_regime(ctx)
    scoring = ctx.get("scoring", {})
    notes = st.session_state.get("pa11r_user_notes", {})
    return {
        "analysis_id": st.session_state.get("loaded_analysis_id"),
        "created_at": st.session_state.get("loaded_analysis_created_at"),
        "decision": {
            "investment_view": scoring.get("recommendation"),
            "swing_view": swing_view,
            "market_regime": regime,
            "final_rating": scoring.get("recommendation"),
            "conviction": scoring.get("conviction"),
            "position_size_guidance": scoring.get("position_size_guidance"),
            "summary": ctx.get("thesis", {}).get("valuation_view", ""),
        },
        "data_sources": _source_metadata_for_save(dataset),
        "dcf": {
            "valuation_basis": st.session_state.get("dcf_valuation_basis", "OCF-based FCF"),
            "scenario_assumptions": dcf_scenarios,
            "scenario_outputs": dcf_outputs,
            "selected_case": "User Case",
            "assumption_update_log": st.session_state.get("assumption_update_log", []),
        },
        "sotp": {
            "enabled": True,
            "segment_assumptions": {"User Case": sotp_segments},
            "scenario_outputs": sotp_outputs,
            "manual_segments": sotp_segments,
            "whole_vs_sum_conclusion": sotp_outputs.get("Base Case", {}).get("whole_vs_sum_interpretation", ""),
        },
        "multiples": {
            "selected_multiple_basis": multiples_basis,
            "selected_multiple": st.session_state.get("pa11r_multiples_selected_multiple", "EV/OCF"),
            "peer_set": ctx.get("peer_df", pd.DataFrame()).to_dict("records") if isinstance(ctx.get("peer_df"), pd.DataFrame) else [],
            "scenario_implied_multiples": scenario_multiples,
            "peer_medians": peer_medians,
            "sector_medians": sector_medians,
            "user_notes": notes.get("valuation", ""),
        },
        "evidence": {
            "clause_mappings": ctx.get("clauses", pd.DataFrame()),
            "applied_evidence": st.session_state.get("assumption_update_log", []),
            "ignored_evidence": [],
            "manual_review_items": _manual_review_items(ctx),
        },
        "business_quality": {
            "moat": ctx.get("moat", {}),
            "risks": ctx.get("risks", {}),
            "thesis_breakers": [],
            "user_notes": notes.get("risks", ""),
        },
        "management_capital_allocation": {
            "management": ctx.get("management", {}),
            "ma_strategy": ctx.get("ma", {}),
            "compensation_sbc": ctx.get("alignment", {}),
            "user_notes": notes.get("management", ""),
        },
        "user_notes": {
            "general": notes.get("general", ""),
            "valuation": notes.get("valuation", ""),
            "thesis": notes.get("thesis", ""),
            "risks": notes.get("risks", ""),
            "manual_review": notes.get("manual_review", ""),
        },
    }


def restore_analysis_to_session_state(payload: dict, use_saved_market_snapshot: bool = False) -> None:
    ticker = str(payload.get("ticker") or "").upper()
    if ticker:
        st.session_state["research_ticker"] = ticker
    st.session_state["loaded_analysis_id"] = payload.get("analysis_id")
    st.session_state["loaded_analysis_name"] = payload.get("analysis_name")
    st.session_state["loaded_analysis_created_at"] = payload.get("created_at")
    st.session_state["loaded_analysis_updated_at"] = payload.get("updated_at")
    st.session_state["loaded_analysis_payload"] = payload
    st.session_state["loaded_analysis_hash"] = compute_state_hash(payload)
    if use_saved_market_snapshot:
        st.session_state["loaded_market_snapshot"] = payload.get("market_snapshot_at_save", {})
    user_case = payload.get("dcf", {}).get("scenario_assumptions", {}).get("User Case", {})
    if ticker and user_case:
        st.session_state[_session_key_for_ticker("assumption_user_case", ticker)] = user_case
    st.session_state["assumption_update_log"] = payload.get("dcf", {}).get("assumption_update_log", [])
    manual_segments = payload.get("sotp", {}).get("manual_segments", [])
    if ticker and manual_segments:
        st.session_state[f"sotp_{ticker}_segments"] = pd.DataFrame(manual_segments)
    multiples = payload.get("multiples", {})
    if multiples.get("selected_multiple_basis"):
        st.session_state["pa11r_multiples_basis"] = multiples.get("selected_multiple_basis")
    if multiples.get("selected_multiple"):
        st.session_state["pa11r_multiples_selected_multiple"] = multiples.get("selected_multiple")
    st.session_state["pa11r_user_notes"] = payload.get("user_notes", {})


def _analysis_state_cards(ctx: dict) -> list[dict]:
    loaded_name = st.session_state.get("loaded_analysis_name") or "Not loaded"
    current_state = collect_dashboard_state(ctx)
    current_hash = compute_state_hash(current_state)
    saved_hash = st.session_state.get("loaded_analysis_hash")
    unsaved = bool(saved_hash and current_hash != saved_hash)
    status = "warning" if unsaved else "supportive" if saved_hash else "neutral"
    value = "Unsaved Changes" if unsaved else "Saved" if saved_hash else "Not Saved"
    return [
        {
            "title": "Analysis State",
            "value": value,
            "subtitle": f"Loaded analysis: {loaded_name}. Last saved: {st.session_state.get('loaded_analysis_updated_at') or 'Never'}.",
            "status": status,
        }
    ]


def _saved_analysis_metadata_table(ctx: dict) -> pd.DataFrame:
    payload = st.session_state.get("loaded_analysis_payload") or {}
    market = ctx.get("dataset", {}).get("market_data", {})
    saved_market = payload.get("market_snapshot_at_save", {})
    return pd.DataFrame(
        [
            {"Field": "Analysis ID", "Value": payload.get("analysis_id")},
            {"Field": "Analysis Name", "Value": payload.get("analysis_name")},
            {"Field": "Created", "Value": payload.get("created_at")},
            {"Field": "Updated", "Value": payload.get("updated_at")},
            {"Field": "Dashboard Version", "Value": payload.get("dashboard_version")},
            {"Field": "Schema Version", "Value": payload.get("schema_version")},
            {"Field": "Price at Save", "Value": saved_market.get("price")},
            {"Field": "Current Price", "Value": market.get("price")},
            {"Field": "Data Sources Used", "Value": ", ".join(ctx.get("dataset", {}).get("sources", []))},
            {"Field": "Manual Review Items at Save", "Value": len(payload.get("evidence", {}).get("manual_review_items", [])) if payload else None},
        ]
    )


def _render_saved_analysis_sidebar(ctx: dict) -> None:
    dataset = ctx.get("dataset", {})
    ticker = dataset.get("ticker", "")
    with st.sidebar.expander("Save / Load Analysis", expanded=False):
        notes = st.session_state.setdefault("pa11r_user_notes", {})
        analysis_name = st.text_input("Analysis name", value=st.session_state.get("loaded_analysis_name") or f"{ticker} User Case", key="analysis_name_input")
        description = st.text_area("Description", value=st.session_state.get("analysis_description", ""), placeholder="What changed in this analysis?", key="analysis_description")
        tags_text = st.text_input("Tags", value=st.session_state.get("analysis_tags", ""), placeholder="watchlist, base-case, earnings-review", key="analysis_tags")
        notes["general"] = st.text_area("General notes", value=notes.get("general", ""), key="analysis_notes_general", height=80)
        notes["valuation"] = st.text_area("Valuation notes", value=notes.get("valuation", ""), key="analysis_notes_valuation", height=80)
        st.session_state["pa11r_user_notes"] = notes
        tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
        state = collect_dashboard_state(ctx)
        payload = build_analysis_payload(ticker, dataset, state, analysis_name, description, tags)
        current_hash = compute_state_hash(payload)
        loaded_hash = st.session_state.get("loaded_analysis_hash")
        if loaded_hash and current_hash != loaded_hash:
            st.warning("Unsaved changes")
        if st.button("Save New Analysis", key="save_new_analysis"):
            result = save_analysis(payload, overwrite=False)
            if result.get("success"):
                st.session_state["loaded_analysis_id"] = result.get("analysis_id")
                st.session_state["loaded_analysis_name"] = analysis_name
                st.session_state["loaded_analysis_hash"] = compute_state_hash(payload)
                st.success(f"Saved successfully: {result.get('analysis_id')}")
            else:
                st.error(result.get("message"))
        loaded_id = st.session_state.get("loaded_analysis_id")
        confirm_update = st.checkbox("Confirm update existing analysis", value=False, key="confirm_update_analysis")
        if st.button("Update Current Analysis", key="update_current_analysis", disabled=not loaded_id or not confirm_update):
            payload["analysis_id"] = loaded_id
            result = update_analysis(loaded_id, payload)
            st.success("Updated successfully." if result.get("success") else result.get("message"))

        show_all = st.checkbox("Show all tickers", value=False, key="show_all_saved_analyses")
        saved = list_saved_analyses(None if show_all else ticker)
        selected = st.selectbox(
            "Load saved analysis",
            options=saved,
            format_func=lambda item: f"{item.get('analysis_name')} | {item.get('updated_at')} | {item.get('final_rating')}",
            key="selected_saved_analysis",
        ) if saved else None
        use_saved_market = st.checkbox("View saved market snapshot instead of only latest", value=False, key="use_saved_market_snapshot")
        if selected and st.button("Load Analysis", key="load_saved_analysis"):
            try:
                loaded = load_analysis(selected["analysis_id"])
                restore_analysis_to_session_state(loaded, use_saved_market_snapshot=use_saved_market)
                st.success("Loaded successfully.")
                st.rerun()
            except Exception as exc:
                st.error(f"Load failed: {exc}")
        if selected:
            if st.button("Duplicate", key="duplicate_analysis"):
                result = duplicate_analysis(selected["analysis_id"], f"{selected.get('analysis_name')} Copy")
                st.success("Duplicated successfully." if result.get("success") else result.get("message"))
            confirm_delete = st.checkbox("Confirm delete selected analysis", value=False, key="confirm_delete_analysis")
            if st.button("Delete", key="delete_analysis", disabled=not confirm_delete):
                result = delete_analysis(selected["analysis_id"])
                st.success("Deleted successfully." if result.get("success") else result.get("message"))
                st.rerun()
            st.download_button(
                "Export Analysis JSON",
                data=export_analysis_json(selected["analysis_id"]),
                file_name=f"{selected['analysis_id']}.json",
                mime="application/json",
                key="export_analysis_json",
            )
        uploaded = st.file_uploader("Import Analysis JSON", type=["json"], key="import_analysis_json")
        if uploaded is not None and st.button("Import Analysis", key="import_analysis_button"):
            result = import_analysis_json(uploaded)
            st.success("Imported successfully." if result.get("success") else result.get("message"))

        if len(saved) >= 2:
            st.markdown("**Compare Saved Analyses**")
            version_a = st.selectbox("Version A", saved, format_func=lambda item: item.get("analysis_name"), key="compare_version_a")
            version_b = st.selectbox("Version B", saved, format_func=lambda item: item.get("analysis_name"), key="compare_version_b")
            if st.button("Show Differences", key="compare_analyses"):
                diff = compare_analyses(load_analysis(version_a["analysis_id"]), load_analysis(version_b["analysis_id"]))
                show_table(pd.DataFrame(diff.get("differences", [])), "No differences available.")


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
    assumptions = _assumption_editor(ctx)
    user_dcf = run_dcf(ctx["historicals"], market, assumptions)
    reverse = run_reverse_dcf(market, ctx["historicals"], assumptions)
    model_table = build_time_axis_financial_model(ctx["historicals"], user_dcf.get("forecast_table"), assumptions)
    derivation_log = build_financial_derivation_log(model_table)
    dcf_output_table = _dcf_forecast_output_table(user_dcf, assumptions, ctx["historicals"])
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

    st.markdown('<div class="pa-section-title">Valuation Readout</div>', unsafe_allow_html=True)
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
            ("MOS Buy Price", user_dcf.get("buy_price_after_margin_of_safety"), "per_share"),
        ]
    )
    metric_row(
        [
            ("Terminal Value % EV", user_dcf.get("terminal_value_weight_pct"), "pct"),
            ("DCF Confidence", ctx.get("accounting_interpretation", {}).get("valuation_confidence"), "text"),
            ("Reverse DCF", reverse.get("market_case"), "text"),
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

    st.markdown('<div class="pa-section-title">Charts & Detailed Review</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fcf_projection_chart(ctx["historicals"], user_dcf["forecast_table"]), width="stretch", key="v2_fcf_projection")
        st.caption("The line chart compares reported FCF with the forecast generated from the current assumption table.")
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
        forecast_period_cols = [col for col in model_table.columns if str(col).endswith(("E", "F"))]
        forecast_rows = [
            "Revenue",
            "Revenue % change",
            "Gross profit",
            "Gross profit % change",
            "Total OPEX",
            "Total OPEX % change",
            "EBIT",
            "EBIT % change",
            "NOPAT",
            "NOPAT % change",
            "Operating cash flow",
            "Operating cash flow % change",
            "Maintenance CAPEX",
            "Maintenance CAPEX % change",
            "Growth CAPEX",
            "Growth CAPEX % change",
            "Total CAPEX",
            "Total CAPEX % change",
            "FCF",
            "FCF % change",
            "D&A",
            "D&A % change",
        ]
        forecast_model_table = model_table[model_table["Line Item"].isin(forecast_rows)][["Line Item", *forecast_period_cols]] if forecast_period_cols else pd.DataFrame()
        _show_financial_table(forecast_model_table, "Forecast unavailable.")
        render_financial_line_chart(
            forecast_model_table,
            "Forecast Table: Selected Line Items",
            default_items=["Revenue", "Gross profit", "Total OPEX", "Operating cash flow", "NOPAT", "FCF"],
            key_prefix="valuation_forecast",
        )

    with st.expander("1. Historical Financials / Operating Model", expanded=True):
        operating_table = model_table[model_table["Line Item"].isin([
            "Revenue",
            "Revenue % change",
            "Revenue growth %",
            "COGS / Cost of sales",
            "COGS / Cost of sales % change",
            "COGS % revenue",
            "Gross profit",
            "Gross profit % change",
            "Gross margin %",
            "Total OPEX",
            "Total OPEX % change",
            "OPEX % revenue",
            "EBIT",
            "EBIT % change",
            "EBIT margin %",
            "D&A",
            "D&A % change",
            "D&A % revenue",
            "EBITDA",
            "EBITDA % change",
            "EBITDA margin %",
            "Tax rate",
            "NOPAT",
            "NOPAT % change",
            "NOPAT margin %",
        ])]
        render_financial_line_chart(operating_table, "Historical Financials / Operating Model: Selected Line Items", key_prefix="valuation_operating")
        _show_financial_table(operating_table)
    with st.expander("2. Cash Flow / CAPEX / NOPAT", expanded=True):
        cash_flow_table = model_table[model_table["Line Item"].isin([
            "Operating cash flow",
            "Operating cash flow % change",
            "OCF margin %",
            "Adjusted OCF",
            "Adjusted OCF % change",
            "Adjusted OCF margin %",
            "Maintenance CAPEX",
            "Maintenance CAPEX % change",
            "Maintenance CAPEX % revenue",
            "Growth CAPEX",
            "Growth CAPEX % change",
            "Growth CAPEX % revenue",
            "Total CAPEX",
            "Total CAPEX % change",
            "Total CAPEX % revenue",
            "FCF",
            "FCF % change",
            "FCF margin %",
            "Adjusted FCF",
            "Adjusted FCF % change",
            "Adjusted FCF margin %",
        ])]
        render_financial_line_chart(cash_flow_table, "Cash Flow / CAPEX / NOPAT: Selected Line Items", key_prefix="valuation_cash_flow")
        _show_financial_table(cash_flow_table)
    with st.expander("3. Forecast Assumptions"):
        show_table(assumptions_table, "Assumptions unavailable.")
        st.subheader("Accounting-Driven Assumption Flags")
        st.caption("These are suggested reviews only. The dashboard does not override your assumptions without confirmation.")
        show_table(accounting_flags, "No accounting-driven assumption flags available.")
    with st.expander("4. DCF Output"):
        render_financial_line_chart(
            dcf_output_table,
            "DCF Scenario Forecast Table: Selected Line Items",
            line_item_col="Metric",
            default_items=["Revenue", "NOPAT", "OCF", "Maintenance CAPEX", "Growth CAPEX", "Total CAPEX", "FCF", "FCFF"],
            key_prefix="valuation_dcf_output",
        )
        _show_financial_table(dcf_output_table, "DCF output unavailable.")
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
    with st.expander("How these rows were calculated"):
        show_table(derivation_log, "No derived rows were required for the current model table.")


def _financial_reports(ctx: dict) -> None:
    historicals = ctx["historicals"]
    st.caption("Financials: Excel-style model layout with financial line items down rows and time periods across columns.")
    if historicals is None or historicals.empty:
        st.info("No reported financial table is available.")
        return
    dcf = run_dcf(historicals, ctx["dataset"].get("market_data", {}), ctx["base_assumptions"])
    model_table = build_time_axis_financial_model(historicals, dcf.get("forecast_table"), ctx["base_assumptions"])
    derivation_log = build_financial_derivation_log(model_table)
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
        render_financial_line_chart(
            model_table,
            "Financial Model: Selected Line Items",
            default_items=["Revenue", "Gross profit", "Total OPEX", "Operating cash flow", "NOPAT", "FCF"],
            key_prefix="reports_financial_model",
        )
        st.dataframe(_style_financial_model_table(_format_financial_table_for_display(model_table)), width="stretch", hide_index=True)

    with st.expander("Row Groups: Operating Model"):
        operating_rows = model_table[model_table["Line Item"].isin([
            "Revenue",
            "Revenue % change",
            "Revenue growth %",
            "COGS / Cost of sales",
            "COGS / Cost of sales % change",
            "COGS % revenue",
            "Gross profit",
            "Gross profit % change",
            "Gross margin %",
            "S&M",
            "S&M % revenue",
            "R&D",
            "R&D % revenue",
            "G&A",
            "G&A % revenue",
            "Total OPEX",
            "Total OPEX % change",
            "OPEX % revenue",
            "EBIT",
            "EBIT % change",
            "EBIT margin %",
        ])]
        render_financial_line_chart(operating_rows, "Financial Reports: Operating Model Lines", key_prefix="reports_operating")
        _show_financial_table(operating_rows)
    with st.expander("Row Groups: Cash Flow / CAPEX / SBC"):
        cash_rows = model_table[model_table["Line Item"].isin([
            "Operating cash flow",
            "Operating cash flow % change",
            "OCF margin %",
            "Adjusted OCF",
            "Adjusted OCF % change",
            "Adjusted OCF margin %",
            "Maintenance CAPEX",
            "Maintenance CAPEX % change",
            "Maintenance CAPEX % revenue",
            "Growth CAPEX",
            "Growth CAPEX % change",
            "Growth CAPEX % revenue",
            "Total CAPEX",
            "Total CAPEX % change",
            "Total CAPEX % revenue",
            "FCF",
            "FCF % change",
            "FCF margin %",
            "Adjusted FCF",
            "Adjusted FCF % change",
            "Adjusted FCF margin %",
            "SBC",
            "SBC % change",
            "SBC % revenue",
            "SBC % gross profit",
            "Diluted shares",
            "Diluted shares % change",
            "Diluted shares growth %",
        ])]
        render_financial_line_chart(cash_rows, "Financial Reports: Cash Flow / CAPEX Lines", key_prefix="reports_cash_flow")
        _show_financial_table(cash_rows)

    with st.expander("SBC / Dilution"):
        _show_financial_table(model_table[model_table["Line Item"].isin(["SBC", "SBC % revenue", "SBC % gross profit", "SBC % OCF", "Diluted shares", "Diluted shares growth %"])])
    with st.expander("How these rows were calculated"):
        show_table(derivation_log, "No derived rows were required for the current model table.")


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
    render_status_grid(_analysis_state_cards(ctx))
    render_section(
        "Valuation Method Reconciliation",
        "DCF, SOTP, and multiples are separate lenses. The snapshot only accepts the valuation read when they can be reconciled.",
        "DCF / SOTP / Multiples",
    )
    render_status_grid(_snapshot_valuation_cards(ctx))

    c1, c2 = st.columns([0.58, 0.42])
    with c1:
        render_section("What Matters Most", "Snapshot keeps only the items most likely to change the decision or model.", "Drivers")
        _mini_list("Top 3 valuation drivers", _top_three_drivers(ctx))
        _mini_list("Top 3 risks", _top_three_risks(ctx))
        st.subheader("Top Model-Changing Evidence")
        show_table(_top_clause_impacts(ctx["clauses"]), "No clause-driven model impacts available yet.")
    with c2:
        render_section("Compact Market / Reinvestment Strip", "Market data and CAPEX are summarized here; full detail lives in the dedicated tabs.", "Snapshot")
        show_table(_finviz_decision_snapshot(market).head(6), "No market summary available.")
        st.subheader("CAPEX Summary")
        show_table(_capex_snapshot_table(ctx), "CAPEX summary unavailable.")
        _accounting_reality_compact(ctx)

    render_decision_summary(_decision_summary(ctx))
    with st.expander("One-page tear sheet", expanded=False):
        render_tearsheet(_tearsheet_summary(ctx))
        render_copy_summary(_tearsheet_summary(ctx))


def _company_story_context(ctx: dict) -> None:
    story = ctx.get("company_story") or {}
    render_section(
        "Company Story & Assumption Context",
        "Business model context is here to help you decide which DCF assumptions deserve adjustment.",
        "Business Model",
    )
    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        st.markdown("**Company Story**")
        st.write(_clip_text(story.get("business_summary") or "Unavailable", 320))
        st.markdown("**How They Make Money**")
        st.write(_clip_text(story.get("how_they_make_money") or "Unavailable", 260))
    with c2:
        st.markdown("**Assumption Implications**")
        implications = story.get("assumption_implications") or []
        if implications:
            for item in implications[:3]:
                implication = _clip_text(item.get("implication"), 130)
                st.write(f"- {item.get('assumption')}: {implication} ({item.get('confidence')})")
        else:
            st.write("Unavailable")
    with st.expander("Show full company story and assumption context", expanded=False):
        st.markdown("**Product story**")
        st.write(_clip_text(story.get("product_story") or "Unavailable", 420))
        st.markdown("**Industry positioning**")
        st.write(_clip_text(story.get("industry_positioning") or "Unavailable", 360))
        st.markdown("**Peers**")
        st.write(_clip_text(story.get("peer_context") or "Unavailable", 260))
        st.markdown("**Buzz/news context**")
        st.write(_clip_text(story.get("buzz_context") or "Social/news buzz unavailable.", 260))
        review_rows = []
        for item in (story.get("manual_review_questions") or [])[:4]:
            review_rows.append(
                {
                    "Question": _clip_text(item.get("Question"), 110),
                    "Why it matters": _clip_text(item.get("Why it matters"), 140),
                    "Model assumption affected": item.get("Model assumption affected"),
                }
            )
        show_table(pd.DataFrame(review_rows), "No manual-review questions generated.")
        show_table(pd.DataFrame({"Sources used": story.get("sources_used") or ["Unavailable"]}), "No story sources available.")


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
    st.markdown(
        """
        <div class="pa-dcf-hero">
            <div class="pa-dcf-kicker">Table-first valuation cockpit</div>
            <div class="pa-dcf-title">DCF Model Workbench</div>
            <div class="pa-dcf-subtitle">Edit forecast assumptions directly in the model table. Locked output rows recalculate below, so the workflow feels like an Excel DCF inside Streamlit.</div>
            <div class="pa-dcf-chips">
                <span class="pa-dcf-chip">Editable forecast assumptions</span>
                <span class="pa-dcf-chip">Locked calculated output</span>
                <span class="pa-dcf-chip">Live fair-value impact</span>
                <span class="pa-dcf-chip">CAPEX / OPEX / OCF / SBC visible</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    market = ctx["dataset"].get("market_data", {})
    dcf = ctx["base_dcf"]
    reverse = ctx["reverse"]
    assumptions = ctx["base_assumptions"]
    profile = infer_stock_profile(ctx["dataset"])
    st.caption(f"Stock-profile assumption group: {profile}.")
    _valuation(ctx)
    with st.expander("Base Valuation Snapshot", expanded=False):
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
                    {"Case": "Bear Case", "Revenue CAGR": max((assumptions.get("revenue_cagr") or 0) - 0.03, -0.2), "Maintenance CAPEX % Revenue": assumptions.get("maintenance_capex_pct_revenue"), "Growth CAPEX % Revenue": (assumptions.get("growth_capex_pct_revenue") or 0) + 0.02, "Total CAPEX % Revenue": (assumptions.get("maintenance_capex_pct_revenue") or 0) + (assumptions.get("growth_capex_pct_revenue") or 0) + 0.02, "FCF Margin": None, "Fair Value / Share": None, "Read": "Stress lower growth / higher reinvestment"},
                    {"Case": "Base Case", "Revenue CAGR": assumptions.get("revenue_cagr"), "Maintenance CAPEX % Revenue": assumptions.get("maintenance_capex_pct_revenue"), "Growth CAPEX % Revenue": assumptions.get("growth_capex_pct_revenue"), "Total CAPEX % Revenue": assumptions.get("total_capex_pct_revenue"), "FCF Margin": None, "Fair Value / Share": dcf.get("fair_value_per_share"), "Read": "Dashboard base case"},
                    {"Case": "Bull Case", "Revenue CAGR": (assumptions.get("revenue_cagr") or 0) + 0.05, "Maintenance CAPEX % Revenue": assumptions.get("maintenance_capex_pct_revenue"), "Growth CAPEX % Revenue": max((assumptions.get("growth_capex_pct_revenue") or 0) - 0.01, 0), "Total CAPEX % Revenue": (assumptions.get("maintenance_capex_pct_revenue") or 0) + max((assumptions.get("growth_capex_pct_revenue") or 0) - 0.01, 0), "FCF Margin": None, "Fair Value / Share": None, "Read": "Evidence-supported growth with CAPEX normalization"},
                    {"Case": "User Case", "Revenue CAGR": assumptions.get("revenue_cagr"), "Maintenance CAPEX % Revenue": assumptions.get("maintenance_capex_pct_revenue"), "Growth CAPEX % Revenue": assumptions.get("growth_capex_pct_revenue"), "Total CAPEX % Revenue": assumptions.get("total_capex_pct_revenue"), "FCF Margin": None, "Fair Value / Share": dcf.get("fair_value_per_share"), "Read": "Editable through assumption controls"},
                    {"Case": "Market-Implied Case", "Revenue CAGR": reverse.get("implied_revenue_cagr"), "Maintenance CAPEX % Revenue": "Not solved", "Growth CAPEX % Revenue": "Not solved", "Total CAPEX % Revenue": "Not solved", "FCF Margin": "Not solved", "Fair Value / Share": market.get("price"), "Read": "Reverse DCF solves growth/margin, not CAPEX directly"},
                ]
            ),
            "Scenario comparison unavailable.",
        )
    with st.expander("SOTP Lens", expanded=False):
        render_sotp_tab(ctx, analyst_details, key_prefix="pa11r_valuation_sotp")
    with st.expander("Multiples / Peer Lens", expanded=False):
        render_multiples_tab(ctx, key_prefix="pa11r_valuation_multiples")
    if analyst_details:
        with st.expander("Assumption Workbench / Update Log", expanded=True):
            _assumption_workbench(ctx, key_prefix="valuation")


def _pa11r_financials_reinvestment_tab(ctx: dict, analyst_details: bool) -> None:
    render_section(
        "Financials & Reinvestment",
        "Historical economics, OCF quality, CAPEX split, D&A proxy quality, working capital, SBC, and the actual-to-forecast FCF bridge live here.",
        "Financials",
    )
    capex = _capex_view(ctx)
    render_status_grid(
        [
            {"title": "CAPEX View", "value": capex["view"], "subtitle": capex["dcf_impact"], "status": "warning" if capex["view"] == "Growth-heavy" else "neutral"},
            {"title": "Maintenance CAPEX", "value": fmt_percent(capex["maintenance"]), "subtitle": capex["method"], "status": "info"},
            {"title": "Growth CAPEX", "value": fmt_percent(capex["growth"]), "subtitle": capex["evidence_grade"], "status": "caution"},
            {"title": "Total CAPEX", "value": fmt_percent(capex["total"]), "subtitle": "Maintenance + growth CAPEX.", "status": "neutral"},
        ],
        numeric=True,
    )
    st.subheader("CAPEX Bridge")
    show_table(_capex_bridge_table(ctx), "CAPEX bridge unavailable.")
    dcf = run_dcf(ctx["historicals"], ctx["dataset"].get("market_data", {}), ctx["base_assumptions"])
    model_table = build_time_axis_financial_model(ctx["historicals"], dcf.get("forecast_table"), ctx["base_assumptions"])
    capex_rows = model_table[model_table["Line Item"].isin(["Maintenance CAPEX", "Growth CAPEX", "Total CAPEX", "D&A", "Operating cash flow", "FCF", "FCF margin %"])] if model_table is not None and not model_table.empty else pd.DataFrame()
    render_financial_line_chart(
        capex_rows,
        "Maintenance vs Growth CAPEX Over Time",
        default_items=["Maintenance CAPEX", "Growth CAPEX", "Total CAPEX"],
        key_prefix="financials_reinvestment_capex_split",
    )
    st.caption("D&A proxy is shown explicitly when used; do not treat it as a disclosed maintenance CAPEX split.")
    _financial_reports(ctx)
    st.subheader("Accounting / Economic Reality")
    _accounting_quality(ctx)


def _pa11r_evidence_assumptions_tab(ctx: dict, analyst_details: bool) -> None:
    render_section(
        "Evidence & Assumptions",
        "Use this tab to connect filing clauses and manual evidence to DCF assumptions. The default view shows only model-changing evidence.",
        "Evidence",
    )
    st.subheader("Top Model-Changing Evidence")
    show_table(_top_clause_impacts(ctx.get("clauses")), "No model-changing evidence loaded yet.")
    _assumption_workbench(ctx, key_prefix="evidence")
    with st.expander("Show full Clause Map", expanded=analyst_details):
        _clause_annotation_map(ctx)
    with st.expander("Filing Metadata and Guidance", expanded=analyst_details):
        _evidence(ctx)


def _pa11r_business_quality_tab(ctx: dict, analyst_details: bool) -> None:
    render_section(
        "Business Quality",
        "Moat, operating leverage, M&A quality, management, SBC alignment, peer quality, and thesis breakers are separated from pure valuation.",
        "Business Quality",
    )
    moat = ctx["moat"]
    render_status_grid(
        [
            {"title": "Moat", "value": _clean_classification(moat.get("classification")), "subtitle": moat.get("terminal_value_implication"), "status": "caution" if "unknown" in _clean_classification(moat.get("classification")).lower() else "supportive", "confidence": moat.get("confidence"), "help_text": f"Moat score: {format_short_score(moat.get('moat_score'))}"},
            {"title": "Management", "value": ctx.get("management", {}).get("management_score"), "subtitle": ctx.get("management", {}).get("style") or "Manual review required.", "status": "neutral"},
            {"title": "Risk Level", "value": _risk_level(ctx)[0], "subtitle": _risk_level(ctx)[1], "status": _risk_level(ctx)[2]},
        ]
    )
    _company_story(ctx)
    _moat_risks(ctx)
    _ma_management_sbc(ctx)
    with st.expander("Peer Quality Comparison", expanded=analyst_details):
        _multiples_peers(ctx)


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
        "Sources & Review",
        "Missing data is a controlled state. It creates a review/fetch plan instead of showing scary top-level errors.",
        "Review",
    )
    render_status_grid(
        [
            {"title": "Data Coverage", "value": confidence, "subtitle": subtitle, "status": status},
            {"title": "Manual Review Items", "value": len(_manual_review_items(ctx)), "subtitle": "Debt, segment, moat, and CAPEX split items are tracked here.", "status": "caution" if _manual_review_items(ctx) else "supportive"},
            {"title": "Provider Sources", "value": ", ".join(ctx["dataset"].get("sources", [])) or "Unavailable", "subtitle": "SEC / Finviz / yfinance availability.", "status": "info"},
        ]
    )
    show_table(_manual_review_plan_table(ctx), "No manual-review plan available.")
    with st.expander("Saved Analysis Metadata", expanded=bool(st.session_state.get("loaded_analysis_id"))):
        if st.session_state.get("loaded_analysis_id"):
            show_table(_saved_analysis_metadata_table(ctx), "No saved-analysis metadata available.")
            saved_market = st.session_state.get("loaded_market_snapshot")
            if saved_market:
                st.caption("Saved market snapshot is shown for comparison only. Latest market data remains the default calculation source.")
                st.json(saved_market)
        else:
            st.info("No saved analysis is currently loaded.")
    with st.expander("Data Coverage Table", expanded=analyst_details):
        show_table(_data_coverage(ctx["dataset"], ctx["historicals"]), "Data coverage unavailable.")
        show_table(_data_quality_table(ctx), "No data-quality notes.")
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
            "DCF Model",
            "Snapshot",
            "Financials & Reinvestment",
            "Evidence & Assumptions",
            "Business Quality & Risks",
            "Sources & Review",
        ]
    )
    with tabs[0]:
        _pa11r_valuation_tab(ctx, analyst_details)
    with tabs[1]:
        _pa11r_snapshot(ctx)
    with tabs[2]:
        _pa11r_financials_reinvestment_tab(ctx, analyst_details)
    with tabs[3]:
        _pa11r_evidence_assumptions_tab(ctx, analyst_details)
    with tabs[4]:
        _pa11r_business_quality_tab(ctx, analyst_details)
    with tabs[5]:
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
    st.session_state.setdefault("research_ticker", "AAPL")

    with st.sidebar:
        st.header("Research Setup")
        dashboard_mode = st.radio("Dashboard", ["PA-11R Hybrid", "MR-1 Lite"], horizontal=False)
        ticker = st.text_input("Ticker", key="research_ticker").upper().strip()
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

    _render_saved_analysis_sidebar(ctx)

    st.caption("Mode: SEC evidence loaded" if include_deep_sec else "Mode: fast SEC JSON snapshot")
    if dashboard_mode == "MR-1 Lite":
        _render_mr1_lite(ctx, analyst_details or debug)
    else:
        _render_pa11r_hybrid(ctx, analyst_details or debug)
    if debug:
        with st.expander("Debug Data Lab", expanded=False):
            _data_lab(ctx, key_prefix="debug")
