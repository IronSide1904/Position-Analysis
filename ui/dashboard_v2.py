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
from models.sotp_model import run_sotp
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
from ui.formatting import UNAVAILABLE, fmt_dollar, fmt_multiple, fmt_percent, fmt_per_share, fmt_score, fmt_shares


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
        .pa-pill {
            display: inline-block;
            border: 1px solid #c8d3df;
            border-radius: 999px;
            padding: 0.16rem 0.55rem;
            margin: 0 0.25rem 0.25rem 0;
            font-size: 0.78rem;
            background: #f7f9fb;
        }
        .pa-pill-ok { border-color: #98d4b4; background: #edf9f2; color: #166534; }
        .pa-pill-warn { border-color: #f0ca79; background: #fff8e7; color: #854d0e; }
        .pa-band {
            border: 1px solid #d9e0e8;
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            background: #fbfcfe;
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
            border: 1px solid #d8e1ea;
            border-radius: 8px;
            padding: 0.55rem;
            margin: 0.75rem 0 0.85rem 0;
            background: #f8fafc;
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
            color: #0f172a;
            font-size: 0.95rem;
            font-weight: 700;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .pa-box {
            border: 1px solid #d9e0e8;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            background: #fbfcfe;
            margin-bottom: 0.75rem;
        }
        .pa-box-title {
            color: #334155;
            font-size: 0.86rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
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
    rows = [
        {"area": "Price history", "status": "Loaded" if not dataset.get("price_history", pd.DataFrame()).empty else "Missing", "source": "yfinance"},
        {"area": "Company profile", "status": "Loaded" if dataset.get("company") else "Missing", "source": ", ".join(dataset.get("sources", []))},
        {"area": "Market cap", "status": "Loaded" if market.get("market_cap") else "Missing", "source": "Finviz / yfinance"},
        {"area": "Enterprise value", "status": "Loaded" if market.get("enterprise_value") else "Missing", "source": "yfinance"},
        {"area": "Shares outstanding", "status": "Loaded" if market.get("shares_outstanding") or sec.get("shares", {}).get("value") else "Missing", "source": "Finviz / yfinance / SEC"},
        {"area": "SEC companyfacts", "status": "Loaded" if sec.get("revenue", {}).get("value") else "Missing", "source": "SEC"},
        {"area": "Full filing text", "status": "Loaded" if dataset.get("filing_texts") else "Not loaded", "source": "SEC evidence mode"},
        {"area": "Historical model table", "status": "Loaded" if historicals is not None and not historicals.empty else "Missing", "source": "SEC / yfinance"},
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
    if value is None or pd.isna(value):
        return UNAVAILABLE
    return f"{float(value):,.0f}"


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


def _finviz_decision_snapshot(market: dict) -> pd.DataFrame:
    rows = [
        ("Share / Float", "Shares outstanding", _fmt_plain(market.get("shares_outstanding"))),
        ("Share / Float", "Shares float", _fmt_plain(market.get("shares_float"))),
        ("Share / Float", "Float / outstanding", fmt_pct(market.get("float_outstanding_pct"))),
        ("Short / Liquidity", "Short float", fmt_pct(market.get("short_float"))),
        ("Short / Liquidity", "Short ratio", _fmt_ratio(market.get("short_ratio"))),
        ("Short / Liquidity", "Average volume", _fmt_plain(market.get("average_volume"))),
        ("Short / Liquidity", "Current volume", _fmt_plain(market.get("volume"))),
        ("Short / Liquidity", "Relative volume", _fmt_ratio(market.get("relative_volume"))),
        ("Volatility / Technical", "Beta", _fmt_ratio(market.get("beta"))),
        ("Volatility / Technical", "ATR", _fmt_ratio(market.get("atr"))),
        ("Volatility / Technical", "Week volatility", fmt_pct(market.get("volatility_week"))),
        ("Volatility / Technical", "Month volatility", fmt_pct(market.get("volatility_month"))),
        ("Volatility / Technical", "Gap", fmt_pct(market.get("gap"))),
        ("Volatility / Technical", "Change", fmt_pct(market.get("change"))),
        ("Volatility / Technical", "SMA20", fmt_pct(market.get("sma20"))),
        ("Volatility / Technical", "SMA50", fmt_pct(market.get("sma50"))),
        ("Volatility / Technical", "SMA200", fmt_pct(market.get("sma200"))),
        ("Valuation", "P/E", _fmt_ratio(market.get("pe"))),
        ("Valuation", "Forward P/E", _fmt_ratio(market.get("forward_pe"))),
        ("Valuation", "PEG", _fmt_ratio(market.get("peg"))),
        ("Valuation", "P/S", _fmt_ratio(market.get("ps"))),
        ("Valuation", "P/B", _fmt_ratio(market.get("pb"))),
        ("Valuation", "P/FCF", _fmt_ratio(market.get("pfcf"))),
        ("Profitability", "ROA", fmt_pct(market.get("roa"))),
        ("Profitability", "ROE", fmt_pct(market.get("roe"))),
        ("Profitability", "ROI / ROIC", fmt_pct(market.get("roi"))),
        ("Profitability", "Gross margin", fmt_pct(market.get("gross_margin"))),
        ("Profitability", "Operating margin", fmt_pct(market.get("operating_margin"))),
        ("Profitability", "Profit margin", fmt_pct(market.get("profit_margin"))),
        ("Balance Sheet", "Current ratio", _fmt_ratio(market.get("current_ratio"))),
        ("Balance Sheet", "Quick ratio", _fmt_ratio(market.get("quick_ratio"))),
        ("Balance Sheet", "LT debt / equity", _fmt_ratio(market.get("lt_debt_to_equity"))),
        ("Balance Sheet", "Total debt / equity", _fmt_ratio(market.get("debt_to_equity"))),
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


def _assumption_editor(base: dict) -> dict:
    def pct_slider(label: str, minimum: float, maximum: float, value: float, step: float = 0.5) -> float:
        try:
            pct_value = float(value) * 100
        except (TypeError, ValueError):
            pct_value = 0.0
        pct_value = min(max(pct_value, minimum * 100), maximum * 100)
        return st.slider(label, minimum * 100, maximum * 100, pct_value, step, format="%.1f%%") / 100

    st.markdown('<div class="pa-section-title">DCF Controls</div>', unsafe_allow_html=True)
    dcf_mode = st.segmented_control("DCF mode", ["FCFF", "FCF"], default=str(base.get("dcf_mode", "FCFF")).upper())
    forecast_years = st.slider("Forecast years", 5, 10, int(base.get("forecast_years", 5)), 1)
    c1, c2 = st.columns(2)
    with c1:
        revenue_cagr = pct_slider("Revenue CAGR", -0.20, 0.60, base.get("revenue_cagr", 0.08))
        gross_margin = pct_slider("Gross margin", -0.20, 0.80, base.get("gross_margin", 0.45))
        ebit_margin = pct_slider("EBIT margin", -0.20, 0.60, base.get("operating_margin", 0.15))
        tax_rate = pct_slider("Tax rate", 0.00, 0.40, base.get("tax_rate", 0.21))
        sm_pct = pct_slider("S&M / revenue", 0.00, 0.50, base.get("sm_pct_revenue", 0.0))
        rd_pct = pct_slider("R&D / revenue", 0.00, 0.50, base.get("rd_pct_revenue", 0.0))
        ga_pct = pct_slider("G&A / revenue", 0.00, 0.50, base.get("ga_pct_revenue", 0.0))
        ocf_margin = pct_slider("OCF margin", -0.20, 0.60, base.get("ocf_margin", 0.16))
    with c2:
        maint_capex = pct_slider("Maintenance CAPEX / revenue", 0.00, 0.25, base.get("maintenance_capex_pct_revenue", 0.03))
        growth_capex = pct_slider("Growth CAPEX / revenue", 0.00, 0.35, base.get("growth_capex_pct_revenue", 0.02))
        working_capital = pct_slider("Working capital / revenue", -0.10, 0.20, base.get("working_capital_pct_revenue", 0.01))
        sbc_pct = pct_slider("SBC / revenue", 0.00, 0.30, base.get("sbc_pct_revenue", 0.0))
        wacc = pct_slider("WACC", 0.04, 0.20, base.get("wacc", 0.095))
        terminal_growth = pct_slider("Terminal growth", -0.02, 0.06, base.get("terminal_growth", 0.025))
        terminal_multiple = st.slider("Terminal multiple", 4.0, 35.0, float(base.get("terminal_multiple", 15.0)), 1.0, format="%.0f")
        share_growth = pct_slider("Diluted share growth", -0.10, 0.20, base.get("diluted_share_growth", 0.0))
    shares = st.number_input("Diluted shares", value=float(base.get("diluted_shares") or 0), min_value=0.0, step=1_000_000.0, format="%.0f")
    margin_of_safety = pct_slider("Margin of safety", 0.0, 0.6, base.get("margin_of_safety", 0.30), step=5.0)
    nopat_margin = ebit_margin * (1 - tax_rate)
    return {
        **base,
        "forecast_years": forecast_years,
        "dcf_mode": dcf_mode or "FCFF",
        "revenue_cagr": revenue_cagr,
        "gross_margin": gross_margin,
        "operating_margin": ebit_margin,
        "nopat_margin": nopat_margin,
        "tax_rate": tax_rate,
        "ocf_margin": ocf_margin,
        "sm_pct_revenue": sm_pct,
        "rd_pct_revenue": rd_pct,
        "ga_pct_revenue": ga_pct,
        "maintenance_capex_pct_revenue": maint_capex,
        "growth_capex_pct_revenue": growth_capex,
        "working_capital_pct_revenue": working_capital,
        "sbc_pct_revenue": sbc_pct,
        "wacc": wacc,
        "terminal_growth": terminal_growth,
        "terminal_multiple": terminal_multiple,
        "diluted_share_growth": share_growth,
        "diluted_shares": shares,
        "margin_of_safety": margin_of_safety,
    }


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
        _accounting_reality_check(ctx, expanded=True)

    with st.expander("Finviz Decision Snapshot"):
        show_table(_finviz_decision_snapshot(market), "No Finviz decision fields available.")


def _valuation(ctx: dict) -> None:
    market = ctx["dataset"].get("market_data", {})
    st.caption("Interactive DCF: assumptions on the left, valuation impact on the right, detailed model below.")
    left, right = st.columns([0.52, 0.48])
    with left:
        assumptions = _assumption_editor(ctx["base_assumptions"])
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
        st.markdown('<div class="pa-section-title">Valuation Output</div>', unsafe_allow_html=True)
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
        if user_dcf.get("terminal_value_weight_pct") and user_dcf.get("terminal_value_weight_pct") > 0.65:
            st.warning(
                f"Terminal Value Warning: terminal value represents {fmt_percent(user_dcf.get('terminal_value_weight_pct'))} of enterprise value. The valuation is highly sensitive to terminal multiple and long-term margin assumptions."
            )
        show_warnings(user_dcf.get("warnings", []))
        show_warnings(ctx.get("accounting_interpretation", {}).get("warnings", []))
        show_table(valuation_summary, "Valuation summary unavailable.")

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


def _ma_management_sbc(ctx: dict) -> None:
    st.caption("M&A / Management / SBC: capital allocation, governance, and dilution signals.")
    management = ctx["management"]
    alignment = ctx["alignment"]
    ma = ctx["ma"]
    metric_row(
        [
            ("Management", management.get("management_score"), "score"),
            ("Alignment", alignment.get("alignment_score"), "score"),
            ("M&A Quality", ma.get("score"), "score"),
            ("SBC Signal", alignment.get("sbc_risk", "Manual review required"), "text"),
        ]
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Management")
        st.write(management.get("summary") or "Management read unavailable.")
        st.write("Style:", management.get("style") or UNAVAILABLE)
        show_warnings(management.get("red_flags", []))
        st.subheader("SBC / Dilution")
        show_table(alignment.get("sbc_table"), "No SBC table available.")
        st.plotly_chart(sbc_vs_buybacks_chart(alignment), width="stretch", key="v2_sbc_management")
    with c2:
        st.subheader("M&A")
        st.write(ma.get("summary") or "M&A read unavailable.")
        st.plotly_chart(ma_timeline_chart(ma), width="stretch", key="v2_ma_management")
        show_table(ma.get("timeline"), "No M&A timeline available.")


def _moat_risks(ctx: dict) -> None:
    st.caption("Moat / Risks: competitive quality and what can break the thesis.")
    moat = ctx["moat"]
    risks = ctx["risks"]
    metric_row(
        [
            ("Moat Score", moat.get("moat_score"), "score"),
            ("Moat Class", moat.get("classification"), "text"),
            ("Moat Confidence", moat.get("confidence"), "text"),
            ("Risk Score", risks.get("risk_score"), "score"),
        ]
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(moat_score_bar(moat.get("moat_sources")), width="stretch", key="v2_moat_risks_bar")
        st.write(moat.get("terminal_value_implication") or "Moat implication unavailable.")
    with c2:
        _mini_list("Top risks", risks.get("top_risks", []) or ["No extracted risks available."])
        _mini_list("Thesis breakers", risks.get("thesis_breakers", []) or ["No thesis breakers extracted."])
        st.write("Bear case:", risks.get("bear_case_implications") or UNAVAILABLE)


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
        for risk in risks.get("top_risks", []):
            st.write("-", risk)
    with c4:
        st.markdown("Thesis breakers")
        for breaker in risks.get("thesis_breakers", []):
            st.write("-", breaker)
        st.write("Bear case:", risks.get("bear_case_implications"))


def _filter_options(df: pd.DataFrame, column: str) -> list[str]:
    if df is None or df.empty or column not in df:
        return []
    return sorted(str(value) for value in df[column].dropna().unique())


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
            out = out[out[column].astype(str).isin(values)]
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
    show_table(
        filtered[compact_cols].rename(
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


def _data_lab(ctx: dict) -> None:
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
    st.plotly_chart(scenario_valuation_bar(ctx["base_dcf"]), width="stretch", key="v2_scenario")

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


def render_dashboard():
    st.set_page_config(page_title="PA-11R Hybrid", layout="wide")
    _css()

    with st.sidebar:
        st.header("Research Setup")
        ticker = st.text_input("Ticker", value="AAPL").upper().strip()
        peer_override = st.text_input("Peer override", value="", help="Comma-separated tickers")
        fetch_peers = st.toggle("Fetch peers", value=True)
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

    dataset = ctx["dataset"]
    st.markdown(
        f"""
        <div class="pa-header">
            <p class="pa-title">{dataset.get("ticker")} - {dataset.get("company") or "Company unavailable"}</p>
            <p class="pa-subtle">{dataset.get("sector") or "Sector unavailable"} / {dataset.get("industry") or "Industry unavailable"}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _source_status(dataset)
    st.caption("Mode: SEC evidence loaded" if include_deep_sec else "Mode: fast SEC JSON snapshot")
    show_warnings(dataset.get("warnings", []))
    _summary_bar(ctx)

    tabs = [
        "Snapshot",
        "Company Story",
        "Financials",
        "Interactive DCF",
        "Reverse DCF",
        "Multiples / Peers",
        "Clause Map",
        "Accounting Quality",
        "M&A / Mgmt / SBC",
        "Moat / Risks",
        "Final Decision",
    ]
    if debug:
        tabs.append("Data Lab")
    selected_tabs = st.tabs(tabs)

    with selected_tabs[0]:
        _overview(ctx)
    with selected_tabs[1]:
        _company_story(ctx)
    with selected_tabs[2]:
        _financial_reports(ctx)
    with selected_tabs[3]:
        _valuation(ctx)
    with selected_tabs[4]:
        _reverse_dcf_tab(ctx)
    with selected_tabs[5]:
        _multiples_peers(ctx)
    with selected_tabs[6]:
        _clause_annotation_map(ctx)
    with selected_tabs[7]:
        _accounting_quality(ctx)
    with selected_tabs[8]:
        _ma_management_sbc(ctx)
    with selected_tabs[9]:
        _moat_risks(ctx)
    with selected_tabs[10]:
        _final_decision(ctx)
    if debug:
        with selected_tabs[11]:
            _data_lab(ctx)
