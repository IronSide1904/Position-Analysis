from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from analysis.capex_ocf_nopat import analyze_capex_ocf_nopat_quality
from analysis.clauses import extract_relevant_clauses
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
from models.dcf_model import build_dcf_sensitivity_table, default_assumptions_from_historicals, run_dcf
from models.financial_model import build_historical_financial_table
from models.reverse_dcf import run_reverse_dcf
from models.scoring import score_investment
from models.sotp_model import run_sotp
from ui.charts import (
    dcf_sensitivity_heatmap,
    fcf_projection_chart,
    ma_timeline_chart,
    moat_score_bar,
    peer_multiple_chart,
    peer_scatter,
    price_action_chart,
    reverse_dcf_chart,
    sbc_vs_buybacks_chart,
    scenario_valuation_bar,
)
from ui.components import fmt_money, fmt_pct, metric_row, show_table, show_warnings


@st.cache_data(show_spinner=False, ttl=3600)
def cached_dataset(ticker: str):
    return load_company_dataset(ticker)


def _slider_assumptions(base: dict, reverse: dict | None = None) -> dict:
    def bounded(value, low, high):
        value = float(value)
        return max(low, min(high, value))

    st.subheader("Assumptions")
    c1, c2, c3 = st.columns(3)
    with c1:
        revenue_cagr = st.slider("Revenue CAGR", -0.20, 0.60, bounded(base.get("revenue_cagr", 0.08), -0.20, 0.60), 0.01, format="%.2f")
        gross_margin = st.slider("Gross margin", 0.00, 0.90, bounded(base.get("gross_margin", 0.45), 0.00, 0.90), 0.01, format="%.2f")
        operating_margin = st.slider("Operating margin", -0.30, 0.60, bounded(base.get("operating_margin", 0.15), -0.30, 0.60), 0.01, format="%.2f")
        nopat_margin = st.slider("NOPAT margin", -0.20, 0.50, bounded(base.get("nopat_margin", 0.12), -0.20, 0.50), 0.01, format="%.2f")
    with c2:
        ocf_margin = st.slider("OCF margin", -0.20, 0.60, bounded(base.get("ocf_margin", 0.16), -0.20, 0.60), 0.01, format="%.2f")
        maintenance_capex = st.slider("Maintenance CAPEX % revenue", 0.00, 0.25, bounded(base.get("maintenance_capex_pct_revenue", 0.03), 0.00, 0.25), 0.005, format="%.3f")
        growth_capex = st.slider("Growth CAPEX % revenue", 0.00, 0.35, bounded(base.get("growth_capex_pct_revenue", 0.02), 0.00, 0.35), 0.005, format="%.3f")
        working_capital = st.slider("Working capital % revenue", -0.10, 0.20, bounded(base.get("working_capital_pct_revenue", 0.01), -0.10, 0.20), 0.005, format="%.3f")
    with c3:
        wacc = st.slider("WACC", 0.04, 0.20, bounded(base.get("wacc", 0.095), 0.04, 0.20), 0.005, format="%.3f")
        terminal_growth = st.slider("Terminal growth", -0.02, 0.06, bounded(base.get("terminal_growth", 0.025), -0.02, 0.06), 0.005, format="%.3f")
        terminal_multiple = st.slider("Terminal multiple", 4.0, 35.0, bounded(base.get("terminal_multiple", 15.0), 4.0, 35.0), 0.5)
        margin_of_safety = st.slider("Margin of safety", 0.0, 0.6, bounded(base.get("margin_of_safety", 0.30), 0.0, 0.6), 0.05)
    shares = st.number_input("Diluted shares", value=float(base.get("diluted_shares") or 0), min_value=0.0, step=1_000_000.0)
    return {
        **base,
        "revenue_cagr": revenue_cagr,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "nopat_margin": nopat_margin,
        "ocf_margin": ocf_margin,
        "maintenance_capex_pct_revenue": maintenance_capex,
        "growth_capex_pct_revenue": growth_capex,
        "working_capital_pct_revenue": working_capital,
        "wacc": wacc,
        "terminal_growth": terminal_growth,
        "terminal_multiple": terminal_multiple,
        "diluted_shares": shares,
        "margin_of_safety": margin_of_safety,
    }


def _assumption_comparison(base: dict, user: dict, reverse: dict, dcf: dict) -> pd.DataFrame:
    rows = []
    for key, label in [
        ("revenue_cagr", "Revenue CAGR"),
        ("nopat_margin", "NOPAT margin"),
        ("ocf_margin", "OCF margin"),
        ("wacc", "WACC"),
        ("terminal_growth", "Terminal growth"),
        ("terminal_multiple", "Terminal multiple"),
    ]:
        market = reverse.get(f"implied_{key}") if key in {"revenue_cagr", "nopat_margin", "ocf_margin", "terminal_multiple"} else None
        rows.append({"Assumption": label, "Base": base.get(key), "User": user.get(key), "Market implied": market, "Difference": (user.get(key) or 0) - (base.get(key) or 0), "Fair-value impact": "Re-run shown above"})
    return pd.DataFrame(rows)


def _add_assumption_log(assumption: str, old, new, reason: str, linked_clause: str):
    if "assumption_log" not in st.session_state:
        st.session_state.assumption_log = []
    st.session_state.assumption_log.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "assumption": assumption,
            "old_value": old,
            "new_value": new,
            "reason": reason,
            "linked_clause": linked_clause,
            "confidence": "Manual",
            "scenario": "User case",
            "fair_value_impact": "Re-run DCF",
        }
    )


def render_dashboard():
    st.set_page_config(page_title="PA-11R Hybrid", layout="wide")
    st.title("PA-11R Hybrid")
    st.caption("Retail-friendly valuation cockpit with institutional-quality evidence behind it.")

    with st.sidebar:
        st.header("Controls")
        ticker = st.text_input("Ticker", value="AAPL").upper().strip()
        benchmark = st.text_input("Benchmark", value="SPY").upper().strip()
        peer_override = st.text_input("Peer override", value="", help="Comma-separated tickers")
        refresh = st.button("Refresh data")
        debug = st.toggle("Debug mode", value=False)
        if refresh:
            cached_dataset.clear()

    if not ticker:
        st.info("Enter a ticker to begin.")
        return

    with st.spinner("Loading SEC, Finviz, and yfinance data..."):
        dataset = cached_dataset(ticker)
        historicals = build_historical_financial_table(dataset)
        clauses = extract_relevant_clauses(dataset.get("filing_texts", {}))
        base_assumptions = default_assumptions_from_historicals(historicals, dataset.get("market_data", {}))
        base_dcf = run_dcf(historicals, dataset.get("market_data", {}), base_assumptions)
        reverse = run_reverse_dcf(dataset.get("market_data", {}), historicals, base_assumptions)

        default_peers = select_peer_candidates(ticker, dataset.get("sector"), dataset.get("industry"))
        peers = [p.strip().upper() for p in peer_override.split(",") if p.strip()] or default_peers
        should_fetch_peers = bool(peer_override.strip()) or "yfinance" in dataset.get("sources", [])
        peer_df = build_peer_comparison(ticker, peers) if peers and should_fetch_peers else pd.DataFrame()

        capex_quality = analyze_capex_ocf_nopat_quality(historicals, clauses)
        leverage = analyze_operating_leverage(historicals, peer_df)
        ma = analyze_ma_strategy(dataset.get("filing_texts", {}), historicals)
        management = analyze_management_and_board(dataset.get("filing_texts", {}), dataset.get("submissions", {}))
        guidance = analyze_guidance_accuracy(dataset.get("filing_texts", {}), historicals)
        alignment = analyze_compensation_alignment(dataset.get("filing_texts", {}), historicals)
        moat = analyze_moat(dataset, historicals, dataset.get("filing_texts", {}), peer_df, clauses)
        risks = analyze_risks_and_thesis_breakers(dataset.get("filing_texts", {}), clauses, historicals)
        scoring = score_investment({"dcf": base_dcf, "reverse_dcf": reverse, "moat": moat, "management": management, "ma": ma, "alignment": alignment, "quality": capex_quality, "operating_leverage": leverage})
        thesis = build_thesis_summary(dataset, base_dcf, reverse, moat, scoring)

    tabs = st.tabs(
        [
            "Snapshot",
            "Thesis & Scorecard",
            "Interactive DCF Lab",
            "Reverse DCF",
            "Clause / Note Map",
            "CAPEX / NOPAT / OCF Quality",
            "Operating Leverage",
            "M&A Strategy",
            "Management & Board",
            "Guidance Tracker",
            "Compensation & SBC",
            "Moat Analyzer",
            "Peers & Valuation",
            "Risks & Thesis Breakers",
            "Final Decision",
        ]
    )

    market = dataset.get("market_data", {})
    with tabs[0]:
        st.subheader(f"{dataset.get('ticker')} - {dataset.get('company') or 'Company unavailable'}")
        metric_row(
            [
                ("Price", market.get("price"), "money"),
                ("Market Cap", market.get("market_cap"), "money"),
                ("Enterprise Value", market.get("enterprise_value"), "money"),
                ("Short Float", market.get("short_float"), "pct"),
            ]
        )
        metric_row(
            [
                ("Sector", dataset.get("sector"), "text"),
                ("Industry", dataset.get("industry"), "text"),
                ("Shares Float", market.get("shares_float"), "text"),
                ("Net Cash / Debt", -historicals["Net Debt"].iloc[-1] if not historicals.empty else None, "money"),
            ]
        )
        show_table(historicals)
        st.plotly_chart(price_action_chart(dataset.get("price_history")), width="stretch", key="price_action_chart")
        st.write("Data source status:", ", ".join(dataset.get("sources", [])) or "Fallback/manual data only")
        show_warnings(dataset.get("warnings", []))

    with tabs[1]:
        st.subheader("Thesis")
        st.write(thesis["what_it_does"])
        st.write(thesis["how_it_makes_money"])
        st.write(thesis["valuation_view"])
        show_table(scoring["scorecard"])

    with tabs[2]:
        user_assumptions = _slider_assumptions(base_assumptions, reverse)
        user_dcf = run_dcf(historicals, market, user_assumptions)
        metric_row(
            [
                ("Fair Value", user_dcf.get("fair_value_per_share"), "money"),
                ("Buy Zone", user_dcf.get("buy_price_after_margin_of_safety"), "money"),
                ("Upside / Downside", user_dcf.get("upside_downside_pct"), "pct"),
                ("Terminal Value Weight", user_dcf.get("terminal_value_weight_pct"), "pct"),
            ]
        )
        show_warnings(user_dcf.get("warnings", []))
        st.plotly_chart(fcf_projection_chart(historicals, user_dcf["forecast_table"]), width="stretch", key="fcf_projection_chart")
        sensitivity = build_dcf_sensitivity_table({**user_assumptions, "historicals": historicals, "market_data": market}, [0.075, 0.085, 0.095, 0.105, 0.115], [0.01, 0.02, 0.03, 0.04])
        st.plotly_chart(dcf_sensitivity_heatmap(sensitivity), width="stretch", key="dcf_sensitivity_heatmap")
        show_table(_assumption_comparison(base_assumptions, user_assumptions, reverse, user_dcf))
        st.subheader("Assumption Update Log")
        if not clauses.empty:
            linked_clause = st.selectbox("Linked clause", clauses["clause_text"].head(50).tolist())
            assumption = st.selectbox("Assumption to update", ["revenue_cagr", "gross_margin", "nopat_margin", "ocf_margin", "maintenance_capex_pct_revenue", "growth_capex_pct_revenue", "wacc", "terminal_growth", "terminal_multiple"])
            reason = st.text_input("Reason", value="Clause suggests manual assumption review.")
            if st.button("Add to assumption log"):
                _add_assumption_log(assumption, base_assumptions.get(assumption), user_assumptions.get(assumption), reason, linked_clause)
        show_table(pd.DataFrame(st.session_state.get("assumption_log", [])), "No assumption changes logged yet.")

    with tabs[3]:
        metric_row(
            [
                ("Market Case", reverse.get("market_case"), "text"),
                ("Implied Revenue CAGR", reverse.get("implied_revenue_cagr"), "pct"),
                ("Implied NOPAT Margin", reverse.get("implied_nopat_margin"), "pct"),
                ("Implied Terminal Multiple", reverse.get("implied_terminal_multiple"), "text"),
            ]
        )
        st.write(reverse.get("interpretation"))
        st.plotly_chart(reverse_dcf_chart(reverse, base_assumptions), width="stretch", key="reverse_dcf_chart")

    with tabs[4]:
        show_table(clauses, "No valuation-relevant clauses extracted. Manual review required.")

    with tabs[5]:
        st.write(capex_quality["summary"])
        metric_row([("Quality Score", capex_quality["quality_score"], "text"), ("CAPEX % Revenue", capex_quality["metrics"].get("capex_pct_revenue"), "pct"), ("OCF Conversion", capex_quality["metrics"].get("ocf_conversion"), "text")])
        show_warnings(capex_quality["red_flags"])
        st.write("Model implications:", capex_quality["model_implications"] or ["Data unavailable"])

    with tabs[6]:
        st.write(leverage["summary"])
        metric_row([("Classification", leverage["classification"], "text"), ("Score", leverage["score"], "text"), ("Operating Margin", leverage["metrics"].get("operating_margin"), "pct")])

    with tabs[7]:
        st.write(ma["summary"])
        metric_row([("M&A Quality", ma["ma_quality_score"], "text"), ("Classification", ma["classification"], "text")])
        show_table(ma["timeline"])
        st.plotly_chart(ma_timeline_chart(ma), width="stretch", key="ma_timeline_chart")

    with tabs[8]:
        st.write(management["summary"])
        metric_row([("Management Score", management["management_score"], "text"), ("Style", management["style"], "text")])
        st.write("Strengths:", management["strengths"] or ["Data unavailable"])
        show_warnings(management["red_flags"])

    with tabs[9]:
        st.write(guidance["summary"])
        show_table(guidance["table"], "Insufficient data.")

    with tabs[10]:
        st.write(alignment["summary"])
        metric_row([("Alignment Score", alignment["alignment_score"], "text")])
        show_table(alignment["compensation_table"])
        show_table(alignment["sbc_table"])
        st.plotly_chart(sbc_vs_buybacks_chart(alignment), width="stretch", key="sbc_vs_buybacks_chart")
        show_warnings(alignment["red_flags"])

    with tabs[11]:
        st.write(moat["summary"])
        metric_row([("Moat Score", moat["moat_score"], "text"), ("Classification", moat["classification"], "text"), ("Confidence", moat["confidence"], "text")])
        st.plotly_chart(moat_score_bar(moat["moat_sources"]), width="stretch", key="moat_score_bar")
        show_table(moat["moat_sources"])
        st.write("New entrant test:", moat["new_entrant_test"])
        st.write(moat["terminal_value_implication"])

    with tabs[12]:
        st.subheader("SOTP Starter")
        st.caption("Use this as a manual segment sandbox when SEC segment data is unavailable.")
        manual_segments = pd.DataFrame(
            [
                {"segment": "Core business", "revenue": historicals["Revenue"].iloc[-1] if not historicals.empty else 0, "margin": base_assumptions.get("nopat_margin", 0.12), "multiple": base_assumptions.get("terminal_multiple", 15.0)}
            ]
        )
        sotp = run_sotp(manual_segments, {"default_margin": base_assumptions.get("nopat_margin", 0.12), "default_multiple": base_assumptions.get("terminal_multiple", 15.0)})
        metric_row([("SOTP Enterprise Value", sotp.get("enterprise_value"), "money")])
        show_table(sotp.get("segment_table"))
        st.subheader("Peers")
        show_table(peer_df, "No peer data available. Add peers in the sidebar.")
        st.plotly_chart(peer_scatter(peer_df), width="stretch", key="peer_scatter")
        st.plotly_chart(peer_multiple_chart(peer_df), width="stretch", key="peer_multiple_chart")

    with tabs[13]:
        metric_row([("Risk Score", risks["risk_score"], "text")])
        st.write("Top risks:", risks["top_risks"])
        st.write("Thesis breakers:", risks["thesis_breakers"])
        st.write("Bear case implications:", risks["bear_case_implications"])

    with tabs[14]:
        st.subheader(scoring["recommendation"])
        metric_row([("Total Score", scoring["total_score"], "text"), ("Conviction", scoring["conviction"], "text"), ("Fair Value", base_dcf.get("fair_value_per_share"), "money")])
        st.write(scoring["position_size_guidance"])
        st.write("Does the moat support valuation?", moat["terminal_value_implication"])
        if (base_dcf.get("terminal_value_weight_pct") or 0) > 0.75 and (moat.get("moat_score") or 0) < 5:
            st.warning("Warning: This valuation depends heavily on terminal value, but moat evidence is weak or unproven.")
        st.plotly_chart(scenario_valuation_bar(base_dcf), width="stretch", key="scenario_valuation_bar")

    if debug:
        st.divider()
        st.subheader("Debug")
        st.json(
            {
                "ticker": ticker,
                "benchmark": benchmark,
                "cik": dataset.get("cik"),
                "sources": dataset.get("sources"),
                "warnings": dataset.get("warnings"),
                "filings": dataset.get("filings"),
                "finviz": dataset.get("finviz"),
                "yfinance": dataset.get("yfinance"),
                "reverse_dcf": reverse,
                "dcf_warnings": base_dcf.get("warnings"),
            }
        )
