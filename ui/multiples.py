from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis.multiples import build_peer_premium_read
from models.dcf_model import run_dcf
from models.multiples_model import (
    MULTIPLE_METRICS,
    build_multiples_table,
    calculate_current_multiples,
    calculate_scenario_implied_multiples,
    peer_median_multiples,
    sector_median_multiples,
)
from models.sotp_model import build_default_segment_data, run_sotp
from ui.charts import premium_discount_heatmap, scenario_multiple_vs_peer_chart
from ui.components import show_table
from ui.design_system import render_section, render_status_grid
from ui.formatting import fmt_multiple, fmt_percent


def _scenario_outputs(ctx: dict, basis: str) -> dict[str, dict]:
    historicals = ctx.get("historicals")
    market = ctx.get("dataset", {}).get("market_data", {})
    base = dict(ctx.get("base_assumptions", {}))
    reverse = ctx.get("reverse", {})
    scenarios = {
        "Bear Case": {"revenue_cagr": -0.03, "nopat_margin": -0.03, "ocf_margin": -0.03, "wacc": 0.015, "terminal_multiple": -2.0},
        "Base Case": {},
        "Bull Case": {"revenue_cagr": 0.05, "nopat_margin": 0.03, "ocf_margin": 0.03, "wacc": -0.01, "terminal_multiple": 2.0},
        "User Case": {},
        "Market-Implied Case": {
            "revenue_cagr": reverse.get("implied_revenue_cagr"),
            "nopat_margin": reverse.get("implied_nopat_margin"),
            "ocf_margin": reverse.get("implied_ocf_margin"),
            "terminal_multiple": reverse.get("implied_terminal_multiple"),
            "wacc": reverse.get("implied_wacc"),
        },
    }
    outputs = {}
    for label, changes in scenarios.items():
        assumptions = dict(base)
        for key, delta in changes.items():
            if delta is None:
                continue
            if label in {"Bear Case", "Bull Case"} and key in assumptions:
                assumptions[key] = (assumptions.get(key) or 0) + delta
            else:
                assumptions[key] = delta
        if assumptions.get("wacc") is not None:
            assumptions["wacc"] = max(float(assumptions.get("wacc") or 0.095), 0.04)
        outputs[label] = run_dcf(historicals, market, assumptions)
    segments = st.session_state.get(
        f"sotp_{ctx.get('dataset', {}).get('ticker', 'default')}_segments",
        build_default_segment_data(historicals, ctx.get("dataset", {}), base),
    )
    outputs["SOTP Case"] = run_sotp(
        segments,
        market,
        base,
        scenario="Base Case",
        dcf_output=ctx.get("base_dcf"),
        peer_multiples=ctx.get("peer_df"),
        sector=ctx.get("dataset", {}).get("sector"),
    )
    return outputs


def _peer_table(ctx: dict, current: dict, peer_medians: dict, sector_medians: dict) -> pd.DataFrame:
    peer_df = ctx.get("peer_df")
    if peer_df is None or peer_df.empty:
        return pd.DataFrame(
            [
                {"Ticker": ctx.get("dataset", {}).get("ticker"), "Company": ctx.get("dataset", {}).get("company"), "Market Cap": ctx.get("dataset", {}).get("market_data", {}).get("market_cap"), **current},
                {"Ticker": "Peer Median", "Company": "Relevant peer set median / fallback", **peer_medians},
                {"Ticker": "Sector Median", "Company": "Sector reference median", **sector_medians},
            ]
        )
    frame = peer_df.copy()
    rename = {
        "ticker": "Ticker",
        "company": "Company",
        "market_cap": "Market Cap",
        "enterprise_value": "Enterprise Value",
        "revenue_growth": "Revenue Growth",
        "gross_margin": "Gross Margin",
        "operating_margin": "Operating Margin",
        "ocf_margin": "OCF Margin",
        "fcf_margin": "FCF Margin",
        "ev_sales": "EV/Revenue",
        "ev_ebitda": "EV/EBITDA",
        "ev_nopat": "EV/NOPAT",
        "ev_ocf": "EV/OCF",
        "ev_fcf": "EV/FCF",
        "pe": "P/E",
        "pb": "P/B",
        "ps": "P/S",
        "fcf_yield": "FCF Yield",
    }
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
    wanted = [
        "Ticker",
        "Company",
        "Market Cap",
        "Revenue Growth",
        "Gross Margin",
        "Operating Margin",
        "OCF Margin",
        "FCF Margin",
        "EV/Revenue",
        "EV/EBITDA",
        "EV/NOPAT",
        "EV/OCF",
        "EV/FCF",
        "P/E",
        "P/B",
        "P/S",
        "FCF Yield",
    ]
    return frame[[column for column in wanted if column in frame.columns]]


def _multiples_context_cards(multiples_table: pd.DataFrame, selected_multiple: str) -> list[dict]:
    row = multiples_table[multiples_table["Metric"].astype(str) == selected_multiple]
    if row.empty:
        return []
    item = row.iloc[0]
    current = item.get("Current Company")
    peer = item.get("Peer Median")
    premium = item.get("Premium / Discount vs Peer")
    risk_status = "warning" if premium is not None and premium > 0.25 else "supportive" if premium is not None and premium < -0.15 else "neutral"
    return [
        {"title": "Selected Multiple", "value": selected_multiple, "subtitle": "Use selector below to compare scenario multiples.", "status": "info"},
        {"title": "Current Company", "value": fmt_percent(current) if "Yield" in selected_multiple else fmt_multiple(current), "subtitle": "Current / TTM provider and financial basis.", "status": "neutral"},
        {"title": "Peer Median", "value": fmt_percent(peer) if "Yield" in selected_multiple else fmt_multiple(peer), "subtitle": "Relevant peer median where available; sector fallback otherwise.", "status": "neutral"},
        {"title": "Premium / Discount", "value": fmt_percent(premium), "subtitle": item.get("Interpretation"), "status": risk_status},
    ]


def render_multiples_tab(ctx: dict, key_prefix: str = "multiples") -> dict:
    render_section(
        "Multiples & Peers",
        "Relative valuation checks whether DCF and SOTP scenarios imply reasonable multiples versus peers and sector references.",
        "Multiples",
    )
    basis = st.selectbox(
        "Multiple Basis",
        ["Normalized Year", "Current / TTM", "Next Year", "Final Forecast Year"],
        index=0,
        key=f"{key_prefix}_basis",
    )
    current = calculate_current_multiples(ctx.get("historicals"), ctx.get("dataset", {}).get("market_data", {}))
    peer_medians, warnings = peer_median_multiples(ctx.get("peer_df"), ctx.get("dataset", {}).get("sector"), ctx.get("dataset", {}).get("industry"))
    sector_medians = sector_median_multiples(ctx.get("dataset", {}).get("sector"), ctx.get("dataset", {}).get("industry"))
    scenario_outputs = _scenario_outputs(ctx, basis)
    scenario_multiples = calculate_scenario_implied_multiples(scenario_outputs, ctx.get("historicals"), ctx.get("dataset", {}).get("market_data", {}), basis)
    table = build_multiples_table(current, scenario_multiples, peer_medians, sector_medians, ctx.get("moat", {}).get("moat_score"))
    peer_read = build_peer_premium_read(current, peer_medians, {"moat_score": ctx.get("moat", {}).get("moat_score")})

    selected_multiple = st.selectbox(
        "Multiple to compare",
        ["EV/Revenue", "EV/EBITDA", "EV/NOPAT", "EV/OCF", "EV/FCF", "P/E", "P/B", "P/S"],
        index=3,
        key=f"{key_prefix}_selected_multiple",
    )
    for warning in warnings:
        st.warning(warning)
    render_status_grid(_multiples_context_cards(table, selected_multiple))
    st.write(peer_read.get("summary"))

    tab_table, tab_charts, tab_peers, tab_connections = st.tabs(["Multiples Table", "Charts", "Peer / Sector Comparison", "DCF & SOTP Connection"])
    with tab_table:
        show_table(table, "Multiples table unavailable.")
    with tab_charts:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(scenario_multiple_vs_peer_chart(table, selected_multiple), width="stretch", key=f"{key_prefix}_scenario_peer_chart")
            st.caption("What this shows: Bear/Base/Bull/User/Market-Implied/SOTP multiples beside peer and sector references. Why it matters: DCF upside is fragile if it requires a stretched multiple.")
        with c2:
            st.plotly_chart(premium_discount_heatmap(table), width="stretch", key=f"{key_prefix}_premium_heatmap")
            st.caption("What this shows: premium/discount versus peer median across scenarios. Why it matters: it highlights where assumptions become hard to defend.")
    with tab_peers:
        show_table(_peer_table(ctx, current, peer_medians, sector_medians), "Peer table unavailable.")
        show_table(peer_read.get("table"), "Peer premium read unavailable.")
        st.write("Premium is justified by higher growth, margins, OCF conversion, operating leverage, moat, lower capital intensity, or stronger capital allocation. Discount is justified by the opposite.")
    with tab_connections:
        selected_row = table[table["Metric"].astype(str) == selected_multiple]
        if selected_row.empty:
            st.info("Select a multiple to see DCF/SOTP connection.")
        else:
            row = selected_row.iloc[0]
            st.write(f"Base DCF implies {selected_multiple}: {fmt_percent(row.get('Base Case')) if 'Yield' in selected_multiple else fmt_multiple(row.get('Base Case'))}.")
            st.write(f"User Case implies {selected_multiple}: {fmt_percent(row.get('User Case')) if 'Yield' in selected_multiple else fmt_multiple(row.get('User Case'))}.")
            st.write(f"SOTP Case implies {selected_multiple}: {fmt_percent(row.get('SOTP Case')) if 'Yield' in selected_multiple else fmt_multiple(row.get('SOTP Case'))}.")
            st.write(row.get("Interpretation") or "Interpretation unavailable.")
            if row.get("Premium / Discount vs Peer") is not None and row.get("Premium / Discount vs Peer") > 0.25:
                st.warning("Valuation warning: the selected case implies a material premium to peer median. Check moat, growth, OCF margin, and capital intensity support.")
    return {"table": table, "peer_read": peer_read, "scenario_multiples": scenario_multiples}
