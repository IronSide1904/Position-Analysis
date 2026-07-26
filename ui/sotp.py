from __future__ import annotations

import pandas as pd
import streamlit as st

from models.sotp_model import (
    SOTP_SCENARIOS,
    VALUATION_METHODS,
    build_default_segment_data,
    normalize_segment_table,
    run_reverse_sotp,
    run_sotp,
    run_sotp_scenarios,
    sotp_assumption_comparison,
    sotp_summary_table,
)
from ui.charts import (
    dcf_vs_sotp_chart,
    sotp_implied_vs_peer_chart,
    sotp_revenue_vs_value_chart,
    sotp_segment_ev_chart,
    sotp_value_mix_chart,
)
from ui.components import metric_row, show_table
from ui.design_system import render_section, render_status_grid
from ui.formatting import fmt_money, fmt_percent, fmt_per_share


SOTP_EDITABLE_COLUMNS = [
    "Segment / Product / Service",
    "Revenue ($B)",
    "Revenue Growth",
    "Gross Margin",
    "OPEX Allocation",
    "OCF Margin",
    "NOPAT Margin",
    "CAPEX % Revenue",
    "Valuation Method",
    "Selected Multiple",
    "Manual Value ($B)",
    "Discount / Premium",
    "Confidence",
    "User Note",
]


def _state_key(ctx: dict, suffix: str) -> str:
    return f"sotp_{ctx.get('dataset', {}).get('ticker', 'default')}_{suffix}"


def _base_segments(ctx: dict) -> pd.DataFrame:
    dataset = ctx.get("dataset", {})
    return build_default_segment_data(ctx.get("historicals"), dataset, ctx.get("base_assumptions", {}))


def get_active_sotp(ctx: dict, scenario: str = "Base Case") -> dict:
    base = _base_segments(ctx)
    segment_state = _state_key(ctx, "segments")
    segments = st.session_state.get(segment_state, base)
    return run_sotp(
        segments,
        ctx.get("dataset", {}).get("market_data", {}),
        ctx.get("base_assumptions", {}),
        scenario=scenario,
        dcf_output=ctx.get("base_dcf"),
        peer_multiples=ctx.get("peer_df"),
        sector=ctx.get("dataset", {}).get("sector"),
    )


def sotp_data_status(segments: pd.DataFrame | None) -> tuple[str, str, str]:
    if segments is None or segments.empty:
        return "Unavailable", "Segment/product/service data is missing. Use the manual builder.", "warning"
    normalized = normalize_segment_table(segments)
    sources = " ".join(normalized.get("Source", pd.Series(dtype=str)).astype(str).tolist()).lower()
    confidences = set(normalized.get("Confidence", pd.Series(dtype=str)).astype(str).str.lower().tolist())
    revenue = pd.to_numeric(normalized.get("Revenue"), errors="coerce").fillna(0)
    if revenue.sum() <= 0:
        return "Manual Builder Needed", "Revenue by segment/product is missing.", "warning"
    if "manual builder fallback" in sources or "manual allocation" in sources or "fallback" in sources:
        return "Partial", "Using product/service fallback or manual allocation; verify against segment disclosures.", "caution"
    if "manual review" in confidences or "low" in confidences:
        return "Partial", "SOTP is available, but some assumptions need manual review.", "caution"
    return "Ready", "Segment assumptions are populated and ready for review.", "supportive"


def _sotp_review_plan(ctx: dict, segments: pd.DataFrame | None) -> pd.DataFrame:
    dataset = ctx.get("dataset", {})
    ticker = str(dataset.get("ticker") or "").upper()
    if ticker == "AAPL":
        rows = [
            ("Product revenue table", "Confirms iPhone, Mac, iPad, Wearables, and Services revenue.", "10-K revenue disaggregation / MD&A", "iPhone, Mac, iPad, Wearables, Services, net sales", "Product/service fallback", "Medium"),
            ("Products vs Services gross margin", "Determines whether Services deserve a higher multiple.", "10-K gross margin discussion / segment margin", "products gross margin, services gross margin", "Consolidated margin split", "Medium"),
            ("Services sub-drivers", "App Store/iCloud/AppleCare/Payments quality affects Services multiple.", "MD&A, earnings call, IR materials", "App Store, iCloud, AppleCare, payments, subscriptions", "Services aggregate", "Medium"),
        ]
    elif ticker == "AMZN":
        rows = [
            ("AWS segment economics", "AWS can dominate SOTP value despite smaller revenue share.", "Segment note / MD&A", "AWS revenue, AWS operating income", "Product/service fallback", "Medium"),
            ("Revenue disaggregation", "Ads, seller services and subscriptions carry different multiples.", "Revenue note", "advertising, third-party seller services, subscriptions", "Manual allocation", "Medium"),
        ]
    elif ticker == "NBIS":
        rows = [
            ("Capacity economics", "Blackwell/Rubin capacity drives revenue and CAPEX intensity.", "10-K / investor presentation / MD&A", "Blackwell, Rubin, GW, utilization, revenue per GW", "Manual capacity fallback", "Low"),
            ("Funding mix", "Debt/equity/prepayments affect EV-to-equity bridge.", "Debt notes / cash flow / financing disclosures", "prepayments, debt, equity, dilution, CAPEX", "Manual review", "Low"),
        ]
    else:
        rows = [
            ("Segment revenue", "SOTP needs revenue by segment/product/service.", "10-K segment note / revenue disaggregation", "segment revenue, product revenue, services revenue", "Manual builder fallback", "Medium"),
            ("Segment margins", "Different margins justify different multiples.", "MD&A / segment profit note", "operating income, gross margin, contribution margin", "Consolidated margin proxy", "Medium"),
            ("Peer multiples", "Selected multiples need external reference.", "Peer comp set / sector multiples", "EV/Revenue, EV/OCF, EV/FCF, P/E", "Sector fallback", "Medium"),
        ]
    return pd.DataFrame(
        [
            {
                "Missing / Review Item": item,
                "Why it matters": why,
                "Where to verify": where,
                "Suggested keywords": keywords,
                "Fallback used": fallback,
                "Confidence impact": confidence,
            }
            for item, why, where, keywords, fallback, confidence in rows
        ]
    )


def _sotp_reconciliation_table(result: dict, dcf_output: dict | None, market_data: dict | None) -> pd.DataFrame:
    dcf_output = dcf_output or {}
    market_data = market_data or {}
    rows = []

    def add(metric: str, dcf_value, sotp_value, interpretation: str) -> None:
        diff = sotp_value - dcf_value if dcf_value is not None and sotp_value is not None else None
        rows.append({"Metric": metric, "DCF": dcf_value, "SOTP": sotp_value, "Difference": diff, "Interpretation": interpretation})

    add("Enterprise Value", dcf_output.get("enterprise_value"), result.get("enterprise_value"), "Positive means SOTP EV exceeds DCF EV.")
    add("Net Debt", dcf_output.get("net_debt"), result.get("net_debt"), "Bridge from enterprise value to equity value.")
    add("Equity Value", dcf_output.get("equity_value"), result.get("equity_value"), "Equity value after net debt.")
    add("Fair Value / Share", dcf_output.get("fair_value_per_share"), result.get("fair_value_per_share"), "Per-share valuation comparison.")
    add("Upside / Downside", dcf_output.get("upside_downside_pct"), result.get("upside_downside_pct"), "Value versus current price.")
    rows.append(
        {
            "Metric": "Terminal Multiple / Selected Multiple",
            "DCF": dcf_output.get("terminal_multiple"),
            "SOTP": "Segment-specific",
            "Difference": None,
            "Interpretation": "SOTP uses one multiple per segment/product/service.",
        }
    )
    rows.append(
        {
            "Metric": "Confidence",
            "DCF": "Model-driven",
            "SOTP": result.get("whole_vs_sum") or "Manual Review",
            "Difference": None,
            "Interpretation": result.get("whole_vs_sum_interpretation"),
        }
    )
    return pd.DataFrame(rows)


def _segment_value_contributors(result: dict, limit: int = 4) -> str:
    segments = result.get("segments", pd.DataFrame())
    if segments is None or segments.empty or "Segment EV" not in segments:
        return "No segment value contributors available."
    top = segments.sort_values("Segment EV", ascending=False).head(limit)
    return "; ".join(f"{row.get('Segment')}: {fmt_money(row.get('Segment EV'))}" for _, row in top.iterrows())


def _workbench(ctx: dict, key_prefix: str) -> pd.DataFrame:
    segment_key = _state_key(ctx, "segments")
    base = _base_segments(ctx)
    if segment_key not in st.session_state:
        st.session_state[segment_key] = base
    st.caption("Segment data unavailable from filings? This manual segment builder stays active so SOTP is never a blank tab.")
    normalized = normalize_segment_table(st.session_state[segment_key], ctx.get("base_assumptions", {})).copy()
    normalized["Revenue ($B)"] = pd.to_numeric(normalized["Revenue"], errors="coerce") / 1e9
    normalized["Manual Value ($B)"] = pd.to_numeric(normalized["Manual Segment Value"], errors="coerce") / 1e9
    normalized = normalized.rename(columns={"Segment": "Segment / Product / Service", "OPEX % Revenue": "OPEX Allocation"})
    editor_input = normalized[SOTP_EDITABLE_COLUMNS].copy()
    pct_columns = ["Revenue Growth", "Gross Margin", "OPEX Allocation", "OCF Margin", "NOPAT Margin", "CAPEX % Revenue", "Discount / Premium"]
    for column in pct_columns:
        editor_input[column] = pd.to_numeric(editor_input[column], errors="coerce") * 100
    edited = st.data_editor(
        editor_input,
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "Valuation Method": st.column_config.SelectboxColumn("Valuation Method", options=VALUATION_METHODS),
            "Confidence": st.column_config.SelectboxColumn("Confidence", options=["Low", "Medium", "High", "Manual Review"]),
            "Revenue ($B)": st.column_config.NumberColumn("Revenue ($B)", min_value=0.0, step=0.1, format="$%.1fB"),
            "Manual Value ($B)": st.column_config.NumberColumn("Manual Value ($B)", min_value=0.0, step=0.1, format="$%.1fB"),
            "Revenue Growth": st.column_config.NumberColumn("Revenue Growth", min_value=-50.0, max_value=100.0, step=1.0, format="%.1f%%"),
            "Gross Margin": st.column_config.NumberColumn("Gross Margin", min_value=-50.0, max_value=100.0, step=1.0, format="%.1f%%"),
            "OPEX Allocation": st.column_config.NumberColumn("OPEX Allocation", min_value=0.0, max_value=100.0, step=1.0, format="%.1f%%"),
            "OCF Margin": st.column_config.NumberColumn("OCF Margin", min_value=-50.0, max_value=100.0, step=1.0, format="%.1f%%"),
            "NOPAT Margin": st.column_config.NumberColumn("NOPAT Margin", min_value=-50.0, max_value=100.0, step=1.0, format="%.1f%%"),
            "CAPEX % Revenue": st.column_config.NumberColumn("CAPEX % Revenue", min_value=0.0, max_value=100.0, step=1.0, format="%.1f%%"),
            "Selected Multiple": st.column_config.NumberColumn("Selected Multiple", min_value=0.0, max_value=80.0, step=0.5, format="%.1fx"),
            "Discount / Premium": st.column_config.NumberColumn("Discount / Premium", min_value=-80.0, max_value=100.0, step=5.0, format="%.1f%%"),
        },
        key=f"{key_prefix}_sotp_segment_editor",
    )
    model_frame = edited.copy()
    model_frame["Revenue"] = pd.to_numeric(model_frame["Revenue ($B)"], errors="coerce") * 1e9
    model_frame["Manual Segment Value"] = pd.to_numeric(model_frame["Manual Value ($B)"], errors="coerce") * 1e9
    model_frame = model_frame.rename(columns={"Segment / Product / Service": "Segment", "OPEX Allocation": "OPEX % Revenue"})
    model_frame = model_frame.drop(columns=["Revenue ($B)", "Manual Value ($B)"])
    for column in pct_columns:
        model_column = "OPEX % Revenue" if column == "OPEX Allocation" else column
        if model_column in model_frame:
            model_frame[model_column] = pd.to_numeric(model_frame[model_column], errors="coerce") / 100
    st.session_state[segment_key] = normalize_segment_table(model_frame, ctx.get("base_assumptions", {}))
    if st.button("Reset SOTP segments to dashboard fallback", key=f"{key_prefix}_reset_sotp"):
        st.session_state[segment_key] = base
        st.rerun()
    return st.session_state[segment_key]


def _assumption_explanation(selected_segment: str, selected_assumption: str, comparison: pd.DataFrame) -> None:
    row = comparison[
        (comparison["Segment"].astype(str) == str(selected_segment))
        & (comparison["Assumption"].astype(str) == str(selected_assumption))
    ]
    if row.empty:
        st.info("Select a segment and assumption to see the comparison.")
        return
    item = row.iloc[0]
    st.markdown(
        f"""
        <div class="pa-box">
            <div class="pa-box-title">SOTP Assumption Explanation</div>
            <strong>{selected_segment} - {selected_assumption}</strong><br/>
            <span class="pa-pill">User Case: {item.get("User Case")}</span>
            <span class="pa-pill">Base: {item.get("Base Case")}</span>
            <span class="pa-pill">Market-Implied: {item.get("Market-Implied")}</span><br/>
            <strong>Why it matters:</strong> this line controls the segment's standalone value and whether the segment deserves a premium or discount versus comparable businesses.<br/>
            <strong>Source badge:</strong> {item.get("Source Badge")}.<br/>
            <strong>Fair-value impact:</strong> review the SOTP summary below; changes flow through segment EV, equity value, and fair value per share.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sotp_tab(ctx: dict, analyst_details: bool = False, key_prefix: str = "sotp") -> dict:
    render_section(
        "SOTP Workbench",
        "Segment-level valuation answers whether the whole company is worth more or less than the sum of its parts.",
        "SOTP",
    )
    segments = _workbench(ctx, key_prefix)
    data_status, data_subtitle, data_card_status = sotp_data_status(segments)
    scenario = st.selectbox("SOTP scenario", SOTP_SCENARIOS, index=SOTP_SCENARIOS.index("User Case"), key=f"{key_prefix}_scenario")
    result = run_sotp(
        segments,
        ctx.get("dataset", {}).get("market_data", {}),
        ctx.get("base_assumptions", {}),
        scenario=scenario,
        dcf_output=ctx.get("base_dcf"),
        peer_multiples=ctx.get("peer_df"),
        sector=ctx.get("dataset", {}).get("sector"),
    )
    scenarios = run_sotp_scenarios(
        segments,
        ctx.get("dataset", {}).get("market_data", {}),
        ctx.get("base_assumptions", {}),
        ctx.get("base_dcf"),
        ctx.get("peer_df"),
        ctx.get("dataset", {}).get("sector"),
    )
    summary = sotp_summary_table(scenarios)
    if not summary.empty:
        summary["DCF Fair Value / Share"] = ctx.get("base_dcf", {}).get("fair_value_per_share")

    render_status_grid(
        [
            {"title": "SOTP Fair Value", "value": fmt_per_share(result.get("fair_value_per_share")), "subtitle": f"{scenario} standalone segment value.", "status": "info"},
            {"title": "Current Price", "value": fmt_per_share(ctx.get("dataset", {}).get("market_data", {}).get("price")), "subtitle": "Provider market price.", "status": "neutral"},
            {"title": "Upside / Downside", "value": fmt_percent(result.get("upside_downside_pct")), "subtitle": "SOTP fair value versus market price.", "status": "supportive" if (result.get("upside_downside_pct") or 0) > 0 else "warning"},
            {"title": "DCF Fair Value", "value": fmt_per_share(ctx.get("base_dcf", {}).get("fair_value_per_share")), "subtitle": "Base DCF comparison anchor.", "status": "neutral"},
            {"title": "SOTP vs DCF Gap", "value": fmt_percent(result.get("sotp_vs_dcf_gap_pct")), "subtitle": "Positive means SOTP EV exceeds DCF EV.", "status": "info"},
            {"title": "Market EV vs SOTP EV", "value": fmt_percent((result.get("current_market_ev") / result.get("enterprise_value") - 1) if result.get("current_market_ev") and result.get("enterprise_value") else None), "subtitle": "Premium or discount versus sum of parts.", "status": "neutral"},
            {"title": "Whole vs Parts", "value": result.get("whole_vs_sum"), "subtitle": result.get("whole_vs_sum_interpretation"), "status": "supportive" if "Hidden" in str(result.get("whole_vs_sum")) or ">" in str(result.get("whole_vs_sum")) else "caution"},
            {"title": "Data Confidence", "value": data_status, "subtitle": data_subtitle, "status": data_card_status},
        ]
    )
    if data_status in {"Unavailable", "Manual Builder Needed"}:
        st.warning(
            "SOTP Analysis Unavailable. Why: segment/product/service data is missing. What to do next: fetch latest 10-K segment disclosure, review revenue disaggregation, add manual segments/products/services, and use peer or manual multiples."
        )
    for warning in result.get("warnings", []):
        st.warning(warning)

    tab_summary, tab_builder, tab_assumptions, tab_valuation, tab_whole, tab_reconcile, tab_charts, tab_quality = st.tabs(
        [
            "SOTP Summary",
            "Segment / Product Builder",
            "Segment Assumptions",
            "Segment Valuation",
            "Whole vs Parts",
            "SOTP vs DCF",
            "SOTP Charts",
            "Data Quality",
        ]
    )
    with tab_summary:
        st.caption("SOTP scenario table: fair value per share, upside/downside, and whole-versus-parts interpretation.")
        show_table(summary, "SOTP scenario summary unavailable.")
        st.markdown("**Main Segment Value Contributors**")
        st.write(_segment_value_contributors(result))
    with tab_builder:
        st.caption("Edit the builder above to change segment/product/service revenue, margins, valuation method, multiples, confidence, and notes.")
        show_table(segments, "Segment/product/service builder is empty. Add manual rows above.")
    with tab_assumptions:
        segment_options = result.get("segments", pd.DataFrame()).get("Segment", pd.Series(dtype=str)).astype(str).tolist()
        selected_segment = st.selectbox("Segment selector", segment_options or ["Core business"], key=f"{key_prefix}_segment_selector")
        selected_assumption = st.selectbox(
            "Assumption group",
            ["Revenue Growth", "OCF Margin", "NOPAT Margin", "CAPEX % Revenue", "Selected Multiple", "Discount / Premium"],
            key=f"{key_prefix}_assumption_selector",
        )
        reverse = run_reverse_sotp(ctx.get("dataset", {}).get("market_data", {}), segments, ctx.get("base_assumptions", {}), ctx.get("peer_df"))
        comparison = sotp_assumption_comparison(_base_segments(ctx), segments, reverse.get("segments"))
        _assumption_explanation(selected_segment, selected_assumption, comparison)
        show_table(comparison, "SOTP assumption comparison unavailable.")
    with tab_valuation:
        st.subheader("Segment Valuation Table")
        show_table(result.get("segments"), "Segment valuation unavailable.")
        reverse = run_reverse_sotp(ctx.get("dataset", {}).get("market_data", {}), segments, ctx.get("base_assumptions", {}), ctx.get("peer_df"))
        st.subheader("Market-Implied Segment Multiples")
        st.warning(reverse.get("warning"))
        metric_row([("Current Market EV", reverse.get("enterprise_value"), "money"), ("SOTP EV", result.get("enterprise_value"), "money")])
        show_table(reverse.get("segments"), "Market-implied SOTP unavailable.")
    with tab_whole:
        st.markdown("**Whole vs Sum of Parts**")
        render_status_grid(
            [
                {"title": "Conclusion", "value": result.get("whole_vs_sum"), "subtitle": result.get("whole_vs_sum_interpretation"), "status": "info"},
                {"title": "SOTP EV", "value": fmt_money(result.get("enterprise_value")), "subtitle": "Sum of standalone segment values.", "status": "neutral"},
                {"title": "Market EV", "value": fmt_money(result.get("current_market_ev")), "subtitle": "Current market enterprise value.", "status": "neutral"},
            ]
        )
        st.caption("If SOTP materially exceeds DCF, identify hidden segment value. If DCF exceeds SOTP, synergies, shared platform economics, operating leverage, or cross-sell must justify the premium.")
        show_table(result.get("segments"), "No segment table available for whole-vs-parts analysis.")
    with tab_reconcile:
        st.markdown("**SOTP vs DCF Reconciliation**")
        reconciliation = _sotp_reconciliation_table(result, ctx.get("base_dcf"), ctx.get("dataset", {}).get("market_data", {}))
        show_table(reconciliation, "SOTP vs DCF reconciliation unavailable.")
        st.plotly_chart(dcf_vs_sotp_chart(summary), width="stretch", key=f"{key_prefix}_dcf_sotp_reconciliation_chart")
        st.caption("What this shows: DCF fair value versus SOTP fair value by scenario. Why it matters: it shows whether consolidated value or standalone segment value is carrying the thesis.")
    with tab_charts:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(sotp_segment_ev_chart(result.get("segments")), width="stretch", key=f"{key_prefix}_segment_ev_chart")
            st.caption("What this shows: which segments contribute most to SOTP EV. Why it matters: one segment can carry the whole thesis. Current interpretation: review the largest contributors first.")
            st.plotly_chart(sotp_revenue_vs_value_chart(result.get("segments")), width="stretch", key=f"{key_prefix}_revenue_value_chart")
            st.caption("What this shows: revenue scale versus value contribution. Why it matters: high-value segments should have better growth, margin, or multiple support. Current interpretation: gaps flag premium assumptions.")
        with c2:
            st.plotly_chart(sotp_value_mix_chart(result.get("segments")), width="stretch", key=f"{key_prefix}_value_mix_chart")
            st.caption("What this shows: segment value mix. Why it matters: concentration raises assumption risk. Current interpretation: concentrated value needs stronger evidence.")
            st.plotly_chart(sotp_implied_vs_peer_chart(result.get("segments")), width="stretch", key=f"{key_prefix}_implied_peer_chart")
            st.caption("What this shows: selected segment multiples versus peer and market-implied references. Why it matters: premiums need evidence. Current interpretation: review outlier multiples.")
        st.plotly_chart(dcf_vs_sotp_chart(summary), width="stretch", key=f"{key_prefix}_dcf_sotp_chart")
        st.caption("What this shows: DCF fair value versus SOTP fair value by scenario. Why it matters: it reconciles intrinsic value with standalone segment value.")
    with tab_quality:
        st.markdown("**Data Quality / Manual Review**")
        render_status_grid([{"title": "SOTP Data Status", "value": data_status, "subtitle": data_subtitle, "status": data_card_status}])
        show_table(_sotp_review_plan(ctx, segments), "No SOTP manual review plan available.")
        st.markdown("**Source / Confidence Table**")
        show_table(segments[[col for col in ["Segment", "Revenue", "Confidence", "Source", "User Note"] if col in segments]], "No segment source table available.")
    return result


def render_sotp_lens(ctx: dict, key_prefix: str = "sotp_lens") -> dict:
    segments = st.session_state.get(_state_key(ctx, "segments"), _base_segments(ctx))
    result = run_sotp(
        segments,
        ctx.get("dataset", {}).get("market_data", {}),
        ctx.get("base_assumptions", {}),
        scenario="User Case",
        dcf_output=ctx.get("base_dcf"),
        peer_multiples=ctx.get("peer_df"),
        sector=ctx.get("dataset", {}).get("sector"),
    )
    data_status, data_subtitle, data_card_status = sotp_data_status(segments)
    render_status_grid(
        [
            {"title": "SOTP Fair Value", "value": fmt_per_share(result.get("fair_value_per_share")), "subtitle": "User Case sum-of-parts read.", "status": "info"},
            {"title": "DCF Fair Value", "value": fmt_per_share(ctx.get("base_dcf", {}).get("fair_value_per_share")), "subtitle": "DCF comparison anchor.", "status": "neutral"},
            {"title": "Current Price", "value": fmt_per_share(ctx.get("dataset", {}).get("market_data", {}).get("price")), "subtitle": "Market price.", "status": "neutral"},
            {"title": "SOTP vs DCF Gap", "value": fmt_percent(result.get("sotp_vs_dcf_gap_pct")), "subtitle": "Positive means SOTP EV exceeds DCF EV.", "status": "info"},
            {"title": "Data Status", "value": data_status, "subtitle": data_subtitle, "status": data_card_status},
        ]
    )
    st.caption(_segment_value_contributors(result))
    if st.button("Open Full SOTP Workbench", key=f"{key_prefix}_open_full_sotp"):
        st.session_state["open_sotp_workbench_hint"] = True
        st.info("Open the SOTP tab in the main navigation. The full SOTP Workbench is now a dedicated tab.")
    return result


def _legacy_render_sotp_charts_removed():
    return None
