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
from ui.formatting import fmt_percent, fmt_per_share


SOTP_EDITABLE_COLUMNS = [
    "Segment",
    "Revenue",
    "Revenue Growth",
    "OCF Margin",
    "NOPAT Margin",
    "CAPEX % Revenue",
    "Valuation Method",
    "Selected Multiple",
    "Discount / Premium",
    "Confidence",
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


def _workbench(ctx: dict, key_prefix: str) -> pd.DataFrame:
    segment_key = _state_key(ctx, "segments")
    base = _base_segments(ctx)
    if segment_key not in st.session_state:
        st.session_state[segment_key] = base
    st.caption("Segment data unavailable from filings? This manual segment builder stays active so SOTP is never a blank tab.")
    editor_input = normalize_segment_table(st.session_state[segment_key], ctx.get("base_assumptions", {}))[SOTP_EDITABLE_COLUMNS].copy()
    pct_columns = ["Revenue Growth", "OCF Margin", "NOPAT Margin", "CAPEX % Revenue", "Discount / Premium"]
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
            "Revenue": st.column_config.NumberColumn("Revenue", min_value=0.0, step=1_000_000.0, format="$%.0f"),
            "Revenue Growth": st.column_config.NumberColumn("Revenue Growth", min_value=-50.0, max_value=100.0, step=1.0, format="%.1f%%"),
            "OCF Margin": st.column_config.NumberColumn("OCF Margin", min_value=-50.0, max_value=100.0, step=1.0, format="%.1f%%"),
            "NOPAT Margin": st.column_config.NumberColumn("NOPAT Margin", min_value=-50.0, max_value=100.0, step=1.0, format="%.1f%%"),
            "CAPEX % Revenue": st.column_config.NumberColumn("CAPEX % Revenue", min_value=0.0, max_value=100.0, step=1.0, format="%.1f%%"),
            "Selected Multiple": st.column_config.NumberColumn("Selected Multiple", min_value=0.0, max_value=80.0, step=0.5, format="%.1fx"),
            "Discount / Premium": st.column_config.NumberColumn("Discount / Premium", min_value=-80.0, max_value=100.0, step=5.0, format="%.1f%%"),
        },
        key=f"{key_prefix}_sotp_segment_editor",
    )
    model_frame = edited.copy()
    for column in pct_columns:
        model_frame[column] = pd.to_numeric(model_frame[column], errors="coerce") / 100
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
    scenario = st.selectbox("SOTP scenario", SOTP_SCENARIOS, index=1, key=f"{key_prefix}_scenario")
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
            {"title": "DCF Fair Value", "value": fmt_per_share(ctx.get("base_dcf", {}).get("fair_value_per_share")), "subtitle": "Base DCF comparison anchor.", "status": "neutral"},
            {"title": "Current Price", "value": fmt_per_share(ctx.get("dataset", {}).get("market_data", {}).get("price")), "subtitle": "Provider market price.", "status": "neutral"},
            {"title": "Whole vs Parts", "value": result.get("whole_vs_sum"), "subtitle": result.get("whole_vs_sum_interpretation"), "status": "supportive" if "Hidden" in str(result.get("whole_vs_sum")) or ">" in str(result.get("whole_vs_sum")) else "caution"},
            {"title": "SOTP vs DCF Gap", "value": fmt_percent(result.get("sotp_vs_dcf_gap_pct")), "subtitle": "Positive means SOTP EV exceeds DCF EV.", "status": "info"},
        ]
    )
    for warning in result.get("warnings", []):
        st.warning(warning)

    tab_summary, tab_segments, tab_reverse, tab_charts = st.tabs(["SOTP Summary", "Segment Assumptions", "Market-Implied SOTP", "Charts"])
    with tab_summary:
        st.caption("SOTP scenario table: fair value per share, upside/downside, and whole-versus-parts interpretation.")
        show_table(summary, "SOTP scenario summary unavailable.")
    with tab_segments:
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
        st.subheader("Segment Valuation Table")
        show_table(result.get("segments"), "Segment valuation unavailable.")
    with tab_reverse:
        reverse = run_reverse_sotp(ctx.get("dataset", {}).get("market_data", {}), segments, ctx.get("base_assumptions", {}), ctx.get("peer_df"))
        st.warning(reverse.get("warning"))
        metric_row([("Current Market EV", reverse.get("enterprise_value"), "money"), ("SOTP EV", result.get("enterprise_value"), "money")])
        show_table(reverse.get("segments"), "Market-implied SOTP unavailable.")
    with tab_charts:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(sotp_segment_ev_chart(result.get("segments")), width="stretch", key=f"{key_prefix}_segment_ev_chart")
            st.caption("What this shows: which segments contribute most to SOTP EV. Why it matters: one segment can carry the whole thesis.")
            st.plotly_chart(sotp_revenue_vs_value_chart(result.get("segments")), width="stretch", key=f"{key_prefix}_revenue_value_chart")
            st.caption("What this shows: revenue scale versus value contribution. Why it matters: high-value segments should have better growth, margin, or multiple support.")
        with c2:
            st.plotly_chart(sotp_value_mix_chart(result.get("segments")), width="stretch", key=f"{key_prefix}_value_mix_chart")
            st.caption("What this shows: segment value mix. Why it matters: concentration raises assumption risk.")
            st.plotly_chart(sotp_implied_vs_peer_chart(result.get("segments")), width="stretch", key=f"{key_prefix}_implied_peer_chart")
            st.caption("What this shows: selected segment multiples versus peer and market-implied references. Why it matters: premiums need evidence.")
        st.plotly_chart(dcf_vs_sotp_chart(summary), width="stretch", key=f"{key_prefix}_dcf_sotp_chart")
        st.caption("What this shows: DCF fair value versus SOTP fair value by scenario. Why it matters: it reconciles intrinsic value with standalone segment value.")
    return result
