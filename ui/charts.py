from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def empty_chart(message: str = "Not enough data available."):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def price_action_chart(price_history: pd.DataFrame):
    if price_history is None or price_history.empty or "Close" not in price_history:
        return empty_chart("Price history unavailable.")
    frame = price_history.copy()
    if "Date" not in frame:
        frame["Date"] = frame.index
    fig = px.line(frame, x="Date", y="Close", title="Price Action")
    fig.update_yaxes(title="Share Price", tickprefix="$", separatethousands=True)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=320)
    return fig


def fcf_projection_chart(historicals: pd.DataFrame, forecast: pd.DataFrame):
    fig = go.Figure()
    if historicals is not None and not historicals.empty and "FCF" in historicals:
        fig.add_trace(go.Scatter(x=historicals["Period"], y=historicals["FCF"], mode="lines+markers", name="Historical FCF"))
    if forecast is not None and not forecast.empty and "FCF" in forecast:
        fig.add_trace(go.Scatter(x=forecast["Year"], y=forecast["FCF"], mode="lines+markers", name="Projected FCF"))
    if not fig.data:
        return empty_chart("FCF projection unavailable.")
    fig.update_yaxes(title="Free Cash Flow", tickprefix="$", separatethousands=True)
    fig.update_xaxes(title="Historical period / forecast year")
    fig.update_layout(title="Historical vs Projected FCF", margin=dict(l=10, r=10, t=40, b=10), height=320)
    return fig


def dcf_sensitivity_heatmap(sensitivity: pd.DataFrame):
    if sensitivity is None or sensitivity.empty:
        return empty_chart("DCF sensitivity unavailable.")
    data = sensitivity.copy()
    data["WACC"] = data["WACC"].map(lambda value: f"{float(value):.1%}")
    matrix = data.set_index("WACC")
    fig = px.imshow(
        matrix,
        aspect="auto",
        title="DCF Sensitivity: Fair Value per Share",
        labels={"x": "Terminal Growth Rate", "y": "WACC / Discount Rate", "color": "Fair Value"},
    )
    fig.update_traces(texttemplate="$%{z:,.0f}", textfont_size=12, hovertemplate="WACC: %{y}<br>Terminal growth: %{x}<br>Fair value: $%{z:,.0f}<extra></extra>")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=40), height=380)
    return fig


def scenario_valuation_bar(dcf: dict):
    value = dcf.get("fair_value_per_share")
    buy = dcf.get("buy_price_after_margin_of_safety")
    if value is None:
        return empty_chart("Scenario valuation unavailable.")
    df = pd.DataFrame({"scenario": ["Fair Value", "Buy Zone"], "price": [value, buy]})
    fig = px.bar(df, x="scenario", y="price", title="Scenario Valuation")
    fig.update_yaxes(title="Price per Share", tickprefix="$", separatethousands=True)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=300)
    return fig


def reverse_dcf_chart(reverse: dict, assumptions: dict):
    implied = reverse.get("implied_revenue_cagr")
    user = assumptions.get("revenue_cagr")
    if implied is None or user is None:
        return empty_chart("Reverse DCF unavailable.")
    df = pd.DataFrame({"case": ["User Case", "Market Implied"], "Revenue CAGR": [user, implied]})
    fig = px.bar(df, x="case", y="Revenue CAGR", title="Reverse DCF: Implied vs User Growth")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=300)
    return fig


def financial_revenue_margin_chart(historicals: pd.DataFrame):
    if historicals is None or historicals.empty or "Revenue" not in historicals:
        return empty_chart("Financial report data unavailable.")
    frame = historicals.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=frame["Period"], y=frame["Revenue"], name="Revenue", yaxis="y"))
    if "Gross Margin" in frame:
        fig.add_trace(go.Scatter(x=frame["Period"], y=frame["Gross Margin"], name="Gross Margin", yaxis="y2", mode="lines+markers"))
    fig.update_layout(
        title="Reported Revenue and Gross Margin",
        yaxis=dict(title="Revenue", tickprefix="$", separatethousands=True),
        yaxis2=dict(title="Gross Margin", overlaying="y", side="right", tickformat=".0%"),
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=50, b=40),
        height=360,
    )
    return fig


def financial_cash_flow_chart(historicals: pd.DataFrame):
    if historicals is None or historicals.empty:
        return empty_chart("Cash flow data unavailable.")
    columns = [col for col in ["OCF", "Total CAPEX", "FCF"] if col in historicals]
    if not columns:
        return empty_chart("Cash flow data unavailable.")
    frame = historicals[["Period", *columns]].melt("Period", var_name="Metric", value_name="Value")
    fig = px.bar(frame, x="Period", y="Value", color="Metric", barmode="group", title="Reported Cash Flow and CAPEX")
    fig.update_yaxes(title="Amount", tickprefix="$", separatethousands=True)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=40), height=360)
    return fig


def financial_profitability_chart(historicals: pd.DataFrame):
    if historicals is None or historicals.empty:
        return empty_chart("Profitability data unavailable.")
    columns = [col for col in ["EBIT", "NOPAT", "Net Income"] if col in historicals]
    if not columns:
        return empty_chart("Profitability data unavailable.")
    frame = historicals[["Period", *columns]].melt("Period", var_name="Metric", value_name="Value")
    fig = px.line(frame, x="Period", y="Value", color="Metric", markers=True, title="Reported Profitability")
    fig.update_yaxes(title="Amount", tickprefix="$", separatethousands=True)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=40), height=360)
    return fig


def moat_score_bar(moat_sources: pd.DataFrame):
    if moat_sources is None or moat_sources.empty:
        return empty_chart("Moat score data unavailable.")
    frame = moat_sources.copy()
    frame["Moat Source"] = frame["moat_source"].astype(str).str.replace(" / ", " /<br>", regex=False)
    frame["Score (1-10)"] = frame["score_1_to_10"]
    fig = px.bar(
        frame,
        x="Moat Source",
        y="Score (1-10)",
        title="Moat Source Scores",
        hover_data=["evidence", "confidence"] if {"evidence", "confidence"}.issubset(frame.columns) else None,
    )
    fig.update_yaxes(title="Score (1-10)", range=[0, 10], dtick=2)
    fig.update_xaxes(title="Moat Source", tickangle=-25, automargin=True)
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=140), height=560)
    return fig


def peer_scatter(peer_df: pd.DataFrame):
    if peer_df is None or peer_df.empty:
        return empty_chart("Peer scatter data unavailable.")
    fig = px.scatter(peer_df, x="market_cap", y="beta", text="ticker", title="Peer Market Cap vs Beta")
    fig.update_traces(textposition="top center")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=320)
    return fig


def peer_multiple_chart(peer_df: pd.DataFrame):
    if peer_df is None or peer_df.empty:
        return empty_chart("Peer multiple data unavailable.")
    fig = px.bar(peer_df, x="ticker", y="enterprise_value", title="Peer Enterprise Value")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=320)
    return fig


def sbc_vs_buybacks_chart(alignment: dict):
    table = alignment.get("sbc_table")
    if table is None or table.empty:
        return empty_chart("SBC and buyback data unavailable.")
    fig = px.bar(table, x="metric", y="value", title="SBC Snapshot")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=300)
    return fig


def ma_timeline_chart(ma: dict):
    timeline = ma.get("timeline")
    if timeline is None or timeline.empty:
        return empty_chart("M&A timeline data unavailable.")
    timeline = timeline.copy()
    timeline["index"] = range(1, len(timeline) + 1)
    fig = px.scatter(timeline, x="index", y="topic", hover_data=["event"], title="M&A Disclosure Timeline")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=300)
    return fig


def sotp_segment_ev_chart(segment_table: pd.DataFrame):
    if segment_table is None or segment_table.empty or "Segment EV" not in segment_table:
        return empty_chart("SOTP segment contribution unavailable.")
    frame = segment_table.copy()
    fig = px.bar(frame, x="Segment", y="Segment EV", color="Confidence" if "Confidence" in frame else None, title="SOTP Segment EV Contribution")
    fig.update_yaxes(title="Segment Enterprise Value", tickprefix="$", separatethousands=True)
    fig.update_xaxes(title="Segment", automargin=True)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=60), height=360)
    return fig


def sotp_value_mix_chart(segment_table: pd.DataFrame):
    if segment_table is None or segment_table.empty or "Segment EV" not in segment_table:
        return empty_chart("SOTP value mix unavailable.")
    frame = segment_table.copy()
    fig = px.pie(frame, names="Segment", values="Segment EV", title="Segment Value Mix", hole=0.45)
    fig.update_traces(textinfo="label+percent", hovertemplate="%{label}<br>EV: $%{value:,.0f}<extra></extra>")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=20), height=360)
    return fig


def sotp_revenue_vs_value_chart(segment_table: pd.DataFrame):
    required = {"Segment", "Revenue", "Segment EV"}
    if segment_table is None or segment_table.empty or not required.issubset(segment_table.columns):
        return empty_chart("SOTP revenue versus value unavailable.")
    frame = segment_table.copy()
    fig = px.scatter(
        frame,
        x="Revenue",
        y="Segment EV",
        size="Segment EV",
        color="Valuation Method" if "Valuation Method" in frame else None,
        text="Segment",
        title="Segment Revenue vs Segment Value",
    )
    fig.update_traces(textposition="top center")
    fig.update_xaxes(title="Segment Revenue", tickprefix="$", separatethousands=True)
    fig.update_yaxes(title="Segment Enterprise Value", tickprefix="$", separatethousands=True)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=40), height=380)
    return fig


def sotp_implied_vs_peer_chart(segment_table: pd.DataFrame):
    required = {"Segment", "Selected Multiple", "Peer Multiple"}
    if segment_table is None or segment_table.empty or not required.issubset(segment_table.columns):
        return empty_chart("Segment multiple comparison unavailable.")
    frame = segment_table.copy()
    melted = frame.melt("Segment", value_vars=["Selected Multiple", "Peer Multiple", "Market-Implied Multiple"], var_name="Multiple Type", value_name="Multiple")
    melted = melted.dropna(subset=["Multiple"])
    if melted.empty:
        return empty_chart("Segment multiple comparison unavailable.")
    fig = px.bar(melted, x="Segment", y="Multiple", color="Multiple Type", barmode="group", title="Segment Multiples vs Peer / Market-Implied")
    fig.update_yaxes(title="Multiple", ticksuffix="x")
    fig.update_xaxes(title="Segment", automargin=True)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=70), height=380)
    return fig


def dcf_vs_sotp_chart(summary_table: pd.DataFrame):
    if summary_table is None or summary_table.empty:
        return empty_chart("DCF vs SOTP comparison unavailable.")
    rows = []
    for _, row in summary_table.iterrows():
        rows.append({"Scenario": row.get("Scenario"), "Method": "SOTP Fair Value", "Fair Value": row.get("Fair Value / Share")})
        if row.get("DCF Fair Value / Share") is not None:
            rows.append({"Scenario": row.get("Scenario"), "Method": "DCF Fair Value", "Fair Value": row.get("DCF Fair Value / Share")})
    frame = pd.DataFrame(rows).dropna(subset=["Fair Value"])
    if frame.empty:
        return empty_chart("DCF vs SOTP comparison unavailable.")
    fig = px.bar(frame, x="Scenario", y="Fair Value", color="Method", barmode="group", title="DCF vs SOTP Fair Value")
    fig.update_yaxes(title="Fair Value per Share", tickprefix="$", separatethousands=True)
    fig.update_xaxes(title="Scenario", automargin=True)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=70), height=380)
    return fig


def scenario_multiple_vs_peer_chart(multiples_table: pd.DataFrame, selected_multiple: str):
    if multiples_table is None or multiples_table.empty:
        return empty_chart("Scenario multiples unavailable.")
    row = multiples_table[multiples_table["Metric"].astype(str) == selected_multiple]
    if row.empty:
        return empty_chart("Selected multiple unavailable.")
    row = row.iloc[0]
    scenarios = ["Bear Case", "Base Case", "Bull Case", "User Case", "Market-Implied Case", "SOTP Case", "Peer Median", "Sector Median"]
    frame = pd.DataFrame({"Case": scenarios, "Multiple": [row.get(case) for case in scenarios]})
    frame = frame.dropna(subset=["Multiple"])
    if frame.empty:
        return empty_chart("Selected multiple unavailable.")
    fig = px.bar(frame, x="Case", y="Multiple", title=f"{selected_multiple}: Scenario Multiples vs Peer Median")
    if "Yield" in selected_multiple:
        fig.update_yaxes(title=selected_multiple, tickformat=".0%")
    else:
        fig.update_yaxes(title=selected_multiple, ticksuffix="x")
    fig.update_xaxes(title="Scenario / Reference", tickangle=-20, automargin=True)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=90), height=390)
    return fig


def premium_discount_heatmap(multiples_table: pd.DataFrame):
    if multiples_table is None or multiples_table.empty:
        return empty_chart("Premium / discount heatmap unavailable.")
    scenarios = ["Current Company", "Bear Case", "Base Case", "Bull Case", "User Case", "Market-Implied Case", "SOTP Case"]
    rows = []
    for _, row in multiples_table.iterrows():
        peer = row.get("Peer Median")
        if peer in (None, 0):
            continue
        item = {"Metric": row.get("Metric")}
        for scenario in scenarios:
            value = row.get(scenario)
            item[scenario] = (value / peer - 1) if value is not None and peer else None
        rows.append(item)
    frame = pd.DataFrame(rows).set_index("Metric") if rows else pd.DataFrame()
    if frame.empty:
        return empty_chart("Premium / discount heatmap unavailable.")
    fig = px.imshow(frame, aspect="auto", color_continuous_scale="RdYlGn_r", title="Premium / Discount vs Peer Median")
    fig.update_traces(texttemplate="%{z:.0%}", hovertemplate="Metric: %{y}<br>Case: %{x}<br>Premium/discount: %{z:.1%}<extra></extra>")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=60), height=420)
    return fig
