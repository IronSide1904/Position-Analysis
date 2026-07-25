from __future__ import annotations


SUPPORTED_DRIVER_PROFILES = [
    "AI Infrastructure / Data Center",
    "SaaS / Software",
    "Semiconductor",
    "Marketplace / Platform",
    "Consumer Brand / Retail",
    "Financial / Fintech",
    "Industrial / Hardware",
    "Energy / Commodity",
    "Biotech / Pharma",
    "Real Estate / REIT",
    "Advertising / Media / Ad-Tech",
    "General",
]


AI_INFRA_DRIVER_ROWS = [
    "blackwell_gw_deployed",
    "rubin_gw_deployed",
    "other_gw_deployed",
    "utilization",
    "revenue_per_blackwell_gw",
    "revenue_per_rubin_gw",
    "revenue_per_other_gw",
    "adjusted_ebitda_margin",
    "hardware_cost_per_blackwell_gw",
    "hardware_cost_per_rubin_gw",
    "hardware_cost_per_other_gw",
    "land_power_cooling_cost_per_blackwell_gw",
    "land_power_cooling_cost_per_rubin_gw",
    "land_power_cooling_cost_per_other_gw",
    "gpu_useful_life",
    "infrastructure_useful_life",
    "maintenance_capex_pct_revenue",
    "customer_prepayment_pct",
    "equity_funding_pct",
    "equity_issue_price",
    "sbc_dilution_pct",
    "share_repurchases",
    "cost_of_debt",
    "risk_free_rate",
    "beta",
    "equity_risk_premium",
    "tax_rate",
    "exit_ebitda_multiple",
    "exit_ebit_multiple",
    "earnings_multiple",
]


GENERAL_DRIVER_ROWS = [
    "capacity_added",
    "utilization",
    "revenue_per_unit",
    "ebitda_margin",
    "maintenance_cost_per_unit",
    "hardware_cost_per_unit",
    "infrastructure_cost_per_unit",
    "land_cost_per_unit",
    "hardware_useful_life",
    "infrastructure_useful_life",
    "customer_prepayment_pct",
    "grant_funding_pct",
    "equity_funding_pct",
    "equity_issue_price",
    "sbc_dilution_pct",
    "share_repurchases",
    "cost_of_debt",
    "risk_free_rate",
    "beta",
    "equity_risk_premium",
    "exit_ebitda_multiple",
    "exit_ebit_multiple",
]


def _generic_template(
    profile: str,
    description: str,
    driver_groups: dict[str, list[str]],
    driver_overrides: dict[str, dict],
    defaults: dict | None = None,
    scenario_rules: dict | None = None,
    manual_review_questions: list[str] | None = None,
) -> dict:
    return {
        "profile": profile,
        "description": description,
        "driver_groups": driver_groups,
        "drivers": {key: driver_overrides.get(key, {}) for rows in driver_groups.values() for key in rows},
        "driver_overrides": driver_overrides,
        "default_driver_rows": [key for rows in driver_groups.values() for key in rows],
        "derived_financial_lines": [
            "Revenue",
            "Gross Profit",
            "OPEX",
            "EBIT",
            "NOPAT",
            "OCF",
            "Maintenance CAPEX",
            "Growth CAPEX",
            "Total CAPEX",
            "FCF",
            "Net Debt",
            "Shares Outstanding",
            "Fair Value Per Share",
        ],
        "default_driver_assumptions": defaults or {},
        "scenario_rules": scenario_rules
        or {
            "Bear Case": {"capacity": 0.85, "unit_economics": 0.92, "margin_delta": -0.03, "capex": 1.12},
            "Base Case": {"capacity": 1.00, "unit_economics": 1.00, "margin_delta": 0.00, "capex": 1.00},
            "Bull Case": {"capacity": 1.15, "unit_economics": 1.08, "margin_delta": 0.03, "capex": 0.95},
        },
        "manual_review_questions": manual_review_questions or [
            "Which operating driver best explains revenue growth?",
            "Which cost bucket most affects cash conversion?",
            "What reinvestment is required to sustain the forecast?",
            "Could dilution or leverage change per-share value materially?",
        ],
    }


SOFTWARE_TEMPLATE = _generic_template(
    "SaaS / Software",
    "ARR/revenue base, customer growth, ARPU, net retention, sales efficiency, R&D, SBC, OCF conversion, and terminal multiple.",
    {
        "Revenue Engine": ["capacity_added", "revenue_per_unit", "utilization"],
        "Margins": ["ebitda_margin"],
        "Cash Conversion": ["customer_prepayment_pct", "maintenance_cost_per_unit"],
        "Dilution": ["sbc_dilution_pct", "share_repurchases"],
        "Valuation": ["risk_free_rate", "beta", "equity_risk_premium", "exit_ebitda_multiple", "exit_ebit_multiple"],
    },
    {
        "capacity_added": {"label": "New Customer / Seat Growth", "impact": "Customer growth -> ARR/revenue -> OCF -> FCF.", "range": "-10% to 40% growth proxy.", "source": "Customer count, ARR, RPO, or analyst estimate."},
        "revenue_per_unit": {"label": "ARPU / ACV", "impact": "ARPU converts customers/seats into recurring revenue.", "range": "Compare ARR/revenue per customer against history and peers.", "source": "ARR, revenue, customer count, or peer economics."},
        "utilization": {"label": "Net Revenue Retention", "impact": "NRR affects revenue durability, expansion, OCF, and terminal value.", "range": "80% to 140%; verify churn and expansion.", "source": "Company retention disclosure or peer benchmark."},
        "ebitda_margin": {"label": "Operating Margin / Sales Efficiency", "impact": "Sales efficiency and R&D/G&A leverage drive EBIT and FCF.", "range": "-20% to 45%.", "source": "Historical margin, S&M efficiency, and peer margins."},
        "customer_prepayment_pct": {"label": "Deferred Revenue / Billing Support", "impact": "Annual billing can support OCF but may not be permanent.", "range": "0% to 30% of reinvestment proxy.", "source": "Deferred revenue and billings trend."},
        "maintenance_cost_per_unit": {"label": "CAPEX / Platform Spend", "impact": "Capitalized platform spend reduces FCF.", "range": "0% to 8% of revenue proxy.", "source": "CAPEX and capitalized software disclosures."},
        "sbc_dilution_pct": {"label": "SBC Dilution", "impact": "SBC raises share count and lowers per-share value.", "range": "-5% to 8%.", "source": "SBC and diluted share trend."},
    },
    defaults={"utilization": 1.05, "ebitda_margin": 0.18, "maintenance_cost_per_unit": 0.02, "sbc_dilution_pct": 0.02},
    manual_review_questions=[
        "Verify ARR/RPO, net revenue retention, churn, and customer growth.",
        "Separate S&M, R&D, G&A leverage rather than relying only on one OPEX line.",
        "Check SBC dilution versus buybacks.",
        "Review deferred revenue quality before treating OCF as recurring.",
    ],
)


SEMICONDUCTOR_TEMPLATE = _generic_template(
    "Semiconductor",
    "Units shipped, ASP, utilization, product mix, R&D intensity, inventory cycle, CAPEX intensity, and cycle-normalized multiples.",
    {
        "Demand": ["capacity_added", "utilization"],
        "Pricing": ["revenue_per_unit"],
        "Margins": ["ebitda_margin"],
        "Reinvestment": ["hardware_cost_per_unit", "infrastructure_cost_per_unit", "maintenance_cost_per_unit"],
        "Cycle / Valuation": ["risk_free_rate", "beta", "equity_risk_premium", "exit_ebitda_multiple", "exit_ebit_multiple"],
    },
    {
        "capacity_added": {"label": "Units Shipped Growth", "impact": "Units shipped -> revenue and factory absorption.", "range": "-20% to 40%; check cycle.", "source": "Shipments, wafers, product volume, or analyst estimate."},
        "utilization": {"label": "Capacity Utilization", "impact": "Utilization affects revenue, gross margin, and inventory risk.", "range": "50% to 100%.", "source": "Fab/load factor, supply commentary, inventory."},
        "revenue_per_unit": {"label": "ASP / Product Mix", "impact": "ASP and mix convert units into revenue and gross margin.", "range": "Compare to historical ASP/mix.", "source": "Product mix, ASP, segment revenue."},
        "ebitda_margin": {"label": "Cycle-Normalized Margin", "impact": "Margin drives EBIT, NOPAT, OCF, and terminal multiple.", "range": "0% to 55%.", "source": "Gross margin, R&D, SG&A, utilization."},
        "hardware_cost_per_unit": {"label": "Growth CAPEX Intensity", "impact": "Capacity CAPEX can pressure FCF before revenue arrives.", "range": "0% to 35% revenue proxy.", "source": "CAPEX guidance and fabs/tooling."},
        "maintenance_cost_per_unit": {"label": "Maintenance CAPEX", "impact": "Required reinvestment reduces FCF.", "range": "0% to 12% revenue proxy.", "source": "CAPEX and depreciation trend."},
    },
    defaults={"utilization": 0.80, "ebitda_margin": 0.28, "hardware_cost_per_unit": 0.08, "maintenance_cost_per_unit": 0.04, "beta": 1.3},
    scenario_rules={"Bear Case": {"capacity": 0.75, "unit_economics": 0.85, "margin_delta": -0.06, "capex": 1.20}, "Base Case": {"capacity": 1.0, "unit_economics": 1.0, "margin_delta": 0.0, "capex": 1.0}, "Bull Case": {"capacity": 1.25, "unit_economics": 1.15, "margin_delta": 0.05, "capex": 0.95}},
)


MARKETPLACE_TEMPLATE = _generic_template(
    "Marketplace / Platform",
    "GMV/activity, take rate, active users, transactions per user, network strength, trust/safety costs, OCF margin, and terminal multiple.",
    {
        "Network Activity": ["capacity_added", "utilization"],
        "Monetization": ["revenue_per_unit"],
        "Margins": ["ebitda_margin", "maintenance_cost_per_unit"],
        "Cash Conversion": ["customer_prepayment_pct"],
        "Valuation": ["risk_free_rate", "beta", "equity_risk_premium", "exit_ebitda_multiple", "exit_ebit_multiple"],
    },
    {
        "capacity_added": {"label": "GMV / Active User Growth", "impact": "GMV and users -> monetizable activity -> revenue.", "range": "-10% to 35%.", "source": "GMV, active users, transactions, or analyst estimate."},
        "utilization": {"label": "Transactions per User / Fill", "impact": "Activity density supports revenue durability and network effects.", "range": "50% to 120% activity proxy.", "source": "Frequency, fill, engagement, retention."},
        "revenue_per_unit": {"label": "Take Rate / Monetization", "impact": "Take rate converts GMV/activity into revenue.", "range": "1% to 30% depending on marketplace.", "source": "Revenue / GMV or platform fee disclosure."},
        "ebitda_margin": {"label": "Platform Margin", "impact": "Scale and trust/safety cost drive EBIT and FCF.", "range": "-10% to 45%.", "source": "Gross margin, trust/safety, S&M, R&D."},
        "maintenance_cost_per_unit": {"label": "Platform CAPEX / Trust Cost", "impact": "Trust/safety and platform investment reduce FCF.", "range": "0% to 10% revenue proxy.", "source": "OPEX/CAPEX disclosures."},
    },
    defaults={"utilization": 0.90, "ebitda_margin": 0.20, "maintenance_cost_per_unit": 0.02},
)


CONSUMER_TEMPLATE = _generic_template(
    "Consumer Brand / Retail",
    "Units/stores, ASP, same-store sales, channel growth, gross margin, marketing intensity, inventory, working capital, and buybacks.",
    {
        "Volume": ["capacity_added", "utilization"],
        "Pricing": ["revenue_per_unit"],
        "Margins": ["ebitda_margin"],
        "Inventory / CAPEX": ["maintenance_cost_per_unit", "hardware_cost_per_unit"],
        "Capital Allocation": ["sbc_dilution_pct", "share_repurchases"],
        "Valuation": ["risk_free_rate", "beta", "equity_risk_premium", "exit_ebitda_multiple", "exit_ebit_multiple"],
    },
    {
        "capacity_added": {"label": "Units / Store Growth", "impact": "Units or store count -> revenue growth.", "range": "-10% to 20%.", "source": "Unit sales, store count, channel growth."},
        "utilization": {"label": "Same-Store Sales / Channel Productivity", "impact": "Productivity supports revenue and margin leverage.", "range": "80% to 120% productivity proxy.", "source": "Same-store sales, traffic, conversion."},
        "revenue_per_unit": {"label": "ASP / Revenue per Store", "impact": "Pricing and mix convert volume into revenue.", "range": "Compare to ASP/revenue per store history.", "source": "Pricing, mix, segment/store data."},
        "ebitda_margin": {"label": "Gross Margin / SG&A Leverage", "impact": "Pricing, mix, supply chain and SG&A drive EBIT.", "range": "0% to 35%.", "source": "Gross margin and SG&A trend."},
        "maintenance_cost_per_unit": {"label": "Inventory / Maintenance CAPEX", "impact": "Inventory and store investment affect OCF and FCF.", "range": "0% to 8% revenue proxy.", "source": "Inventory days, CAPEX per store."},
    },
    defaults={"utilization": 0.95, "ebitda_margin": 0.16, "maintenance_cost_per_unit": 0.03, "hardware_cost_per_unit": 0.02},
)


FINANCIAL_TEMPLATE = _generic_template(
    "Financial / Fintech",
    "AUM/deposits/loan book, net interest margin, fee rate, credit losses, efficiency ratio, ROE, book value growth, capital ratio, and P/B.",
    {
        "Assets / Flows": ["capacity_added", "revenue_per_unit"],
        "Spread / Fees": ["utilization"],
        "Efficiency": ["ebitda_margin"],
        "Capital / Dilution": ["sbc_dilution_pct", "share_repurchases"],
        "Valuation": ["risk_free_rate", "beta", "equity_risk_premium", "exit_ebitda_multiple", "exit_ebit_multiple"],
    },
    {
        "capacity_added": {"label": "AUM / Loan Book Growth", "impact": "Balance sheet or flow growth drives revenue base.", "range": "-10% to 25%.", "source": "AUM, deposits, loans, payment volume."},
        "revenue_per_unit": {"label": "Fee Rate / Yield", "impact": "Yield and fee rate convert assets/flows into revenue.", "range": "Review NIM, fee rate, take rate.", "source": "NIM, fee yield, transaction economics."},
        "utilization": {"label": "Transaction / Spread Utilization", "impact": "Utilization proxy captures spread, fee capture, and usage.", "range": "50% to 120% proxy.", "source": "Volume, spread, activity."},
        "ebitda_margin": {"label": "Efficiency / ROE Proxy", "impact": "Efficiency and credit losses drive earnings and P/B.", "range": "0% to 45%.", "source": "Efficiency ratio, loss rate, ROE."},
    },
    defaults={"utilization": 0.85, "ebitda_margin": 0.22, "maintenance_cost_per_unit": 0.01},
)


INDUSTRIAL_TEMPLATE = _generic_template(
    "Industrial / Hardware",
    "Backlog, book-to-bill, backlog conversion, production utilization, ASP, gross margin, inventory, receivables, CAPEX, and OCF conversion.",
    {
        "Orders / Backlog": ["capacity_added", "utilization"],
        "Production / Pricing": ["revenue_per_unit"],
        "Margins": ["ebitda_margin"],
        "Working Capital / CAPEX": ["maintenance_cost_per_unit", "hardware_cost_per_unit", "infrastructure_cost_per_unit"],
        "Valuation": ["risk_free_rate", "beta", "equity_risk_premium", "exit_ebitda_multiple", "exit_ebit_multiple"],
    },
    {
        "capacity_added": {"label": "Backlog / Unit Growth", "impact": "Backlog conversion and units shipped drive revenue.", "range": "-10% to 30%.", "source": "Backlog, orders, book-to-bill, RPO."},
        "utilization": {"label": "Backlog Conversion / Utilization", "impact": "Conversion affects revenue, inventory, OCF, and working capital.", "range": "40% to 100%.", "source": "Backlog conversion, utilization, production data."},
        "revenue_per_unit": {"label": "ASP / Mix", "impact": "Pricing and mix convert production into revenue.", "range": "Compare with segment/unit trend.", "source": "ASP, mix, segment revenue."},
        "ebitda_margin": {"label": "Gross Margin / OPEX Leverage", "impact": "Manufacturing margin drives EBIT and FCF.", "range": "0% to 35%.", "source": "Gross margin and OPEX trend."},
        "maintenance_cost_per_unit": {"label": "Inventory / Maintenance CAPEX", "impact": "Inventory and sustaining investment reduce OCF/FCF.", "range": "0% to 10% revenue proxy.", "source": "Inventory, receivables, CAPEX."},
        "hardware_cost_per_unit": {"label": "Growth CAPEX % Revenue", "impact": "Capacity expansion can pressure near-term FCF.", "range": "0% to 20% revenue proxy.", "source": "CAPEX guidance and equipment needs."},
    },
    defaults={"utilization": 0.75, "ebitda_margin": 0.18, "maintenance_cost_per_unit": 0.03, "hardware_cost_per_unit": 0.04},
)


ENERGY_TEMPLATE = _generic_template(
    "Energy / Commodity",
    "Production volume, realized price, operating cost per unit, decline rate, reserve life, hedging, maintenance/growth CAPEX, leverage, and FCF yield.",
    {
        "Production": ["capacity_added", "utilization"],
        "Commodity Price": ["revenue_per_unit"],
        "Costs / Margins": ["ebitda_margin", "maintenance_cost_per_unit"],
        "CAPEX / Balance Sheet": ["hardware_cost_per_unit", "cost_of_debt"],
        "Valuation": ["risk_free_rate", "beta", "equity_risk_premium", "exit_ebitda_multiple", "exit_ebit_multiple"],
    },
    {
        "capacity_added": {"label": "Production Volume Growth", "impact": "Production volume drives revenue and reserve depletion.", "range": "-15% to 20%.", "source": "Production, reserves, decline rates."},
        "utilization": {"label": "Realization / Decline Offset", "impact": "Realized production and decline affect revenue durability.", "range": "60% to 110%.", "source": "Decline, hedging, uptime."},
        "revenue_per_unit": {"label": "Realized Commodity Price", "impact": "Price converts production into revenue and FCF.", "range": "Use strip price and realized differential.", "source": "Commodity price, hedges, realizations."},
        "ebitda_margin": {"label": "Operating Cost / EBITDA Margin", "impact": "Cost per unit drives NOPAT and FCF.", "range": "5% to 60%.", "source": "Lifting cost, transportation, royalties."},
        "hardware_cost_per_unit": {"label": "Growth CAPEX / Development Cost", "impact": "Development CAPEX determines reserve replacement and FCF.", "range": "0% to 35% revenue proxy.", "source": "CAPEX plan and reserves."},
    },
    defaults={"utilization": 0.85, "ebitda_margin": 0.32, "maintenance_cost_per_unit": 0.06, "hardware_cost_per_unit": 0.10, "beta": 1.2},
)


BIOTECH_TEMPLATE = _generic_template(
    "Biotech / Pharma",
    "Commercial revenue, pipeline probability, TAM, peak sales, launch timing, patent life, R&D burn, cash runway, dilution, and probability-adjusted NPV.",
    {
        "Commercial / Pipeline": ["capacity_added", "utilization", "revenue_per_unit"],
        "Probability / Margins": ["ebitda_margin"],
        "R&D / Cash Runway": ["maintenance_cost_per_unit", "equity_funding_pct"],
        "Dilution": ["equity_issue_price", "sbc_dilution_pct"],
        "Valuation": ["risk_free_rate", "beta", "equity_risk_premium", "exit_ebitda_multiple", "exit_ebit_multiple"],
    },
    {
        "capacity_added": {"label": "Pipeline / Product Growth", "impact": "Launches and product uptake drive revenue.", "range": "Probability-weighted; verify stage.", "source": "Pipeline, approvals, launch timing."},
        "utilization": {"label": "Probability of Success", "impact": "Probability weighting changes revenue and NPV.", "range": "0% to 100% by clinical stage.", "source": "Clinical stage and regulatory status."},
        "revenue_per_unit": {"label": "Peak Sales / Product Revenue", "impact": "Peak sales convert pipeline success into revenue.", "range": "Compare TAM and penetration.", "source": "TAM, pricing, patient population."},
        "ebitda_margin": {"label": "Commercial Margin / R&D Burden", "impact": "R&D and SG&A ramp affect NOPAT and cash runway.", "range": "-100% to 45%.", "source": "R&D burn and commercial margin."},
        "equity_funding_pct": {"label": "External Funding Need", "impact": "Cash burn may require equity issuance and dilution.", "range": "0% to 100% of funding gap.", "source": "Cash runway and financing plan."},
    },
    defaults={"utilization": 0.35, "ebitda_margin": -0.25, "maintenance_cost_per_unit": 0.05, "equity_funding_pct": 0.20, "beta": 1.4},
)


REIT_TEMPLATE = _generic_template(
    "Real Estate / REIT",
    "Occupancy, rent per unit, same-store NOI growth, cap rates, leverage, maintenance CAPEX, AFFO, dividend payout, and NAV.",
    {
        "Properties / Occupancy": ["capacity_added", "utilization"],
        "Rent / NOI": ["revenue_per_unit", "ebitda_margin"],
        "Leverage / CAPEX": ["maintenance_cost_per_unit", "cost_of_debt"],
        "Dividends / Valuation": ["risk_free_rate", "beta", "equity_risk_premium", "exit_ebitda_multiple", "exit_ebit_multiple"],
    },
    {
        "capacity_added": {"label": "Property / Sq Ft Growth", "impact": "Property growth and development drive rent base.", "range": "-5% to 15%.", "source": "Properties, sq ft, acquisitions/developments."},
        "utilization": {"label": "Occupancy", "impact": "Occupancy converts capacity into rental revenue and NOI.", "range": "70% to 100%.", "source": "Occupancy disclosure."},
        "revenue_per_unit": {"label": "Rent per Unit", "impact": "Rent converts occupied units into revenue.", "range": "Compare lease rates and same-store NOI.", "source": "Rent, lease rates, NOI."},
        "ebitda_margin": {"label": "NOI / AFFO Margin", "impact": "NOI/AFFO margin drives NAV and dividend capacity.", "range": "30% to 75%.", "source": "NOI, AFFO, operating expenses."},
        "maintenance_cost_per_unit": {"label": "Maintenance CAPEX", "impact": "Sustaining property investment reduces AFFO/FCF.", "range": "0% to 10% revenue proxy.", "source": "Maintenance CAPEX and recurring capex."},
    },
    defaults={"utilization": 0.92, "ebitda_margin": 0.55, "maintenance_cost_per_unit": 0.04, "cost_of_debt": 0.055, "beta": 0.9},
)


ADTECH_TEMPLATE = _generic_template(
    "Advertising / Media / Ad-Tech",
    "Traffic/supply, fill rate, CPM/CPC/CPA pricing, take rate, advertiser spend, publisher supply, retention, gross margin, OPEX, SBC, OCF, and working capital.",
    {
        "Traffic / Supply": ["capacity_added", "utilization"],
        "Demand / Pricing": ["revenue_per_unit"],
        "Margins": ["ebitda_margin", "maintenance_cost_per_unit"],
        "Cash Conversion": ["customer_prepayment_pct"],
        "Dilution / Valuation": ["sbc_dilution_pct", "risk_free_rate", "beta", "equity_risk_premium", "exit_ebitda_multiple", "exit_ebit_multiple"],
    },
    {
        "capacity_added": {"label": "Traffic / Impression Growth", "impact": "Traffic and supply create monetizable ad inventory.", "range": "-15% to 35%.", "source": "Impressions, MAU/DAU, publisher supply."},
        "utilization": {"label": "Fill Rate / Retention", "impact": "Fill rate and retention convert inventory into paid impressions.", "range": "50% to 100%.", "source": "Fill rate, advertiser retention, supply quality."},
        "revenue_per_unit": {"label": "CPM / Take Rate", "impact": "Pricing and take rate convert impressions/spend into revenue.", "range": "Compare CPM/CPC/take-rate history.", "source": "CPM/CPC/CPA, take rate, ad spend."},
        "ebitda_margin": {"label": "Platform Margin", "impact": "Supply costs and OPEX efficiency drive EBIT and FCF.", "range": "-10% to 40%.", "source": "Traffic acquisition cost, gross margin, OPEX."},
        "sbc_dilution_pct": {"label": "SBC Dilution", "impact": "SBC affects per-share valuation.", "range": "-5% to 8%.", "source": "SBC and diluted share count."},
    },
    defaults={"utilization": 0.80, "ebitda_margin": 0.18, "maintenance_cost_per_unit": 0.02, "sbc_dilution_pct": 0.02},
)


GENERAL_TEMPLATE = _generic_template(
    "General",
    "Fallback business-driver model: revenue growth, margin leverage, cash conversion, reinvestment, balance sheet, dilution, WACC, and terminal value.",
    {
        "Growth": ["capacity_added", "revenue_per_unit", "utilization"],
        "Margins": ["ebitda_margin"],
        "Cash Conversion": ["customer_prepayment_pct"],
        "Reinvestment": ["maintenance_cost_per_unit", "hardware_cost_per_unit"],
        "Balance Sheet / Dilution": ["cost_of_debt", "sbc_dilution_pct", "share_repurchases"],
        "Valuation": ["risk_free_rate", "beta", "equity_risk_premium", "exit_ebitda_multiple", "exit_ebit_multiple"],
    },
    {
        "capacity_added": {"label": "Revenue Growth Driver", "impact": "Demand/volume driver feeds revenue growth.", "range": "-10% to 25%.", "source": "Historical growth, guidance, peers, analyst estimate."},
        "revenue_per_unit": {"label": "Price / Revenue per Unit", "impact": "Pricing and mix convert demand into revenue.", "range": "Compare to historical revenue per unit or revenue base.", "source": "Revenue history and business description."},
        "utilization": {"label": "Volume / Retention Quality", "impact": "Retention, utilization, or volume quality affects revenue durability.", "range": "60% to 120% proxy.", "source": "Filings, guidance, peer benchmarks."},
        "ebitda_margin": {"label": "Operating Margin", "impact": "Margin converts revenue into EBIT, NOPAT, OCF, and FCF.", "range": "-20% to 45%.", "source": "Historical margin and peer range."},
        "maintenance_cost_per_unit": {"label": "Maintenance CAPEX % Revenue", "impact": "Maintenance reinvestment reduces FCF.", "range": "0% to 10%.", "source": "CAPEX and depreciation trend."},
        "hardware_cost_per_unit": {"label": "Growth CAPEX % Revenue", "impact": "Growth reinvestment reduces near-term FCF.", "range": "0% to 20%.", "source": "CAPEX guidance and asset intensity."},
        "sbc_dilution_pct": {"label": "Diluted Share Growth", "impact": "Share count changes affect fair value per share.", "range": "-5% to 8%.", "source": "Diluted shares, SBC, repurchases."},
    },
)


DRIVER_TEMPLATES = {
    "AI Infrastructure / Data Center": {
        "profile": "AI Infrastructure / Data Center",
        "description": "Capacity, chip generation mix, utilization, revenue per GW, build cost per GW, prepayments, funding, and dilution drive valuation.",
        "driver_groups": {
            "Capacity": ["blackwell_gw_deployed", "rubin_gw_deployed", "other_gw_deployed"],
            "Unit Economics": ["revenue_per_blackwell_gw", "revenue_per_rubin_gw", "revenue_per_other_gw"],
            "Utilization": ["utilization"],
            "Margins": ["adjusted_ebitda_margin", "tax_rate"],
            "Reinvestment / CAPEX": [
                "hardware_cost_per_blackwell_gw",
                "hardware_cost_per_rubin_gw",
                "hardware_cost_per_other_gw",
                "land_power_cooling_cost_per_blackwell_gw",
                "land_power_cooling_cost_per_rubin_gw",
                "land_power_cooling_cost_per_other_gw",
                "maintenance_capex_pct_revenue",
            ],
            "Depreciation": ["gpu_useful_life", "infrastructure_useful_life"],
            "Funding": ["customer_prepayment_pct", "equity_funding_pct", "cost_of_debt"],
            "Dilution": ["equity_issue_price", "sbc_dilution_pct", "share_repurchases"],
            "Valuation": ["risk_free_rate", "beta", "equity_risk_premium", "exit_ebitda_multiple", "exit_ebit_multiple", "earnings_multiple"],
        },
        "derived_financial_lines": [
            "Total energized GW",
            "Average total GW",
            "Revenue",
            "Adjusted EBITDA",
            "Hardware depreciation",
            "EBIT",
            "NOPAT",
            "Operating cash flow",
            "Growth CAPEX",
            "Total CAPEX",
            "Free cash flow",
            "Ending net debt",
            "Shares outstanding",
        ],
        "default_driver_assumptions": {
            "blackwell_gw_deployed": 0.0,
            "rubin_gw_deployed": 0.0,
            "other_gw_deployed": 0.2,
            "utilization": 0.70,
            "revenue_per_blackwell_gw": 1_500_000_000.0,
            "revenue_per_rubin_gw": 1_800_000_000.0,
            "revenue_per_other_gw": 1_000_000_000.0,
            "adjusted_ebitda_margin": 0.35,
            "hardware_cost_per_blackwell_gw": 1_000_000_000.0,
            "hardware_cost_per_rubin_gw": 1_150_000_000.0,
            "hardware_cost_per_other_gw": 800_000_000.0,
            "land_power_cooling_cost_per_blackwell_gw": 650_000_000.0,
            "land_power_cooling_cost_per_rubin_gw": 700_000_000.0,
            "land_power_cooling_cost_per_other_gw": 450_000_000.0,
            "gpu_useful_life": 5.0,
            "infrastructure_useful_life": 15.0,
            "maintenance_capex_pct_revenue": 0.03,
            "customer_prepayment_pct": 0.10,
            "equity_funding_pct": 0.10,
            "sbc_dilution_pct": 0.02,
            "cost_of_debt": 0.08,
            "risk_free_rate": 0.04,
            "beta": 1.4,
            "equity_risk_premium": 0.055,
            "tax_rate": 0.21,
            "exit_ebitda_multiple": 12.0,
            "exit_ebit_multiple": 16.0,
            "earnings_multiple": 25.0,
        },
        "scenario_rules": {
            "Bear Case": {"capacity": 0.75, "unit_economics": 0.85, "margin_delta": -0.05, "capex": 1.15},
            "Base Case": {"capacity": 1.00, "unit_economics": 1.00, "margin_delta": 0.00, "capex": 1.00},
            "Bull Case": {"capacity": 1.25, "unit_economics": 1.15, "margin_delta": 0.05, "capex": 0.90},
        },
        "default_driver_rows": AI_INFRA_DRIVER_ROWS,
        "manual_review_questions": [
            "Verify energized GW by chip generation and timing.",
            "Validate revenue per GW against contracts and utilization.",
            "Separate growth CAPEX, maintenance CAPEX, customer prepayments, debt funding, and equity dilution.",
            "Review hardware useful life and depreciation burden.",
        ],
    },
    "SaaS / Software": SOFTWARE_TEMPLATE,
    "Semiconductor": SEMICONDUCTOR_TEMPLATE,
    "Marketplace / Platform": MARKETPLACE_TEMPLATE,
    "Consumer Brand / Retail": CONSUMER_TEMPLATE,
    "Financial / Fintech": FINANCIAL_TEMPLATE,
    "Industrial / Hardware": INDUSTRIAL_TEMPLATE,
    "Energy / Commodity": ENERGY_TEMPLATE,
    "Biotech / Pharma": BIOTECH_TEMPLATE,
    "Real Estate / REIT": REIT_TEMPLATE,
    "Advertising / Media / Ad-Tech": ADTECH_TEMPLATE,
    "General": GENERAL_TEMPLATE,
}

PROFILE_ALIASES = {
    "Consumer Brand": "Consumer Brand / Retail",
    "Retail / Store": "Consumer Brand / Retail",
    "Subscription / SaaS": "SaaS / Software",
    "Marketplace / Transaction": "Marketplace / Platform",
    "Standard Financial": "General",
}


def driver_profile_options() -> list[str]:
    return list(SUPPORTED_DRIVER_PROFILES)


def get_driver_template(profile: str) -> dict:
    profile = PROFILE_ALIASES.get(profile, profile)
    return DRIVER_TEMPLATES.get(profile, DRIVER_TEMPLATES["General"])
