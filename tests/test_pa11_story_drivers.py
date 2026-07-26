import pandas as pd

from analysis.company_story import build_pa11_story
from models.business_drivers import infer_business_driver_profile
from models.driver_templates import get_driver_template


def test_pa11_story_uses_ai_infrastructure_economic_engine():
    dataset = {
        "ticker": "AIDC",
        "company": "AI Data Center Co",
        "sector": "Technology",
        "industry": "Cloud Infrastructure",
        "company_description": "AI Data Center Co provides GPU cloud infrastructure, data center capacity, Blackwell GPU clusters, and customer compute contracts.",
    }
    profile = infer_business_driver_profile(dataset)
    story = build_pa11_story(dataset, business_driver_profile=profile, driver_template=get_driver_template(profile["profile"]))
    text = " ".join(
        [
            story["economic_engine_summary"],
            " ".join(story["core_capex_drivers"]),
            " ".join(story["core_dilution_or_balance_sheet_drivers"]),
            " ".join(row["driver"] for row in story["driver_to_assumption_map"]),
        ]
    ).lower()

    assert story["business_model_type"] == "AI Infrastructure / Data Center"
    assert "capacity" in text
    assert "gpu" in text or "chip" in text
    assert "power" in text or "energy" in text
    assert "debt" in text
    assert "equity" in text or "dilution" in text
    assert any("Growth CAPEX" in row["affected_assumptions"] for row in story["driver_to_assumption_map"])


def test_pa11_story_uses_retail_membership_driver_language():
    dataset = {
        "ticker": "RETL",
        "company": "Membership Retail Co",
        "sector": "Consumer Defensive",
        "industry": "Membership Retail",
        "company_description": "Membership Retail Co operates warehouse stores with membership renewals, traffic, basket size, private label products, supplier scale, and inventory turns.",
    }
    profile = {"profile": "Consumer Brand / Retail", "confidence": "High", "reason": "Test profile."}
    story = build_pa11_story(dataset, business_driver_profile=profile, driver_template=get_driver_template(profile["profile"]))
    text = " ".join(
        [
            story["economic_engine_summary"],
            " ".join(story["core_revenue_drivers"]),
            " ".join(story["core_ocf_drivers"]),
            " ".join(story["core_terminal_value_drivers"]),
        ]
    ).lower()

    assert story["business_model_type"] == "Retail / Membership Retail"
    assert "membership" in text
    assert "renewal" in text
    assert "inventory" in text
    assert "supplier" in text
    assert "terminal" in " ".join(row["suggested_action"] for row in story["driver_to_assumption_map"]).lower()


def test_pa11_story_sector_peer_context_and_no_buzz_hallucination():
    dataset = {
        "ticker": "SOFT",
        "company": "Software Co",
        "sector": "Technology",
        "industry": "Software",
        "company_description": "Software Co sells subscription workflow software to enterprise customers.",
    }
    peer_df = pd.DataFrame([{"ticker": "PEER1", "EV/Revenue": 8.0}, {"ticker": "PEER2", "EV/Revenue": 6.0}])
    profile = {"profile": "SaaS / Software", "confidence": "High", "reason": "Test profile."}
    story = build_pa11_story(dataset, peer_data=peer_df, business_driver_profile=profile, driver_template=get_driver_template(profile["profile"]))

    assert story["sector_theme_peer_context"]["peer_positioning"].startswith("Peer set loaded with 2 rows")
    assert story["social_buzz_context"] == "Social/news buzz unavailable."
    assert story["latest_updates"] == "Dashboard has not fetched recent news/social data yet."
    assert story["driver_reflection_map"]
    assert all(row["DCF line affected"] for row in story["driver_reflection_map"])


def test_key_driver_discovery_uses_ticker_specific_product_lines_for_apple():
    from ui.dashboard_v2 import _product_service_lines_from_context

    ctx = {
        "dataset": {
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "company_description": "Apple sells iPhone, Mac, iPad, wearables and services including App Store, iCloud, AppleCare and Apple Pay.",
        },
        "pa11_story": {},
    }

    rows = _product_service_lines_from_context(ctx, "Consumer Brand / Retail")
    text = " ".join(f"{row['product_line']} {row['specific_driver']}" for row in rows).lower()

    assert "iphone" in text
    assert "services" in text or "app store" in text
    assert "buybacks" in text or "diluted shares" in text
    assert all(row["model_impact"] for row in rows)
    assert all(row["source_basis"] for row in rows)


def test_key_driver_discovery_uses_profile_product_lines_for_unknown_software():
    from ui.dashboard_v2 import _product_service_lines_from_context

    ctx = {
        "dataset": {
            "ticker": "SOFT",
            "company": "Software Co",
            "sector": "Technology",
            "industry": "Software",
            "company_description": "Software Co sells subscription workflow software with recurring revenue, seats, retention, and cloud hosting costs.",
        },
        "pa11_story": {},
    }

    rows = _product_service_lines_from_context(ctx, "SaaS / Software")
    text = " ".join(f"{row['driver_group']} {row['product_line']} {row['specific_driver']}" for row in rows).lower()

    assert "subscription" in text
    assert "retention" in text
    assert "sbc" in text or "buybacks" in text
    assert "revenue growth %" not in text


def test_aapl_key_driver_default_rows_are_product_service_focused():
    from ui.dashboard_v2 import _build_profile_key_driver_table

    assumptions = {"revenue_cagr": 0.08, "gross_margin": 0.46, "diluted_share_growth": -0.02, "forecast_years": 5}
    specs = [(1, "FY2026E"), (2, "FY2027F"), (3, "FY2028F"), (4, "FY2029F"), (5, "FY2030F")]
    original_matrix = pd.DataFrame(
        [
            {"Row Key": "revenue_cagr", "Assumption": "Revenue Growth %", "Row Type": "Input", "Evidence": "Scenario-based", "Confidence": "Medium"},
            {"Row Key": "revenue_amount", "Assumption": "Revenue ($)", "Row Type": "Calculated", "Evidence": "Calculated", "Confidence": "High"},
        ]
    )
    ctx = {
        "dataset": {
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "company_description": "Apple sells iPhone, Mac, iPad, Wearables and Services.",
            "market_data": {},
        },
        "historicals": pd.DataFrame([{"Period": "FY2025A", "Revenue": 416_000_000_000}]),
        "pa11_story": {},
    }

    table = _build_profile_key_driver_table(ctx, assumptions, pd.DataFrame(), {}, original_matrix, specs, ["FY2025A"], "Consumer Brand / Retail", False)
    text = " ".join(table["Product / Service Line"].astype(str).tolist() + table["Specific Driver"].astype(str).tolist()).lower()

    assert "iphone" in text
    assert "mac" in text
    assert "ipad" in text
    assert "wearables" in text
    assert "services" in text
    assert "buybacks" in text or "diluted shares" in text
    assert "store growth" not in text
    assert "same-store" not in text
    assert "backlog" not in text
    assert len(table) <= 12


def test_product_driver_rollup_sets_consolidated_revenue_growth():
    from ui.dashboard_v2 import _apply_product_driver_rollup

    specs = [(1, "FY2026E")]
    assumptions = {"revenue_cagr": 0.08, "gross_margin": 0.45, "forecast_years": 1}
    ctx = {
        "dataset": {"ticker": "AAPL", "market_data": {}},
        "historicals": pd.DataFrame([{"Period": "FY2025A", "Revenue": 100.0}]),
    }
    table = pd.DataFrame(
        [
            {"Row Key": "pline:iphone:revenue_growth", "Product / Service Line": "iPhone", "Specific Driver": "Revenue Growth %", "Row Type": "Input", "FY2026E": "20.0%"},
            {"Row Key": "pline:mac:revenue_growth", "Product / Service Line": "Mac", "Specific Driver": "Revenue Growth %", "Row Type": "Input", "FY2026E": "0.0%"},
            {"Row Key": "pline:ipad:revenue_growth", "Product / Service Line": "iPad", "Specific Driver": "Revenue Growth %", "Row Type": "Input", "FY2026E": "0.0%"},
            {"Row Key": "pline:wearables:revenue_growth", "Product / Service Line": "Wearables", "Specific Driver": "Revenue Growth %", "Row Type": "Input", "FY2026E": "0.0%"},
            {"Row Key": "pline:services:revenue_growth", "Product / Service Line": "Services", "Specific Driver": "Revenue Growth %", "Row Type": "Input", "FY2026E": "0.0%"},
        ]
    )

    updated = _apply_product_driver_rollup(ctx, "Consumer Brand / Retail", table, assumptions, specs)

    assert round(updated["forecast_assumptions_by_year"]["1"]["revenue_cagr"], 4) == 0.10


def test_aapl_pa11_story_is_product_specific_and_analytical():
    dataset = {
        "ticker": "AAPL",
        "company": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "company_description": "Apple sells smartphones, personal computers, tablets, wearables, accessories, and services.",
    }
    story = build_pa11_story(dataset)
    combined = " ".join(
        [
            story["economic_engine_summary"],
            story["growth_driver_story"],
            " ".join(row["driver"] for row in story["driver_to_assumption_map"]),
            " ".join(section["Read"] for section in story["detailed_story_sections"]),
        ]
    ).lower()

    assert story["business_model_type"] == "Premium Consumer Technology Ecosystem"
    assert "iphone" in combined
    assert "services" in combined
    assert "installed base" in combined
    assert "buybacks" in combined
    assert "terminal multiple" in combined
    assert story["story_to_driver_mapping"]
