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
