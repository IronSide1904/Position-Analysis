import json
import uuid
from pathlib import Path

from persistence.analysis_store import (
    build_analysis_payload,
    compare_analyses,
    delete_analysis,
    duplicate_analysis,
    export_analysis_json,
    import_analysis_json,
    list_saved_analyses,
    load_analysis,
    save_analysis,
)


def _dataset():
    return {
        "ticker": "ABC",
        "company": "ABC Corp",
        "sources": ["SEC/EDGAR", "Finviz Elite", "yfinance"],
        "market_data": {"price": 12.34, "market_cap": 1000.0, "enterprise_value": 1100.0, "shares_outstanding": 100.0},
        "finviz": {"available": True},
        "yfinance": {"available": True},
        "cik": "0000000000",
    }


def _state():
    return {
        "decision": {"investment_view": "Watchlist", "final_rating": "Watchlist", "swing_view": "Neutral"},
        "dcf": {
            "scenario_assumptions": {"User Case": {"revenue_cagr": 0.08, "ocf_margin": 0.2}},
            "scenario_outputs": {"User Case": {"fair_value_per_share": 18.5}},
            "assumption_update_log": [{"model_line": "revenue_cagr", "new_value": 0.08}],
        },
        "sotp": {"manual_segments": [{"Segment": "Core", "Revenue": 1000.0}], "scenario_outputs": {"Base Case": {"fair_value_per_share": 17.0}}},
        "multiples": {"selected_multiple_basis": "Normalized Year", "peer_medians": {"EV/OCF": 15.0}},
        "evidence": {"manual_review_items": [{"Data Needed": "Debt detail"}]},
        "user_notes": {"general": "Initial review"},
        "FINVIZ_AUTH_TOKEN": "do-not-save",
    }


def _store_root():
    path = Path(".cache") / f"analysis-store-tests-{uuid.uuid4().hex[:8]}"
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_save_load_index_and_secret_scrub(monkeypatch):
    monkeypatch.setenv("PA11R_ANALYSIS_STORE_ROOT", str(_store_root()))
    payload = build_analysis_payload("ABC", _dataset(), _state(), "ABC Base", "desc", ["watchlist"])
    result = save_analysis(payload)

    assert result["success"]
    saved = load_analysis(result["analysis_id"])
    assert saved["ticker"] == "ABC"
    assert saved["dcf"]["scenario_assumptions"]["User Case"]["revenue_cagr"] == 0.08
    assert "FINVIZ_AUTH_TOKEN" not in json.dumps(saved)
    index = list_saved_analyses("ABC")
    assert index[0]["analysis_id"] == result["analysis_id"]
    assert index[0]["fair_value_user_case"] == 18.5


def test_duplicate_delete_export_import_and_compare(monkeypatch):
    monkeypatch.setenv("PA11R_ANALYSIS_STORE_ROOT", str(_store_root()))
    payload = build_analysis_payload("ABC", _dataset(), _state(), "ABC Base")
    result = save_analysis(payload)
    duplicate = duplicate_analysis(result["analysis_id"], "ABC Copy")

    assert duplicate["success"]
    exported = export_analysis_json(result["analysis_id"])
    imported = import_analysis_json(exported)
    assert imported["success"]
    diff = compare_analyses(load_analysis(result["analysis_id"]), load_analysis(duplicate["analysis_id"]))
    assert "differences" in diff
    deleted = delete_analysis(duplicate["analysis_id"])
    assert deleted["success"]
