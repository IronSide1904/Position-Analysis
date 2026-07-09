from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from persistence.schema import ANALYSIS_SCHEMA_VERSION, DASHBOARD_VERSION, migrate_analysis_payload, validate_analysis_payload


SECRET_KEYS = {"FINVIZ_AUTH_TOKEN", "SEC_USER_AGENT", "api_key", "apikey", "token", "secret", "password", "authorization"}
INDEX_FILE = "index.json"


def get_storage_root() -> Path:
    root = os.environ.get("PA11R_ANALYSIS_STORE_ROOT")
    path = Path(root) if root else Path(__file__).resolve().parents[1] / "saved_analyses"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "analysis"


def generate_analysis_id(ticker: str, analysis_name: str | None = None) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    slug = sanitize_filename(analysis_name or "user_case").lower()
    return f"{sanitize_filename(ticker).upper()}_{now}_{slug}"


def make_jsonable(value: Any):
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return make_jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def scrub_secrets(payload: dict) -> dict:
    def scrub(value):
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                key_text = str(key)
                if key_text in SECRET_KEYS or any(secret.lower() in key_text.lower() for secret in SECRET_KEYS):
                    continue
                out[key] = scrub(item)
            return out
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return scrub(make_jsonable(payload))


def _index_path() -> Path:
    return get_storage_root() / INDEX_FILE


def _read_index() -> list[dict]:
    path = _index_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _write_index(items: list[dict]) -> None:
    _index_path().write_text(json.dumps(items, indent=2, sort_keys=True), encoding="utf-8")


def _payload_path(ticker: str, analysis_id: str) -> Path:
    ticker_dir = get_storage_root() / sanitize_filename(ticker).upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)
    return ticker_dir / f"{sanitize_filename(analysis_id)}.json"


def _find_analysis_path(path_or_id: str) -> Path:
    candidate = Path(path_or_id)
    if candidate.exists():
        return candidate
    for item in _read_index():
        if item.get("analysis_id") == path_or_id:
            path = Path(item.get("path", ""))
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[1] / path
            return path
    for path in get_storage_root().glob(f"*/{sanitize_filename(path_or_id)}.json"):
        return path
    raise FileNotFoundError(f"Saved analysis not found: {path_or_id}")


def _index_item(payload: dict, path: Path) -> dict:
    dcf = payload.get("dcf", {})
    user_output = dcf.get("scenario_outputs", {}).get("User Case", {})
    decision = payload.get("decision", {})
    market = payload.get("market_snapshot_at_save", {})
    root = Path(__file__).resolve().parents[1]
    try:
        rel_path = path.resolve().relative_to(root)
    except ValueError:
        rel_path = path.resolve()
    return {
        "analysis_id": payload.get("analysis_id"),
        "ticker": payload.get("ticker"),
        "company_name": payload.get("company_name"),
        "analysis_name": payload.get("analysis_name"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "final_rating": decision.get("final_rating"),
        "investment_view": decision.get("investment_view"),
        "swing_view": decision.get("swing_view"),
        "price_at_save": market.get("price"),
        "fair_value_user_case": user_output.get("fair_value_per_share"),
        "tags": payload.get("tags", []),
        "path": str(rel_path).replace("\\", "/"),
    }


def _update_index(payload: dict, path: Path) -> None:
    item = _index_item(payload, path)
    items = [existing for existing in _read_index() if existing.get("analysis_id") != item.get("analysis_id")]
    items.append(item)
    items.sort(key=lambda row: (row.get("ticker") or "", row.get("updated_at") or ""), reverse=True)
    _write_index(items)


def build_analysis_payload(
    ticker: str,
    dataset: dict,
    dashboard_state: dict,
    analysis_name: str,
    description: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    now = dt.datetime.now().replace(microsecond=0).isoformat()
    market = dataset.get("market_data", {}) if isinstance(dataset, dict) else {}
    payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "dashboard": "PA-11R Hybrid",
        "dashboard_version": DASHBOARD_VERSION,
        "analysis_id": dashboard_state.get("analysis_id") or generate_analysis_id(ticker, analysis_name),
        "ticker": str(ticker or dataset.get("ticker") or "").upper(),
        "company_name": dataset.get("company"),
        "created_at": dashboard_state.get("created_at") or now,
        "updated_at": now,
        "analysis_name": analysis_name,
        "description": description or "",
        "tags": tags or [],
        "user_profile": dashboard_state.get("user_profile", {"mode": "Investor", "time_horizon": "Long-term", "risk_level": "Medium"}),
        "decision": dashboard_state.get("decision", {}),
        "market_snapshot_at_save": {
            "price": market.get("price"),
            "market_cap": market.get("market_cap"),
            "enterprise_value": market.get("enterprise_value"),
            "shares_outstanding": market.get("shares_outstanding"),
            "net_debt": dashboard_state.get("dcf", {}).get("scenario_assumptions", {}).get("User Case", {}).get("net_debt"),
            "date": now,
        },
        "data_sources": dashboard_state.get("data_sources", {}),
        "dcf": dashboard_state.get("dcf", {}),
        "sotp": dashboard_state.get("sotp", {}),
        "multiples": dashboard_state.get("multiples", {}),
        "evidence": dashboard_state.get("evidence", {}),
        "business_quality": dashboard_state.get("business_quality", {}),
        "management_capital_allocation": dashboard_state.get("management_capital_allocation", {}),
        "user_notes": dashboard_state.get("user_notes", {}),
    }
    return scrub_secrets(migrate_analysis_payload(payload))


def save_analysis(payload: dict, overwrite: bool = False) -> dict:
    payload = scrub_secrets(migrate_analysis_payload(payload))
    valid, errors = validate_analysis_payload(payload)
    if not valid:
        return {"success": False, "path": None, "analysis_id": payload.get("analysis_id"), "message": "; ".join(errors)}
    path = _payload_path(payload["ticker"], payload["analysis_id"])
    if path.exists() and not overwrite:
        payload["analysis_id"] = f"{payload['analysis_id']}_{uuid.uuid4().hex[:6]}"
        payload["created_at"] = dt.datetime.now().replace(microsecond=0).isoformat()
        path = _payload_path(payload["ticker"], payload["analysis_id"])
    payload["updated_at"] = dt.datetime.now().replace(microsecond=0).isoformat()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _update_index(payload, path)
    return {"success": True, "path": str(path), "analysis_id": payload["analysis_id"], "message": "Analysis saved."}


def load_analysis(path_or_id: str) -> dict:
    path = _find_analysis_path(path_or_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = migrate_analysis_payload(payload)
    valid, errors = validate_analysis_payload(payload)
    if not valid:
        raise ValueError("; ".join(errors))
    return payload


def list_saved_analyses(ticker: str | None = None) -> list[dict]:
    items = _read_index()
    if not items:
        items = rebuild_index()
    if ticker:
        symbol = ticker.upper().strip()
        items = [item for item in items if str(item.get("ticker", "")).upper() == symbol]
    return items


def update_analysis(analysis_id: str, payload: dict) -> dict:
    existing = load_analysis(analysis_id)
    payload = migrate_analysis_payload({**existing, **payload, "analysis_id": analysis_id, "created_at": existing.get("created_at")})
    return save_analysis(payload, overwrite=True)


def duplicate_analysis(analysis_id: str, new_name: str) -> dict:
    payload = load_analysis(analysis_id)
    payload["analysis_name"] = new_name
    payload["analysis_id"] = generate_analysis_id(payload.get("ticker"), new_name)
    now = dt.datetime.now().replace(microsecond=0).isoformat()
    payload["created_at"] = now
    payload["updated_at"] = now
    return save_analysis(payload, overwrite=False)


def delete_analysis(analysis_id: str) -> dict:
    path = _find_analysis_path(analysis_id)
    if path.exists():
        try:
            path.unlink()
        except PermissionError:
            tombstone = {"deleted": True, "analysis_id": analysis_id, "deleted_at": dt.datetime.now().replace(microsecond=0).isoformat()}
            path.write_text(json.dumps(tombstone, indent=2), encoding="utf-8")
    _write_index([item for item in _read_index() if item.get("analysis_id") != analysis_id])
    return {"success": True, "analysis_id": analysis_id, "message": "Analysis deleted."}


def export_analysis_json(analysis_id: str) -> bytes:
    payload = load_analysis(analysis_id)
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def import_analysis_json(uploaded_file) -> dict:
    raw = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    payload = json.loads(raw)
    payload = migrate_analysis_payload(payload)
    valid, errors = validate_analysis_payload(payload)
    if not valid:
        return {"success": False, "message": "Import failed: invalid PA-11R analysis file. " + "; ".join(errors)}
    return save_analysis(payload, overwrite=False)


def rebuild_index() -> list[dict]:
    items = []
    for path in get_storage_root().glob("*/*.json"):
        if path.name == INDEX_FILE:
            continue
        try:
            payload = migrate_analysis_payload(json.loads(path.read_text(encoding="utf-8")))
            valid, _errors = validate_analysis_payload(payload)
            if valid:
                items.append(_index_item(payload, path))
        except Exception:
            continue
    _write_index(items)
    return items


def compute_state_hash(payload_or_state: dict) -> str:
    focus = {
        "decision": payload_or_state.get("decision"),
        "dcf": payload_or_state.get("dcf"),
        "sotp": payload_or_state.get("sotp"),
        "multiples": payload_or_state.get("multiples"),
        "evidence": payload_or_state.get("evidence"),
        "business_quality": payload_or_state.get("business_quality"),
        "management_capital_allocation": payload_or_state.get("management_capital_allocation"),
        "user_notes": payload_or_state.get("user_notes"),
    }
    raw = json.dumps(scrub_secrets(focus), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compare_analyses(payload_a: dict, payload_b: dict) -> dict:
    payload_a = migrate_analysis_payload(payload_a)
    payload_b = migrate_analysis_payload(payload_b)
    fields = [
        ("Investment View", ("decision", "investment_view")),
        ("Swing View", ("decision", "swing_view")),
        ("Final Rating", ("decision", "final_rating")),
        ("User Case Fair Value", ("dcf", "scenario_outputs", "User Case", "fair_value_per_share")),
        ("Revenue CAGR", ("dcf", "scenario_assumptions", "User Case", "revenue_cagr")),
        ("OPEX % Revenue", ("dcf", "scenario_assumptions", "User Case", "opex_pct_revenue")),
        ("OCF Margin", ("dcf", "scenario_assumptions", "User Case", "ocf_margin")),
        ("Maintenance CAPEX %", ("dcf", "scenario_assumptions", "User Case", "maintenance_capex_pct_revenue")),
        ("Growth CAPEX %", ("dcf", "scenario_assumptions", "User Case", "growth_capex_pct_revenue")),
        ("WACC", ("dcf", "scenario_assumptions", "User Case", "wacc")),
        ("Terminal Multiple", ("dcf", "scenario_assumptions", "User Case", "terminal_multiple")),
        ("SOTP Fair Value", ("sotp", "scenario_outputs", "Base Case", "fair_value_per_share")),
        ("Moat View", ("business_quality", "moat", "classification")),
        ("Risk View", ("business_quality", "risks", "classification")),
        ("Final Notes", ("user_notes", "thesis")),
    ]

    def dig(payload, path):
        value = payload
        for part in path:
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    rows = []
    for label, path in fields:
        a = dig(payload_a, path)
        b = dig(payload_b, path)
        delta = None
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = b - a
        elif a != b:
            delta = "Changed"
        rows.append({"Field": label, "Version A": a, "Version B": b, "Delta": delta})
    return {"differences": rows, "changed_count": sum(1 for row in rows if row["Delta"] not in (None, 0, ""))}
