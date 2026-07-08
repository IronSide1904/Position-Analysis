from __future__ import annotations

import io
import os
import re
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from config import FINVIZ_COLUMNS, FINVIZ_CONFIG, FINVIZ_DECISION_COLUMNS_STRING, FINVIZ_DISCOVERY_COLUMNS_STRING
from data_sources.runtime_env import prepare_external_data_env


SNAPSHOT_KEYS = [
    "no",
    "ticker",
    "company",
    "sector",
    "industry",
    "country",
    "market_cap",
    "pe",
    "forward_pe",
    "peg",
    "ps",
    "pb",
    "pc",
    "pfcf",
    "shares_outstanding",
    "shares_float",
    "float_outstanding_pct",
    "short_float",
    "short_ratio",
    "roa",
    "roe",
    "roi",
    "current_ratio",
    "quick_ratio",
    "lt_debt_to_equity",
    "debt_to_equity",
    "gross_margin",
    "operating_margin",
    "profit_margin",
    "beta",
    "atr",
    "volatility_week",
    "volatility_month",
    "sma20",
    "sma50",
    "sma200",
    "high_52w",
    "low_52w",
    "rsi",
    "gap",
    "relative_volume",
    "average_volume",
    "volume",
    "price",
    "change",
    "earnings_date",
]

HEADER_ALIASES = {
    "no": "no",
    "ticker": "ticker",
    "symbol": "ticker",
    "company": "company",
    "company_name": "company",
    "sector": "sector",
    "industry": "industry",
    "country": "country",
    "market_cap": "market_cap",
    "marketcap": "market_cap",
    "p_e": "pe",
    "forward_p_e": "forward_pe",
    "peg": "peg",
    "p_s": "ps",
    "p_b": "pb",
    "p_c": "pc",
    "p_cash": "pc",
    "p_fcf": "pfcf",
    "p_free_cash_flow": "pfcf",
    "shares_outstanding": "shares_outstanding",
    "shs_outstand": "shares_outstanding",
    "shs_outstanding": "shares_outstanding",
    "shares_out": "shares_outstanding",
    "shares_float": "shares_float",
    "shs_float": "shares_float",
    "float": "shares_float",
    "float_outstanding": "float_outstanding_pct",
    "float_pct": "float_outstanding_pct",
    "short_float": "short_float",
    "float_short": "short_float",
    "short_interest_share": "short_float",
    "short_ratio": "short_ratio",
    "short_interest_ratio": "short_ratio",
    "roa": "roa",
    "return_on_assets": "roa",
    "roe": "roe",
    "return_on_equity": "roe",
    "roi": "roi",
    "roic": "roi",
    "return_on_invested_capital": "roi",
    "current_ratio": "current_ratio",
    "quick_ratio": "quick_ratio",
    "debt_eq": "debt_to_equity",
    "total_debt_equity": "debt_to_equity",
    "lt_debt_equity": "lt_debt_to_equity",
    "lt_debt_eq": "lt_debt_to_equity",
    "gross_margin": "gross_margin",
    "operating_margin": "operating_margin",
    "profit_margin": "profit_margin",
    "beta": "beta",
    "atr": "atr",
    "average_true_range": "atr",
    "volatility_week": "volatility_week",
    "volatility_w": "volatility_week",
    "volatility_month": "volatility_month",
    "volatility_m": "volatility_month",
    "sma20": "sma20",
    "20_day_sma": "sma20",
    "20_day_simple_moving_average": "sma20",
    "sma50": "sma50",
    "50_day_sma": "sma50",
    "50_day_simple_moving_average": "sma50",
    "sma200": "sma200",
    "200_day_sma": "sma200",
    "200_day_simple_moving_average": "sma200",
    "52w_high": "high_52w",
    "52_week_high": "high_52w",
    "high_52w": "high_52w",
    "52w_low": "low_52w",
    "52_week_low": "low_52w",
    "low_52w": "low_52w",
    "rsi": "rsi",
    "relative_strength_index": "rsi",
    "relative_strength_index_14": "rsi",
    "gap": "gap",
    "relative_volume": "relative_volume",
    "rel_volume": "relative_volume",
    "average_volume": "average_volume",
    "avg_volume": "average_volume",
    "volume": "volume",
    "current_volume": "volume",
    "price": "price",
    "change": "change",
    "earnings_date": "earnings_date",
    "earnings": "earnings_date",
}

FINVIZ_EXPORT_TEMPLATES = {
    "stock_daily": {
        "label": "Stock Daily",
        "base_url": FINVIZ_CONFIG["stock_daily_base_url"],
        "params": {"p": "d"},
        "type": "ticker",
        "output": "csv",
    },
    "screener_decision_snapshot": {
        "label": "Screener Decision Snapshot",
        "base_url": FINVIZ_CONFIG["screener_base_url"],
        "params": {"v": "152", "ft": FINVIZ_CONFIG["filter_type"], "c": FINVIZ_DECISION_COLUMNS_STRING},
        "type": "ticker",
        "output": "csv",
    },
    "screener_discovery": {
        "label": "Screener Discovery",
        "base_url": FINVIZ_CONFIG["screener_base_url"],
        "params": {"v": "152", "ft": FINVIZ_CONFIG["filter_type"], "c": FINVIZ_DISCOVERY_COLUMNS_STRING},
        "type": "ticker",
        "output": "csv",
    },
}

PERCENT_COLUMNS = {
    "float_outstanding_pct",
    "short_float",
    "roa",
    "roe",
    "roi",
    "gross_margin",
    "operating_margin",
    "profit_margin",
    "volatility_week",
    "volatility_month",
    "sma20",
    "sma50",
    "sma200",
    "high_52w",
    "low_52w",
    "gap",
    "change",
}

NUMBER_COLUMNS = {
    "no",
    "market_cap",
    "pe",
    "forward_pe",
    "peg",
    "ps",
    "pb",
    "pc",
    "pfcf",
    "shares_outstanding",
    "shares_float",
    "short_ratio",
    "current_ratio",
    "quick_ratio",
    "lt_debt_to_equity",
    "debt_to_equity",
    "beta",
    "atr",
    "rsi",
    "relative_volume",
    "average_volume",
    "volume",
    "price",
} | PERCENT_COLUMNS


def _streamlit_secret(name: str) -> str | None:
    try:
        import streamlit as st

        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


def get_finviz_auth_token() -> str | None:
    """
    Load Finviz token safely.

    Priority:
    1. Streamlit secrets
    2. Environment variable
    3. .env file loaded by python-dotenv

    Never hardcode token.
    """
    prepare_external_data_env()
    token = _streamlit_secret("FINVIZ_AUTH_TOKEN")
    if token:
        return token
    token = os.getenv("FINVIZ_AUTH_TOKEN")
    if token:
        return token
    load_dotenv()
    return os.getenv("FINVIZ_AUTH_TOKEN")


def _configured_columns() -> str | None:
    ids = [str(v) for v in FINVIZ_COLUMNS.values() if v is not None]
    return ",".join(ids) if ids else None


def _template(name: str) -> dict:
    return FINVIZ_EXPORT_TEMPLATES.get(name, FINVIZ_EXPORT_TEMPLATES["screener_decision_snapshot"])


def build_finviz_export_url(
    columns: str | None = None,
    filters: str | None = None,
    ticker: str | None = None,
    template: str = "screener_decision_snapshot",
    include_auth: bool = True,
) -> tuple[str, dict]:
    """
    Build Finviz Elite export request.
    Return base_url and params.
    """
    template_config = _template(template)
    token = get_finviz_auth_token() if include_auth else None
    params: dict[str, Any] = dict(template_config["params"])
    if filters and template.startswith("screener"):
        params["f"] = filters
    elif not ticker and template == "screener_decision_snapshot":
        params["f"] = filters or FINVIZ_CONFIG["default_filters"]
    if ticker:
        params["t"] = ticker.upper().strip()
    column_ids = columns or (None if template_config["params"].get("c") else _configured_columns())
    if column_ids:
        params["c"] = column_ids
    if token and include_auth:
        params["auth"] = token
    return template_config["base_url"], params


def build_finviz_preview_url(ticker: str, template: str = "screener_decision_snapshot", columns: str | None = None) -> str:
    """
    Return an auth-free URL suitable for UI/debug display.
    """
    url, params = build_finviz_export_url(columns=columns, ticker=ticker, template=template, include_auth=False)
    return requests.Request("GET", url, params=params).prepare().url or url


def _unavailable(ticker: str | None = None, error: str = "Finviz unavailable") -> dict:
    payload = {key: None for key in SNAPSHOT_KEYS}
    payload.update({"available": False, "source": "finviz", "error": error})
    if ticker:
        payload["ticker"] = ticker.upper()
    return payload


def _looks_like_html(text: str) -> bool:
    sample = text[:500].lower()
    return "<html" in sample or "<!doctype html" in sample or "login" in sample and "<form" in sample


def fetch_finviz_export(
    columns: str | None = None,
    filters: str = "geo_usa",
    ticker: str | None = None,
    template: str = "screener_decision_snapshot",
) -> pd.DataFrame:
    """
    Fetch Finviz export CSV.

    Detects missing/invalid tokens, HTML/login pages, empty responses, timeouts,
    invalid CSV, and rate limits. Returns an empty DataFrame with an error in attrs
    rather than raising.
    """
    if not FINVIZ_CONFIG["enabled"]:
        df = pd.DataFrame()
        df.attrs["error"] = "Finviz disabled"
        return df
    prepare_external_data_env()
    if not get_finviz_auth_token():
        df = pd.DataFrame()
        df.attrs["error"] = "FINVIZ_AUTH_TOKEN missing or invalid"
        return df

    url, params = build_finviz_export_url(columns=columns, filters=filters, ticker=ticker, template=template)
    try:
        response = requests.get(url, params=params, timeout=FINVIZ_CONFIG["timeout"])
        if response.status_code in {401, 403}:
            df = pd.DataFrame()
            df.attrs["error"] = "FINVIZ_AUTH_TOKEN missing or invalid"
            return df
        if response.status_code == 429:
            df = pd.DataFrame()
            df.attrs["error"] = "Finviz rate limit reached"
            return df
        response.raise_for_status()
        text = response.text.strip()
        if not text:
            df = pd.DataFrame()
            df.attrs["error"] = "Finviz returned empty response"
            return df
        if _looks_like_html(text):
            df = pd.DataFrame()
            df.attrs["error"] = "Finviz returned HTML/login page"
            return df
        df = pd.read_csv(io.StringIO(text))
        if df.empty:
            df.attrs["error"] = "Finviz CSV contained no rows"
            return df
        out = normalize_finviz_dataframe(df)
        out.attrs["template"] = template
        out.attrs["preview_url"] = build_finviz_preview_url(ticker or "", template=template, columns=columns) if ticker else None
        out.attrs["headers"] = list(out.columns)
        return out
    except requests.Timeout:
        df = pd.DataFrame()
        df.attrs["error"] = "Finviz request timed out"
        return df
    except Exception as exc:
        df = pd.DataFrame()
        df.attrs["error"] = f"Finviz unavailable: {exc}"
        return df


def _snake(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")
    return HEADER_ALIASES.get(name, name)


def normalize_finviz_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Finviz column names to snake_case.
    Convert human-readable numbers.
    """
    out = df.copy()
    out.columns = [_snake(c) for c in out.columns]
    for col in out.columns:
        if col in NUMBER_COLUMNS:
            out[col] = out[col].map(parse_human_number)
    if "market_cap" in out.columns:
        out["market_cap"] = out["market_cap"].map(_normalize_finviz_market_cap)
    for col in ["shares_outstanding", "shares_float"]:
        if col in out.columns:
            out[col] = out[col].map(_normalize_finviz_share_count)
    if "average_volume" in out.columns:
        out["average_volume"] = out["average_volume"].map(_normalize_finviz_average_volume)
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype(str).str.upper()
    return out


def parse_human_number(value):
    """
    Convert:
    1.2B -> 1200000000
    450M -> 450000000
    800K -> 800000
    3.4% -> 0.034
    - / N/A -> None
    """
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "N/A", "NA", "nan"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    is_pct = text.endswith("%")
    if is_pct:
        text = text[:-1]
    multiplier = 1.0
    suffix = text[-1:].upper()
    if suffix in {"T", "B", "M", "K"}:
        multiplier = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}[suffix]
        text = text[:-1]
    try:
        number = float(text) * multiplier
        if is_pct:
            number /= 100
        return -number if negative else number
    except ValueError:
        return value


def _normalize_finviz_market_cap(value):
    """
    Finviz Elite's broad export can return market cap in millions without a suffix.
    Convert those values to dollars while leaving already-scaled values alone.
    """
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if 0 < abs(value) < 10_000_000:
        return value * 1_000_000
    return value


def _normalize_finviz_share_count(value):
    """
    Screener share-count columns can be unsuffixed but expressed in millions.
    """
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if 0 < abs(value) < 1_000_000:
        return value * 1_000_000
    return value


def _normalize_finviz_average_volume(value):
    """
    Screener average-volume values can be unsuffixed but expressed in thousands.
    """
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if 0 < abs(value) < 100_000:
        return value * 1_000
    return value


def fetch_finviz_ticker_snapshot(ticker: str) -> dict:
    """
    Return a normalized Finviz ticker snapshot. Finviz is optional; failures use
    a clean unavailable object.
    """
    symbol = ticker.upper().strip()
    df = fetch_finviz_export(ticker=symbol, template="screener_decision_snapshot")
    if df.empty:
        return _unavailable(symbol, df.attrs.get("error", "Finviz unavailable"))
    if "ticker" not in df.columns:
        return _unavailable(symbol, "Finviz CSV missing ticker column")
    row = df[df["ticker"].astype(str).str.upper() == symbol]
    if row.empty:
        return _unavailable(symbol, "Ticker not found in Finviz export")
    data = {key: None for key in SNAPSHOT_KEYS}
    data.update(row.iloc[0].to_dict())
    data = {key: data.get(key) for key in SNAPSHOT_KEYS}
    data.update({"available": True, "source": "finviz", "error": None})
    data["ticker"] = symbol
    data["template"] = df.attrs.get("template", "screener_decision_snapshot")
    data["preview_url"] = df.attrs.get("preview_url") or build_finviz_preview_url(symbol)
    data["available_headers"] = df.attrs.get("headers", list(df.columns))
    return data


def discover_finviz_headers(ticker: str) -> dict:
    """
    Debug-only schema discovery for Finviz column IDs 0..110.
    """
    symbol = ticker.upper().strip()
    df = fetch_finviz_export(ticker=symbol, template="screener_discovery")
    return {
        "ticker": symbol,
        "available": not df.empty,
        "error": df.attrs.get("error") if df.empty else None,
        "headers": list(df.columns),
        "preview_url": build_finviz_preview_url(symbol, template="screener_discovery"),
    }
