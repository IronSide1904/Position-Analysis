from __future__ import annotations

import hashlib
import json
import os
import re
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dotenv import load_dotenv

from config import DEBUG_MODE, SEC_CONFIG
from data_sources.runtime_env import prepare_external_data_env


US_GAAP_TAGS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForProceedsFromProductiveAssets",
    ],
    "depreciation_amortization": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "debt_current": ["LongTermDebtCurrent", "LongTermDebtAndFinanceLeaseObligationsCurrent", "ShortTermBorrowings"],
    "debt_noncurrent": ["LongTermDebtNoncurrent", "LongTermDebtAndFinanceLeaseObligationsNoncurrent"],
    "shares_outstanding": [
        "EntityCommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
    "sbc": [
        "ShareBasedCompensation",
        "ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriodTotal",
    ],
    "stock_repurchases": ["PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity"],
    "dividends": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "goodwill": ["Goodwill"],
    "intangibles": ["FiniteLivedIntangibleAssetsNet", "IndefiniteLivedIntangibleAssetsExcludingGoodwill"],
}


def _streamlit_secret(name: str) -> str | None:
    try:
        import streamlit as st

        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


def get_sec_user_agent() -> str:
    prepare_external_data_env()
    value = _streamlit_secret("SEC_USER_AGENT") or os.getenv("SEC_USER_AGENT")
    if value:
        return value
    load_dotenv()
    return os.getenv("SEC_USER_AGENT") or "PA-11R Hybrid research app contact@example.com"


def _sec_headers(url: str) -> dict[str, str]:
    headers = {"User-Agent": get_sec_user_agent(), "Accept-Encoding": "gzip, deflate"}
    if "data.sec.gov" in url:
        headers["Host"] = "data.sec.gov"
    return headers


def _sec_cache_path(*parts: str) -> Path:
    path = Path(SEC_CONFIG["cache_dir"]).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _fresh(path: Path, ttl_seconds: int) -> bool:
    return path.exists() and time.time() - path.stat().st_mtime < ttl_seconds


def _safe_error(exc: Exception) -> str:
    return repr(exc) if DEBUG_MODE else str(exc)


def sec_get_json(url: str, cache_path: Path, ttl_seconds: int) -> dict:
    prepare_external_data_env()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if _fresh(cache_path, ttl_seconds):
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("_cache", {})["hit"] = True
                return data
        except Exception:
            pass

    try:
        response = requests.get(url, headers=_sec_headers(url), timeout=SEC_CONFIG["timeout"])
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return {"error": "SEC returned non-object JSON"}
        data["_cache"] = {"hit": False, "url": url}
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception as exc:
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("warnings", []).append(f"Using stale SEC cache after fetch failure: {_safe_error(exc)}")
                    data.setdefault("_cache", {})["hit"] = True
                    data["_cache"]["stale"] = True
                    return data
            except Exception:
                pass
        return {"error": f"SEC data unavailable: {_safe_error(exc)}", "_cache": {"hit": False, "url": url}}


@lru_cache(maxsize=1)
def fetch_ticker_cik_map() -> pd.DataFrame:
    data = sec_get_json(
        SEC_CONFIG["ticker_cik_url"],
        _sec_cache_path("cik_map.json"),
        SEC_CONFIG["cik_map_ttl_seconds"],
    )
    if data.get("error"):
        df = pd.DataFrame(columns=["ticker", "title", "cik", "cik_str"])
        df.attrs["error"] = data["error"]
        return df

    rows = [row for key, row in data.items() if str(key).isdigit() and isinstance(row, dict)]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "title", "cik", "cik_str"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["cik"] = df["cik_str"].astype(int).astype(str).str.zfill(10)
    df["cik_str"] = df["cik"]
    return df[["ticker", "title", "cik", "cik_str"]]


def get_ticker_cik_map_error() -> str | None:
    df = fetch_ticker_cik_map()
    return df.attrs.get("error") if df.empty else None


def get_cik_for_ticker(ticker: str) -> str | None:
    df = fetch_ticker_cik_map()
    if df.empty:
        return None
    row = df[df["ticker"] == ticker.upper().strip()]
    if row.empty:
        return None
    cik_value = row.iloc[0].get("cik") if "cik" in row.columns else row.iloc[0].get("cik_str")
    return str(cik_value).zfill(10) if cik_value is not None else None


def fetch_company_submissions(cik: str) -> dict:
    cik = str(cik).zfill(10)
    url = f"{SEC_CONFIG['submissions_base_url']}/CIK{cik}.json"
    return sec_get_json(url, _sec_cache_path("submissions", f"{cik}.json"), SEC_CONFIG["submissions_ttl_seconds"])


def fetch_company_facts(cik: str) -> dict:
    cik = str(cik).zfill(10)
    url = f"{SEC_CONFIG['companyfacts_base_url']}/CIK{cik}.json"
    return sec_get_json(url, _sec_cache_path("companyfacts", f"{cik}.json"), SEC_CONFIG["companyfacts_ttl_seconds"])


def _document_url(cik: str, accession_number: str, primary_document: str) -> str:
    compact = accession_number.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{compact}/{primary_document}"


def get_latest_filings(
    submissions: dict,
    forms: tuple[str, ...] = ("10-K", "10-Q", "8-K", "DEF 14A"),
    limit_per_form: int = 3,
) -> list[dict]:
    if not isinstance(submissions, dict):
        return []
    cik = str(submissions.get("cik") or "").zfill(10)
    recent = submissions.get("filings", {}).get("recent", {})
    counts = {form: 0 for form in forms}
    rows = []
    for i, form in enumerate(recent.get("form", [])):
        if form not in forms or counts[form] >= limit_per_form:
            continue
        accession = recent.get("accessionNumber", [None])[i]
        primary = recent.get("primaryDocument", [None])[i]
        if not cik or not accession or not primary:
            continue
        url = _document_url(cik, accession, primary)
        rows.append(
            {
                "form": form,
                "filing_date": recent.get("filingDate", [None])[i],
                "report_date": recent.get("reportDate", [None])[i],
                "accession_number": accession,
                "primary_document": primary,
                "filing_url": url,
                "document_url": url,
            }
        )
        counts[form] += 1
    return rows


def get_fact_units(companyfacts: dict, namespace: str, tag: str) -> dict | None:
    return companyfacts.get("facts", {}).get(namespace, {}).get(tag, {}).get("units")


def extract_fact_dataframe(
    companyfacts: dict,
    tag_options: list[str],
    preferred_units: tuple[str, ...] = ("USD", "shares"),
) -> pd.DataFrame:
    columns = ["tag", "tag_priority", "unit", "val", "fy", "fp", "form", "filed", "start", "end", "frame", "source", "confidence"]
    frames = []
    for priority, tag in enumerate(tag_options):
        units = get_fact_units(companyfacts, "us-gaap", tag)
        if not units:
            continue
        unit_name = next((unit for unit in preferred_units if unit in units), None) or next(iter(units), None)
        rows = units.get(unit_name, []) if unit_name else []
        if not rows:
            continue
        df = pd.DataFrame(rows)
        for col in ["fy", "fp", "form", "filed", "start", "end", "frame"]:
            if col not in df:
                df[col] = None
        df["tag"] = tag
        df["tag_priority"] = priority
        df["unit"] = unit_name
        df["source"] = "SEC companyfacts"
        df["confidence"] = "reported"
        if "val" in df and tag in set(US_GAAP_TAGS["capex"]):
            df["val"] = df["val"].map(lambda value: abs(value) if pd.notna(value) else value)
        frames.append(df[columns])
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=columns)


def normalize_companyfacts_financials(companyfacts: dict) -> dict:
    if not isinstance(companyfacts, dict) or companyfacts.get("error"):
        error = companyfacts.get("error", "SEC companyfacts unavailable") if isinstance(companyfacts, dict) else "SEC companyfacts unavailable"
        return {"available": False, "source": "SEC companyfacts", "metrics": {}, "warnings": [error]}
    metrics = {name: extract_fact_dataframe(companyfacts, tags) for name, tags in US_GAAP_TAGS.items()}
    warnings_out = [f"SEC companyfacts missing {name}" for name, df in metrics.items() if df.empty]
    return {"available": any(not df.empty for df in metrics.values()), "source": "SEC companyfacts", "metrics": metrics, "warnings": warnings_out}


def _latest_metric_value(metrics: dict[str, pd.DataFrame], name: str) -> Any:
    df = metrics.get(name, pd.DataFrame())
    if df.empty or "val" not in df:
        return None
    usable = df[df["val"].notna()].copy()
    if usable.empty:
        return None
    usable["_sort"] = usable["end"].fillna("").astype(str) + "|" + usable["filed"].fillna("").astype(str)
    return usable.sort_values("_sort").iloc[-1]["val"]


def extract_core_financials_from_companyfacts(companyfacts: dict) -> dict:
    normalized = normalize_companyfacts_financials(companyfacts)
    metrics = normalized.get("metrics", {})
    out = {}
    debt_current = _latest_metric_value(metrics, "debt_current") or 0
    debt_noncurrent = _latest_metric_value(metrics, "debt_noncurrent") or 0
    for name in list(US_GAAP_TAGS) + ["debt_total"]:
        value = debt_current + debt_noncurrent if name == "debt_total" else _latest_metric_value(metrics, name)
        out[name] = {
            "value": value,
            "period": None,
            "form": None,
            "filed": None,
            "source": "SEC companyfacts",
            "confidence": "reported" if value is not None else "unavailable",
        }
    out["shares"] = out["shares_outstanding"]
    out["debt"] = out["debt_total"]
    return out


def download_filing_text(document_url: str, cache_key: str | None = None) -> str:
    cache_name = cache_key or hashlib.sha256(document_url.encode("utf-8")).hexdigest()
    cache_path = _sec_cache_path("filings", f"{cache_name}.txt")
    if _fresh(cache_path, SEC_CONFIG["filing_text_ttl_seconds"]):
        try:
            return cache_path.read_text(encoding="utf-8")
        except Exception:
            pass
    try:
        prepare_external_data_env()
        response = requests.get(document_url, headers=_sec_headers(document_url), timeout=SEC_CONFIG["timeout"])
        response.raise_for_status()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(response.text, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = re.sub(r"[ \t]{2,}", " ", re.sub(r"\n{3,}", "\n\n", soup.get_text("\n"))).strip()
        cache_path.write_text(text, encoding="utf-8")
        return text
    except Exception:
        return ""


def fetch_sec_fast_snapshot(ticker: str) -> dict:
    symbol = ticker.upper().strip()
    warnings_out: list[str] = []
    cik = get_cik_for_ticker(symbol)
    if not cik:
        err = get_ticker_cik_map_error()
        return {
            "available": False,
            "source": "SEC",
            "ticker": symbol,
            "cik": None,
            "company_name": None,
            "sic": None,
            "sic_description": None,
            "latest_filings": [],
            "financials": {"available": False, "source": "SEC companyfacts", "metrics": {}, "warnings": []},
            "warnings": [f"SEC ticker map unavailable: {err}" if err else "SEC CIK not found"],
        }
    submissions = fetch_company_submissions(cik)
    companyfacts = fetch_company_facts(cik)
    if submissions.get("error"):
        warnings_out.append(submissions["error"])
    if companyfacts.get("error"):
        warnings_out.append(companyfacts["error"])
    normalized = normalize_companyfacts_financials(companyfacts)
    warnings_out.extend(normalized.get("warnings", [])[:5])
    return {
        "available": not submissions.get("error") or normalized.get("available"),
        "source": "SEC",
        "ticker": symbol,
        "cik": cik,
        "company_name": submissions.get("name") or companyfacts.get("entityName"),
        "sic": submissions.get("sic"),
        "sic_description": submissions.get("sicDescription"),
        "latest_filings": get_latest_filings(submissions),
        "financials": normalized,
        "submissions": submissions,
        "companyfacts": companyfacts,
        "warnings": warnings_out,
    }


def fetch_sec_deep_filings(
    ticker: str,
    forms: tuple[str, ...] = ("10-K", "10-Q", "DEF 14A"),
    limit_per_form: int = 1,
) -> dict:
    symbol = ticker.upper().strip()
    warnings_out: list[str] = []
    cik = get_cik_for_ticker(symbol)
    if not cik:
        return {"ticker": symbol, "filing_texts": {}, "filings": [], "warnings": ["SEC CIK not found"]}
    submissions = fetch_company_submissions(cik)
    filings = get_latest_filings(submissions, forms=forms, limit_per_form=limit_per_form)
    filing_texts = {}
    for filing in filings:
        form = filing["form"]
        cache_key = f"{cik}_{filing['accession_number'].replace('-', '')}_{filing['primary_document']}"
        text = download_filing_text(filing["document_url"], cache_key=cache_key)
        if text:
            filing_texts.setdefault(form, text)
        else:
            warnings_out.append(f"SEC filing text unavailable for {form} {filing.get('filing_date')}")
    return {"ticker": symbol, "filing_texts": filing_texts, "filings": filings, "warnings": warnings_out}
