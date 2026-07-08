from __future__ import annotations

import pandas as pd

from data_sources.finviz_fetcher import fetch_finviz_ticker_snapshot
from data_sources.sec_edgar_fetcher import (
    extract_core_financials_from_companyfacts,
    fetch_sec_deep_filings,
    fetch_sec_fast_snapshot,
)
from data_sources.yfinance_fetcher import fetch_price_history, fetch_yfinance_financials, fetch_yfinance_snapshot


def _pick(*values):
    for value in values:
        if value not in (None, "", []):
            return value
    return None


def _legacy_filings(latest_filings: list[dict]) -> dict:
    filings = {"latest_10k": None, "latest_10q": None, "latest_proxy": None, "latest_8ks": []}
    for filing in latest_filings or []:
        form = filing.get("form")
        if form == "10-K" and not filings["latest_10k"]:
            filings["latest_10k"] = filing
        elif form == "10-Q" and not filings["latest_10q"]:
            filings["latest_10q"] = filing
        elif form == "DEF 14A" and not filings["latest_proxy"]:
            filings["latest_proxy"] = filing
        elif form == "8-K" and len(filings["latest_8ks"]) < 5:
            filings["latest_8ks"].append(filing)
    return filings


def _sec_core_has_data(sec_financials: dict) -> bool:
    required = ("revenue", "operating_income", "operating_cash_flow", "capex", "shares")
    return any(sec_financials.get(key, {}).get("value") not in (None, "") for key in required)


def load_market_data(ticker: str, period: str = "5y") -> dict:
    symbol = ticker.upper().strip()
    warnings: list[str] = []
    price_history = fetch_price_history(symbol, period=period)
    finviz = fetch_finviz_ticker_snapshot(symbol)
    yfinance_snapshot = fetch_yfinance_snapshot(symbol)

    if price_history.empty:
        warnings.append(price_history.attrs.get("error") or "yfinance OHLCV unavailable")
    if not finviz.get("available"):
        warnings.append(finviz.get("error") or "Finviz data unavailable")
    if not yfinance_snapshot.get("available"):
        warnings.append(yfinance_snapshot.get("error") or "yfinance snapshot unavailable")

    market_data = {
        "price": _pick(finviz.get("price"), yfinance_snapshot.get("current_price"), yfinance_snapshot.get("price")),
        "market_cap": _pick(finviz.get("market_cap"), yfinance_snapshot.get("market_cap")),
        "enterprise_value": yfinance_snapshot.get("enterprise_value"),
        "cash": yfinance_snapshot.get("cash"),
        "debt": yfinance_snapshot.get("debt"),
        "shares_outstanding": _pick(finviz.get("shares_outstanding"), yfinance_snapshot.get("shares_outstanding")),
        "shares_float": _pick(finviz.get("shares_float"), yfinance_snapshot.get("float_shares")),
        "short_float": finviz.get("short_float"),
        "relative_volume": finviz.get("relative_volume"),
        "beta": _pick(finviz.get("beta"), yfinance_snapshot.get("beta")),
        "country": finviz.get("country"),
        "average_volume": finviz.get("average_volume"),
        "volume": finviz.get("volume"),
        "float_outstanding_pct": finviz.get("float_outstanding_pct"),
        "short_ratio": finviz.get("short_ratio"),
        "atr": finviz.get("atr"),
        "volatility_week": finviz.get("volatility_week"),
        "volatility_month": finviz.get("volatility_month"),
        "gap": finviz.get("gap"),
        "change": finviz.get("change"),
        "sma20": finviz.get("sma20"),
        "sma50": finviz.get("sma50"),
        "sma200": finviz.get("sma200"),
        "high_52w": finviz.get("high_52w"),
        "low_52w": finviz.get("low_52w"),
        "rsi": finviz.get("rsi"),
        "pe": finviz.get("pe"),
        "forward_pe": finviz.get("forward_pe"),
        "peg": finviz.get("peg"),
        "ps": finviz.get("ps"),
        "pb": finviz.get("pb"),
        "pc": finviz.get("pc"),
        "pfcf": finviz.get("pfcf"),
        "roa": finviz.get("roa"),
        "roe": finviz.get("roe"),
        "roi": finviz.get("roi"),
        "current_ratio": finviz.get("current_ratio"),
        "quick_ratio": finviz.get("quick_ratio"),
        "lt_debt_to_equity": finviz.get("lt_debt_to_equity"),
        "debt_to_equity": finviz.get("debt_to_equity"),
        "gross_margin": finviz.get("gross_margin"),
        "operating_margin": finviz.get("operating_margin"),
        "profit_margin": finviz.get("profit_margin"),
        "earnings_date": finviz.get("earnings_date"),
    }
    return {
        "ticker": symbol,
        "price_history": price_history,
        "finviz": finviz,
        "yfinance_snapshot": yfinance_snapshot,
        "market_data": market_data,
        "warnings": warnings,
    }


def load_sec_data_fast(ticker: str) -> dict:
    return fetch_sec_fast_snapshot(ticker)


def load_sec_deep_data(ticker: str) -> dict:
    return fetch_sec_deep_filings(ticker)


def load_company_dataset(ticker: str, include_deep_sec: bool = False) -> dict:
    symbol = ticker.upper().strip()
    warnings: list[str] = []
    sources: list[str] = []

    market = load_market_data(symbol)
    sec = load_sec_data_fast(symbol)
    deep_sec = load_sec_deep_data(symbol) if include_deep_sec else {"ticker": symbol, "filing_texts": {}, "filings": [], "warnings": []}

    finviz = market["finviz"]
    yfinance_snapshot = market["yfinance_snapshot"]
    sec_financials = extract_core_financials_from_companyfacts(sec.get("companyfacts", {}))
    needs_yfinance_financials = not _sec_core_has_data(sec_financials)
    yfinance_financials = fetch_yfinance_financials(symbol) if needs_yfinance_financials else {
        "available": False,
        "source": "yfinance",
        "income_stmt": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
        "balance_sheet": pd.DataFrame(),
        "error": "Not fetched; SEC companyfacts available",
    }

    if sec.get("available"):
        sources.append("SEC/EDGAR")
    if finviz.get("available"):
        sources.append("Finviz Elite")
    if yfinance_snapshot.get("available") or not market["price_history"].empty:
        sources.append("yfinance")

    warnings.extend(sec.get("warnings", []))
    warnings.extend(market.get("warnings", []))
    warnings.extend(deep_sec.get("warnings", []))
    if needs_yfinance_financials and not yfinance_financials.get("available"):
        warnings.append(yfinance_financials.get("error") or "yfinance financials unavailable")

    latest_filings = sec.get("latest_filings", [])
    return {
        "ticker": symbol,
        "company": _pick(sec.get("company_name"), finviz.get("company"), yfinance_snapshot.get("company")),
        "company_description": yfinance_snapshot.get("description"),
        "sector": _pick(finviz.get("sector"), yfinance_snapshot.get("sector")),
        "industry": _pick(finviz.get("industry"), yfinance_snapshot.get("industry")),
        "cik": sec.get("cik"),
        "market_data": market["market_data"],
        "price_history": market["price_history"],
        "sec": sec,
        "finviz": finviz,
        "yfinance": yfinance_snapshot,
        "financials": {"sec": sec_financials, "yfinance": yfinance_financials, "sec_normalized": sec.get("financials", {})},
        "filings": _legacy_filings(latest_filings),
        "latest_filings": latest_filings,
        "filing_texts": deep_sec.get("filing_texts", {}),
        "deep_filings": deep_sec.get("filings", []),
        "submissions": sec.get("submissions", {}),
        "companyfacts": sec.get("companyfacts", {}),
        "sources": sorted(set(sources)),
        "warnings": list(dict.fromkeys([warning for warning in warnings if warning])),
        "evidence_loaded": include_deep_sec,
    }
