from __future__ import annotations

from data_sources.loader import load_company_dataset


def get_ticker_dataset(ticker: str, include_deep_sec: bool = False) -> dict:
    """
    Public data function used by app/dashboard.
    """
    return load_company_dataset(ticker, include_deep_sec=include_deep_sec)
