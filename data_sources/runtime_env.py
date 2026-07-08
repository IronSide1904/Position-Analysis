from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def load_local_env_files() -> None:
    """
    Load local secrets without overriding values already supplied by the host.
    """
    load_dotenv(override=False)


def _is_dead_loopback_proxy(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value if "://" in value else f"http://{value}")
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"} and parsed.port == 9


def remove_dead_proxy_env() -> list[str]:
    """
    Remove only the local discard-port proxy that blocks outbound market-data calls.
    Other proxy settings are left intact.
    """
    removed = []
    for key in PROXY_ENV_KEYS:
        if _is_dead_loopback_proxy(os.getenv(key)):
            os.environ.pop(key, None)
            removed.append(key)
    return removed


def prepare_external_data_env() -> list[str]:
    load_local_env_files()
    return remove_dead_proxy_env()


def yfinance_cache_dir() -> str:
    cache_dir = Path(tempfile.gettempdir()) / "pa11r-hybrid-dashboard" / "yfinance-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)
