from __future__ import annotations

from persistence.schema import migrate_analysis_payload


def migrate(payload: dict) -> dict:
    return migrate_analysis_payload(payload)
