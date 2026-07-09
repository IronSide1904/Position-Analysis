from __future__ import annotations

import pandas as pd

from models.multiples_model import multiples_interpretation


def premium_discount_decision(metric: str, value: float | None, peer_value: float | None, context: dict | None = None) -> str:
    context = context or {}
    return multiples_interpretation(metric, value, peer_value, context.get("moat_score"))


def build_peer_premium_read(current_multiples: dict, peer_medians: dict, quality_context: dict | None = None) -> dict:
    quality_context = quality_context or {}
    checks = []
    for metric in ["EV/Revenue", "EV/OCF", "EV/NOPAT", "EV/FCF", "P/E"]:
        current = current_multiples.get(metric)
        peer = peer_medians.get(metric)
        if current is None or not peer:
            continue
        premium = current / peer - 1
        checks.append({"Metric": metric, "Premium / Discount": premium, "Read": premium_discount_decision(metric, current, peer, quality_context)})
    if not checks:
        return {
            "classification": "Unavailable",
            "summary": "Peer premium/discount cannot be calculated with the current provider fields.",
            "table": pd.DataFrame(),
        }
    table = pd.DataFrame(checks)
    avg = table["Premium / Discount"].mean()
    if avg > 0.2:
        classification = "Premium"
        summary = "Company trades at a premium to available peer references; require growth, margin, OCF, moat, and capital-allocation evidence."
    elif avg < -0.2:
        classification = "Discount"
        summary = "Company trades at a discount to available peer references; review whether risk explains it or whether hidden value exists."
    else:
        classification = "In line"
        summary = "Company trades broadly in line with available peer references."
    return {"classification": classification, "summary": summary, "table": table}
