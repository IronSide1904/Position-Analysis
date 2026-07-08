from __future__ import annotations

from ui.formatting import UNAVAILABLE, fmt_per_share


def build_thesis_summary(dataset: dict, dcf: dict, reverse_dcf: dict, moat: dict, score: dict) -> dict:
    return {
        "what_it_does": f"{dataset.get('company') or dataset.get('ticker')} operates in {dataset.get('industry') or 'an industry requiring manual classification'}.",
        "how_it_makes_money": "Review revenue recognition and segment notes in the Clause Map for primary-source support.",
        "valuation_view": f"Fair value estimate: {fmt_per_share(dcf.get('fair_value_per_share'))}. Market case: {reverse_dcf.get('market_case') or UNAVAILABLE}.",
        "decision": score.get("recommendation"),
        "moat_support": moat.get("terminal_value_implication"),
    }
