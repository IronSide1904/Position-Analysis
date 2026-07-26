from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd
import streamlit as st


def _state_bucket(name: str) -> dict:
    bucket = st.session_state.setdefault(name, {})
    if not isinstance(bucket, dict):
        bucket = {}
        st.session_state[name] = bucket
    return bucket


def _clean_value(value: Any):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "inf", "-inf", "none", "n.m.", "not meaningful", "not calculated", "not applicable", "unavailable"}:
            return None
        return text
    return value


def _number(value: Any) -> float | None:
    value = _clean_value(value)
    if value is None:
        return None
    if isinstance(value, str):
        text = value.replace(",", "").replace("$", "").replace("shares", "").strip()
        is_percent = "%" in text
        text = text.replace("%", "").replace("x", "").strip()
        if text in {"-", "--", "—"}:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return number / 100 if is_percent else number
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _equalish(old: Any, new: Any) -> bool:
    old_num = _number(old)
    new_num = _number(new)
    if old_num is not None and new_num is not None:
        return abs(old_num - new_num) <= 1e-9
    return str(_clean_value(old) or "") == str(_clean_value(new) or "")


def _delta(old: Any, new: Any):
    old_num = _number(old)
    new_num = _number(new)
    if old_num is None or new_num is None:
        return None
    return new_num - old_num


def _editable_period_columns(frame: pd.DataFrame) -> list[str]:
    static = {
        "Row Key",
        "Driver Group",
        "Product / Service Line",
        "Specific Driver",
        "Assumption",
        "Evidence",
        "Confidence",
        "Row Type",
        "Scenario",
        "Status",
        "Model Impact",
        "Source / Basis",
        "Suggested Keywords",
        "Suggested Filing Section",
        "Fallback Used",
        "Manual Review Needed",
        "Affected Assumptions",
        "User Note",
    }
    return [col for col in frame.columns if col not in static]


def _model_impact_text(assumption: str, row_type: str) -> str:
    text = str(assumption or "").lower()
    if row_type == "Override":
        return "Override active: implied percentage / ratio recalculates downstream model rows."
    if "opex" in text:
        return "OPEX, EBIT, NOPAT, FCF, fair value."
    if "cogs" in text or "gross" in text:
        return "Gross profit, EBIT, NOPAT, FCF, fair value."
    if "tax" in text:
        return "Tax expense, NOPAT, FCF, fair value."
    if "nopat" in text:
        return "NOPAT, FCF, fair value."
    if "ocf" in text:
        return "OCF, FCF, fair value."
    if "capex" in text:
        return "CAPEX, FCF, fair value."
    if "working capital" in text:
        return "Working capital, FCF, fair value."
    if "sbc" in text:
        return "SBC, dilution, per-share value."
    if "share" in text:
        return "Diluted shares and fair value per share."
    if "revenue" in text:
        return "Revenue, margins, cash flow, fair value."
    return "DCF output and scenario comparison."


def _restore_locked_rows(committed: pd.DataFrame, edited: pd.DataFrame) -> pd.DataFrame:
    if committed is None or edited is None or committed.empty or edited.empty or "Row Key" not in committed or "Row Key" not in edited:
        return edited
    out = edited.copy()
    committed_index = committed.set_index("Row Key", drop=False)
    period_cols = _editable_period_columns(committed)
    for idx, row in out.iterrows():
        row_key = row.get("Row Key")
        if row_key not in committed_index.index:
            continue
        row_type = str(committed_index.loc[row_key].get("Row Type", "Input"))
        if row_type in {"Input", "Override"}:
            continue
        for period in period_cols:
            if period in out.columns:
                out.at[idx, period] = committed_index.loc[row_key].get(period)
    return out


def detect_assumption_changes(committed: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    """
    Compare committed vs draft assumptions.

    Return table:
    Assumption, Period, Old Value, New Value, Delta, Scenario, Status.
    """
    if committed is None or draft is None or committed.empty or draft.empty:
        return pd.DataFrame(columns=["Assumption", "Period", "Old Value", "New Value", "Delta", "Row Type", "Model Impact", "Scenario", "Status"])
    key_col = "Row Key" if "Row Key" in committed.columns and "Row Key" in draft.columns else None
    left = committed.set_index(key_col, drop=False) if key_col else committed.copy()
    right = draft.set_index(key_col, drop=False) if key_col else draft.copy()
    rows = []
    for idx, old_row in left.iterrows():
        if idx not in right.index:
            continue
        new_row = right.loc[idx]
        if isinstance(new_row, pd.DataFrame):
            new_row = new_row.iloc[0]
        assumption = old_row.get("Assumption") if hasattr(old_row, "get") else str(idx)
        row_type = str(old_row.get("Row Type", "Input")) if hasattr(old_row, "get") else "Input"
        if row_type not in {"Input", "Override"}:
            continue
        for period in _editable_period_columns(committed):
            if period not in draft.columns:
                continue
            old_value = old_row.get(period)
            new_value = new_row.get(period)
            if _equalish(old_value, new_value):
                continue
            rows.append(
                {
                    "Assumption": assumption,
                    "Period": period,
                    "Old Value": old_value,
                    "New Value": new_value,
                    "Delta": _delta(old_value, new_value),
                    "Row Type": row_type,
                    "Model Impact": _model_impact_text(assumption, row_type),
                    "Scenario": new_row.get("Scenario", ""),
                    "Status": "Pending",
                }
            )
    return pd.DataFrame(rows)


def apply_assumption_changes(committed: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    """
    Replace committed assumptions with draft assumptions after user clicks Apply.
    """
    if draft is None or draft.empty:
        return committed.copy() if committed is not None else pd.DataFrame()
    return draft.copy()


def discard_assumption_changes(committed: pd.DataFrame) -> pd.DataFrame:
    """
    Reset draft assumptions back to committed assumptions.
    """
    return committed.copy() if committed is not None else pd.DataFrame()


def render_editable_assumption_table(
    committed_assumptions: pd.DataFrame,
    scenario_scope: str,
    key: str = "dcf_assumptions_editor",
    *,
    disabled_columns: list[str] | None = None,
    column_config: dict | None = None,
    height: int = 520,
    read_only: bool = False,
) -> dict:
    """
    Render editable assumptions table.

    Must:
    - Use st.data_editor.
    - Store edits in draft state.
    - Not automatically recalculate DCF on every edit.
    - Return draft assumptions and pending changes.
    """
    committed_assumptions = committed_assumptions.copy() if committed_assumptions is not None else pd.DataFrame()
    committed_bucket = _state_bucket("assumptions_committed")
    draft_bucket = _state_bucket("assumptions_draft")
    pending_bucket = _state_bucket("assumptions_pending_changes")
    last_applied_bucket = _state_bucket("last_applied_assumptions")

    fingerprint = str(hash(tuple(committed_assumptions.astype(str).to_numpy().ravel()))) if not committed_assumptions.empty else "empty"
    fingerprint_key = f"{key}_fingerprint"
    if committed_bucket.get(key) is None or st.session_state.get(fingerprint_key) != fingerprint:
        committed_bucket[key] = committed_assumptions.copy()
        draft_bucket[key] = committed_assumptions.copy()
        pending_bucket[key] = pd.DataFrame()
        last_applied_bucket.setdefault(key, committed_assumptions.copy())
        st.session_state[fingerprint_key] = fingerprint

    committed_df = committed_bucket[key].copy()
    draft_df = draft_bucket.get(key, committed_df).copy()
    st.markdown("**Editable Assumptions**")
    st.caption("You can edit multiple cells. Changes are not applied until you click Apply Changes & Recalculate.")

    pending_before = detect_assumption_changes(committed_df, draft_df)
    pending_count = len(pending_before)
    status_class = "pa-pill-warn" if pending_count else "pa-pill-ok"
    status_text = f"Pending changes: {pending_count}" if pending_count else "No pending changes"
    st.markdown(f'<span class="pa-pill {status_class}">{status_text}</span>', unsafe_allow_html=True)

    editor_disabled = disabled_columns or []
    if read_only:
        editor_disabled = list(committed_df.columns)
    edited = st.data_editor(
        draft_df,
        width="stretch",
        height=height,
        hide_index=True,
        column_config=column_config or {},
        disabled=editor_disabled,
        key=key,
    )
    edited = _restore_locked_rows(committed_df, edited)
    draft_bucket[key] = edited.copy()
    pending = detect_assumption_changes(committed_df, edited)
    pending_bucket[key] = pending.copy()

    if not pending.empty:
        st.warning("These changes are not included in the valuation until you click Apply Changes & Recalculate.")
        with st.expander("Pending Changes", expanded=True):
            st.dataframe(pending, width="stretch", hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    applied = False
    discarded = False
    reset_to_base = False
    save_draft = False
    if c1.button("Apply Changes & Recalculate", key=f"{key}_apply", disabled=read_only or pending.empty):
        committed_bucket[key] = apply_assumption_changes(committed_df, edited)
        draft_bucket[key] = committed_bucket[key].copy()
        last_applied_bucket[key] = committed_bucket[key].copy()
        pending_bucket[key] = pd.DataFrame()
        applied = True
    if c2.button("Discard Pending Changes", key=f"{key}_discard", disabled=read_only or pending.empty):
        draft_bucket[key] = discard_assumption_changes(committed_df)
        pending_bucket[key] = pd.DataFrame()
        discarded = True
    if c3.button("Reset User Case to Base", key=f"{key}_reset_base", disabled=read_only):
        reset_to_base = True
    if c4.button("Save Draft Analysis", key=f"{key}_save_draft"):
        save_draft = True

    return {
        "committed": committed_bucket[key].copy(),
        "draft": draft_bucket[key].copy(),
        "pending_changes": pending.copy(),
        "pending_count": len(pending),
        "applied": applied,
        "discarded": discarded,
        "reset_to_base": reset_to_base,
        "save_draft": save_draft,
    }
