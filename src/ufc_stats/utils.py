"""Utilities for UFC stats library."""

import re
from typing import Any

import pandas as pd


# Fight detail field definitions (for reference)
# rev = Reversals: when a fighter reverses position in grappling (e.g., from bottom to top)
# ctrl = Control time: time spent in dominant position (e.g., "3:35")


def parse_ratio_value(val: Any) -> dict | None:
    """
    Parse "X/Y" or "X of Y" format to success, attempt, ratio, pct.
    Returns None if not a ratio format.
    """
    if not isinstance(val, str):
        return None
    m = re.search(r"(\d+)\s*/\s*(\d+)", val)
    if not m:
        m = re.search(r"(\d+)\s+of\s+(\d+)", val, re.IGNORECASE)
    if m:
        success = int(m.group(1))
        attempt = int(m.group(2))
        pct = round(success / attempt, 4) if attempt > 0 else None
        ratio = f"{success}/{attempt}"
        return {"success": success, "attempt": attempt, "ratio": ratio, "pct": pct}
    return None


def format_accuracy_value(val: Any) -> Any:
    """
    Reformat "58 of 105" style strings to "58/105".
    Returns value unchanged if not matching.
    """
    if not isinstance(val, str):
        return val
    m = re.search(r"(\d+)\s+of\s+(\d+)", val, re.IGNORECASE)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return val


def _flatten_fight(fight: dict) -> dict:
    """Flatten a single fight dict to flat row (no nested objects)."""
    row = dict(fight)
    if "bonuses" in row and isinstance(row["bonuses"], list):
        row["bonuses"] = ", ".join(row["bonuses"]) if row["bonuses"] else ""
    elif row.get("bonuses") is None:
        row["bonuses"] = ""
    return row


def flatten_json_for_df(obj: dict) -> pd.DataFrame:
    """
    Recursively flatten nested dict into a single-row DataFrame.
    Nested dicts get prefixed keys; lists of primitives become comma-separated strings.
    """
    row = {}

    def _flatten(value: Any, key_prefix: str) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                new_key = f"{key_prefix}_{k}" if key_prefix else k
                _flatten(v, new_key)
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                # Don't explode - caller should handle list of dicts separately
                row[key_prefix] = str(value)
            else:
                row[key_prefix] = ", ".join(str(x) for x in value) if value else ""
        else:
            row[key_prefix] = value

    for k, v in obj.items():
        _flatten(v, k)

    return pd.DataFrame([row])
