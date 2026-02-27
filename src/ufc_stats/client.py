"""UFC Stats API client."""

import json
import re
from contextlib import contextmanager, nullcontext as _nullcontext
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .utils import (
    _flatten_fight,
    format_accuracy_value,
    flatten_json_for_df,
    normalize_fight_stats,
    parse_ratio_value,
)

API_BASE = "https://ufcapi.aristotle.me"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "notebooks" / "data"

# Fight detail field glossary (fd_* columns)
# rev = Reversals: position reversals in grappling (e.g., bottom to top)
# ctrl = Control time: time in dominant position (e.g., "3:35")


class UFCStatsClient:
    """Client for UFC Stats API (https://ufcapi.aristotle.me)."""

    def __init__(
        self,
        base_url: str = API_BASE,
        cache_path: Optional[Path | str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        if cache_path is False:
            self._cache_path = None
        elif cache_path is not None:
            self._cache_path = Path(cache_path)
            self._cache_path.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_path = DEFAULT_CACHE_PATH
            self._cache_path.mkdir(parents=True, exist_ok=True)
        self._skip_cache = False

    @contextmanager
    def force_refresh(self):
        """
        Context manager to bypass cache and overwrite with fresh API data.
        Use when data looks stale; fresh responses are written to cache.
        """
        prev = self._skip_cache
        self._skip_cache = True
        try:
            yield
        finally:
            self._skip_cache = prev

    def _path_to_cache_key(self, path: str) -> str:
        """Convert API path to filesystem-safe cache filename."""
        # /api/fighters?search=oliveira -> api_fighters_search_oliveira.json
        key = path.lstrip("/").replace("/", "_").replace("?", "-").replace("&", "-").replace("=", "_")
        key = re.sub(r"[^\w\-]", "_", key)
        return f"{key}.json"

    def _get_cached(self, path: str) -> dict | None:
        """Load response from cache if it exists."""
        if self._cache_path is None:
            return None
        cache_file = self._cache_path / self._path_to_cache_key(path)
        if cache_file.exists():
            with open(cache_file, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _set_cached(self, path: str, data: dict) -> None:
        """Save response to cache."""
        if self._cache_path is None:
            return
        cache_file = self._cache_path / self._path_to_cache_key(path)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _get(self, path: str) -> dict:
        """GET request to API. Uses cache when cache_path is set (unless force_refresh)."""
        if not self._skip_cache:
            cached = self._get_cached(path)
            if cached is not None:
                return cached
        url = f"{self.base_url}{path}"
        resp = self._session.get(url)
        resp.raise_for_status()
        data = resp.json()
        self._set_cached(path, data)
        return data

    def load_from_cache(self, path: str) -> dict | None:
        """
        Load raw JSON from cache only. Does not call the API.
        Returns None if not cached. Use for parsing cached data manually.
        Example paths: '/api/fighters?search=oliveira', '/api/fighters/07225ba28ae309b6'
        """
        return self._get_cached(path)

    def search_fighter_by_name(self, name: str) -> list[dict]:
        """Search fighters by name, returns list of fighter objects."""
        data = self._get(f"/api/fighters?search={name}")
        return data.get("data", [])

    def get_fighter_info_df(self, name: str) -> pd.DataFrame:
        """
        Given a fighter's name, return a flat DataFrame with their info.
        Uses first search result. Excludes nested 'fights' for brevity.
        """
        fighters = self.search_fighter_by_name(name)
        if not fighters:
            return pd.DataFrame()
        fighter = fighters[0]
        # Exclude fights from basic info - use get_fighter_fights_df for that
        fighter_info = {k: v for k, v in fighter.items() if k != "fights"}
        fighter_info = self._format_fighter_metrics(fighter_info)
        df = flatten_json_for_df(fighter_info)
        return df

    STAT_LABELS = {
        "slpm": "Significant Strikes Landed per Min",
        "str_acc": "Striking Accuracy (%)",
        "sapm": "Significant Strikes Absorbed per Min",
        "str_def": "Strike Defense (%)",
        "td_avg": "Takedowns per 15 Min",
        "td_acc": "Takedown Accuracy (%)",
        "td_def": "Takedown Defense (%)",
        "sub_avg": "Submission Attempts per 15 Min",
    }

    def compare_fighters_df(self, fighter1_id: str, fighter2_id: str) -> pd.DataFrame:
        """
        Given two fighter IDs, compare fighters and return a flat DataFrame.
        Uses fighter names as column headers and expands stat acronyms.
        """
        data = self._get(
            f"/api/fighters/compare?fighter1={fighter1_id}&fighter2={fighter2_id}"
        )
        f1 = data.get("fighter1", {})
        f2 = data.get("fighter2", {})
        name1 = f1.get("name", "Fighter 1")
        name2 = f2.get("name", "Fighter 2")

        rows = []
        stats = data.get("stats_comparison", {})
        for stat_name, values in stats.items():
            stat_label = self.STAT_LABELS.get(
                stat_name, stat_name.replace("_", " ").title()
            )
            row = {
                "Stat": stat_label,
                name1: values.get("fighter1"),
                name2: values.get("fighter2"),
            }
            rows.append(row)

        advantages = data.get("advantages", {})
        adv1 = ", ".join(advantages.get("fighter1", [])) or None
        adv2 = ", ".join(advantages.get("fighter2", [])) or None
        if adv1 or adv2:
            rows.append({
                "Stat": "Advantages",
                name1: adv1,
                name2: adv2,
            })

        return pd.DataFrame(rows)

    def get_fighter_fights_df(self, fighter_id: str) -> pd.DataFrame:
        """
        Given a fighter ID, get all fights they've been in, return flat DataFrame.
        Each row includes base fight data plus full fight details (totals, significant strikes
        for both fighters). Fight details are cached separately at api_fights_{fight_id}.json.
        """
        data = self._get(f"/api/fighters/{fighter_id}")
        fights = data.get("fights", [])
        if not fights:
            return pd.DataFrame()
        rows = []
        for f in fights:
            flat = _flatten_fight(f)
            flat["fighter_id"] = data.get("id")
            flat["fighter_name"] = data.get("name")
            # Fetch fight details (cached separately at api_fights_{fight_id}.json)
            fight_id = f.get("fight_id")
            if fight_id:
                try:
                    details = self.get_fight_details(fight_id)
                    flat.update(self._flatten_fight_details_to_row(details))
                except requests.HTTPError:
                    pass  # No details available for this fight, keep base data only
            rows.append(flat)
        return pd.DataFrame(rows)

    def _flatten_fight_details_to_row(self, data: dict) -> dict:
        """
        Flatten fight details (totals, significant_strikes) into a dict for merging into a row.
        Ratio fields (X/Y) are expanded to _success, _attempt, _pct for graphing/AI.
        rev = Reversals (position reversals in grappling); ctrl = Control time.
        """
        row = {}
        row["fd_referee"] = data.get("referee")
        row["fd_bout_type"] = data.get("bout_type")
        row["fd_time_format"] = data.get("time_format")

        def _process_value(base_key: str, val) -> None:
            parsed = parse_ratio_value(val)
            if parsed:
                row[f"{base_key}_success"] = parsed["success"]
                row[f"{base_key}_attempt"] = parsed["attempt"]
                row[f"{base_key}"] = parsed["ratio"]
                row[f"{base_key}_pct"] = parsed["pct"]
            else:
                ratio = format_accuracy_value(val) if isinstance(val, str) else val
                row[base_key] = ratio

        for key in ("fighter1", "fighter2"):
            prefix = f"fd_{key}_"
            t = data.get("totals", {}).get(key, {})
            s = data.get("significant_strikes", {}).get(key, {})
            row[f"{prefix}fighter_name"] = t.get("fighter") or s.get("fighter")
            for k, v in t.items():
                if k != "fighter":
                    _process_value(f"{prefix}totals_{k}", v)
            for k, v in s.items():
                if k != "fighter":
                    _process_value(f"{prefix}sig_str_{k}", v)
        return row

    def get_fighter_id(self, name: str) -> Optional[str]:
        """Get fighter ID by name (first search match)."""
        fighters = self.search_fighter_by_name(name)
        return fighters[0]["id"] if fighters else None

    def get_fighter_full(self, fighter_id: str) -> dict:
        """Get full fighter details by ID."""
        return self._get(f"/api/fighters/{fighter_id}")

    def get_fighter_full_df(self, name: str) -> pd.DataFrame:
        """Like get_fighter_info_df but uses full fighter detail (includes more data)."""
        fighters = self.search_fighter_by_name(name)
        if not fighters:
            return pd.DataFrame()
        fid = fighters[0]["id"]
        data = self.get_fighter_full(fid)
        fighter_info = {k: v for k, v in data.items() if k != "fights"}
        fighter_info = self._format_fighter_metrics(fighter_info)
        return flatten_json_for_df(fighter_info)

    def get_fight_details(self, fight_id: str) -> dict:
        """Get fight details with round-by-round stats (includes 'X of Y' style metrics)."""
        return self._get(f"/api/fights/{fight_id}")

    def get_fight_details_df(self, fight_id: str) -> pd.DataFrame:
        """
        Get fight details as flat DataFrame.
        Reformats 'X of Y' metrics to 'X/Y' in totals and significant_strikes.
        """
        data = self.get_fight_details(fight_id)
        return self._flatten_fight_details(data)

    def _flatten_fight_details(self, data: dict) -> pd.DataFrame:
        """Flatten fight details (totals, significant_strikes) into DataFrame."""
        from .utils import format_accuracy_value

        base = {
            "fight_id": data.get("id"),
            "url": data.get("url"),
            "round": data.get("round"),
            "time": data.get("time"),
            "time_format": data.get("time_format"),
            "referee": data.get("referee"),
            "bout_type": data.get("bout_type"),
        }
        totals = data.get("totals", {})
        sig_str = data.get("significant_strikes", {})
        rows = []
        for key in ("fighter1", "fighter2"):
            t = totals.get(key, {})
            s = sig_str.get(key, {})
            row = dict(base)
            row["fighter"] = t.get("fighter") or s.get("fighter")
            for k, v in t.items():
                if k != "fighter":
                    row[f"totals_{k}"] = format_accuracy_value(v) if isinstance(v, str) else v
            for k, v in s.items():
                if k != "fighter":
                    row[f"sig_str_{k}"] = format_accuracy_value(v) if isinstance(v, str) else v
            rows.append(row)
        return pd.DataFrame(rows)

    def get_events_df(self, limit: int = 50) -> pd.DataFrame:
        """
        Get events list as flat DataFrame.
        Nested fights are expanded into event_id, event_name, etc. per fight row.
        """
        data = self._get(f"/api/events?limit={limit}")
        events = data.get("data", [])
        rows = []
        for ev in events:
            ev_id = ev.get("id")
            ev_name = ev.get("name")
            ev_date = ev.get("date")
            ev_loc = ev.get("location")
            ev_url = ev.get("url")
            for fight in ev.get("fights", []):
                row = {
                    "event_id": ev_id,
                    "event_name": ev_name,
                    "event_date": ev_date,
                    "event_location": ev_loc,
                    "event_url": ev_url,
                    "fight_id": fight.get("fight_id"),
                    "fight_url": fight.get("fight_url"),
                    "weight_class": fight.get("weight_class"),
                    "method": fight.get("method"),
                    "round": fight.get("round"),
                    "time": fight.get("time"),
                }
                fighters = fight.get("fighters", [])
                for i, f in enumerate(fighters[:2]):
                    row[f"fighter{i + 1}_name"] = f.get("name") if isinstance(f, dict) else None
                    row[f"fighter{i + 1}_url"] = f.get("url") if isinstance(f, dict) else None
                rows.append(row)
        return pd.DataFrame(rows)

    def get_all_data_for_fighters(
        self,
        fighter1_name: str,
        fighter2_name: str,
        *,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Convenience: get basic info, fights, compare, and combined fights for two fighters.
        Returns dict with keys: fighter1_info, fighter2_info, fighter1_fights, fighter2_fights,
        compare, all_fights

        If force_refresh=True, bypasses cache, fetches fresh data from API, and overwrites
        the cache. Use when data looks stale.
        """
        fetcher = self.force_refresh() if force_refresh else _nullcontext()
        with fetcher:
            id1 = self.get_fighter_id(fighter1_name)
            id2 = self.get_fighter_id(fighter2_name)
            if not id1 or not id2:
                raise ValueError(f"Could not find fighters: {fighter1_name!r}, {fighter2_name!r}")
            f1_fights = normalize_fight_stats(self.get_fighter_fights_df(id1))
            f2_fights = normalize_fight_stats(self.get_fighter_fights_df(id2))
            fighter1_info = self.get_fighter_info_df(fighter1_name)
            fighter2_info = self.get_fighter_info_df(fighter2_name)
            compare = self.compare_fighters_df(id1, id2)
        return {
            "fighter1_info": fighter1_info,
            "fighter2_info": fighter2_info,
            "fighter1_fights": f1_fights,
            "fighter2_fights": f2_fights,
            "compare": compare,
            "all_fights": pd.concat([f1_fights, f2_fights], ignore_index=True),
        }

    def _format_fighter_metrics(self, d: dict) -> dict:
        """Reformat 'X of Y' style metrics to 'X/Y'."""
        out = dict(d)
        for key, val in list(out.items()):
            if isinstance(val, str) and " of " in val.lower():
                out[key] = format_accuracy_value(val)
        return out
