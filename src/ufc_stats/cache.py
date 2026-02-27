"""Cache backends for UFC Stats API responses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from supabase import Client


class CacheBackend(Protocol):
    """Protocol for cache backends (filesystem, Supabase, etc.)."""

    def get(self, key: str) -> dict | None:
        """Load cached JSON by key. Returns None if not found."""
        ...

    def set(self, key: str, data: dict) -> None:
        """Save JSON to cache."""
        ...


class FilesystemCache:
    """Filesystem-based cache (local or mounted volume)."""

    def __init__(self, cache_path: Path | str):
        self._path = Path(cache_path)
        self._path.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        cache_file = self._path / key
        if not cache_file.exists():
            return None
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    def set(self, key: str, data: dict) -> None:
        cache_file = self._path / key
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class SupabaseStorageCache:
    """Supabase Storage–backed cache for Streamlit Community Cloud / ephemeral envs."""

    def __init__(self, supabase: "Client", bucket: str = "fighter-cache"):
        self._client = supabase
        self._bucket = bucket

    def get(self, key: str) -> dict | None:
        try:
            data = self._client.storage.from_(self._bucket).download(key)
            return json.loads(data.decode("utf-8"))
        except Exception:
            return None

    def set(self, key: str, data: dict) -> None:
        content = json.dumps(data, indent=2).encode("utf-8")
        self._client.storage.from_(self._bucket).upload(
            key,
            content,
            file_options={"content_type": "application/json", "upsert": "true"},
        )
