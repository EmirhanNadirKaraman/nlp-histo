"""
BlacklistManager

Thread-safe JSON-backed set of PMC IDs that previously caused processing
failures.  Persisted to disk after every mutation.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


class BlacklistManager:
    """
    Thread-safe blacklist for failed documents.

    Parameters
    ----------
    path:
        Path to the JSON file.  Created automatically if it does not exist.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or Path("out/failed_pdfs_blacklist.json")
        self._lock = threading.Lock()
        self._entries: set = self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def contains(self, pmcid: str) -> bool:
        with self._lock:
            return pmcid in self._entries

    def add(self, pmcid: str, reason: str = "") -> None:
        with self._lock:
            self._entries.add(pmcid)
            self._persist(reason_hint=f"{pmcid}: {reason}" if reason else pmcid)

    def remove(self, pmcid: str) -> None:
        with self._lock:
            self._entries.discard(pmcid)
            self._persist()

    def all(self) -> frozenset:
        with self._lock:
            return frozenset(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> set:
        if not self._path.exists():
            return set()
        try:
            data = json.loads(self._path.read_text())
            return set(data.get("blacklisted", []))
        except (json.JSONDecodeError, OSError):
            return set()

    def _persist(self, reason_hint: str = "") -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "blacklisted":   sorted(self._entries),
            "last_updated":  datetime.now().isoformat(),
            "last_change":   reason_hint,
        }
        self._path.write_text(json.dumps(data, indent=2))
