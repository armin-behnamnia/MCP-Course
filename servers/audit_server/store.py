# audit_server/store.py
"""
Append-only storage for audit records.
Each record is one JSON line in audit.jsonl — easy to tail, grep, or ingest.
Swap this class out for SQLite / Postgres without touching server.py.
"""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

AUDIT_FILE = Path(__file__).parent / "audit.jsonl"


class AuditStore:
    """Thread-safe append-only JSON-Lines store."""

    def __init__(self, path: Path = AUDIT_FILE):
        self._path = path
        self._lock = threading.Lock()
        self._path.touch(exist_ok=True)

    # ── write ─────────────────────────────────────────────────────────────────

    def append(self, record: dict) -> str:
        """Add a record; injects id + timestamp; returns the record id."""
        record_id = str(uuid.uuid4())
        record["id"]        = record_id
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        return record_id

    # ── read ──────────────────────────────────────────────────────────────────

    def all(self) -> list[dict]:
        with self._lock:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        return [json.loads(l) for l in lines if l.strip()]

    def by_id(self, record_id: str) -> Optional[dict]:
        return next((r for r in self.all() if r["id"] == record_id), None)

    def by_session(self, session_id: str) -> list[dict]:
        return [r for r in self.all() if r.get("session_id") == session_id]

    def by_tool(self, tool_name: str) -> list[dict]:
        return [r for r in self.all() if r.get("tool_called") == tool_name]

    def recent(self, n: int = 20) -> list[dict]:
        return self.all()[-n:]

    def failures(self) -> list[dict]:
        return [r for r in self.all() if r.get("type") == "tool_trace"
                and not r.get("success", True)]