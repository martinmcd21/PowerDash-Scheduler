import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional


CREATE_AUDIT_TABLE = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT,
    candidate_email TEXT,
    hiring_manager_email TEXT,
    recruiter_email TEXT,
    role_title TEXT,
    event_id TEXT,
    payload_json TEXT,
    status TEXT,
    error_message TEXT
);
"""

CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS scheduled_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE,
    candidate_email TEXT,
    hiring_manager_email TEXT,
    recruiter_email TEXT,
    role_title TEXT,
    subject TEXT,
    start_utc TEXT,
    end_utc TEXT,
    timezone TEXT,
    last_status TEXT,
    created_at_utc TEXT
);
"""


class AuditLogger:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(CREATE_AUDIT_TABLE)
            conn.execute(CREATE_EVENTS_TABLE)
            conn.commit()

    def log(
        self,
        action: str,
        status: str,
        actor: Optional[str] = None,
        candidate_email: Optional[str] = None,
        hiring_manager_email: Optional[str] = None,
        recruiter_email: Optional[str] = None,
        role_title: Optional[str] = None,
        event_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        payload_json = json.dumps(payload or {})
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_log (
                    timestamp_utc, action, actor, candidate_email, hiring_manager_email,
                    recruiter_email, role_title, event_id, payload_json, status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    action,
                    actor,
                    candidate_email,
                    hiring_manager_email,
                    recruiter_email,
                    role_title,
                    event_id,
                    payload_json,
                    status,
                    error_message,
                ),
            )
            conn.commit()

    def save_event(
        self,
        event_id: str,
        candidate_email: str,
        hiring_manager_email: str,
        recruiter_email: str,
        role_title: str,
        subject: str,
        start_utc: str,
        end_utc: str,
        timezone: str,
        status: str,
    ) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scheduled_events (
                    event_id, candidate_email, hiring_manager_email, recruiter_email,
                    role_title, subject, start_utc, end_utc, timezone, last_status, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    candidate_email=excluded.candidate_email,
                    hiring_manager_email=excluded.hiring_manager_email,
                    recruiter_email=excluded.recruiter_email,
                    role_title=excluded.role_title,
                    subject=excluded.subject,
                    start_utc=excluded.start_utc,
                    end_utc=excluded.end_utc,
                    timezone=excluded.timezone,
                    last_status=excluded.last_status
                """,
                (
                    event_id,
                    candidate_email,
                    hiring_manager_email,
                    recruiter_email,
                    role_title,
                    subject,
                    start_utc,
                    end_utc,
                    timezone,
                    status,
                    created_at,
                ),
            )
            conn.commit()

    def remove_event(self, event_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM scheduled_events WHERE event_id = ?", (event_id,))
            conn.commit()

    def recent_audit_entries(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT timestamp_utc, action, status, event_id, candidate_email, hiring_manager_email, recruiter_email, role_title, error_message FROM audit_log ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        entries = []
        for row in rows:
            entries.append(
                {
                    "timestamp_utc": row[0],
                    "action": row[1],
                    "status": row[2],
                    "event_id": row[3],
                    "candidate_email": row[4],
                    "hiring_manager_email": row[5],
                    "recruiter_email": row[6],
                    "role_title": row[7],
                    "error_message": row[8],
                }
            )
        return entries

    def scheduled_events(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT event_id, candidate_email, hiring_manager_email, recruiter_email, role_title, subject, start_utc, end_utc, timezone, last_status FROM scheduled_events ORDER BY created_at_utc DESC"
            )
            rows = cur.fetchall()
        events = []
        for row in rows:
            events.append(
                {
                    "event_id": row[0],
                    "candidate_email": row[1],
                    "hiring_manager_email": row[2],
                    "recruiter_email": row[3],
                    "role_title": row[4],
                    "subject": row[5],
                    "start_utc": row[6],
                    "end_utc": row[7],
                    "timezone": row[8],
                    "last_status": row[9],
                }
            )
        return events


def redact_payload(payload: Dict[str, Any], keys_to_redact: Iterable[str] | None = None) -> Dict[str, Any]:
    keys_to_redact = set(keys_to_redact or [])
    redacted: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in keys_to_redact:
            redacted[key] = "[redacted]"
        else:
            redacted[key] = value
    return redacted
