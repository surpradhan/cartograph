import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "history.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            query     TEXT    NOT NULL,
            depth     TEXT    NOT NULL,
            model     TEXT    NOT NULL,
            report    TEXT    NOT NULL
        )
    """)
    conn.commit()
    return conn


def save_run(query: str, depth: str, model: str, report: str) -> None:
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO runs (timestamp, query, depth, model, report) VALUES (?,?,?,?,?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M"), query, depth, model, report),
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save run to history: %s", exc)


def load_recent(limit: int = 10) -> list[dict]:
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT id, timestamp, query, depth, model, report "
                "FROM runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"id": r[0], "timestamp": r[1], "query": r[2],
             "depth": r[3], "model": r[4], "report": r[5]}
            for r in rows
        ]
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load history: %s", exc)
        return []


def load_by_id(run_id: int) -> str:
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT report FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return row[0] if row else ""
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load run %d: %s", run_id, exc)
        return ""
