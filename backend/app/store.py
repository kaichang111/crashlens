"""Case files on disk + SQLite for cases, disputes and outcomes (spec §6.3)."""
from __future__ import annotations

import datetime
import json
import sqlite3
import os
import threading
import time
from pathlib import Path

# CRASHLENS_DATA lets you keep case data out of a synced folder (OneDrive locks files mid-write)
DATA = Path(os.environ.get("CRASHLENS_DATA") or Path(__file__).resolve().parent.parent / "data")
CASES = DATA / "cases"
DB_PATH = DATA / "crashlens.db"
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
  id TEXT PRIMARY KEY, evidence_sha256 TEXT, received_at TEXT, location_type TEXT,
  your_vehicle TEXT, status TEXT, stage TEXT, percent REAL, error TEXT, cached INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS disputes (
  id INTEGER PRIMARY KEY, case_id TEXT, fact_id TEXT, actor TEXT,
  old_value TEXT, new_value TEXT, evidence_type TEXT, evidence_sha256 TEXT,
  old_split REAL, new_split REAL, note TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS outcomes (  -- filled in later when the real decision is known
  case_id TEXT PRIMARY KEY, decided_by TEXT,  -- adjuster | arbitration | court
  decided_fault_A REAL, notes TEXT
);
"""


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def db() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def case_dir(cid: str) -> Path:
    d = CASES / cid
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(cid: str, name: str, obj) -> None:
    """Atomic write (tmp + rename). Sync clients like OneDrive briefly lock a file they just saw
    created, which makes the rename fail with 'Access is denied' — retry, then write in place."""
    p = case_dir(cid) / name
    data = json.dumps(obj, indent=1)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(data, encoding="utf-8")
    for attempt in range(20):
        try:
            tmp.replace(p)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    p.write_text(data, encoding="utf-8")
    try:
        tmp.unlink()
    except OSError:
        pass


def read_json(cid: str, name: str):
    p = CASES / cid / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def create_case(cid: str, sha: str, received: str, form: dict, cached: bool) -> None:
    with _lock, db() as con:
        con.execute("INSERT OR REPLACE INTO cases VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (cid, sha, received, form.get("location_type"), form.get("your_vehicle", "A"),
                     "uploaded", "uploaded", 0.0, None, int(cached)))
    write_json(cid, "case.json", {"id": cid, "evidence_sha256": sha, "received_at": received,
                                  "form": form, "cached": cached})


def set_status(cid: str, stage: str, percent: float, status: str = "processing", error: str | None = None):
    with _lock, db() as con:
        con.execute("UPDATE cases SET stage=?, percent=?, status=?, error=? WHERE id=?",
                    (stage, percent, status, error, cid))


def get_case(cid: str) -> dict | None:
    with db() as con:
        r = con.execute("SELECT * FROM cases WHERE id=?", (cid,)).fetchone()
    return dict(r) if r else None


def list_cases() -> list[dict]:
    with db() as con:
        return [dict(r) for r in con.execute("SELECT * FROM cases ORDER BY received_at DESC")]


def add_dispute(cid: str, d: dict) -> int:
    with _lock, db() as con:
        cur = con.execute(
            "INSERT INTO disputes (case_id,fact_id,actor,old_value,new_value,evidence_type,evidence_sha256,"
            "old_split,new_split,note,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (cid, d["fact_id"], d.get("actor"), json.dumps(d.get("old_value")), json.dumps(d["new_value"]),
             d["evidence_type"], d.get("evidence_sha256"), d.get("old_split"), d.get("new_split"),
             d.get("note", ""), now()))
        return int(cur.lastrowid)


def update_dispute_split(did: int, new_split: float | None) -> None:
    with _lock, db() as con:
        con.execute("UPDATE disputes SET new_split=? WHERE id=?", (new_split, did))


def disputes(cid: str) -> list[dict]:
    with db() as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM disputes WHERE case_id=? ORDER BY id", (cid,))]
    for r in rows:
        for k in ("old_value", "new_value"):
            try:
                r[k] = json.loads(r[k]) if r[k] is not None else None
            except json.JSONDecodeError:
                pass
    return rows


def clear_disputes(cid: str) -> None:
    with _lock, db() as con:
        con.execute("DELETE FROM disputes WHERE case_id=?", (cid,))
