"""Job store — SQLite-backed persistent queue for long-running scheme conversions.

Single-writer concurrency pattern: one background worker task owns the
database's mutation rights; HTTP request handlers only read and insert.
This sidesteps SQLite's single-writer-at-a-time limitation without any
explicit locking in the app layer.

Schema:
    jobs(
        id              TEXT PRIMARY KEY,      -- uuid4 hex
        created_at      REAL NOT NULL,         -- unix epoch (UTC)
        started_at      REAL,                  -- set when worker picks up
        finished_at     REAL,                  -- set on success or failure
        status          TEXT NOT NULL,         -- queued|running|done|failed
        scheme          TEXT NOT NULL,
        output_mode     TEXT NOT NULL,
        source_filenames TEXT NOT NULL,        -- JSON list of original filenames
        page_count      INTEGER,               -- total pages across all inputs, for ETA
        ocr_expected    INTEGER NOT NULL DEFAULT 0,  -- 0 or 1, for ETA
        estimate_sec    REAL,                  -- seconds, populated at submit
        input_blob      BLOB NOT NULL,         -- pickled dict[str, bytes] of raw inputs
        result_json     TEXT,                  -- JSON-serialised result envelope
        error_detail    TEXT,                  -- non-null on failure
        engine_used     TEXT                   -- docling|heuristic|mixed|none (post-hoc)
    )

Retention: cleanup_expired() drops any row with finished_at older than
RETENTION_SECONDS. Called by a periodic task in the main app.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RETENTION_SECONDS = 7 * 24 * 60 * 60   # 7 days
CLEANUP_INTERVAL_SECONDS = 60 * 60     # run cleanup hourly

# "Small enough to run synchronously" heuristic for the router in main.py.
# Documents at or under this threshold bypass the queue and return inline.
SYNC_PAGE_THRESHOLD = 5
SYNC_SIZE_THRESHOLD_BYTES = 2 * 1024 * 1024   # 2 MB

# Default ETA coefficients. Recalibrated from history automatically by
# recalibrate_eta(). Conservative initial values assume CPU inference.
# ETA = base_seconds + per_native_page * pages_native + per_ocr_page * pages_ocr
ETA_DEFAULTS = {
    "base_seconds": 4.0,        # model warm-up + exporter overhead
    "per_native_page": 12.0,    # native-text page on CPU with layout + TableFormer
    "per_ocr_page": 45.0,       # OCR-heavy page on CPU
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    created_at: float
    status: str                         # queued | running | done | failed
    scheme: str
    output_mode: str
    source_filenames: List[str]
    estimate_sec: Optional[float] = None
    page_count: Optional[int] = None
    ocr_expected: bool = False
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[dict] = None       # only for status=done
    error_detail: Optional[str] = None  # only for status=failed
    engine_used: Optional[str] = None

    def elapsed_sec(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return max(0.0, end - self.started_at)

    def to_public_dict(self, include_result: bool = True) -> dict:
        d = {
            "id": self.id,
            "status": self.status,
            "scheme": self.scheme,
            "output_mode": self.output_mode,
            "source_filenames": self.source_filenames,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "estimate_sec": self.estimate_sec,
            "elapsed_sec": round(self.elapsed_sec(), 1),
            "page_count": self.page_count,
            "ocr_expected": self.ocr_expected,
            "engine_used": self.engine_used,
        }
        if self.status == "failed":
            d["error_detail"] = self.error_detail
        if include_result and self.status == "done":
            d["result"] = self.result
        return d


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class JobStore:
    """Thin wrapper over SQLite. Not thread-safe by itself; callers are
    expected to be either the HTTP request loop (single-threaded via FastAPI
    + asyncio) or the worker task (single-threaded by construction).

    Connections are short-lived per-call — avoids the 'SQLite objects
    created in a thread can only be used in that same thread' footgun when
    FastAPI bounces between threads via run_in_executor.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._init_schema()
        logger.info(f"JobStore initialised at {db_path}")

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA busy_timeout=30000")
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    status TEXT NOT NULL,
                    scheme TEXT NOT NULL,
                    output_mode TEXT NOT NULL,
                    source_filenames TEXT NOT NULL,
                    page_count INTEGER,
                    ocr_expected INTEGER NOT NULL DEFAULT 0,
                    estimate_sec REAL,
                    input_blob BLOB NOT NULL,
                    result_json TEXT,
                    error_detail TEXT,
                    engine_used TEXT
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_finished ON jobs(finished_at)")
            # ETA calibration table
            c.execute("""
                CREATE TABLE IF NOT EXISTS eta_history (
                    job_id TEXT PRIMARY KEY,
                    page_count INTEGER NOT NULL,
                    ocr_expected INTEGER NOT NULL,
                    actual_sec REAL NOT NULL,
                    finished_at REAL NOT NULL
                )
            """)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    def submit(
        self,
        *,
        scheme: str,
        output_mode: str,
        inputs: Dict[str, bytes],
        page_count: Optional[int],
        ocr_expected: bool,
        estimate_sec: Optional[float],
    ) -> Job:
        job_id = uuid.uuid4().hex
        now = time.time()
        filenames = list(inputs.keys())
        blob = pickle.dumps(inputs, protocol=pickle.HIGHEST_PROTOCOL)
        with self._conn() as c:
            c.execute(
                """INSERT INTO jobs
                   (id, created_at, status, scheme, output_mode,
                    source_filenames, page_count, ocr_expected,
                    estimate_sec, input_blob)
                   VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, now, scheme, output_mode,
                 json.dumps(filenames), page_count, int(bool(ocr_expected)),
                 estimate_sec, blob),
            )
        return Job(
            id=job_id, created_at=now, status="queued",
            scheme=scheme, output_mode=output_mode,
            source_filenames=filenames,
            page_count=page_count, ocr_expected=ocr_expected,
            estimate_sec=estimate_sec,
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get(self, job_id: str) -> Optional[Job]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_recent(self, limit: int = 20) -> List[Job]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def queue_position(self, job_id: str) -> Optional[int]:
        """1-based position among queued jobs; None if not queued.
        Position 1 means 'next up' (no queued jobs ahead of this one; a
        currently-running job doesn't count as 'ahead').
        A currently-running job returns 0.
        """
        with self._conn() as c:
            row = c.execute("SELECT status, created_at FROM jobs WHERE id = ?",
                            (job_id,)).fetchone()
            if row is None:
                return None
            if row["status"] == "running":
                return 0
            if row["status"] != "queued":
                return None
            ahead = c.execute(
                "SELECT COUNT(*) FROM jobs WHERE status='queued' AND created_at < ?",
                (row["created_at"],),
            ).fetchone()[0]
            return ahead + 1

    # ------------------------------------------------------------------
    # Worker-side mutations
    # ------------------------------------------------------------------
    def claim_next_queued(self) -> Optional[Job]:
        """Atomically pick the oldest queued job and flip it to running."""
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                row = c.execute(
                    "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
                ).fetchone()
                if row is None:
                    c.execute("COMMIT")
                    return None
                now = time.time()
                c.execute(
                    "UPDATE jobs SET status='running', started_at=? WHERE id=?",
                    (now, row["id"]),
                )
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise
            job = self._row_to_job(row)
            job.status = "running"
            job.started_at = now
            return job

    def load_inputs(self, job_id: str) -> Optional[Dict[str, bytes]]:
        """Fetch the pickled inputs blob for a job."""
        with self._conn() as c:
            row = c.execute("SELECT input_blob FROM jobs WHERE id=?",
                            (job_id,)).fetchone()
        if row is None:
            return None
        return pickle.loads(row["input_blob"])

    def mark_done(self, job_id: str, result: dict, engine_used: Optional[str] = None):
        now = time.time()
        result_json = json.dumps(result)
        with self._conn() as c:
            c.execute(
                """UPDATE jobs
                   SET status='done', finished_at=?, result_json=?, engine_used=?,
                       input_blob=?
                   WHERE id=?""",
                # Drop the input blob on success to reclaim space.
                (now, result_json, engine_used, b"", job_id),
            )
            # Record ETA history
            r = c.execute("SELECT page_count, ocr_expected, started_at FROM jobs WHERE id=?",
                          (job_id,)).fetchone()
            if r and r["started_at"] and r["page_count"]:
                actual = now - r["started_at"]
                c.execute(
                    """INSERT OR REPLACE INTO eta_history
                       (job_id, page_count, ocr_expected, actual_sec, finished_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (job_id, r["page_count"], r["ocr_expected"], actual, now),
                )

    def mark_failed(self, job_id: str, error_detail: str):
        now = time.time()
        with self._conn() as c:
            c.execute(
                """UPDATE jobs SET status='failed', finished_at=?, error_detail=?,
                                   input_blob=?
                   WHERE id=?""",
                (now, error_detail, b"", job_id),
            )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def cleanup_expired(self) -> int:
        cutoff = time.time() - RETENTION_SECONDS
        with self._conn() as c:
            cur = c.execute(
                "DELETE FROM jobs WHERE finished_at IS NOT NULL AND finished_at < ?",
                (cutoff,),
            )
            deleted = cur.rowcount
            c.execute(
                "DELETE FROM eta_history WHERE finished_at < ?", (cutoff,),
            )
        if deleted:
            logger.info(f"cleanup: removed {deleted} expired job(s)")
        return deleted

    def requeue_stuck_running_jobs(self):
        """On startup, any row still marked 'running' is stale — the process
        died mid-job. Flip them to 'failed' so they don't block the worker.
        (Alternative: re-queue; but re-running a job that may have been
        half-complete is error-prone, so fail-safe is better.)
        """
        with self._conn() as c:
            cur = c.execute(
                """UPDATE jobs
                   SET status='failed',
                       finished_at=?,
                       error_detail='Process died while job was running'
                   WHERE status='running'""",
                (time.time(),),
            )
            if cur.rowcount:
                logger.warning(f"startup: failed {cur.rowcount} stuck 'running' job(s)")

    # ------------------------------------------------------------------
    # ETA calibration
    # ------------------------------------------------------------------
    def compute_eta(self, page_count: Optional[int], ocr_expected: bool) -> float:
        """Return seconds estimate for a (page_count, ocr) pair.

        Uses recent history if available (at least 3 similar-type samples
        within the last 30 days). Falls back to ETA_DEFAULTS.
        """
        if not page_count or page_count <= 0:
            return ETA_DEFAULTS["base_seconds"]

        cutoff = time.time() - 30 * 24 * 60 * 60
        with self._conn() as c:
            rows = c.execute(
                """SELECT page_count, actual_sec FROM eta_history
                   WHERE ocr_expected=? AND finished_at >= ?""",
                (int(bool(ocr_expected)), cutoff),
            ).fetchall()
        if len(rows) >= 3:
            # Fit per_page as median of (actual_sec / page_count).
            rates = sorted(r["actual_sec"] / max(1, r["page_count"]) for r in rows)
            per_page = rates[len(rates) // 2]
            return ETA_DEFAULTS["base_seconds"] + per_page * page_count

        per_page = (ETA_DEFAULTS["per_ocr_page"] if ocr_expected
                    else ETA_DEFAULTS["per_native_page"])
        return ETA_DEFAULTS["base_seconds"] + per_page * page_count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _row_to_job(self, row: sqlite3.Row) -> Job:
        result = None
        if row["result_json"]:
            try:
                result = json.loads(row["result_json"])
            except Exception:
                result = None
        return Job(
            id=row["id"],
            created_at=row["created_at"],
            status=row["status"],
            scheme=row["scheme"],
            output_mode=row["output_mode"],
            source_filenames=json.loads(row["source_filenames"]),
            page_count=row["page_count"],
            ocr_expected=bool(row["ocr_expected"]),
            estimate_sec=row["estimate_sec"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            result=result,
            error_detail=row["error_detail"],
            engine_used=row["engine_used"],
        )
