from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

try:
    from .folding import (
        CampaignConfig,
        KNOWN_SQUARE_COUNTS,
        chunked,
        count_payload_raw,
        generate_prefixes,
    )
except ImportError:  # pragma: no cover - allows `python app.py` from this folder
    from folding import (  # type: ignore
        CampaignConfig,
        KNOWN_SQUARE_COUNTS,
        chunked,
        count_payload_raw,
        generate_prefixes,
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "mfah.sqlite3"


def make_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MFAH_DB"] = os.environ.get("MFAH_DB", str(DEFAULT_DB))
    app.config["MFAH_LEASE_SECONDS"] = int(os.environ.get("MFAH_LEASE_SECONDS", "300"))
    app.config["MFAH_VERIFY_RESULTS"] = os.environ.get("MFAH_VERIFY_RESULTS", "0") == "1"
    n = int(os.environ.get("MFAH_N", "5"))
    stop_depth_text = os.environ.get("MFAH_STOP_DEPTH")
    stop_depth = int(stop_depth_text) if stop_depth_text else None
    campaign = CampaignConfig(
        n=n,
        prefix_depth=int(os.environ.get("MFAH_PREFIX_DEPTH", "14")),
        stop_depth=stop_depth,
        prefixes_per_unit=int(os.environ.get("MFAH_PREFIXES_PER_UNIT", "64")),
    )
    if campaign.prefix_depth < 3 or campaign.prefix_depth > campaign.n2:
        raise ValueError(f"MFAH_PREFIX_DEPTH must be between 3 and {campaign.n2}")
    if campaign.effective_stop_depth < campaign.prefix_depth or campaign.effective_stop_depth > campaign.n2:
        raise ValueError(
            f"MFAH_STOP_DEPTH must be between {campaign.prefix_depth} and {campaign.n2}"
        )
    app.config["MFAH_CAMPAIGN"] = campaign

    with app.app_context():
        init_db(app)
        seed_campaign(app)

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/config")
    def api_config():
        cfg: CampaignConfig = app.config["MFAH_CAMPAIGN"]
        return jsonify(
            {
                "campaign": cfg.campaign_id,
                "n": cfg.n,
                "n2": cfg.n2,
                "prefixDepth": cfg.prefix_depth,
                "stopDepth": cfg.effective_stop_depth,
                "prefixesPerUnit": cfg.prefixes_per_unit,
                "factor": cfg.factor,
                "isFullSearch": cfg.is_full_search,
                "resultLabel": cfg.result_label,
                "knownAnswer": str(KNOWN_SQUARE_COUNTS.get(cfg.n, "")),
                "leaseSeconds": app.config["MFAH_LEASE_SECONDS"],
            }
        )

    @app.post("/api/client")
    def api_client():
        body = request.get_json(silent=True) or {}
        client_id = clean_id(body.get("clientId")) or uuid.uuid4().hex
        display_name = str(body.get("displayName") or "anonymous")[:80]
        user_agent = str(request.headers.get("User-Agent", ""))[:240]
        now = now_ts()
        with connect(app) as db:
            db.execute(
                """
                INSERT INTO clients(client_id, display_name, user_agent, first_seen, last_seen)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(client_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    user_agent=excluded.user_agent,
                    last_seen=excluded.last_seen
                """,
                (client_id, display_name, user_agent, now, now),
            )
        return jsonify({"clientId": client_id})

    @app.get("/api/stats")
    def api_stats():
        client_id = clean_id(request.args.get("client_id"))
        return jsonify(read_stats(app, client_id))

    @app.post("/api/work")
    def api_work():
        body = request.get_json(silent=True) or {}
        client_id = clean_id(body.get("clientId"))
        if not client_id:
            return jsonify({"error": "clientId is required"}), 400

        lease = lease_work(app, client_id)
        if lease is None:
            return jsonify({"work": None, "stats": read_stats(app, client_id)})
        return jsonify({"work": lease, "stats": read_stats(app, client_id)})

    @app.post("/api/result")
    def api_result():
        body = request.get_json(silent=True) or {}
        client_id = clean_id(body.get("clientId"))
        lease_id = clean_id(body.get("leaseId"))
        work_unit_id = body.get("workUnitId")
        raw_count_text = str(body.get("rawCount", "")).strip()
        elapsed_ms = int(float(body.get("elapsedMs") or 0))
        nodes = int(float(body.get("nodes") or 0))

        if not client_id or not lease_id or work_unit_id is None or not raw_count_text.isdigit():
            return jsonify({"error": "clientId, leaseId, workUnitId, and rawCount are required"}), 400

        raw_count = int(raw_count_text)
        accepted = accept_result(app, client_id, lease_id, int(work_unit_id), raw_count, elapsed_ms, nodes)
        status = 200 if accepted["ok"] else 409
        return jsonify({**accepted, "stats": read_stats(app, client_id)}), status

    @app.post("/api/heartbeat")
    def api_heartbeat():
        body = request.get_json(silent=True) or {}
        client_id = clean_id(body.get("clientId"))
        if client_id:
            with connect(app) as db:
                db.execute("UPDATE clients SET last_seen=? WHERE client_id=?", (now_ts(), client_id))
        return jsonify({"ok": True})

    return app


def connect(app: Flask) -> sqlite3.Connection:
    db = sqlite3.connect(app.config["MFAH_DB"], timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init_db(app: Flask) -> None:
    Path(app.config["MFAH_DB"]).parent.mkdir(parents=True, exist_ok=True)
    with connect(app) as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS work_units (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign TEXT NOT NULL,
                unit_index INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                prefix_count INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                lease_id TEXT,
                client_id TEXT,
                lease_until REAL,
                attempts INTEGER NOT NULL DEFAULT 0,
                raw_count TEXT,
                elapsed_ms INTEGER,
                nodes INTEGER,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(campaign, unit_index)
            );

            CREATE INDEX IF NOT EXISTS idx_work_units_campaign_status
                ON work_units(campaign, status, id);

            CREATE TABLE IF NOT EXISTS clients (
                client_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                user_agent TEXT,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL
            );
            """
        )


def seed_campaign(app: Flask) -> None:
    cfg: CampaignConfig = app.config["MFAH_CAMPAIGN"]
    now = now_ts()
    with connect(app) as db:
        existing = db.execute(
            "SELECT COUNT(*) FROM work_units WHERE campaign=?", (cfg.campaign_id,)
        ).fetchone()[0]
        if existing:
            return

        prefixes = generate_prefixes(cfg.n, cfg.prefix_depth)
        rows = []
        for unit_index, batch in enumerate(chunked(prefixes, cfg.prefixes_per_unit)):
            payload = {
                "campaign": cfg.campaign_id,
                "unitIndex": unit_index,
                "n": cfg.n,
                "depth": cfg.prefix_depth,
                "stopDepth": cfg.effective_stop_depth,
                "factor": cfg.factor,
                "isFullSearch": cfg.is_full_search,
                "prefixes": batch,
            }
            rows.append(
                (
                    cfg.campaign_id,
                    unit_index,
                    json.dumps(payload, separators=(",", ":")),
                    len(batch),
                    now,
                    now,
                )
            )

        db.executemany(
            """
            INSERT INTO work_units(
                campaign, unit_index, payload_json, prefix_count, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def lease_work(app: Flask, client_id: str) -> dict[str, Any] | None:
    cfg: CampaignConfig = app.config["MFAH_CAMPAIGN"]
    now = now_ts()
    lease_until = now + app.config["MFAH_LEASE_SECONDS"]
    lease_id = uuid.uuid4().hex
    with connect(app) as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            """
            UPDATE work_units
            SET status='pending', lease_id=NULL, client_id=NULL, lease_until=NULL, updated_at=?
            WHERE campaign=? AND status='leased' AND lease_until < ?
            """,
            (now, cfg.campaign_id, now),
        )
        row = db.execute(
            """
            SELECT * FROM work_units
            WHERE campaign=? AND status='pending'
            ORDER BY id
            LIMIT 1
            """,
            (cfg.campaign_id,),
        ).fetchone()
        if row is None:
            db.commit()
            return None
        db.execute(
            """
            UPDATE work_units
            SET status='leased', lease_id=?, client_id=?, lease_until=?,
                attempts=attempts + 1, updated_at=?
            WHERE id=?
            """,
            (lease_id, client_id, lease_until, now, row["id"]),
        )
        db.execute("UPDATE clients SET last_seen=? WHERE client_id=?", (now, client_id))
        db.commit()

    payload = json.loads(row["payload_json"])
    return {
        "workUnitId": row["id"],
        "leaseId": lease_id,
        "leaseUntil": lease_until,
        "payload": payload,
    }


def accept_result(
    app: Flask,
    client_id: str,
    lease_id: str,
    work_unit_id: int,
    raw_count: int,
    elapsed_ms: int,
    nodes: int,
) -> dict[str, Any]:
    now = now_ts()
    cfg: CampaignConfig = app.config["MFAH_CAMPAIGN"]
    with connect(app) as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM work_units WHERE id=?", (work_unit_id,)).fetchone()
        if row is None or row["campaign"] != cfg.campaign_id:
            db.commit()
            return {"ok": False, "error": "unknown work unit"}
        if row["status"] == "complete":
            db.commit()
            return {"ok": True, "duplicate": True}
        if row["status"] != "leased" or row["lease_id"] != lease_id or row["client_id"] != client_id:
            db.commit()
            return {"ok": False, "error": "lease is no longer active"}

        error = None
        if app.config["MFAH_VERIFY_RESULTS"]:
            expected = count_payload_raw(json.loads(row["payload_json"]))
            if expected != raw_count:
                error = f"expected {expected}, got {raw_count}"

        if error:
            db.execute(
                """
                UPDATE work_units
                SET status='pending', lease_id=NULL, client_id=NULL, lease_until=NULL,
                    error=?, updated_at=?
                WHERE id=?
                """,
                (error, now, work_unit_id),
            )
            db.commit()
            return {"ok": False, "error": error}

        db.execute(
            """
            UPDATE work_units
            SET status='complete', raw_count=?, elapsed_ms=?, nodes=?,
                lease_until=NULL, updated_at=?
            WHERE id=?
            """,
            (str(raw_count), elapsed_ms, nodes, now, work_unit_id),
        )
        db.execute("UPDATE clients SET last_seen=? WHERE client_id=?", (now, client_id))
        db.commit()

    return {"ok": True, "answerContribution": str(raw_count * cfg.factor)}


def read_stats(app: Flask, client_id: str | None = None) -> dict[str, Any]:
    cfg: CampaignConfig = app.config["MFAH_CAMPAIGN"]
    stale_cutoff = now_ts() - 30
    with connect(app) as db:
        rows = db.execute(
            """
            SELECT status, COUNT(*) AS units, COALESCE(SUM(prefix_count), 0) AS prefixes
            FROM work_units
            WHERE campaign=?
            GROUP BY status
            """,
            (cfg.campaign_id,),
        ).fetchall()
        status = {row["status"]: {"units": row["units"], "prefixes": row["prefixes"]} for row in rows}
        totals = db.execute(
            """
            SELECT COUNT(*) AS units, COALESCE(SUM(prefix_count), 0) AS prefixes
            FROM work_units
            WHERE campaign=?
            """,
            (cfg.campaign_id,),
        ).fetchone()
        complete = db.execute(
            """
            SELECT
                COALESCE(SUM(elapsed_ms), 0) AS elapsed_ms,
                COALESCE(SUM(nodes), 0) AS nodes
            FROM work_units
            WHERE campaign=? AND status='complete'
            """,
            (cfg.campaign_id,),
        ).fetchone()
        complete_raw_rows = db.execute(
            "SELECT raw_count FROM work_units WHERE campaign=? AND status='complete'",
            (cfg.campaign_id,),
        ).fetchall()
        active_clients = db.execute(
            "SELECT COUNT(*) FROM clients WHERE last_seen >= ?", (stale_cutoff,)
        ).fetchone()[0]
        personal = None
        if client_id:
            personal = db.execute(
                """
                SELECT
                    COUNT(*) AS units,
                    COALESCE(SUM(prefix_count), 0) AS prefixes,
                    COALESCE(SUM(elapsed_ms), 0) AS elapsed_ms,
                    COALESCE(SUM(nodes), 0) AS nodes
                FROM work_units
                WHERE campaign=? AND status='complete' AND client_id=?
                """,
                (cfg.campaign_id, client_id),
            ).fetchone()
            personal_raw_rows = db.execute(
                """
                SELECT raw_count
                FROM work_units
                WHERE campaign=? AND status='complete' AND client_id=?
                """,
                (cfg.campaign_id, client_id),
            ).fetchall()
        else:
            personal_raw_rows = []

    raw_done = sum(int(row["raw_count"] or 0) for row in complete_raw_rows)
    answer_done = raw_done * cfg.factor
    total_units = int(totals["units"] or 0)
    completed_units = int(status.get("complete", {}).get("units", 0))
    progress = completed_units / total_units if total_units else 0.0
    known_answer = KNOWN_SQUARE_COUNTS.get(cfg.n)

    return {
        "campaign": cfg.campaign_id,
        "n": cfg.n,
        "prefixDepth": cfg.prefix_depth,
        "stopDepth": cfg.effective_stop_depth,
        "isFullSearch": cfg.is_full_search,
        "resultLabel": cfg.result_label,
        "totalUnits": total_units,
        "totalPrefixes": int(totals["prefixes"] or 0),
        "status": status,
        "progress": progress,
        "activeClients": active_clients,
        "rawCompleted": str(raw_done),
        "answerCompleted": str(answer_done),
        "knownAnswer": str(known_answer or ""),
        "isComplete": completed_units == total_units and total_units > 0,
        "elapsedMs": int(complete["elapsed_ms"] or 0),
        "nodes": int(complete["nodes"] or 0),
        "personal": row_to_personal(personal, personal_raw_rows, cfg.factor) if personal is not None else None,
    }


def row_to_personal(row: sqlite3.Row, raw_rows: list[sqlite3.Row], factor: int) -> dict[str, Any]:
    raw_count = sum(int(raw_row["raw_count"] or 0) for raw_row in raw_rows)
    return {
        "units": int(row["units"] or 0),
        "prefixes": int(row["prefixes"] or 0),
        "rawCount": str(raw_count),
        "answerContribution": str(raw_count * factor),
        "elapsedMs": int(row["elapsed_ms"] or 0),
        "nodes": int(row["nodes"] or 0),
    }


def clean_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    keep = "".join(ch for ch in text if ch.isalnum() or ch in "-_")
    return keep[:96] or None


def now_ts() -> float:
    return time.time()


app = make_app()


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5050")), debug=True)
