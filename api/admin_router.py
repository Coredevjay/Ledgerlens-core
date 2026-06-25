"""Admin REST API for model lifecycle and system configuration (Issue #160)."""

import glob
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_admin_key
from config.settings import settings, _runtime_cache
from detection.model_registry import get_current_version, list_model_versions

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin_key)])

_MODEL_NAMES = ["random_forest", "xgboost", "lightgbm"]


# ---------------------------------------------------------------------------
# GET /admin/models
# ---------------------------------------------------------------------------


@router.get("/models", include_in_schema=False)
def list_models() -> list[dict]:
    """List all versioned model files with active/inactive deployment status."""
    model_dir = settings.model_dir
    result: dict[str, dict] = {}

    for name in _MODEL_NAMES:
        current = get_current_version(name, model_dir)
        try:
            versions = list_model_versions(name, model_dir)
        except (FileNotFoundError, OSError):
            versions = []
        for v in versions:
            key = v
            if key not in result:
                result[key] = {"version": v, "models": [], "active": v == current}
            result[key]["models"].append(name)
            if v == current:
                result[key]["active"] = True

    return list(result.values())


# ---------------------------------------------------------------------------
# POST /admin/models/{version}/promote
# ---------------------------------------------------------------------------


@router.post("/models/{version}/promote", include_in_schema=False)
def promote_model(version: str) -> dict:
    """Promote ``version`` to active for all three model types."""
    model_dir = settings.model_dir
    missing = [
        name
        for name in _MODEL_NAMES
        if not os.path.isfile(os.path.join(model_dir, f"{name}_v{version}.joblib"))
    ]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Model files not found for version {version!r}: {missing}",
        )

    for name in _MODEL_NAMES:
        latest_path = os.path.join(model_dir, f"{name}_latest.txt")
        with open(latest_path, "w") as f:
            f.write(version)

    return {"promoted": version, "models": _MODEL_NAMES}


# ---------------------------------------------------------------------------
# GET /admin/config
# ---------------------------------------------------------------------------


@router.get("/config", include_in_schema=False)
def get_config() -> dict:
    """Return the current runtime configuration from the `runtime_config` table."""
    config: dict = {}
    try:
        with sqlite3.connect(settings.db_path) as conn:
            for key, value in conn.execute("SELECT key, value FROM runtime_config"):
                config[key] = value
    except sqlite3.OperationalError:
        pass
    return config


# ---------------------------------------------------------------------------
# PATCH /admin/config
# ---------------------------------------------------------------------------


class ConfigPatch(BaseModel):
    updates: dict[str, str]


@router.patch("/config", include_in_schema=False)
def patch_config(body: ConfigPatch) -> dict:
    """Persist config key/value updates to SQLite and invalidate the in-process cache."""
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(settings.db_path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS runtime_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        for key, value in body.updates.items():
            conn.execute(
                "INSERT INTO runtime_config (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, now),
            )

    # Invalidate the in-process cache so next load_runtime_config() re-reads from DB
    _runtime_cache["ts"] = 0
    _runtime_cache["config"] = {}

    return {"updated": list(body.updates.keys())}


# ---------------------------------------------------------------------------
# POST /admin/retrain
# ---------------------------------------------------------------------------


def _ensure_retrain_jobs_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS retrain_jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT
        )"""
    )


def _run_retrain(job_id: str) -> None:
    """Background task: run retraining and update job status in SQLite."""
    started_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(settings.db_path) as conn:
        _ensure_retrain_jobs_table(conn)
        conn.execute(
            "INSERT INTO retrain_jobs (job_id, status, started_at) VALUES (?, ?, ?)",
            (job_id, "running", started_at),
        )

    try:
        from detection.model_training import train_models
        from ingestion.synthetic_data import generate_synthetic_trades

        trades = generate_synthetic_trades()
        train_models(trades, model_dir=settings.model_dir)
        status = "completed"
    except Exception:
        status = "failed"

    completed_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(settings.db_path) as conn:
        _ensure_retrain_jobs_table(conn)
        conn.execute(
            "UPDATE retrain_jobs SET status=?, completed_at=? WHERE job_id=?",
            (status, completed_at, job_id),
        )


@router.post("/retrain", include_in_schema=False)
def trigger_retrain(background_tasks: BackgroundTasks) -> dict:
    """Enqueue an async retraining job and return its job ID."""
    job_id = str(uuid.uuid4())
    background_tasks.add_task(_run_retrain, job_id)
    return {"job_id": job_id, "status": "queued"}


# ---------------------------------------------------------------------------
# GET /admin/red-team/latest  (Issue-133)
# ---------------------------------------------------------------------------


@router.get("/red-team/latest", include_in_schema=False)
def get_latest_red_team_report(report_dir: str = "./red_team_reports") -> dict:
    """Return the most recent red-team campaign summary JSON."""
    import json as _json
    report_files = sorted(glob.glob(os.path.join(report_dir, "*.json")))
    if not report_files:
        raise HTTPException(status_code=404, detail="No red-team campaign reports found")
    with open(report_files[-1], "r", encoding="utf-8") as fh:
        return _json.load(fh)


# ---------------------------------------------------------------------------
# GET /admin/drift-reports/latest  (Issue-135)
# ---------------------------------------------------------------------------


@router.get("/drift-reports/latest", include_in_schema=False)
def get_latest_drift_report() -> dict:
    """Return the most recent per-feature PSI drift report."""
    from detection.drift_monitor import DriftMonitor
    monitor = DriftMonitor(db_path=settings.db_path)
    return monitor.get_latest_report()


# ---------------------------------------------------------------------------
# GET /admin/ensemble-weights  (Issue-136)
# GET /admin/label-feedback
# POST /admin/label-feedback
# ---------------------------------------------------------------------------


class LabelFeedbackRequest(BaseModel):
    wallet: str
    asset_pair: str
    true_label: int
    model_scores: dict[str, float]
    ensemble_score: int
    analyst_id: str


@router.get("/ensemble-weights", include_in_schema=False)
def get_ensemble_weights() -> dict:
    """Return current adaptive ensemble weights."""
    from detection.adaptive_reweighter import AdaptiveReweighter
    reweighter = AdaptiveReweighter(db_path=settings.db_path)
    return {"weights": reweighter.current_weights(), "updated_at": datetime.now(timezone.utc).isoformat()}


@router.post("/label-feedback", include_in_schema=False)
def submit_label_feedback(body: LabelFeedbackRequest) -> dict:
    """Submit analyst-confirmed label feedback and update ensemble weights."""
    from detection.adaptive_reweighter import AdaptiveReweighter, LabelFeedback
    if body.true_label not in (0, 1):
        raise HTTPException(status_code=422, detail="true_label must be 0 or 1")
    feedback = LabelFeedback(
        wallet=body.wallet,
        asset_pair=body.asset_pair,
        true_label=body.true_label,
        model_scores=body.model_scores,
        ensemble_score=body.ensemble_score,
        analyst_id=body.analyst_id,
    )
    reweighter = AdaptiveReweighter(db_path=settings.db_path)
    reweighter.record_feedback(feedback)
    reweighter.update_weights()
    return {"status": "accepted", "weights": reweighter.current_weights()}
