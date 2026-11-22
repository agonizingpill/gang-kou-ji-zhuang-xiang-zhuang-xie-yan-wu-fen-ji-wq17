import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from redis import Redis
from sqlalchemy import text

from .database import Base, SessionLocal, engine
from .model_activation import ensure_schema, heartbeat, register_process, unregister_process
from .model_registry import DEFAULT_SPEC_KEY, MODEL_SPECS, MissingFeatureError, build_model, predict
from .models import AnalysisJob, DelayCase

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
SHARED_DIR = Path(os.getenv("SHARED_DIR", "/shared"))
queue = Redis.from_url(REDIS_URL, decode_responses=True)
PROCESS_ID = f"grading-{socket.gethostname()}-{uuid4().hex[:8]}"


def prepare_specs():
    """Build every shipped version; only advertise versions that truly loaded."""
    ready = []
    for key in sorted(MODEL_SPECS):
        try:
            build_model(key)
            ready.append(key)
        except Exception as exc:  # noqa: BLE001 - surfaced through readiness, not crash
            print(f"model {key} not prepared: {exc}", flush=True)
    return ready


def claim_job(db, job_id):
    """Atomically move a queued job to running and pin the model version it will use.

    A job already pinned (e.g. a queued job retargeted by activation) keeps that
    version; a legacy queued job adopts the current active version at claim time.
    """
    row = db.execute(text("""
        UPDATE analysis_jobs SET status='running',
            model_version = COALESCE(model_version,
                (SELECT active_spec_key FROM model_registry_state WHERE id=1))
        WHERE id=:job_id AND status='queued'
        RETURNING model_version
    """), {"job_id": int(job_id)}).first()
    db.commit()
    return row[0] if row else None


def pinned_version(db, job):
    """Resolve the model version this job must use.

    A queued job is claimed atomically and pinned (honouring a version already
    assigned by activation).  A job left ``running`` by an interrupted worker
    resumes on the version it pinned at claim time, which is the old model if
    activation happened afterwards.
    """
    if job.status == "queued":
        return claim_job(db, job.id)
    if job.status == "running":
        # Legacy in-flight jobs predate version pinning: they were graded by the
        # original model, so finish them on that version rather than the new one.
        return job.model_version or DEFAULT_SPEC_KEY
    return None


def validate_result(spec_key, result):
    spec = MODEL_SPECS[spec_key]
    missing = [level for level in spec["levels"] if level not in result["probabilities"]]
    if missing:
        raise ValueError("模型概率缺少等级：" + "、".join(missing))
    total = sum(result["probabilities"].values())
    if abs(total - 1) > 0.01:
        raise ValueError(f"模型概率合计异常：{round(total, 4)}")


def run_job(job_id, ready_specs):
    db = SessionLocal()
    try:
        job = db.get(AnalysisJob, int(job_id))
        if not job:
            return
        spec_key = pinned_version(db, job)
        db.refresh(job)
        if spec_key is None:
            # Not queued any more (already claimed or cancelled); do nothing.
            return
        case = db.get(DelayCase, job.case_id)
        if not case:
            raise ValueError("延误案件不存在")
        if spec_key not in ready_specs:
            raise ValueError(f"当前分级进程未准备模型版本 {spec_key}")
        base = dict(job.input_summary or {})
        result = predict(spec_key, base)
        validate_result(spec_key, result)
        result["model_version"] = spec_key
        result["model_label"] = MODEL_SPECS[spec_key]["label"]
        result["formula"] = MODEL_SPECS[spec_key]["formula"]
        job.result = result
        job.status = "succeeded"
        job.model_version = spec_key
        job.finished_at = datetime.now(timezone.utc)
        case.model_output = result
        case.model_version = spec_key
        case.final_level = result["label"]
        case.status = "classified"
        case.exception_reason = None
        db.commit()
    except MissingFeatureError as exc:
        db.rollback()
        job = db.get(AnalysisJob, int(job_id))
        if job:
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
    except Exception as exc:
        db.rollback()
        job = db.get(AnalysisJob, int(job_id))
        if job:
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


def main():
    Base.metadata.create_all(engine)
    ensure_schema()
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    ready_specs = prepare_specs()
    register_process(PROCESS_ID, ready_specs)
    last_beat = 0.0
    try:
        while True:
            Path(SHARED_DIR / "worker-heartbeat").touch()
            now = time.time()
            if now - last_beat > 5:
                heartbeat(PROCESS_ID, ready_specs)
                last_beat = now
            item = queue.blpop("container_delay_triage_jobs", timeout=3)
            if item:
                run_job(item[1], ready_specs)
    finally:
        unregister_process(PROCESS_ID)


if __name__ == "__main__":
    main()
