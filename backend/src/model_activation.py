"""Model acceptance, activation and grading-process readiness.

The active model version lives in ``model_registry_state`` and is the single
pointer every grading process resolves when it starts a job.  Activation runs
acceptance against existing vessel features first, then flips the pointer,
retargets still-queued jobs and writes one activation record in a single
transaction -- an interrupted activation therefore either happened completely
or not at all, and retrying the same version is idempotent.
"""

import json
from datetime import datetime, timedelta, timezone
from threading import Lock

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from .database import engine
from .model_registry import DEFAULT_SPEC_KEY, MODEL_SPECS, MissingFeatureError, predict

PROCESS_ALIVE_SECONDS = 30
PROBABILITY_TOLERANCE = 0.01

_schema_lock = Lock()
_schema_ready = False


def _conn_execute(sql, parameters=()):
    with engine.begin() as conn:
        result = conn.exec_driver_sql(sql, parameters)
        return [tuple(row) for row in result] if result.returns_rows else []


def ensure_schema():
    global _schema_ready
    with _schema_lock:
        if _schema_ready:
            return
        _conn_execute("""
            CREATE TABLE IF NOT EXISTS model_registry_state (
                id INTEGER PRIMARY KEY,
                active_spec_key VARCHAR(64) NOT NULL,
                updated_by VARCHAR(160) NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        _conn_execute("""
            CREATE TABLE IF NOT EXISTS model_activations (
                id SERIAL PRIMARY KEY,
                spec_key VARCHAR(64) NOT NULL,
                previous_spec_key VARCHAR(64),
                actor VARCHAR(160) NOT NULL,
                checks JSONB NOT NULL,
                summary JSONB NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        _conn_execute("""
            CREATE TABLE IF NOT EXISTS model_processes (
                process_id VARCHAR(96) PRIMARY KEY,
                kind VARCHAR(24) NOT NULL,
                spec_keys JSONB NOT NULL,
                started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                heartbeat_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Columns added to pre-existing tables; required for persisted volumes.
        _conn_execute("ALTER TABLE delay_cases ADD COLUMN IF NOT EXISTS model_version VARCHAR(64)")
        _conn_execute("ALTER TABLE analysis_jobs ADD COLUMN IF NOT EXISTS model_version VARCHAR(64)")
        _conn_execute(
            "INSERT INTO model_registry_state(id,active_spec_key,updated_by) VALUES(1,%s,%s) "
            "ON CONFLICT (id) DO NOTHING",
            (DEFAULT_SPEC_KEY, "system"),
        )
        _schema_ready = True


def active_spec_key(db: Session) -> str:
    ensure_schema()
    row = db.execute(text("SELECT active_spec_key FROM model_registry_state WHERE id=1")).first()
    return row[0] if row else DEFAULT_SPEC_KEY


def register_process(process_id: str, spec_keys=None):
    ensure_schema()
    keys = sorted(MODEL_SPECS if spec_keys is None else spec_keys)
    _conn_execute("""
        INSERT INTO model_processes(process_id, kind, spec_keys, heartbeat_at)
        VALUES (%s,%s,%s,CURRENT_TIMESTAMP)
        ON CONFLICT (process_id) DO UPDATE
        SET kind=EXCLUDED.kind, spec_keys=EXCLUDED.spec_keys, heartbeat_at=CURRENT_TIMESTAMP
    """, (process_id, "grading", json.dumps(keys)))


def heartbeat(process_id: str, spec_keys=None):
    register_process(process_id, spec_keys)


def unregister_process(process_id: str):
    ensure_schema()
    _conn_execute("DELETE FROM model_processes WHERE process_id=%s", (process_id,))


def live_processes(db: Session, kind: str = "grading"):
    ensure_schema()
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=PROCESS_ALIVE_SECONDS)
    rows = db.execute(text("""
        SELECT process_id, spec_keys, heartbeat_at FROM model_processes
        WHERE kind=:kind AND heartbeat_at >= :cutoff ORDER BY process_id
    """), {"kind": kind, "cutoff": cutoff.replace(tzinfo=None)}).all()
    return [{"process_id": row[0], "spec_keys": list(row[1] or []), "heartbeat_at": row[2]} for row in rows]


def _vessel_feature_samples(db: Session):
    from .models import VesselCall
    from .services import build_features

    vessels = db.query(VesselCall).order_by(VesselCall.id).all()
    return [
        {
            "vessel": vessel.call_no,
            "vessel_name": vessel.vessel_name,
            "features": build_features(db, vessel),
        }
        for vessel in vessels
    ]


def run_acceptance(db: Session, spec_key: str):
    """Validate a candidate against existing vessel features without writing cases."""
    if spec_key not in MODEL_SPECS:
        raise HTTPException(404, "模型版本不存在")
    spec = MODEL_SPECS[spec_key]
    checks = []

    samples = _vessel_feature_samples(db)
    checks.append({
        "name": "存在可验收船次特征",
        "passed": bool(samples),
        "detail": f"读取 {len(samples)} 个船次的现有特征",
    })

    outcomes = []
    levels = set(spec["levels"])
    probability_issues = []
    total = 0
    for sample in samples:
        try:
            result = predict(spec_key, sample["features"])
        except MissingFeatureError as exc:
            probability_issues.append(f"{sample['vessel']} 缺少特征 {exc.missing}")
            continue
        total += 1
        probabilities = result["probabilities"]
        missing_levels = sorted(levels - set(probabilities))
        extra_levels = sorted(set(probabilities) - levels)
        broken = [level for level, value in probabilities.items()
                  if not isinstance(value, (int, float)) or value != value or value < 0]
        probability_sum = round(sum(probabilities.values()), 6)
        if missing_levels or extra_levels or broken or abs(probability_sum - 1) > PROBABILITY_TOLERANCE:
            reasons = []
            if missing_levels:
                reasons.append("缺等级 " + "/".join(missing_levels))
            if extra_levels:
                reasons.append("多等级 " + "/".join(extra_levels))
            if broken:
                reasons.append("非法概率 " + "/".join(broken))
            if abs(probability_sum - 1) > PROBABILITY_TOLERANCE:
                reasons.append(f"概率合计 {probability_sum}")
            probability_issues.append(f"{sample['vessel']} " + "，".join(reasons))
        outcomes.append({
            "vessel": sample["vessel"],
            "label": result["label"],
            "confidence": result["confidence"],
            "probabilities": probabilities,
        })

    checks.append({
        "name": "四级等级概率齐全",
        "passed": total > 0 and not probability_issues,
        "detail": ("全部船次均覆盖 " + "/".join(spec["levels"]) if total > 0 and not probability_issues
                   else ("没有可评估的船次特征" if total == 0
                         else "；".join(probability_issues[:8]))),
    })

    processes = live_processes(db)
    unprepared = [p["process_id"] for p in processes if spec_key not in p["spec_keys"]]
    checks.append({
        "name": "分级进程均已完成准备",
        "passed": bool(processes) and not unprepared,
        "detail": (f"{len(processes)} 个在线分级进程均已装载 {spec_key}" if processes and not unprepared
                   else ("没有在线分级进程" if not processes
                         else "未准备进程：" + "、".join(unprepared))),
    })

    return {
        "spec_key": spec_key,
        "label": spec["label"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "samples": outcomes,
        "sample_count": total,
        "processes": processes,
    }


def registry_view(db: Session):
    ensure_schema()
    active = active_spec_key(db)
    processes = live_processes(db)
    rows = db.execute(text("""
        SELECT id, spec_key, previous_spec_key, actor, created_at
        FROM model_activations ORDER BY id DESC LIMIT 20
    """)).all()
    return {
        "active_spec_key": active,
        "active_label": MODEL_SPECS.get(active, {}).get("label", active),
        "versions": [
            {
                "spec_key": key,
                "label": spec["label"],
                "levels": spec["levels"],
                "feature_count": len(spec["feature_keys"]),
                "active": key == active,
            }
            for key, spec in MODEL_SPECS.items()
        ],
        "processes": processes,
        "activations": [
            {
                "id": row[0], "spec_key": row[1], "previous_spec_key": row[2],
                "actor": row[3], "created_at": row[4].isoformat() if row[4] else None,
            }
            for row in rows
        ],
    }


def activate(db: Session, spec_key: str, user, note: str = ""):
    """Acceptance + atomic switch.  Repeated activation of the live version is a no-op."""
    ensure_schema()
    if spec_key not in MODEL_SPECS:
        raise HTTPException(404, "模型版本不存在")
    # Serialise activations so a retried or concurrent request cannot register twice.
    db.execute(text("SELECT id FROM model_registry_state WHERE id=1 FOR UPDATE"))
    current = active_spec_key(db)
    if current == spec_key:
        db.rollback()
        return {
            "status": "already_active",
            "message": "该版本已是生效模型，无需重复登记",
            "active_spec_key": current,
            "acceptance": None,
        }

    acceptance = run_acceptance(db, spec_key)
    if not acceptance["passed"]:
        db.rollback()
        failed = [check["name"] for check in acceptance["checks"] if not check["passed"]]
        return {
            "status": "rejected",
            "message": "验收未通过，沿用旧版：" + "、".join(failed),
            "active_spec_key": current,
            "acceptance": acceptance,
        }

    actor = getattr(user, "username", None) or str(user)
    summary = {
        "samples": acceptance["sample_count"],
        "processes": len(acceptance["processes"]),
        "levels": MODEL_SPECS[spec_key]["levels"],
    }
    # Single transaction: activation record, pointer flip and queued-job retarget.
    # Running jobs keep the version they already pinned, so they finish on the old model.
    db.execute(text("""
        INSERT INTO model_activations(spec_key, previous_spec_key, actor, checks, summary)
        VALUES (:spec_key, :previous, :actor, CAST(:checks AS JSONB), CAST(:summary AS JSONB))
    """), {
        "spec_key": spec_key,
        "previous": current,
        "actor": actor,
        "checks": json.dumps(acceptance["checks"], ensure_ascii=False),
        "summary": json.dumps(summary, ensure_ascii=False),
    })
    db.execute(text("""
        UPDATE analysis_jobs SET model_version=:spec_key
        WHERE status='queued'
    """), {"spec_key": spec_key})
    db.execute(text("""
        UPDATE model_registry_state SET active_spec_key=:spec_key, updated_by=:actor,
            updated_at=CURRENT_TIMESTAMP WHERE id=1
    """), {"spec_key": spec_key, "actor": actor})
    from .models import AuditLog
    db.add(AuditLog(
        actor=actor, role=getattr(user, "role", "supervisor"),
        action="activate_model_version", entity="model_registry", entity_id=None,
        detail=f"{current} → {spec_key}；{note}".strip("；"),
    ))
    db.commit()
    return {
        "status": "activated",
        "message": "验收通过，所有分级进程已切换到新版",
        "active_spec_key": spec_key,
        "previous_spec_key": current,
        "acceptance": acceptance,
    }
