"""Versioned delay-triage model definitions.

A model version is identified by a spec key.  Every worker build ships the
spec keys it can execute, which lets acceptance verify that *all* live grading
processes are prepared before the supervisor switches the registry pointer.
"""

from threading import Lock

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_FEATURE_KEYS = [
    "container_count", "avg_gate_dwell_min", "max_gate_dwell_min",
    "reefer_ratio", "hazmat_ratio", "customs_open_count",
    "gate_event_count", "dry_count", "reefer_count", "hazmat_count",
    "port_call_hour", "window_minutes",
]
DERIVED_FEATURE_KEYS = ["dwell_x_reefer", "gate_x_count"]
FEATURE_KEYS = BASE_FEATURE_KEYS + DERIVED_FEATURE_KEYS
LEVELS = ["normal", "watch", "major", "critical"]
DEFAULT_SPEC_KEY = "delay-triage-lr-v1"


def complete_features(base):
    """Add the manual cross features shared by every supported version."""
    features = {key: base[key] for key in BASE_FEATURE_KEYS if key in base}
    features["dwell_x_reefer"] = round(
        float(base.get("avg_gate_dwell_min", 0)) * float(base.get("reefer_ratio", 0)), 3
    )
    features["gate_x_count"] = (
        float(base.get("gate_event_count", 0)) * float(base.get("container_count", 0))
    )
    return features


class MissingFeatureError(ValueError):
    def __init__(self, missing):
        self.missing = list(missing)
        super().__init__("缺少特征：" + "、".join(self.missing))


def feature_vector(spec, base):
    features = complete_features(base)
    missing = [key for key in spec["feature_keys"] if key not in features]
    if missing:
        raise MissingFeatureError(missing)
    vector = np.array([[float(features[key]) for key in spec["feature_keys"]]], dtype=float)
    return features, vector


def _build_pipeline(spec):
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=500,
            random_state=spec["seed"],
            class_weight=spec.get("class_weight"),
        )),
    ])
    rows = spec["training_rows"]
    features = []
    labels = []
    for values, label in rows:
        base = dict(zip(BASE_FEATURE_KEYS, values))
        features.append([complete_features(base)[key] for key in spec["feature_keys"]])
        labels.append(label)
    pipeline.fit(features, labels)
    return pipeline


_model_cache = {}
_cache_lock = Lock()


def build_model(spec_key):
    """Build (and memoise) the sklearn pipeline for a registered spec key."""
    spec = MODEL_SPECS[spec_key]
    with _cache_lock:
        if spec_key not in _model_cache:
            _model_cache[spec_key] = _build_pipeline(spec)
        return _model_cache[spec_key]


def predict(spec_key, base):
    spec = MODEL_SPECS[spec_key]
    pipeline = build_model(spec_key)
    features, vector = feature_vector(spec, base)
    probabilities = pipeline.predict_proba(vector)[0]
    classes = pipeline.named_steps["lr"].classes_
    prob_map = {str(level): round(float(p), 4) for level, p in zip(classes, probabilities)}
    label = max(prob_map, key=prob_map.get)
    return {
        "label": label,
        "confidence": prob_map[label],
        "probabilities": prob_map,
        "features": features,
        "feature_keys": list(spec["feature_keys"]),
    }


MODEL_SPECS = {
    "delay-triage-lr-v1": {
        "label": "延误分级 LR v1（现行）",
        "pipeline": "standard-scaler-logistic",
        "seed": 42,
        "levels": list(LEVELS),
        "feature_keys": list(FEATURE_KEYS),
        "formula": "手工交叉特征 = avg_gate_dwell_min×reefer_ratio, gate_event_count×container_count；StandardScaler + LogisticRegression(random_state=42)",
        "training_rows": [
            ([4, 8, 12, .0, .0, 0, 4, 4, 0, 0, 6, 480], "normal"),
            ([6, 16, 22, .17, .0, 0, 6, 5, 1, 0, 8, 420], "watch"),
            ([8, 28, 40, .25, .12, 0, 8, 5, 2, 1, 10, 360], "major"),
            ([9, 42, 66, .33, .22, 1, 9, 4, 3, 2, 13, 300], "critical"),
            ([5, 10, 18, .2, .0, 0, 5, 4, 1, 0, 7, 520], "normal"),
            ([7, 22, 31, .29, .0, 0, 7, 4, 2, 0, 9, 390], "watch"),
            ([10, 35, 51, .3, .1, 0, 10, 6, 3, 1, 11, 330], "major"),
            ([12, 50, 78, .42, .17, 0, 12, 6, 5, 2, 14, 280], "critical"),
            ([3, 7, 10, .0, .0, 0, 3, 3, 0, 0, 5, 600], "normal"),
            ([8, 25, 36, .38, .0, 0, 8, 5, 3, 0, 12, 360], "major"),
            ([5, 18, 24, .2, .2, 0, 5, 3, 1, 1, 9, 440], "watch"),
            ([11, 45, 70, .45, .25, 0, 11, 4, 5, 2, 15, 260], "critical"),
        ],
    },
    "delay-triage-lr-v2": {
        "label": "延误分级 LR v2（新版）",
        "pipeline": "standard-scaler-logistic",
        "seed": 2026,
        "levels": list(LEVELS),
        "feature_keys": list(FEATURE_KEYS),
        "formula": "沿用 14 项手工交叉特征；扩充分级样本并重新标定 StandardScaler + LogisticRegression(random_state=2026)",
        "training_rows": [
            ([4, 8, 12, .0, .0, 0, 4, 4, 0, 0, 6, 480], "normal"),
            ([3, 6, 9, .0, .0, 0, 3, 3, 0, 0, 4, 620], "normal"),
            ([5, 11, 17, .2, .0, 0, 5, 4, 1, 0, 7, 500], "normal"),
            ([4, 9, 14, .25, .0, 0, 4, 3, 1, 0, 8, 470], "normal"),
            ([2, 5, 8, .0, .0, 0, 2, 2, 0, 0, 3, 650], "normal"),
            ([6, 16, 22, .17, .0, 0, 6, 5, 1, 0, 8, 420], "watch"),
            ([6, 15, 21, .17, .0, 0, 6, 5, 1, 0, 8, 430], "watch"),
            ([7, 19, 27, .14, .14, 1, 7, 4, 1, 1, 9, 400], "watch"),
            ([7, 22, 31, .29, .0, 0, 7, 4, 2, 0, 9, 390], "watch"),
            ([5, 18, 24, .2, .2, 0, 5, 3, 1, 1, 9, 440], "watch"),
            ([8, 28, 40, .25, .12, 0, 8, 5, 2, 1, 10, 360], "major"),
            ([9, 30, 44, .22, .11, 0, 9, 5, 2, 1, 11, 340], "major"),
            ([10, 35, 51, .3, .1, 0, 10, 6, 3, 1, 11, 330], "major"),
            ([10, 33, 48, .3, .1, 0, 10, 6, 3, 1, 12, 320], "major"),
            ([8, 25, 36, .38, .0, 0, 8, 5, 3, 0, 12, 360], "major"),
            ([9, 42, 66, .33, .22, 1, 9, 4, 3, 2, 13, 300], "critical"),
            ([12, 50, 78, .42, .17, 0, 12, 6, 5, 2, 14, 280], "critical"),
            ([13, 55, 84, .38, .23, 1, 13, 5, 6, 3, 16, 250], "critical"),
            ([12, 48, 74, .42, .17, 0, 12, 6, 5, 2, 15, 270], "critical"),
            ([11, 45, 70, .45, .25, 0, 11, 4, 5, 2, 15, 260], "critical"),
        ],
    },
}
