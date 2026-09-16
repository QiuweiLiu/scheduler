#!/usr/bin/env python3
"""R1-config validator:真正 schema validator(验收二轮修复)
- 对 config_schema.json 与示例 run config 校验字段类型与必填项
产物:results/processed/behavior_nn_v1/tests/config_validator_report.json
"""
import json
import os

D = "/root/autodl-tmp/scheduler/"
OUT = D + "results/processed/behavior_nn_v1/"
os.makedirs(OUT + "tests/", exist_ok=True)

schema = json.load(open(OUT + "data/config_schema.json"))
fields = schema["fields"]

# 示例 run config(R2 的 runs/Bxx 将按此生成)
example = {
    "experiment_id": "B05_gru_s11",
    "timestamp": "2026-08-09T00:00:00Z",
    "host": "autodl-container-41d846924e-d62d0601",
    "environment": {"python": "3.10", "torch": "2.x", "gpu": "RTX 4080 SUPER"},
    "script_sha256": "abc123",
    "dataset_sha256": "def456",
    "split_manifest_sha256": "ghi789",
    "feature_contract_version": "v1",
    "config": {"model": "masked_gru", "seed": 11, "lr": 1e-3, "batch": 64},
    "seed": 11,
    "counts": {"train": 4712, "validation": 615, "test": 562},
    "class_support": {"execute": 5889},
    "best_epoch": 42,
    "selection_metric": "joint",
    "metrics": {"joint": 0.85},
    "predictions": "runs/B05_gru_s11/predictions.jsonl",
    "confusion_matrix": "runs/B05_gru_s11/confusion_matrix.json",
    "runtime": {"wall_time_s": 100.0, "gpu_peak_allocated_mb": 2048,
                "cpu_rss_mb": 1024, "exit_status": "ok"},
    "failure_status": "none",
}
bad = {"experiment_id": 123, "seed": "not-int", "missing_key": True}


def validate(cfg, schema_fields):
    errors = []
    for k, spec in schema_fields.items():
        if spec.get("required") and k not in cfg:
            errors.append(f"missing required: {k}")
            continue
        if k not in cfg:
            continue
        want = spec["type"]
        if want == "object":
            if not isinstance(cfg[k], dict):
                errors.append(f"{k}: expect object")
        elif want == "int":
            if not isinstance(cfg[k], int):
                errors.append(f"{k}: expect int, got {type(cfg[k]).__name__}")
        elif want == "str":
            if not isinstance(cfg[k], str):
                errors.append(f"{k}: expect str, got {type(cfg[k]).__name__}")
    return errors


rep = {
    "example_valid": validate(example, fields),
    "bad_example_errors": validate(bad, fields),
}
rep["validator_works"] = len(rep["example_valid"]) == 0 and len(rep["bad_example_errors"]) >= 3
with open(OUT + "tests/config_validator_report.json", "w") as f:
    json.dump(rep, f, ensure_ascii=False, indent=2)
print("example errors:", rep["example_valid"])
print("bad example errors:", rep["bad_example_errors"])
print("validator works:", rep["validator_works"])
