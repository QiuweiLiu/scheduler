#!/usr/bin/env python3
"""Derive a small, explicit resource profile from C1 calibration output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_plateau(samples: list[dict[str, Any]], threshold_mb: float = 1000.0) -> float | None:
    for sample in samples:
        value = sample.get("memory_used_mb")
        if value is not None and float(value) >= threshold_mb:
            return float(value)
    return None


def build(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = {str(row["case"]): row for row in manifest.get("cases", [])}

    def case_peak(name: str) -> float | None:
        value = cases[name].get("gpu_sampling", {}).get("max_memory_used_mb")
        return float(value) if value is not None else None

    def worker_peak(name: str) -> float | None:
        responses = cases[name].get("measurement", {}).get("responses", [])
        values = [float(row["peak_allocated_mb"]) for row in responses if row.get("peak_allocated_mb") is not None]
        return max(values) if values else None

    def resident(name: str) -> float | None:
        return first_plateau(cases[name].get("gpu_sampling", {}).get("samples", []))

    model_cases = {
        "Qwen3-VL-8B-Instruct": "qwen3_vl_8b_standalone",
        "Qwen2.5-VL-3B-Instruct": "qwen2_5_vl_3b_standalone",
        "Qwen3-4B": "qwen3_4b_text_standalone",
    }
    model_profiles: dict[str, dict[str, Any]] = {}
    for model_id, case_name in model_cases.items():
        model_profiles[model_id] = {
            "resident_model_mb": resident(case_name),
            "standalone_peak_memory_mb": case_peak(case_name),
            "worker_peak_allocated_mb": worker_peak(case_name),
            "resident_rule": "first post-baseline NVML plateau sample >=1000MB",
            "source_case": case_name,
        }

    yolo_case = "yolo11x_batch_1"
    yolo32_case = "yolo11x_batch_32"
    yolo_resident = case_peak(yolo_case)
    yolo_profile = {
        "resident_model_mb": yolo_resident,
        "standalone_peak_memory_mb": case_peak(yolo_case),
        "batch_profiles": {},
        "resident_rule": "conservative batch-1 NVML peak",
        "source_case": yolo_case,
    }
    for batch in (1, 8, 16, 32, 64):
        name = f"yolo11x_batch_{batch}"
        worker = cases[name].get("measurement", {}).get("worker", {})
        peak = case_peak(name)
        yolo_profile["batch_profiles"][str(batch)] = {
            "peak_memory_mb": peak,
            "peak_allocated_mb": worker.get("peak_allocated_mb"),
            "peak_reserved_mb": worker.get("peak_reserved_mb"),
            "active_workspace_mb": max(0.0, float(peak or 0.0) - float(yolo_resident or 0.0)),
            "source_case": name,
        }
    model_profiles["yolo11x.pt"] = yolo_profile

    overlap_profiles: dict[str, dict[str, Any]] = {}
    for name, case in cases.items():
        if name.startswith("mixed_") or name.startswith("two_vlm_"):
            overlap_profiles[name] = {
                "peak_memory_mb": case.get("gpu_sampling", {}).get("max_memory_used_mb"),
                "max_gpu_utilization_percent": case.get("gpu_sampling", {}).get("max_gpu_utilization_percent"),
                "max_power_w": case.get("gpu_sampling", {}).get("max_power_w"),
                "overlap_confirmed": bool(case.get("measurement", {}).get("yolo_resident_during_qwen", False)) or name.startswith("two_vlm_"),
                "source_case": name,
            }

    profile = {
        "schema_version": "c1-resource-profile-v1",
        "profile_type": "single_gpu_contention_calibration",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "gpu": manifest.get("system", {}),
        "repeats": manifest.get("repeats"),
        "controlled_oom_test": bool(manifest.get("controlled_oom_test", False)),
        "model_profiles": model_profiles,
        "overlap_profiles": overlap_profiles,
        "semantics": {
            "resident_model_mb": "conservative measured post-load plateau used for cache admission",
            "active_workspace_mb": "standalone or batch peak minus conservative resident estimate",
            "overlap_peak_memory_mb": "direct NVML peak during confirmed concurrent workers",
            "scope": "single-GPU contention/stress; not a multi-GPU throughput claim",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = build(args.manifest, args.output)
    print(json.dumps({"output": str(args.output), "schema_version": profile["schema_version"], "models": sorted(profile["model_profiles"]), "overlap_cases": len(profile["overlap_profiles"])}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
