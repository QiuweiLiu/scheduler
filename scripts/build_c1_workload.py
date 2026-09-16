#!/usr/bin/env python3
"""Rebuild v0.2 templates/episodes with the measured C1 resident profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tracing.workloads import build_workload_v02 as v02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != "c1-resource-profile-v1":
        raise ValueError(f"unsupported profile schema: {profile.get('schema_version')}")
    models = profile.get("model_profiles") or {}
    for model_id, row in models.items():
        resident = row.get("resident_model_mb")
        if resident is not None:
            v02.MODEL_RESIDENT_MB[str(model_id)] = float(resident)
    v02.RESIDENT_PROVENANCE = f"c1_resource_profile_v1:{sha256(path)[:16]}"
    return profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--compute", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--arrival-source", type=Path, required=True)
    parser.add_argument("--arrival-manifest", type=Path, required=True)
    parser.add_argument("--resource-contract", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=20000)
    parser.add_argument("--validation-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=6750)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    template_summary = v02.compile_templates(args.project_root, args.snapshot, args.compute, args.templates)
    arrival_summary = v02.build_arrival_manifest(args.arrival_source, args.arrival_manifest)
    summaries = {
        "train": v02.generate_split(
            args.templates,
            args.train,
            split="train",
            count=args.train_count,
            seed=args.seed,
            arrival_source=args.arrival_source,
            resource_contract=args.resource_contract,
        ),
        "validation": v02.generate_split(
            args.templates,
            args.validation,
            split="validation",
            count=args.validation_count,
            seed=args.seed + 1,
            arrival_source=args.arrival_source,
            resource_contract=args.resource_contract,
        ),
        "test": v02.generate_split(
            args.templates,
            args.test,
            split="test",
            count=args.test_count,
            seed=args.seed + 2,
            arrival_source=args.arrival_source,
            resource_contract=args.resource_contract,
            fixed_test_matrix=True,
        ),
    }
    manifest = {
        "schema_version": "c1-workload-build-v1",
        "profile": str(args.profile),
        "profile_sha256": sha256(args.profile),
        "profile_gpu": profile.get("gpu"),
        "template_summary": template_summary,
        "arrival_summary": arrival_summary,
        "episode_summaries": summaries,
        "inputs": {
            "snapshot": str(args.snapshot),
            "compute": str(args.compute),
            "arrival_source": str(args.arrival_source),
            "resource_contract": str(args.resource_contract),
        },
        "semantics": "measured_trace_templates plus C1 resident cache profile plus train-only resource quantiles plus Alibaba/synthetic arrivals",
    }
    manifest_path = args.templates.parent / "c1_workload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "templates": template_summary.get("templates"), "train": summaries["train"].get("episodes"), "validation": summaries["validation"].get("episodes"), "test": summaries["test"].get("episodes"), "profile_sha256": manifest["profile_sha256"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
