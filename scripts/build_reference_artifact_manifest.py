#!/usr/bin/env python3
"""Build the immutable reference-artifact manifest for the F0 comparison runs.

The F0 reference arm loads a base pack plus an overlay (``f0_seed11``).  Those
packs live under ``outputs/`` (not committed), so the formal run must pin their
exact bytes in the repository: this manifest lists every file with its SHA256 and
size, a combined tree hash, and the overlay's own internal provenance fields
(artifact id, producer, checkpoint hash, base pack hash).  The runner verifies
the packs against this manifest before starting and fails closed on any mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PACK = ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"
OVERLAY_PACK = ROOT / "outputs/resource_v2_artifacts/f0_seed11"
OUT_DEFAULT = ROOT / "data/manifests/f0_reference_artifacts_manifest_v1.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe_pack(pack: Path) -> dict:
    files = []
    for path in sorted(pack.rglob("*")):
        if path.is_file():
            files.append({
                "path": str(path.relative_to(pack)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            })
    tree = hashlib.sha256()
    for entry in files:
        tree.update(f"{entry['path']}\0{entry['sha256']}\0{entry['size_bytes']}\n".encode())
    return {
        "pack": str(pack.relative_to(ROOT)).replace("\\", "/"),
        "tree_sha256": tree.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    base = describe_pack(BASE_PACK)
    overlay = describe_pack(OVERLAY_PACK)
    internal = json.loads((OVERLAY_PACK / "resource_v2_manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schema": "f0-reference-artifacts-manifest-v1",
        "note": "byte-level pin of the F0 reference packs used by the main-table formal runs; "
                "the runner verifies these hashes and fails closed on any mismatch",
        "base_pack": base,
        "overlay_pack": overlay,
        "overlay_provenance": {
            key: internal.get(key)
            for key in ("schema_version", "artifact_id", "arm", "seed", "artifact_sha256",
                        "producer", "producer_checkpoint_sha256", "producer_checkpoint",
                        "frozen_j3_sha256", "base_pack_sha256", "bin_spec_sha256",
                        "bin_schema_id")
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out.relative_to(ROOT)),
        "base_tree": base["tree_sha256"],
        "overlay_tree": overlay["tree_sha256"],
        "overlay_artifact_sha256": manifest["overlay_provenance"]["artifact_sha256"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
