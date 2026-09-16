#!/usr/bin/env python3
"""Derive a topology-compatible episode subset for a measured GPU profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_rows(rows: list[dict[str, Any]], topology_mb: list[float]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        actual = [float(value) for value in row.get("gpu_topology_mb", [])]
        if actual == topology_mb:
            selected.append(row)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--topology-mb", type=float, nargs="+", required=True)
    args = parser.parse_args()

    source_bytes = args.input.read_bytes()
    rows = [json.loads(line) for line in source_bytes.decode().splitlines() if line.strip()]
    selected = select_rows(rows, args.topology_mb)
    if not selected:
        raise SystemExit(f"no episodes match gpu_topology_mb={args.topology_mb}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in selected)
    output_bytes = output_text.encode()
    args.output.write_bytes(output_bytes)
    meta = {
        "schema_version": "transition-profile-episode-filter-v0.1",
        "source_path": str(args.input),
        "source_sha256": sha256_bytes(source_bytes),
        "source_episode_count": len(rows),
        "filter": {"gpu_topology_mb": args.topology_mb},
        "selected_episode_count": len(selected),
        "selected_episode_ids": [row.get("episode_id") for row in selected],
        "output_path": str(args.output),
        "output_sha256": sha256_bytes(output_bytes),
    }
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
