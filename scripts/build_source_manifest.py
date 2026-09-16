#!/usr/bin/env python3
"""Record hashes for the current tracing source tree without copying code/data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(root: Path) -> dict[str, object]:
    source_root = root / "src" / "tracing"
    if not source_root.is_dir():
        raise SystemExit(f"missing source root: {source_root}")
    files = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name == ".DS_Store":
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "schema_version": "source-sync-registry-v0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": "src/tracing",
        "python_import_name": "tracing",
        "runtime_path_contract": "PYTHONPATH=src",
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
