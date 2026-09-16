#!/usr/bin/env python3
"""Add a full-slot sidecar to packed artifacts without changing the main chain.

The packer can emit ``--min-steps 5`` chains, which destroys the implicit
length signal that point consumers rely on (the emitted chain length *is* the
predicted length).  This normalizer restores the original semantics:

* ``future_h{horizon}``      -> truncated to ``predicted_future_length`` (as before)
* ``future_h{horizon}_full`` -> all emitted slots (for length-distribution sampling)

Run once over the --min-steps 5 pack; the scheduler chooses whichever key it
needs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict


def read_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_gz(path: Path, rows) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    args.target.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {"rows": 0, "truncated_rows": 0, "slots_kept": 0, "slots_dropped": 0}
    for path in sorted(args.source.glob("*.jsonl.gz")):
        rows = []
        for row in read_gz(path):
            counts["rows"] += 1
            for key in list(row.keys()):
                if not (isinstance(key, str) and key.startswith("future_h")) or key.endswith("_full"):
                    continue
                scenarios = row.get(key) or []
                if not scenarios:
                    continue
                scenario = scenarios[0]
                steps = scenario.get("steps") or []
                full_key = f"{key}_full"
                row[full_key] = [dict(scenario, steps=None)]
                row[full_key][0]["steps"] = steps
                predicted_length = row.get("predicted_future_length")
                if isinstance(predicted_length, (int, float)):
                    keep = max(0, min(int(predicted_length), len(steps)))
                    if keep < len(steps):
                        counts["truncated_rows"] += 1
                        counts["slots_dropped"] += len(steps) - keep
                    scenario["steps"] = steps[:keep]
                    counts["slots_kept"] += keep
            rows.append(row)
        write_gz(args.target / path.name, rows)
    manifest_src = args.source / "b05_artifact_manifest.json"
    if manifest_src.is_file():
        manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
        files = {}
        for path in sorted(args.target.glob("*.jsonl.gz")):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            files[path.name] = digest.hexdigest()
        manifest["files"] = files
        manifest.setdefault("notes", []).append(
            "full-slot sidecar normalisation: file hashes recomputed for the rewritten gzip payloads"
        )
        (args.target / "b05_artifact_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    (args.target / "full_slot_sidecar.json").write_text(
        json.dumps(
            {
                "schema_version": "future-artifact-full-slot-sidecar-v1",
                "source": str(args.source),
                "counts": counts,
                "note": "future_h* keeps the argmax chain; future_h*_full carries all emitted slots for length sampling",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
