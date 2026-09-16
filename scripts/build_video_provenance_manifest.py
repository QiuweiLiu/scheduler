#!/usr/bin/env python3
"""Build verified video provenance records from source manifests and real files.

The command never invents file metadata.  Missing files, source URLs, archive
entries, probe results, and hash mismatches are reported in a sidecar coverage
report and are not emitted as valid provenance records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VIDEO_PROVENANCE_VERSION = "video-provenance-v0.1"
ARCHIVE_INDEX_VERSION = "videomme-archive-index-v0.1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def video_id(row: dict[str, Any]) -> str:
    value = row.get("video_id") or row.get("source_video_id")
    if value:
        return str(value).strip()
    return Path(str(row.get("video_path") or "")).stem


def merge_sources(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            identifier = video_id(row)
            if not identifier:
                continue
            item = merged.setdefault(identifier, {"video_id": identifier, "source_paths": []})
            item["source_paths"].append(str(path))
            for key in ("dataset", "dataset_revision", "revision", "source_url", "url", "video_path"):
                value = row.get(key)
                if value not in (None, "") and item.get(key) in (None, ""):
                    item[key] = value
            expected = row.get("video_sha256") or row.get("sha256")
            if expected:
                item.setdefault("expected_sha256", set()).add(str(expected).lower())
    return merged


def supplement_sources(merged: dict[str, dict[str, Any]], paths: Iterable[Path]) -> None:
    """Fill metadata for IDs already selected by the primary sources."""

    for path in paths:
        for row in read_jsonl(path):
            identifier = video_id(row)
            if not identifier or identifier not in merged:
                continue
            item = merged[identifier]
            item["source_paths"].append(str(path))
            for key in ("dataset", "dataset_revision", "revision", "source_url", "url", "video_path"):
                value = row.get(key)
                if value not in (None, "") and item.get(key) in (None, ""):
                    item[key] = value
            expected = row.get("video_sha256") or row.get("sha256")
            if expected:
                item.setdefault("expected_sha256", set()).add(str(expected).lower())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    stream = (payload.get("streams") or [{}])[0]
    format_data = payload.get("format") or {}
    width = stream.get("width")
    height = stream.get("height")
    duration = stream.get("duration") or format_data.get("duration")
    if width is None or height is None or duration is None:
        raise ValueError(f"ffprobe did not return width/height/duration for {path}")
    return {"width": int(width), "height": int(height), "duration_s": float(duration)}


def relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_file_index(roots: Iterable[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.mp4"):
            index.setdefault(path.stem, []).append(path)
    for paths in index.values():
        paths.sort()
    return index


def choose_file(
    identifier: str,
    metadata: dict[str, Any],
    file_index: dict[str, list[Path]],
) -> tuple[Path | None, list[str]]:
    candidates: list[Path] = []
    explicit = str(metadata.get("video_path") or "")
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.exists():
            candidates.append(explicit_path)
    for path in file_index.get(identifier, []):
        if path not in candidates:
            candidates.append(path)
    expected = metadata.get("expected_sha256") or set()
    if len(expected) > 1:
        if candidates:
            return candidates[0], ["conflicting_expected_sha256"]
        return None, ["conflicting_expected_sha256", "video_file_missing"]
    if expected:
        expected_hash = next(iter(expected))
        for path in candidates:
            if sha256_file(path) == expected_hash:
                return path, []
        if candidates:
            return None, ["sha256_mismatch"]
    if candidates:
        reasons = ["multiple_candidate_paths"] if len(candidates) > 1 else []
        return candidates[0], reasons
    return None, ["video_file_missing"]


def archive_record(entry: dict[str, Any] | None, repository: str | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    return {
        "repository": str(entry.get("repository") or repository or ""),
        "shard": str(entry["shard"]),
        "member": str(entry.get("member") or entry.get("name") or ""),
        "shard_size_bytes": int(entry.get("shard_size_bytes", entry.get("shard_size"))),
        "compressed_size_bytes": int(entry.get("compressed_size_bytes", entry.get("compressed_size"))),
        "uncompressed_size_bytes": int(entry.get("uncompressed_size_bytes", entry.get("uncompressed_size"))),
        "crc32": int(entry["crc32"]),
        "local_offset": int(entry["local_offset"]),
    }


def derivative_record(path: Path, parent_sha256: str, project_root: Path, variant_id: str, encoder: str, ffprobe: str) -> dict[str, Any]:
    metadata = probe_video(path, ffprobe)
    return {
        "variant_id": variant_id,
        "path": relative_path(path, project_root),
        "parent_sha256": parent_sha256,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        **metadata,
        "encoder": encoder,
    }


def load_archive_index(path: Path | None) -> tuple[str | None, dict[str, dict[str, Any]]]:
    if path is None:
        return None, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries") or payload.get("videos") or {}
    if isinstance(entries, list):
        entries = {str(item["video_id"]): item for item in entries}
    if not isinstance(entries, dict):
        raise ValueError(f"archive index has no object entries: {path}")
    return payload.get("source"), {str(key): value for key, value in entries.items()}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve()
    merged = merge_sources(args.source)
    for item in merged.values():
        if isinstance(item.get("expected_sha256"), set):
            item["expected_sha256"] = set(item["expected_sha256"])
    archive_repository, archive_entries = load_archive_index(args.archive_index)
    file_index = build_file_index([p.resolve() for p in args.video_root])
    derivative_index = build_file_index([p.resolve() for p in args.derivative_root])
    verified_at = datetime.now(timezone.utc).isoformat()
    supplement_sources(merged, args.metadata_source)
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for identifier in sorted(merged):
        metadata = merged[identifier]
        reasons: list[str] = []
        path, path_reasons = choose_file(identifier, metadata, file_index)
        reasons.extend(path_reasons)
        source_url = str(metadata.get("source_url") or metadata.get("url") or "").strip()
        if not source_url:
            reasons.append("source_url_missing")
        archive = archive_record(archive_entries.get(identifier), archive_repository)
        if args.require_archive and archive is None:
            reasons.append("official_archive_entry_missing")
        if path is None:
            missing.append({"video_id": identifier, "reasons": sorted(set(reasons)), "source_paths": sorted(set(metadata["source_paths"]))})
            continue
        try:
            original_sha256 = sha256_file(path)
            expected = metadata.get("expected_sha256") or set()
            if expected and original_sha256 not in expected:
                reasons.append("sha256_mismatch")
            probe = probe_video(path, args.ffprobe)
        except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            reasons.append(f"probe_or_hash_error:{type(exc).__name__}")
            probe = None
            original_sha256 = None
        if original_sha256 is None or "sha256_mismatch" in reasons or probe is None or reasons and any(r.endswith("missing") for r in reasons):
            missing.append({"video_id": identifier, "reasons": sorted(set(reasons)), "path": str(path), "source_paths": sorted(set(metadata["source_paths"]))})
            continue
        if reasons:
            warnings.append({"video_id": identifier, "reasons": sorted(set(reasons)), "path": str(path)})
        derivative_rows: list[dict[str, Any]] = []
        derivative_path = next(iter(derivative_index.get(identifier, [])), None)
        if derivative_path is not None:
            try:
                derivative_rows.append(derivative_record(derivative_path, original_sha256, project_root, args.derivative_variant, args.derivative_encoder, args.ffprobe))
            except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
                reasons.append(f"derivative_probe_error:{type(exc).__name__}")
        record = {
            "schema_version": VIDEO_PROVENANCE_VERSION,
            "collection_id": args.collection_id,
            "video_id": identifier,
            "source": {
                "dataset": str(metadata.get("dataset") or "unknown"),
                "dataset_revision": metadata.get("dataset_revision") or metadata.get("revision") or ("main" if str(metadata.get("dataset")) == "videomme" else None),
                "source_url": source_url,
                "official_archive": archive,
            },
            "original": {
                "path": relative_path(path, project_root),
                "sha256": original_sha256,
                "size_bytes": path.stat().st_size,
                **probe,
            },
            "derivatives": derivative_rows,
            "retrieval": {"retrieved_at": None, "verified_at": verified_at, "method": "existing_file_sha256_ffprobe"},
            "retention": {
                "original_status": "present",
                "archive_path": None,
                "archive_sha256": None,
                "archive_restore_tested": False,
                "deletion_eligible": False,
            },
        }
        records.append(record)
    write_jsonl(args.output, records)
    coverage = {
        "schema_version": "video-provenance-coverage-v0.1",
        "collection_id": args.collection_id,
        "generated_at": verified_at,
        "source_paths": [str(path) for path in args.source],
        "target_unique_video_count": len(merged),
        "valid_record_count": len(records),
        "missing_record_count": len(missing),
        "derivative_record_count": sum(len(row["derivatives"]) for row in records),
        "warning_count": len(warnings),
        "warnings": warnings,
        "missing": missing,
        "output": str(args.output),
    }
    args.coverage_report.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_report.write_text(json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.require_archive and any("official_archive_entry_missing" in row["reasons"] for row in missing):
        raise SystemExit(f"official archive coverage incomplete; see {args.coverage_report}")
    return coverage


def build_archive_index(video_ids: set[str], cache_dir: Path, output: Path, workers: int) -> dict[str, Any]:
    try:
        from download_videomme_subset import REPO_BASE, _index_shards
    except ImportError:
        from scripts.download_videomme_subset import REPO_BASE, _index_shards
    entries = _index_shards(video_ids, cache_dir, max(1, workers))
    payload = {
        "schema_version": ARCHIVE_INDEX_VERSION,
        "source": REPO_BASE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "video_count": len(entries),
        "entries": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"output": str(output), "video_count": len(entries)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--metadata-source", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--coverage-report", required=True, type=Path)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--video-root", action="append", required=True, type=Path)
    parser.add_argument("--derivative-root", action="append", type=Path, default=[])
    parser.add_argument("--derivative-variant", default="derived_640x360_crf23")
    parser.add_argument("--derivative-encoder", default="ffmpeg libx264 preset=fast crf=23; aac 128k")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--archive-index", type=Path)
    parser.add_argument("--require-archive", action="store_true")
    parser.add_argument("--build-archive-index", type=Path)
    parser.add_argument("--archive-cache-dir", type=Path, default=Path("/tmp/videomme_shard_tails"))
    parser.add_argument("--archive-workers", type=int, default=8)
    args = parser.parse_args()
    if args.build_archive_index:
        merged = merge_sources(args.source)
        print(json.dumps(build_archive_index(set(merged), args.archive_cache_dir, args.build_archive_index, args.archive_workers), ensure_ascii=False))
        args.archive_index = args.build_archive_index
    print(json.dumps(build(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
