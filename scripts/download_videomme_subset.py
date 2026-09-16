#!/usr/bin/env python3
"""Extract selected Video-MME MP4s from official Hugging Face ZIP shards.

The public Video-MME repository stores videos in large ZIP shards.  This
script reads only each shard's ZIP tail (central directory), then downloads
the compressed byte range for requested video IDs and verifies the result by
size and CRC32.  It deliberately does not download or unpack a whole shard.

Network access is intentionally delegated to the system ``curl`` executable;
the caller can run this script in an environment where curl has approved
network access.  No authentication, cookies, or API key is used.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Any, Iterable


REPO_BASE = "https://huggingface.co/datasets/lmms-eval/Video-MME/resolve/main"
TAIL_BYTES = 1_048_576
CHUNK_SIZES = {
    "01": 5_179_377_199,
    "02": 5_326_504_615,
    "03": 4_864_542_112,
    "04": 5_163_884_703,
    "05": 5_095_830_066,
    "06": 5_274_280_690,
    "07": 5_217_070_182,
    "08": 5_275_876_595,
    "09": 5_093_027_546,
    "10": 5_251_346_789,
    "11": 5_123_466_459,
    "12": 5_316_829_715,
    "13": 5_295_651_815,
    "14": 5_291_514_653,
    "15": 5_283_263_670,
    "16": 5_105_597_041,
    "17": 5_277_591_301,
    "18": 5_343_301_110,
    "19": 4_973_938_854,
    "20": 2_241_427_390,
}

_CENTRAL_FMT = "<4s6H3I5H2I"
_LOCAL_FMT = "<4s5H3I2H"


def _curl_range(url: str, start: int, end: int, output: Path) -> None:
    expected = end - start + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl",
        "--ipv4",
        "--http1.1",
        "-sS",
        "-L",
        "--retry",
        "5",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--max-time",
        "600",
        "-r",
        f"{start}-{end}",
        "-o",
        str(output),
        url,
    ]
    subprocess.run(command, check=True)
    actual = output.stat().st_size
    if actual != expected:
        raise RuntimeError(f"range size mismatch for {url}: expected {expected}, got {actual}")


def _download_range_parallel(
    url: str,
    start: int,
    end: int,
    output: Path,
    temp_dir: Path,
    workers: int,
    segment_bytes: int,
) -> None:
    """Download one member using independent ranges and join them in order."""

    segments: list[tuple[int, int, Path]] = []
    cursor = start
    part_index = 0
    while cursor <= end:
        part_end = min(end, cursor + segment_bytes - 1)
        part_path = temp_dir / f"{output.name}.part{part_index:05d}"
        segments.append((cursor, part_end, part_path))
        cursor = part_end + 1
        part_index += 1

    def fetch(segment: tuple[int, int, Path]) -> None:
        _curl_range(url, segment[0], segment[1], segment[2])

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(fetch, segments))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        for _, _, part_path in segments:
            with part_path.open("rb") as part:
                shutil.copyfileobj(part, handle)
            part_path.unlink()


def _zip64_values(extra: bytes, fields: list[int]) -> list[int]:
    data: bytes | None = None
    pos = 0
    while pos + 4 <= len(extra):
        field_id, length = struct.unpack_from("<HH", extra, pos)
        part = extra[pos + 4 : pos + 4 + length]
        pos += 4 + length
        if field_id == 1:
            data = part
            break
    result: list[int] = []
    cursor = 0
    for field in fields:
        if field == 0xFFFFFFFF:
            if data is None or cursor + 8 > len(data):
                raise RuntimeError("ZIP64 extra field is missing a required value")
            result.append(struct.unpack_from("<Q", data, cursor)[0])
            cursor += 8
        else:
            result.append(field)
    return result


def _read_central_entries(tail: bytes, shard_size: int) -> list[dict[str, Any]]:
    base = shard_size - len(tail)
    eocd_pos = tail.rfind(b"PK\x06\x06")
    if eocd_pos < 0:
        raise RuntimeError("ZIP64 end-of-central-directory record not found")
    record = struct.unpack_from("<4sQ2H2I2Q2Q", tail, eocd_pos)
    central_size, central_offset = record[-2:]
    start = central_offset - base
    end = start + central_size
    if start < 0 or end > len(tail):
        raise RuntimeError("central directory is not fully contained in downloaded tail")

    entries: list[dict[str, Any]] = []
    pos = start
    while pos < end:
        values = struct.unpack_from(_CENTRAL_FMT, tail, pos)
        if values[0] != b"PK\x01\x02":
            raise RuntimeError(f"invalid central-directory signature at offset {pos}")
        name_length, extra_length, comment_length = values[10:13]
        name_start = pos + 46
        name = tail[name_start : name_start + name_length].decode("utf-8")
        extra_start = name_start + name_length
        extra = tail[extra_start : extra_start + extra_length]
        uncompressed_size, compressed_size, local_offset = _zip64_values(
            extra, [values[9], values[8], values[16]]
        )
        entries.append(
            {
                "name": name,
                "video_id": Path(name).stem,
                "compression": values[4],
                "crc32": values[7],
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_offset": local_offset,
            }
        )
        pos += 46 + name_length + extra_length + comment_length
    return entries


def _index_shards(video_ids: set[str], cache_dir: Path, workers: int) -> dict[str, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, Path]] = []
    for shard, size in CHUNK_SIZES.items():
        tail_path = cache_dir / f"tail_{shard}.bin"
        tasks.append((shard, tail_path))

    def fetch(item: tuple[str, Path]) -> tuple[str, list[dict[str, Any]]]:
        shard, tail_path = item
        if not tail_path.exists() or tail_path.stat().st_size != TAIL_BYTES:
            size = CHUNK_SIZES[shard]
            _curl_range(
                f"{REPO_BASE}/videos_chunked_{shard}.zip?download=true",
                size - TAIL_BYTES,
                size - 1,
                tail_path,
            )
        return shard, _read_central_entries(tail_path.read_bytes(), CHUNK_SIZES[shard])

    matches: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for shard, entries in pool.map(fetch, tasks):
            for entry in entries:
                if entry["video_id"] in video_ids:
                    entry = dict(entry)
                    entry["shard"] = shard
                    entry["shard_size"] = CHUNK_SIZES[shard]
                    matches[entry["video_id"]] = entry
    missing = sorted(video_ids - matches.keys())
    if missing:
        raise RuntimeError(f"requested video IDs are absent from official shards: {missing}")
    return matches


def _download_one(
    entry: dict[str, Any],
    output_dir: Path,
    temp_dir: Path,
    *,
    force: bool,
    segment_workers: int,
    segment_bytes: int,
) -> dict[str, Any]:
    video_id = str(entry["video_id"])
    output_path = output_dir / f"{video_id}.mp4"
    if output_path.exists() and not force:
        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
        if output_path.stat().st_size == int(entry["uncompressed_size"]):
            return {**entry, "output_path": str(output_path), "sha256": digest, "skipped": True}

    # A local ZIP header is tiny; its filename/extra lengths determine the
    # first byte of the compressed member payload.
    header_path = temp_dir / f"{video_id}.local-header.bin"
    # Keep the original Hugging Face URL for every range request.  In the
    # current local-proxy environment, reusing the short-lived redirected CDN
    # URL causes intermittent SSL failures, while curl following this URL's
    # redirect succeeds and preserves the same byte-range semantics.
    shard_url = f"{REPO_BASE}/videos_chunked_{entry['shard']}.zip?download=true"
    _curl_range(
        shard_url,
        int(entry["local_offset"]),
        int(entry["local_offset"]) + 4095,
        header_path,
    )
    local = struct.unpack_from(_LOCAL_FMT, header_path.read_bytes(), 0)
    if local[0] != b"PK\x03\x04":
        raise RuntimeError(f"invalid local header for {video_id}")
    filename_length, extra_length = local[9:11]
    data_start = int(entry["local_offset"]) + 30 + filename_length + extra_length
    compressed_end = data_start + int(entry["compressed_size"]) - 1
    compressed_path = temp_dir / f"{video_id}.compressed"
    _download_range_parallel(
        shard_url,
        data_start,
        compressed_end,
        compressed_path,
        temp_dir,
        segment_workers,
        segment_bytes,
    )
    payload = compressed_path.read_bytes()
    if int(entry["compression"]) == 0:
        data = payload
    elif int(entry["compression"]) == 8:
        data = zlib.decompress(payload, -15)
    else:
        raise RuntimeError(f"unsupported ZIP compression {entry['compression']} for {video_id}")
    if len(data) != int(entry["uncompressed_size"]):
        raise RuntimeError(f"uncompressed size mismatch for {video_id}")
    crc32 = zlib.crc32(data) & 0xFFFFFFFF
    if crc32 != int(entry["crc32"]):
        raise RuntimeError(f"CRC32 mismatch for {video_id}: {crc32:#x} != {entry['crc32']:#x}")

    part_path = output_path.with_suffix(".mp4.part")
    part_path.write_bytes(data)
    part_path.replace(output_path)
    return {
        **entry,
        "output_path": str(output_path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "skipped": False,
    }


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"manifest line {line_number} is not an object")
        video_path = str(item.get("video_path") or "")
        video_id = str(item.get("video_id") or Path(video_path).stem)
        if not video_id:
            raise ValueError(f"manifest line {line_number} has no video_id/video_path")
        item["video_id"] = video_id
        records.append(item)
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/videomme_shard_tails"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--segment-workers", type=int, default=8)
    parser.add_argument("--segment-bytes", type=int, default=1_048_576)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    records = _read_manifest(args.manifest)
    video_ids = {str(item["video_id"]) for item in records}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matches = _index_shards(video_ids, args.cache_dir, max(1, int(args.workers)))
    temp_dir = Path(tempfile.mkdtemp(prefix="videomme_extract_"))
    try:
        results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = [
                pool.submit(
                    _download_one,
                    matches[video_id],
                    args.output_dir,
                    temp_dir,
                    force=args.force,
                    segment_workers=max(1, int(args.segment_workers)),
                    segment_bytes=max(65_536, int(args.segment_bytes)),
                )
                for video_id in sorted(video_ids)
            ]
            for future in futures:
                results.append(future.result())
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    report_path = args.report or args.output_dir.parent / "videomme_subset_download_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "source": REPO_BASE,
                "requested_video_count": len(video_ids),
                "downloaded_video_count": sum(not item["skipped"] for item in results),
                "skipped_video_count": sum(item["skipped"] for item in results),
                "total_bytes": sum(int(item["uncompressed_size"]) for item in results),
                "videos": sorted(results, key=lambda item: item["video_id"]),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"videos": len(results), "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
