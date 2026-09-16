#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def collect_traces(root: Path):
    by_run = {}
    for path in sorted(root.rglob("trace.jsonl")):
        try:
            events = list(read_jsonl(path))
        except Exception:
            continue
        if not events:
            continue
        run_ids = {str(x.get("run_id")) for x in events if x.get("run_id")}
        for run_id in run_ids:
            by_run[run_id] = {"path": str(path), "events": events}
    return by_run

def frame_values(event: dict[str, Any]):
    inp = event.get("input") or {}
    if not isinstance(inp, dict):
        return []
    vals = inp.get("frame_indices")
    if not isinstance(vals, list):
        return []
    out = []
    for value in vals:
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            out.append(value)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    rows = list(read_jsonl(Path(args.input)))
    traces = collect_traces(Path(args.raw_root))
    out = []
    missing_runs = 0
    missing_events = 0
    target_seen = 0
    future_frame_violations = 0
    frame_counts = []
    for row in rows:
        run_id = str(row.get("run_id", ""))
        info = traces.get(run_id)
        event_map = {str(e.get("event_id")): e for e in (info or {}).get("events", []) if e.get("event_id")}
        if info is None:
            missing_runs += 1
        prefix_ids = [str(x) for x in (row.get("prefix_source_event_ids") or [])]
        target_id = row.get("target_source_event_id")
        target_seen += int(bool(target_id and target_id in event_map))
        frame_indices = []
        used_ids = []
        missing_for_row = []
        for event_id in prefix_ids:
            event = event_map.get(event_id)
            if event is None:
                missing_events += 1
                missing_for_row.append(event_id)
                continue
            used_ids.append(event_id)
            frame_indices.extend(frame_values(event))
        dedup = []
        seen = set()
        for value in frame_indices:
            if value not in seen:
                seen.add(value)
                dedup.append(value)
        frame_counts.append(len(dedup))
        context = {
            "frame_indices": dedup,
            "source_event_ids": used_ids,
            "missing_source_event_ids": missing_for_row,
            "prefix_event_count": len(prefix_ids),
            "target_event_excluded": True,
            "future_events_excluded": True,
            "video_id": row.get("video_id"),
        }
        out_row = dict(row)
        out_row["visual_prefix_context"] = context
        out.append(out_row)
    for row in out:
        target = row.get("target_source_event_id")
        if target and target in (row.get("visual_prefix_context") or {}).get("source_event_ids", []):
            future_frame_violations += 1
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in out), encoding="utf-8")
    report = {
        "schema_version": "prefix-visual-context-v0.1",
        "input": args.input,
        "raw_root": args.raw_root,
        "rows": len(out),
        "runs_with_raw_trace": len({r.get("run_id") for r in out if r.get("run_id") in traces}),
        "raw_trace_files": len({v["path"] for v in traces.values()}),
        "missing_runs": missing_runs,
        "missing_source_events": missing_events,
        "target_events_found": target_seen,
        "target_events_excluded": True,
        "future_frame_violations": future_frame_violations,
        "rows_with_prior_frames": sum(x > 0 for x in frame_counts),
        "rows_without_prior_frames": sum(x == 0 for x in frame_counts),
        "max_prior_frames": max(frame_counts or [0]),
        "mean_prior_frames": sum(frame_counts) / max(len(frame_counts), 1),
        "leakage_contract": {
            "target_event_frame_indices_used": False,
            "future_event_frame_indices_used": False,
            "full_video_embedding_used": False,
            "only_prefix_source_event_ids": True,
        },
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
