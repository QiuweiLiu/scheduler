#!/usr/bin/env python3
"""Run a bounded Phase 2 manifest through the Phase 1 VideoTool adapter.

The manifest is JSONL so each public benchmark record, repetition, and
baseline is explicit.  Repetitions are labelled as such; they are not claimed
to be new benchmark questions.  This is intentional when a public media
subset is smaller than the Phase 2 preliminary sample target.
"""

from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable

from tracing.collectors.videotool_phase1 import DEFAULT_OPTIONS, run_one


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"manifest line {line_number} is not an object")
        agent_input = payload.get("agent_input")
        if isinstance(agent_input, dict):
            payload.setdefault("question", agent_input.get("question"))
            payload.setdefault("options", agent_input.get("options"))
        for key in ("task_id", "dataset", "video_path", "question"):
            if not payload.get(key):
                raise ValueError(f"manifest line {line_number} missing {key}")
        payload.setdefault("options", DEFAULT_OPTIONS)
        payload.setdefault("baselines", ["st_fixed", "star", "langgraph_react"])
        payload.setdefault("repetitions", 1)
        records.append(payload)
    return records


def _args_for_record(args: argparse.Namespace, record: dict[str, Any], task_id: str) -> Namespace:
    options = record.get("options", DEFAULT_OPTIONS)
    if isinstance(options, list):
        options = "\n".join(f"({chr(65 + index)}) {item}" for index, item in enumerate(options))
    return Namespace(
        video_path=Path(str(record["video_path"])),
        output_root=args.output_root,
        videotool_root=args.videotool_root,
        question=str(record["question"]),
        options=str(options),
        dataset=str(record["dataset"]),
        task_id=task_id,
        model_name=str(record.get("model_name", args.model_name)),
        planner_mode=str(record.get("planner_mode", args.planner_mode)),
        max_iterations=args.max_iterations,
        yolo_model=_optional_path(record.get("yolo_model", args.yolo_model)),
        yolo_python=record.get("yolo_python", args.yolo_python),
        yolo_batch=int(record.get("yolo_batch", args.yolo_batch)),
        yolo_preobserve=bool(record.get("yolo_preobserve", args.yolo_preobserve)),
        qwen_model=_optional_path(record.get("qwen_model", args.qwen_model)),
        qwen_python=record.get("qwen_python", args.qwen_python),
        qwen_max_new_tokens=int(record.get("qwen_max_new_tokens", args.qwen_max_new_tokens)),
        planner_model=_optional_path(record.get("planner_model", args.planner_model)),
        planner_python=record.get("planner_python", args.planner_python),
        planner_max_new_tokens=int(record.get("planner_max_new_tokens", args.planner_max_new_tokens)),
        planner_constrained_json=bool(
            record.get("planner_constrained_json", args.planner_constrained_json)
        ),
        answer_model=_optional_path(record.get("answer_model", args.answer_model)),
        answer_python=record.get("answer_python", args.answer_python),
        answer_max_new_tokens=int(record.get("answer_max_new_tokens", args.answer_max_new_tokens)),
        model_stack_id=str(record.get("model_stack_id", args.model_stack_id)),
        planner_model_id=record.get("planner_model_id", args.planner_model_id),
        visual_model_id=record.get("visual_model_id", args.visual_model_id),
        answer_model_id=record.get("answer_model_id", args.answer_model_id),
        detector_model_id=record.get("detector_model_id", args.detector_model_id),
        task_structure=record.get("task_structure"),
    )


def run_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = _read_manifest(args.manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "phase2_batch_summary.jsonl"
    completed: set[tuple[str, str, int]] = set()
    if getattr(args, "resume", False) and summary_path.is_file():
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("status") == "success":
                completed.add(
                    (
                        str(item.get("source_task_id")),
                        str(item.get("baseline")),
                        int(item.get("repetition", 0)),
                    )
                )
    summaries: list[dict[str, Any]] = []
    with summary_path.open("a", encoding="utf-8") as summary_file:
        for record in records:
            repetitions = max(1, int(record.get("repetitions", 1)))
            baselines = [str(item) for item in record.get("baselines", [])]
            for repetition in range(1, repetitions + 1):
                for baseline in baselines:
                    key = (str(record["task_id"]), baseline, repetition)
                    if key in completed:
                        continue
                    task_id = f"{record['task_id']}_r{repetition:02d}"
                    run_args = _args_for_record(args, record, task_id)
                    started = time.perf_counter()
                    run_dir = run_one(run_args, baseline)
                    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
                    item = {
                        "task_id": task_id,
                        "source_task_id": record["task_id"],
                        "dataset": record["dataset"],
                        "baseline": baseline,
                        "repetition": repetition,
                        "run_dir": str(run_dir),
                        "status": status.get("status"),
                        "trace_path": str(run_dir / "trace.jsonl"),
                        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                        "error": status.get("error"),
                    }
                    summaries.append(item)
                    summary_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    summary_file.flush()
    return summaries


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--videotool-root", required=True, type=Path)
    parser.add_argument("--planner-mode", choices=["scripted", "api", "local_qwen", "local_split"], default="scripted")
    parser.add_argument("--model-name", default="qwen3-vl-plus")
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--yolo-model", type=Path, default=None)
    parser.add_argument("--yolo-python", default=None)
    parser.add_argument("--yolo-batch", type=int, default=1)
    parser.add_argument("--yolo-preobserve", action="store_true")
    parser.add_argument("--qwen-model", type=Path, default=None)
    parser.add_argument("--qwen-python", default=None)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=96)
    parser.add_argument("--planner-model", type=Path, default=None)
    parser.add_argument("--planner-python", default=None)
    parser.add_argument("--planner-max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--planner-constrained-json",
        action="store_true",
        help="enforce the PlannerDecision JSON Schema for local_split planner output",
    )
    parser.add_argument("--answer-model", type=Path, default=None)
    parser.add_argument("--answer-python", default=None)
    parser.add_argument("--answer-max-new-tokens", type=int, default=96)
    parser.add_argument("--model-stack-id", default="stack_a_qwen3_vl8b")
    parser.add_argument("--planner-model-id", default=None)
    parser.add_argument("--visual-model-id", default=None)
    parser.add_argument("--answer-model-id", default=None)
    parser.add_argument("--detector-model-id", default=None)
    parser.add_argument("--resume", action="store_true", help="skip success rows already recorded in the summary")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    summaries = run_manifest(args)
    print(json.dumps({"runs": len(summaries), "output_root": str(args.output_root)}, ensure_ascii=False))
    return 0 if all(item["status"] == "success" for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
