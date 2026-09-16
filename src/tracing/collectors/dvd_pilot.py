#!/usr/bin/env python3
"""Collect a bounded DVD-style local pilot in the shared trace schema.

The official Deep Video Discovery checkout remains read-only.  This adapter
keeps its four-tool vocabulary and search/inspect loop, while using the local
Qwen VLM worker and a small per-video caption index.  It is deliberately
labelled ``dvd_local`` and must not be reported as an exact official-DVD run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VIDEO_TOOL_ROOT = PROJECT_ROOT / "third_party" / "videotool"
if VIDEO_TOOL_ROOT.is_dir() and str(VIDEO_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEO_TOOL_ROOT))

import tracing.collectors.videotool_phase1 as videotool  # noqa: E402
from tracing.collectors.qwen3_vl_worker import QwenVLClient  # noqa: E402
from tracing.collectors.videoseek_wrapper import (  # noqa: E402
    SCHEMA_VERSION,
    TraceRecorder,
    _environment_metadata,
    _git_provenance,
    _json_safe,
    _round_ms,
    _sha256,
    _utc_now,
)


DVD_TOOL_CATEGORIES = {
    "DVDGlobalBrowse": "dvd_global",
    "DVDClipSearch": "dvd_search",
    "DVDFrameInspect": "dvd_visual",
    "DVDFinish": "dvd_finish",
}
videotool._CATEGORY.update(DVD_TOOL_CATEGORIES)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _options_text(options: Any) -> str:
    if isinstance(options, list):
        return "\n".join(str(item) for item in options)
    return str(options or "")


def _tokens(text: str) -> set[str]:
    return {item for item in re.findall(r"[a-z0-9]{3,}", str(text).lower()) if item not in {"the", "and", "what", "which", "video"}}


class CaptionIndex:
    """Small lexical index over locally generated timestamped captions."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.entries = entries

    def search(self, query: str, top_k: int = 6) -> list[dict[str, Any]]:
        query_tokens = _tokens(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in self.entries:
            caption_tokens = _tokens(entry.get("caption", ""))
            overlap = len(query_tokens & caption_tokens)
            scored.append((float(overlap), entry))
        scored.sort(key=lambda item: (-item[0], float(item[1].get("time_start_secs", 0.0))))
        return [entry for _, entry in scored[: max(1, int(top_k))]]

    def render(self, query: str, top_k: int = 6) -> str:
        results = self.search(query, top_k=top_k)
        lines = [
            f"[{float(item.get('time_start_secs', 0.0)):.2f}-{float(item.get('time_end_secs', 0.0)):.2f}s] {item.get('caption', '')}"
            for item in results
        ]
        return "Here are the most relevant timestamped clip captions:\n" + "\n".join(lines)


def _record_preprocess_caption(
    ctx: videotool.ToolContext,
    visible: videotool.VisibleFrames,
    qwen_client: QwenVLClient,
) -> CaptionIndex:
    """Generate the offline caption index used by global/clip search."""

    indices = visible.sample_indices(6)
    frame_paths = visible.ensure_frame_files(indices, limit=6)
    started = time.perf_counter()
    start_wall = _utc_now()
    prompt = (
        "Create concise timestamp-aware captions for these sampled video frames. "
        "List one short factual caption per frame in order; do not answer the question."
    )
    status = "success"
    error: Optional[str] = None
    raw = ""
    metrics: dict[str, Any] = {}
    try:
        response = qwen_client.describe(frame_paths, prompt)
        raw = str(response.get("text", ""))[:4000]
        metrics = videotool._worker_resource_metrics(response)
        if not raw.strip():
            raise RuntimeError("empty caption response")
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = _round_ms(time.perf_counter() - started)
    ctx.recorder.record_event(
        event_type="action",
        step_id=0,
        action="dvd_caption_preprocess",
        node_type="dvd_preprocess",
        parent_step_ids=[],
        input_data={
            "frame_indices": indices,
            "frame_count": len(frame_paths),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_model_output": raw,
            "implementation": "local_reimplementation",
        },
        start_time=start_wall,
        end_time=_utc_now(),
        runtime_ms=elapsed_ms,
        api_wait_ms=0.0,
        status=status,
        output_summary_ref=f"trace.jsonl:dvd-caption-{ctx.recorder.event_count + 1}",
        model_id=ctx.qwen_model_id,
        error=error,
        resource_overrides=metrics,
    )
    if status != "success":
        raise RuntimeError(error or "DVD caption preprocessing failed")

    duration = float(visible.video_info.get("duration") or 0.0)
    entries = []
    for position, index in enumerate(indices):
        start = index / float(visible.video_info.get("fps") or 1.0)
        if position + 1 < len(indices):
            end = indices[position + 1] / float(visible.video_info.get("fps") or 1.0)
        else:
            end = duration or start + 1.0
        entries.append(
            {
                "time_start_secs": round(start, 3),
                "time_end_secs": round(max(start, end), 3),
                "caption": raw,
            }
        )
    return CaptionIndex(entries)


class DVDGlobalBrowse(videotool.BaseTool):
    name = "global_browse_tool"
    description = "Browse a global timestamped overview of the video caption database."
    replacement_model_id = "dvd-local-caption-index-v0"

    def __init__(self, ctx: videotool.ToolContext, index: CaptionIndex) -> None:
        super().__init__(ctx)
        self.index = index

    def _run(self, tool_input: str) -> str:
        return json.dumps(
            {
                "subject_registry": [],
                "query_related_event": self.index.render(tool_input, top_k=6),
            },
            ensure_ascii=False,
        )


class DVDClipSearch(videotool.BaseTool):
    name = "clip_search_tool"
    description = "Search timestamped caption clips for a described event."
    replacement_model_id = "dvd-local-caption-index-v0"

    def __init__(self, ctx: videotool.ToolContext, index: CaptionIndex) -> None:
        super().__init__(ctx)
        self.index = index

    def _run(self, tool_input: str) -> str:
        return self.index.render(tool_input, top_k=4)


class DVDFrameInspect(videotool.BaseTool):
    name = "frame_inspect_tool"
    description = "Inspect selected time ranges with the local Qwen VLM."
    replacement_model_id = "qwen3-vl-8b-instruct"

    def _run(self, tool_input: str) -> str:
        # Use the collector's safe 95%-of-duration sampler.  The generic
        # fraction sampler may request the nominal final VFR frame, which is
        # sometimes one frame beyond what ffmpeg can decode.
        indices = self.ctx.visible_frames.sample_indices(6)
        frame_paths = self.ctx.visible_frames.ensure_frame_files(indices, limit=6)
        if self.ctx.qwen_client is None:
            raise RuntimeError("DVD frame inspection requires a local VLM worker")
        response = self.ctx.qwen_client.describe(
            frame_paths,
            "Inspect these video frames for concrete evidence relevant to the request. "
            "Mention timestamps, objects, actions, text, and temporal order only when visible.\n"
            + tool_input,
        )
        self.ctx.record_tool_metrics(videotool._worker_resource_metrics(response))
        answer = str(response.get("text", ""))[:3000]
        self.ctx.visible_frames.add_qa(self.name, answer, indices)
        return f"Frame inspection over indices {indices}: {answer}"


class DVDFinish(videotool.BaseTool):
    name = "finish"
    description = "Finish after confirming the answer to the user's question."
    replacement_model_id = "dvd-terminal-v0"

    def _run(self, tool_input: str) -> str:
        return tool_input.strip()[:1000] or "A"


def _standard_args(tool_name: str, tool_input: str) -> dict[str, Any]:
    if tool_name == "global_browse_tool":
        return {"query": tool_input}
    if tool_name == "clip_search_tool":
        return {"event_description": tool_input, "top_k": 16}
    if tool_name == "frame_inspect_tool":
        return {"question": tool_input, "time_ranges_hhmmss": []}
    return {"answer": tool_input}


def _run_dvd_loop(
    ctx: videotool.ToolContext,
    tools: list[videotool.BaseTool],
    question: str,
    options: str,
    max_iterations: int,
    planner_model_id: str,
) -> tuple[str, dict[str, Any]]:
    planner = videotool.PlannerModel(ctx, "local_qwen", planner_model_id)
    observations: list[dict[str, Any]] = []
    allowed = [
        {
            "tool_name": tool.inference.name,
            "description": tool.inference.description,
            "arguments": "one short string in tool_input",
        }
        for tool in tools
    ]
    by_name = {tool.inference.name: tool for tool in tools}
    final_answer = ""
    executed: list[str] = []
    for iteration in range(1, max(1, int(max_iterations)) + 1):
        force_finish = iteration == max(1, int(max_iterations))
        iteration_allowed = (
            [item for item in allowed if item["tool_name"] == "finish"] if force_finish else allowed
        )
        prompt = (
            "You are the DVD video discovery orchestrator. Follow THINK -> ACT -> OBSERVE. "
            "Choose exactly one tool from the allowed list. Use global browse for a broad overview, "
            "clip search for a targeted event, frame inspect to verify visual evidence, and finish only "
            "when the answer is confirmed. The final iteration must call finish. Return one JSON object "
            "with reasoning, tool_name, tool_input, info_sufficient.\n"
            f"Current iteration: {iteration}\n"
            f"Question: {question}\nOptions:\n{options}\n"
            f"Allowed tools: {json.dumps(iteration_allowed, ensure_ascii=False)}\n"
            f"Prior observations: {json.dumps(observations[-4:], ensure_ascii=False)}\n"
            f"Evidence state: {json.dumps(ctx.state_evidence.snapshot(), ensure_ascii=False)}\n"
            f"Must finish now: {str(force_finish).lower()}\n"
            '{"reasoning":"short reason","tool_name":"clip_search_tool",'
            '"tool_input":"short query or final answer","info_sufficient":false}'
        )
        decision = planner.generate(prompt)
        if decision is None:
            raise RuntimeError("DVD planner failed to produce a valid decision after retry")
        tool_name = str(decision.tool_name).strip()
        tool_input = str(decision.tool_input or "").strip()[:1200]
        tool = by_name.get(tool_name)
        if tool is None:
            raise RuntimeError(f"DVD planner selected unknown tool: {tool_name}")
        ctx.current_step = iteration
        ctx.current_standard_tool_call = {
            "name": tool_name,
            "args": _standard_args(tool_name, tool_input),
            "id": f"dvd-local-step-{iteration}",
            "type": "function_call",
        }
        observation = tool.inference(input=tool_input)
        observations.append(
            {
                "iteration": iteration,
                "tool": tool_name,
                "tool_input": tool_input,
                "observation": observation[:1200],
                "info_sufficient": bool(decision.info_sufficient),
            }
        )
        executed.append(tool_name)
        if tool_name == "finish":
            final_answer = observation
            break
    if not final_answer:
        raise RuntimeError("DVD run ended without a finish tool")
    return final_answer, {
        "agent_family": "dvd",
        "implementation": "local_reimplementation",
        "engine": "dvd_local_search_loop",
        "tool_history": observations,
        "executed_tools": executed,
    }


def run_one(row: dict[str, Any], args: argparse.Namespace) -> Path:
    video_path = Path(str(row["video_path"])).expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{video_path.stem}_dvd_local_{int(time.time() * 1000)}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    question = str(row.get("question", ""))
    options = _options_text(row.get("options"))
    task_id = str(row.get("task_id") or video_path.stem)
    model_id = str(args.model_id)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "framework": "dvd_local_adapter",
        "agent_family": "dvd",
        "implementation": "local_reimplementation",
        "baseline": "dvd_local",
        "dataset": str(row.get("dataset", "videomme")),
        "task_id": task_id,
        "video_id": str(row.get("video_id") or video_path.stem),
        "video_path": str(video_path),
        "video_sha256": _sha256(video_path),
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "question_id": row.get("question_id"),
        "dvd_source_commit": args.dvd_commit,
        "dvd_root": str(args.dvd_root),
        "dvd_tool_names": ["global_browse_tool", "clip_search_tool", "frame_inspect_tool", "finish"],
        "model_stack_id": args.model_stack_id,
        "planner_model_id": model_id,
        "visual_model_id": model_id,
        "answer_model_id": "finish_argument",
        "qwen_model": str(args.qwen_model),
        "qwen_python": args.qwen_python,
        "max_iterations": args.max_iterations,
        "source_provenance": _git_provenance(args.dvd_root),
        "environment": _environment_metadata(),
        "started_at": _utc_now(),
        "status": "running",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    recorder = TraceRecorder(
        run_dir,
        run_id=run_id,
        framework="dvd_local_adapter",
        dataset=str(row.get("dataset", "videomme")),
        task_id=task_id,
        model_id=model_id,
    )
    qwen_client: Optional[QwenVLClient] = None
    ctx: Optional[videotool.ToolContext] = None
    started = time.perf_counter()
    answer = ""
    error: Optional[str] = None
    status = "error"
    try:
        qwen_client = QwenVLClient(
            python_bin=args.qwen_python,
            model_path=args.qwen_model,
            log_path=run_dir / "qwen_worker.log",
            max_new_tokens=args.qwen_max_new_tokens,
        )
        recorder.record_event(
            event_type="run",
            step_id=0,
            action="baseline_start",
            node_type="run_control",
            parent_step_ids=[],
            input_data={
                "baseline": "dvd_local",
                "agent_family": "dvd",
                "implementation": "local_reimplementation",
                "model_stack_id": args.model_stack_id,
            },
            start_time=manifest["started_at"],
            end_time=_utc_now(),
            runtime_ms=0.0,
            api_wait_ms=0.0,
            status="success",
            output_summary_ref="trace.jsonl:run-start",
            model_id=args.model_stack_id,
        )
        visible = videotool.VisibleFrames(video_path, run_dir)
        ctx = videotool.ToolContext(
            visible,
            video_path,
            run_dir,
            recorder,
            None,
            None,
            qwen_client=qwen_client,
            qwen_model_id=model_id,
            planner_model_id=model_id,
            answer_model_id="finish_argument",
            model_stack_id=args.model_stack_id,
            baseline="dvd_local",
            task_structure=videotool.derive_task_structure({"question": question, "options": options}),
            max_steps=args.max_iterations,
        )
        ctx.planner_mode = "local_qwen"
        index = _record_preprocess_caption(ctx, visible, qwen_client)
        tools = [
            DVDGlobalBrowse(ctx, index),
            DVDClipSearch(ctx, index),
            DVDFrameInspect(ctx),
            DVDFinish(ctx),
        ]
        answer, details = _run_dvd_loop(ctx, tools, question, options, args.max_iterations, model_id)
        recorder.record_event(
            event_type="run",
            step_id=max(1, ctx.current_step) + 1,
            action="answer",
            node_type="answer_generation",
            parent_step_ids=[max(1, ctx.current_step)],
            input_data={"baseline": "dvd_local", "answer": answer[:500], "details": details},
            start_time=_utc_now(),
            end_time=_utc_now(),
            runtime_ms=_round_ms(time.perf_counter() - started),
            api_wait_ms=0.0,
            status="success",
            output_summary_ref="trace.jsonl:run-answer",
            model_id="finish_argument",
        )
        status = "success"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if ctx is not None:
            recorder.record_event(
                event_type="run",
                step_id=max(1, ctx.current_step),
                action="baseline_error",
                node_type="run_control",
                parent_step_ids=[],
                input_data={"baseline": "dvd_local", "agent_family": "dvd"},
                start_time=_utc_now(),
                end_time=_utc_now(),
                runtime_ms=_round_ms(time.perf_counter() - started),
                api_wait_ms=0.0,
                status="error",
                output_summary_ref=None,
                model_id=args.model_stack_id,
                error=error,
            )
    finally:
        if qwen_client is not None:
            qwen_client.close()
        recorder.close()
    manifest.update(
        {
            "status": status,
            "finished_at": _utc_now(),
            "elapsed_ms": _round_ms(time.perf_counter() - started),
            "answer": answer[:500],
            "error": error,
            "trace_event_count": sum(1 for _ in (run_dir / "trace.jsonl").open(encoding="utf-8")),
        }
    )
    (run_dir / "run_manifest.json").write_text(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": status,
                "run_id": run_id,
                "baseline": "dvd_local",
                "agent_family": "dvd",
                "answer": answer[:500],
                "error": error,
                "trace_path": str(run_dir / "trace.jsonl"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dvd-root", type=Path, required=True)
    parser.add_argument("--dvd-commit", required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--qwen-python", required=True)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=96)
    parser.add_argument("--model-id", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--model-stack-id", default="dvd_local_qwen3vl8b")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    all_rows = _read_jsonl(args.manifest)
    start = max(0, int(args.offset))
    rows = all_rows[start : start + max(1, int(args.limit))]
    if not rows:
        raise SystemExit("manifest contains no rows")
    exit_code = 0
    for row in rows:
        run_dir = run_one(row, args)
        status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))["status"]
        print(json.dumps({"run_dir": str(run_dir), "status": status}, ensure_ascii=False), flush=True)
        if status != "success":
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
