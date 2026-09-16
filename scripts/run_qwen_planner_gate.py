#!/usr/bin/env python3
"""Run the Qwen3-4B PlannerDecision parsing gate.

The gate keeps the model's raw output and never replaces an invalid decision
with a scripted tool.  A single strict retry is allowed, matching the trace
collector's retry contract.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tracing.collectors.qwen_text_worker import QwenTextClient


TOOLS = [
    "frame-selector",
    "temporal-grounding",
    "temporal-qa",
    "image-grid-selector",
    "image-qa",
    "image-grid-qa",
    "patch-zoomer",
    "yolo-tracker",
    "summarization-tool",
]


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_object(raw: str) -> tuple[dict[str, Any] | None, str]:
    cleaned = _strip_code_fence(raw)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload, "json"
        return None, "schema_error"
    except json.JSONDecodeError:
        if cleaned.startswith("{") and "}" not in cleaned:
            return None, "truncated_json"
    starts = [index for index, char in enumerate(cleaned) if char == "{"]
    objects: list[dict[str, Any]] = []
    for start in starts:
        end = cleaned.find("}", start)
        if end < 0:
            continue
        try:
            candidate = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            objects.append(candidate)
    if len(objects) > 1:
        return None, "multiple_objects"
    if objects:
        # A valid object embedded in prose is not a strict one-object response.
        return objects[0], "non_json"
    return None, "non_json"


def _validate(payload: dict[str, Any] | None) -> tuple[bool, str, dict[str, Any] | None]:
    if payload is None:
        return False, "schema_error", None
    required = {"reasoning", "tool_name", "tool_input", "info_sufficient"}
    if not required.issubset(payload):
        return False, "schema_error", payload
    if not isinstance(payload["reasoning"], str) or not isinstance(payload["tool_name"], str):
        return False, "schema_error", payload
    if not isinstance(payload["tool_input"], str):
        return False, "wrong_argument_type", payload
    if not isinstance(payload["info_sufficient"], bool):
        return False, "schema_error", payload
    if payload["tool_name"].strip().lower() not in TOOLS:
        return False, "unknown_tool", payload
    return True, "json", payload


def _prompt(index: int) -> str:
    baseline = "STAR" if index % 2 else "LangGraph ReAct"
    length = index % 3
    prior = "none" if length == 0 else "frame-selector observed one temporal segment"
    if length == 2:
        prior += "; image-qa described labels and objects; temporal-qa compared order"
    question = [
        "Which object appears first in the video?",
        "What is the main content of the video?",
        "Which option matches the visible action?",
        "How many people are visible in the relevant scene?",
    ][index % 4]
    return (
        f"You are the {baseline} planner. Current iteration: {index % 6 + 1}.\n"
        f"Question: {question}\nPrevious observations ({'short' if length == 0 else 'long' if length == 2 else 'medium'}): {prior}\n"
        f"Allowed tools: {json.dumps(TOOLS)}\n"
        "Keep reasoning concise (at most 12 words) and tool_input concise (at most 20 words).\n"
        "Return exactly one JSON object with reasoning, tool_name, tool_input, info_sufficient."
    )


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = QwenTextClient(
        python_bin=args.python,
        model_path=args.model_path,
        log_path=args.output.with_suffix(".worker.log"),
        max_new_tokens=args.max_new_tokens,
        timeout_seconds=args.timeout,
        constrained_json=args.constrained_json,
    )
    rows: list[dict[str, Any]] = []
    try:
        for index in range(1, max(1, int(args.count)) + 1):
            prompt = _prompt(index)
            attempts: list[dict[str, Any]] = []
            for attempt in range(2):
                started = time.perf_counter()
                try:
                    response = client.generate(
                        prompt
                        if attempt == 0
                        else prompt
                        + "\nRetry: output exactly one complete JSON object; no markdown, prose, or multiple objects."
                    )
                    raw = str(response.get("text", ""))
                    payload, parse_category = _parse_object(raw)
                    if parse_category != "json":
                        valid, category, parsed = False, parse_category, None
                    else:
                        valid, validation_category, parsed = _validate(payload)
                        category = "json" if valid else validation_category
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "raw_output": raw,
                            "category": category,
                            "parsed": parsed,
                            "response": response,
                            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                        }
                    )
                    if valid:
                        break
                except TimeoutError as exc:
                    attempts.append({"attempt": attempt + 1, "category": "timeout", "error": str(exc)})
                except Exception as exc:
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "category": "worker_exit" if client._process is None else "worker_error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            final = attempts[-1] if attempts else {"category": "worker_error"}
            rows.append(
                {
                    "index": index,
                    "prompt": prompt,
                    "first_category": attempts[0].get("category") if attempts else "worker_error",
                    "final_category": final.get("category"),
                    "retry_used": len(attempts) > 1,
                    "success": final.get("category") == "json",
                    "attempts": attempts,
                }
            )
    finally:
        client.close()
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    first_success = sum(row["first_category"] == "json" for row in rows)
    final_success = sum(row["success"] for row in rows)
    categories: dict[str, int] = {}
    for row in rows:
        category = str(row["final_category"])
        categories[category] = categories.get(category, 0) + 1
    return {
        "count": len(rows),
        "first_success": first_success,
        "first_success_rate": first_success / len(rows) if rows else 0.0,
        "final_success": final_success,
        "final_success_rate": final_success / len(rows) if rows else 0.0,
        "categories": categories,
        "output": str(args.output),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--constrained-json",
        action="store_true",
        help="enforce the PlannerDecision JSON Schema inside the local worker",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    summary = run_gate(args)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["final_success_rate"] >= 0.98 else 1


if __name__ == "__main__":
    raise SystemExit(main())
