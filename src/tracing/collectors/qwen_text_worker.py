#!/usr/bin/env python3
"""Serial local causal-LM worker for the split planner stack.

The trace collector is intentionally kept independent of the heavy model
runtime.  This process loads a local Qwen causal language model once and
exchanges bounded JSON requests over stdin/stdout.  It is suitable for a
Qwen3-4B planner, while the visual worker remains in ``qwen3_vl_worker``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PLANNER_DECISION_TOOLS = (
    "frame-selector",
    "temporal-grounding",
    "temporal-qa",
    "image-grid-selector",
    "image-qa",
    "image-grid-qa",
    "patch-zoomer",
    "yolo-tracker",
    "summarization-tool",
)

PLANNER_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 256},
        "tool_name": {"type": "string", "enum": list(PLANNER_DECISION_TOOLS)},
        "tool_input": {"type": "string", "maxLength": 512},
        "info_sufficient": {"type": "boolean"},
    },
    "required": ["reasoning", "tool_name", "tool_input", "info_sufficient"],
    "additionalProperties": False,
}


def _json_line(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _worker(model_path: Path, max_new_tokens: int, constrained_json: bool = False) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_started = time.perf_counter()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    load_ms = round((time.perf_counter() - load_started) * 1000.0, 3)
    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device

    prefix_allowed_tokens_fn = None
    constraint_setup_ms: float | None = None
    if constrained_json:
        # lm-format-enforcer 0.11.x imports this symbol from the pre-5.x
        # Transformers module.  Transformers 5.8 moved it to
        # tokenization_utils_base; expose the compatible alias before the
        # optional integration import.  This does not alter tokenizer logic.
        import transformers.tokenization_utils as tokenization_utils
        import transformers.tokenization_utils_base as tokenization_utils_base

        if not hasattr(tokenization_utils, "PreTrainedTokenizerBase"):
            tokenization_utils.PreTrainedTokenizerBase = (
                tokenization_utils_base.PreTrainedTokenizerBase
            )
        try:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import (
                build_transformers_prefix_allowed_tokens_fn,
            )
        except ImportError as exc:
            raise RuntimeError(
                "constrained JSON requested but lm-format-enforcer is unavailable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        constraint_started = time.perf_counter()
        parser = JsonSchemaParser(PLANNER_DECISION_SCHEMA)
        prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
            tokenizer, parser
        )
        constraint_setup_ms = round((time.perf_counter() - constraint_started) * 1000.0, 3)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if request.get("op") == "health":
                _json_line(
                    {
                        "ok": True,
                        "model_path": str(model_path),
                        "initial_load_ms": load_ms,
                        "device": str(model_device),
                        "model_type": getattr(getattr(model, "config", None), "model_type", None),
                        "json_schema_constrained": bool(constrained_json),
                        "constraint_setup_ms": constraint_setup_ms,
                    }
                )
                continue
            if request.get("op") != "generate":
                raise ValueError(f"unknown worker operation: {request.get('op')!r}")
            prompt = str(request.get("prompt", ""))[:8000]
            messages = [{"role": "user", "content": prompt}]
            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            try:
                inputs = tokenizer.apply_chat_template(
                    messages, enable_thinking=False, **template_kwargs
                )
            except (TypeError, ValueError):
                # Older tokenizer revisions do not expose enable_thinking.
                inputs = tokenizer.apply_chat_template(messages, **template_kwargs)
            inputs = inputs.to(model_device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max(1, int(max_new_tokens)),
                    do_sample=False,
                    **(
                        {"prefix_allowed_tokens_fn": prefix_allowed_tokens_fn}
                        if prefix_allowed_tokens_fn is not None
                        else {}
                    ),
                )
            input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
            generated_trimmed = [
                output_ids[len(input_ids_row) :]
                for input_ids_row, output_ids in zip(input_ids, generated)
            ]
            text = tokenizer.batch_decode(
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            inference_ms = round((time.perf_counter() - started) * 1000.0, 3)
            payload: dict[str, Any] = {
                "ok": True,
                "text": str(text[0] if text else "").strip()[:8000],
                "model_load_ms": load_ms,
                "inference_ms": inference_ms,
                "model_resident": True,
                "json_schema_constrained": bool(constrained_json),
                "constraint_setup_ms": constraint_setup_ms,
            }
            if torch.cuda.is_available():
                payload.update(
                    {
                        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 3),
                        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 3),
                    }
                )
                torch.cuda.reset_peak_memory_stats()
            _json_line(payload)
        except Exception as exc:  # keep the protocol alive for the next request
            _json_line({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


class QwenTextClient:
    """Parent-side JSON-line client with one serially resident text model."""

    def __init__(
        self,
        *,
        python_bin: str,
        model_path: Path,
        log_path: Path,
        max_new_tokens: int = 96,
        timeout_seconds: float = 300.0,
        constrained_json: bool = False,
    ) -> None:
        self.python_bin = python_bin
        self.model_path = model_path.expanduser().resolve()
        self.log_path = log_path
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.constrained_json = bool(constrained_json)
        self._process: Any = None
        self._startup_ms: float | None = None
        self._request_count = 0

    @property
    def request_count(self) -> int:
        return self._request_count

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        import select
        import subprocess

        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Qwen text model directory does not exist: {self.model_path}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = self.log_path.open("a", encoding="utf-8")
        started = time.perf_counter()
        worker_cmd = [
            self.python_bin,
            str(Path(__file__).resolve()),
            "--worker",
            "--model-path",
            str(self.model_path),
            "--max-new-tokens",
            str(self.max_new_tokens),
        ]
        if self.constrained_json:
            worker_cmd.append("--constrained-json")
        self._process = subprocess.Popen(
            worker_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_file,
            text=True,
            bufsize=1,
        )
        self._process._phase2_log_file = log_file
        self._send({"op": "health"})
        response = self._read_response(select)
        if not response.get("ok"):
            raise RuntimeError(f"Qwen text worker health check failed: {response.get('error')}")
        self._startup_ms = round((time.perf_counter() - started) * 1000.0, 3)
        self._health = response

    def _send(self, request: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Qwen text worker is not running")
        self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._process.stdin.flush()

    def _read_response(self, select_module: Any | None = None) -> dict[str, Any]:
        import select as select_impl

        select_impl = select_module or select_impl
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Qwen text worker is not running")
        ready, _, _ = select_impl.select([self._process.stdout], [], [], self.timeout_seconds)
        if not ready:
            raise TimeoutError("timed out waiting for Qwen text worker")
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(f"Qwen text worker exited with code {self._process.poll()}")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("Qwen text worker response is not a JSON object")
        return payload

    def generate(self, prompt: str) -> dict[str, Any]:
        self._ensure_process()
        self._send({"op": "generate", "prompt": str(prompt)[:8000]})
        response = self._read_response()
        self._request_count += 1
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Qwen text worker request failed"))
        response["worker_startup_ms"] = self._startup_ms
        response["request_index"] = self._request_count
        return response

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=10)
        except Exception:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                process.kill()
        log_file = getattr(process, "_phase2_log_file", None)
        if log_file is not None:
            log_file.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--constrained-json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.worker:
        raise SystemExit("QwenTextClient is the parent-side API; use --worker only for the child process")
    return _worker(
        args.model_path.expanduser().resolve(),
        args.max_new_tokens,
        constrained_json=args.constrained_json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
