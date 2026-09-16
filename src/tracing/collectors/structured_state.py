#!/usr/bin/env python3
"""Auditable structured task/state helpers for Phase 3.

The old Phase 3 predictor used raw question tokens and a hand-capped prefix
length.  This module defines the next representation without changing any
existing agent.  It has two responsibilities:

* convert a public benchmark record into a structured task manifest while
  deliberately dropping answer labels from the runtime input; and
* build a per-prefix ``state_t`` snapshot from already observed actions and
  structured evidence supplied by tools.

The module is dependency-free so it can be imported by the remote Python 3.10
environment and exercised locally before a collector is changed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


STATE_SCHEMA_VERSION = "0.2"
TASK_MANIFEST_SCHEMA_VERSION = "0.2"
DEFAULT_COVERAGE_BINS = 32

_OPTION_RE = re.compile(r"^\s*\(?([A-Z])\s*[.)]\s*(.+?)\s*$", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _question_text(record: Mapping[str, Any]) -> str:
    return _text(record.get("question") or record.get("prompt"))


def _option_items(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    if isinstance(value, tuple):
        return [_text(item) for item in value if _text(item)]
    text = _text(value)
    if not text:
        return []
    items: list[str] = []
    for line in text.splitlines():
        match = _OPTION_RE.match(line)
        items.append(match.group(2).strip() if match else line.strip())
    return [item for item in items if item]


def _question_type(question: str) -> str:
    text = question.lower()
    if re.search(r"\bhow many\b|\bnumber of\b|\bcount of\b", text):
        return "count"
    if re.search(r"\bwhy\b|\breason\b|\bbecause\b", text):
        return "causal"
    if re.search(r"\blargest\b|\bsmallest\b|\bmost\b|\bleast\b|\bcompare\b|\bdifference\b", text):
        return "comparison"
    if re.search(r"\bhow long\b|\bwhen\b|\bbefore\b|\bafter\b|\bduring\b|\bsequence\b", text):
        return "temporal"
    if re.search(r"\bwhere\b|\blocation\b|\bposition\b", text):
        return "spatial"
    if re.search(r"\bbest describes?\b|\bcontent of the video\b|\bsummary\b", text):
        return "summary"
    if re.search(r"\bwhat kind\b|\bwhich (?:item|object|person|animal)\b|\bwhat is\b", text):
        return "identification"
    return "unknown"


def _temporal_scope(question: str) -> str:
    text = question.lower()
    if re.search(r"\bhow long\b|\bduration\b|\bfor how many minutes?\b", text):
        return "duration"
    if re.search(r"\bbefore\b|\bafter\b|\bwhen\b|\bduring\b|\bfirst\b|\bthen\b", text):
        return "ordered_events"
    if re.search(r"\bcontent of the video\b|\boverall\b|\baccurately describes\b", text):
        return "global"
    return "unspecified"


def _required_modalities(question: str) -> list[str]:
    text = question.lower()
    modalities: set[str] = set()
    if re.search(r"\bwhen\b|\bbefore\b|\bafter\b|\bduring\b|\bsequence\b|\bhow long\b", text):
        modalities.add("temporal")
    if re.search(r"\bsign\b|\bwritten\b|\btext\b|\bword\b|\bsubtitle\b|\blabel\b", text):
        modalities.add("text")
    if re.search(r"\bobject\b|\bitem\b|\banimal\b|\bperson\b|\bpeople\b|\bdecoration\b|\bcolor\b", text):
        modalities.add("object")
    if re.search(r"\bdance\b|\bscene\b|\bbuilding\b|\bvideo\b|\bcontent\b|\bappears?\b", text):
        modalities.add("scene")
    if not modalities:
        modalities.add("unknown")
    return sorted(modalities)


def derive_task_structure(record: Mapping[str, Any]) -> dict[str, Any]:
    """Derive fixed-schema task features without using the answer label."""

    question = _question_text(record)
    options = _option_items(record.get("options"))
    option_lengths = [len(option) for option in options]
    question_type = _question_type(question)
    result = {
        "question_type": question_type,
        "answer_type": "multiple_choice" if options else "free_form",
        "temporal_scope": _temporal_scope(question),
        "required_modalities": _required_modalities(question),
        "option_count": len(options),
        "question_chars": len(question),
        "question_tokens": len(_WORD_RE.findall(question)),
        "option_chars_mean": fmean(option_lengths) if option_lengths else 0.0,
        "domain": _text(record.get("domain")) or "unknown",
        "sub_category": _text(record.get("sub_category")) or "unknown",
        "derivation": {
            "version": "heuristic_v0.2",
            "question_type": "rule_based",
            "domain": "manifest_field_or_unknown",
            "answer_label_used": False,
        },
    }
    official_task_type = _text(record.get("official_task_type"))
    if official_task_type:
        # This is a fixed public-dataset annotation, not model-generated
        # reasoning and not an answer label.  Keeping it explicit lets Phase
        # 3 stratify by the benchmark's task taxonomy without using raw
        # question text as a proxy.
        result["official_task_type"] = official_task_type
    return result


def structured_task_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Create a runtime task record and intentionally omit ``answer``."""

    task_id = _text(record.get("task_id"))
    dataset = _text(record.get("dataset"))
    video_path = _text(record.get("video_path"))
    question = _question_text(record)
    if not task_id or not dataset or not video_path or not question:
        raise ValueError("task_id, dataset, video_path, and question are required")
    options = _option_items(record.get("options"))
    result: dict[str, Any] = {
        "manifest_schema": f"phase3-task-{TASK_MANIFEST_SCHEMA_VERSION}",
        "task_id": task_id,
        "dataset": dataset,
        "video_id": Path(video_path).stem,
        "video_path": video_path,
        "agent_input": {
            "question": question,
            "options": options,
        },
        "task_structure": derive_task_structure(record),
        "baselines": [str(value) for value in (record.get("baselines") or [])],
        "repetitions": int(record.get("repetitions") or 1),
        "source": {
            "source": _text(record.get("source")) or "unknown",
            "source_url": _text(record.get("source_url")) or None,
        },
    }
    if "question_id" in record and _text(record.get("question_id")):
        result["question_id"] = _text(record.get("question_id"))
    return result


def convert_manifest_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [structured_task_record(record) for record in records]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        records.append(payload)
    return records


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


_ACTION_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("retry", ("retry", "repair", "fallback")),
    ("temporal_qa", ("temporal", "videoqa", "time")),
    ("object_detection", ("yolo", "detect", "object", "track")),
    ("ocr", ("ocr", "text_read", "text")),
    ("spatial_qa", ("patch-zoomer", "patchzoomer", "image-qa", "imageqa", "image_grid", "grid-qa", "gridqa", "spatial")),
    ("sample_seek", ("image-grid-selector", "image-grid-select", "imagegridselector", "imagegridselect", "frame-selector", "frameselector", "frame_select", "sample", "seek", "skim", "focus", "overview")),
    ("summarize", ("summar", "caption")),
    ("answer", ("answer", "final")),
)


def canonical_action(action: Any) -> str:
    """Map framework-specific action names to a small scheduler action family."""

    text = _text(action).lower().replace("_", "-")
    for family, patterns in _ACTION_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return family
    return "other"


def _nonnegative_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else default


@dataclass
class EvidenceAccumulator:
    """Cumulative evidence state updated only after an observed tool result."""

    duration_s: float | None = None
    bin_count: int = DEFAULT_COVERAGE_BINS
    coverage_bins: list[float] = field(default_factory=list)
    observed_intervals: list[list[float]] = field(default_factory=list)
    frames_seen: int = 0
    modality_counts: Counter[str] = field(default_factory=Counter)
    object_counts: Counter[str] = field(default_factory=Counter)
    ocr_chars: int = 0
    temporal_relation_count: int = 0
    confidences: list[float] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.bin_count <= 0:
            raise ValueError("bin_count must be positive")
        if not self.coverage_bins:
            self.coverage_bins = [0.0] * self.bin_count
        if len(self.coverage_bins) != self.bin_count:
            raise ValueError("coverage_bins length must equal bin_count")
        if self.duration_s is not None and self.duration_s < 0.0:
            raise ValueError("duration_s must be non-negative")

    def observe(
        self,
        *,
        intervals: Sequence[Sequence[float]] = (),
        frames_seen: int = 0,
        modality: str | None = None,
        object_counts: Mapping[str, Any] | None = None,
        ocr_chars: int = 0,
        temporal_relation_count: int = 0,
        confidence: float | None = None,
        source_event_id: str | None = None,
    ) -> None:
        """Merge one completed tool observation into the cumulative state."""

        self.frames_seen += max(0, int(frames_seen))
        self.ocr_chars += max(0, int(ocr_chars))
        self.temporal_relation_count += max(0, int(temporal_relation_count))
        if modality:
            self.modality_counts[_text(modality)] += 1
        if object_counts:
            for name, count in object_counts.items():
                value = _nonnegative_float(count)
                if value:
                    self.object_counts[_text(name)] += value
        if confidence is not None and math.isfinite(float(confidence)):
            self.confidences.append(min(1.0, max(0.0, float(confidence))))
        if source_event_id:
            self.source_event_ids.append(_text(source_event_id))
        for interval in intervals:
            if len(interval) < 2:
                continue
            start = _nonnegative_float(interval[0])
            end = _nonnegative_float(interval[1])
            if end < start:
                start, end = end, start
            if self.duration_s is not None:
                start = min(start, self.duration_s)
                end = min(end, self.duration_s)
            if end <= start:
                continue
            self.observed_intervals.append([start, end])
            if self.duration_s and self.duration_s > 0.0:
                first = max(0, min(self.bin_count - 1, int(start / self.duration_s * self.bin_count)))
                last = max(0, min(self.bin_count - 1, int((end - 1e-12) / self.duration_s * self.bin_count)))
                for index in range(first, last + 1):
                    self.coverage_bins[index] = 1.0

    def snapshot(self) -> dict[str, Any]:
        coverage_ratio = fmean(self.coverage_bins) if self.coverage_bins else 0.0
        return {
            "coverage_bins": list(self.coverage_bins),
            "coverage_ratio": coverage_ratio,
            "observed_intervals": [list(interval) for interval in self.observed_intervals],
            "frames_seen": self.frames_seen,
            "modality_counts": dict(sorted(self.modality_counts.items())),
            "object_counts": dict(sorted(self.object_counts.items())),
            "ocr_chars": self.ocr_chars,
            "temporal_relation_count": self.temporal_relation_count,
            "confidence_mean": fmean(self.confidences) if self.confidences else None,
            "source_event_ids": list(dict.fromkeys(self.source_event_ids)),
            "future_events_included": False,
            "ground_truth_included": False,
        }


def build_state_snapshot(
    *,
    run_id: str,
    task_id: str,
    video_id: str,
    baseline: str,
    step_id: int,
    task_structure: Mapping[str, Any],
    video_metadata: Mapping[str, Any] | None,
    prefix_actions: Sequence[str],
    local_runtime_ms: float = 0.0,
    api_wait_ms: float = 0.0,
    retry_count: int = 0,
    error_count: int = 0,
    max_steps: int | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the state available immediately before the next decision."""

    raw_actions = [_text(action) for action in prefix_actions]
    canonical_actions = [canonical_action(action) for action in raw_actions]
    observed_steps = len(canonical_actions)
    effective_max_steps = max_steps if max_steps and max_steps > 0 else max(observed_steps, 1)
    prefix = {
        "observed_step_count": observed_steps,
        "raw_actions": raw_actions,
        "canonical_actions": canonical_actions,
        "action_histogram": dict(sorted(Counter(canonical_actions).items())),
        "last_actions": canonical_actions[-2:],
        "max_steps": effective_max_steps,
        "progress_ratio": min(1.0, observed_steps / effective_max_steps),
        "local_runtime_ms": _nonnegative_float(local_runtime_ms),
        "api_wait_ms": _nonnegative_float(api_wait_ms),
        "retry_count": max(0, int(retry_count)),
        "error_count": max(0, int(error_count)),
    }
    video = {
        "duration_s": video_metadata.get("duration_s") if video_metadata else None,
        "fps": video_metadata.get("fps") if video_metadata else None,
        "width": video_metadata.get("width") if video_metadata else None,
        "height": video_metadata.get("height") if video_metadata else None,
        "subtitle_available": video_metadata.get("subtitle_available") if video_metadata else None,
        "shot_count": video_metadata.get("shot_count") if video_metadata else None,
    }
    evidence_payload = dict(evidence or {})
    evidence_payload.setdefault("coverage_bins", [0.0] * DEFAULT_COVERAGE_BINS)
    evidence_payload.setdefault("coverage_ratio", fmean(evidence_payload["coverage_bins"]))
    evidence_payload.setdefault("observed_intervals", [])
    evidence_payload.setdefault("frames_seen", 0)
    evidence_payload.setdefault("modality_counts", {})
    evidence_payload.setdefault("object_counts", {})
    evidence_payload.setdefault("ocr_chars", 0)
    evidence_payload.setdefault("temporal_relation_count", 0)
    evidence_payload.setdefault("confidence_mean", None)
    evidence_payload.setdefault("source_event_ids", [])
    evidence_payload.setdefault("future_events_included", False)
    evidence_payload.setdefault("ground_truth_included", False)
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "state_id": f"{run_id}:state:{step_id}",
        "run_id": _text(run_id),
        "task_id": _text(task_id),
        "video_id": _text(video_id),
        "baseline": _text(baseline),
        "step_id": int(step_id),
        "task": dict(task_structure),
        "video": video,
        "prefix": prefix,
        "evidence": evidence_payload,
        "leakage_guard": {
            "future_events_excluded": evidence_payload.get("future_events_included") is False,
            "ground_truth_excluded": evidence_payload.get("ground_truth_included") is False,
            "video_id_used_as_feature": False,
            "answer_text_used_as_feature": False,
        },
    }


def validate_state_snapshot(state: Mapping[str, Any]) -> list[str]:
    """Return human-readable structural errors without requiring jsonschema."""

    errors: list[str] = []
    for key in ("schema_version", "state_id", "run_id", "task_id", "video_id", "baseline", "step_id", "task", "video", "prefix", "evidence", "leakage_guard"):
        if key not in state:
            errors.append(f"missing {key}")
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        errors.append("schema_version must be 0.2")
    if not isinstance(state.get("step_id"), int) or state.get("step_id", -1) < 0:
        errors.append("step_id must be a non-negative integer")
    prefix = state.get("prefix")
    if isinstance(prefix, Mapping):
        progress = prefix.get("progress_ratio")
        if not isinstance(progress, (int, float)) or not 0.0 <= float(progress) <= 1.0:
            errors.append("prefix.progress_ratio must be in [0, 1]")
        if prefix.get("observed_step_count", -1) != len(prefix.get("canonical_actions", [])):
            errors.append("prefix.observed_step_count must match canonical_actions")
    evidence = state.get("evidence")
    if isinstance(evidence, Mapping):
        bins = evidence.get("coverage_bins", [])
        if not isinstance(bins, list) or not bins:
            errors.append("evidence.coverage_bins must be non-empty")
        elif any(not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0 for value in bins):
            errors.append("evidence.coverage_bins values must be in [0, 1]")
        if evidence.get("future_events_included") is not False:
            errors.append("evidence.future_events_included must be false")
        if evidence.get("ground_truth_included") is not False:
            errors.append("evidence.ground_truth_included must be false")
    guard = state.get("leakage_guard")
    if isinstance(guard, Mapping):
        for key in ("future_events_excluded", "ground_truth_excluded", "video_id_used_as_feature", "answer_text_used_as_feature"):
            if guard.get(key) is not (False if key.endswith("used_as_feature") else True):
                errors.append(f"leakage_guard.{key} has an unsafe value")
    return errors


def write_state_sidecar(path: Path, state: Mapping[str, Any]) -> None:
    """Validate and persist one immutable ``state_t`` JSON sidecar."""

    errors = validate_state_snapshot(state)
    if errors:
        raise ValueError("invalid state snapshot: " + "; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="source benchmark JSONL")
    parser.add_argument("--output", required=True, type=Path, help="structured runtime JSONL")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    records = convert_manifest_records(read_jsonl(args.input))
    write_jsonl(args.output, records)
    print(json.dumps({"records": len(records), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
