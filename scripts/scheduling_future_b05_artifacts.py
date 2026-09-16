#!/usr/bin/env python3
"""Export frozen B05 probabilities and causal H-step scenario artifacts.

The script reuses the accepted R2 preprocessing/model modules.  It fits no
parameters: encoders are fitted on the locked B05 train split and checkpoints
are loaded from the three final-holdout seed directories.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path("/root/autodl-tmp/scheduler")
R2_ROOT = ROOT / "scripts_behavior_r2_v5"
R2B_ROOT = R2_ROOT / "r2_v5"
for _path in (R2B_ROOT, R2_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from r2 import constants, data, preprocessing  # noqa: E402
from r2b import models_seq  # noqa: E402


ROLE_LABELS = tuple(constants.ROLES)
FAMILY_LABELS = tuple(constants.FAMILIES)
FAMILY_TO_RAW = {
    "select_frames": "frame-selector",
    "visual_qa": "image-qa",
    "temporal_ops": "temporal-grounding",
    "summarize": "summarization-tool",
    "detect": "object_detection",
    "other": "other",
}
RAW_TO_FAMILY = {
    "sample_seek": "select_frames",
    "frame-selector": "select_frames",
    "image-grid-selector": "select_frames",
    "spatial_qa": "visual_qa",
    "image-qa": "visual_qa",
    "image-grid-qa": "visual_qa",
    "patch-zoomer": "visual_qa",
    "temporal-qa": "temporal_ops",
    "temporal-grounding": "temporal_ops",
    "summarize": "summarize",
    "summarization-tool": "summarize",
    "object_detection": "detect",
    "yolo-tracker": "detect",
}
NODE_ROLE = {
    "planner": "plan",
    "answer_generation": "aggregate",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(values: Iterable[Any]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def top_probabilities(values: np.ndarray, labels: tuple[str, ...], k: int) -> list[dict[str, Any]]:
    order = np.argsort(-values)[:k]
    return [{"label": labels[int(i)], "probability": float(values[int(i)])} for i in order]


def finite_probability(values: np.ndarray) -> None:
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("non-finite probability matrix")
    if (values < -1e-7).any() or (np.abs(values.sum(axis=1) - 1.0) > 1e-5).any():
        raise ValueError("probability rows are not normalized")


def clean_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Keep model-input fields while dropping labels and audit targets."""
    result = copy.deepcopy(sample)
    for key in (
        "next_role",
        "target_source_event_id",
        "target_raw_action",
        "family_label",
        "target_event_index",
        "y_true",
    ):
        result.pop(key, None)
    result.pop("source_event_id", None)
    return result


def node_role(node: dict[str, Any]) -> str | None:
    node_type = str(node.get("node_type") or "")
    if node_type in NODE_ROLE:
        return NODE_ROLE[node_type]
    if node_type.startswith("videotool_"):
        return "execute"
    return None


def node_family(node: dict[str, Any]) -> str | None:
    return RAW_TO_FAMILY.get(str(node.get("raw_action") or "")) or RAW_TO_FAMILY.get(str(node.get("activity") or ""))


def choose_mode(rows: list[dict[str, Any]], key: str, default: str) -> str:
    values = [str(row.get(key) or "") for row in rows if row.get(key)]
    return Counter(values).most_common(1)[0][0] if values else default


class B05Runner:
    def __init__(self, dataset_dir: Path, checkpoint_paths: list[Path], device: str) -> None:
        self.dataset_dir = dataset_dir
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        role = data.load_role_samples(str(dataset_dir / "role_event_samples.jsonl"))
        tool = data.load_tool_samples(str(dataset_dir / "semantic_tool_samples.jsonl"))
        self.role_train = [row for row in role if row["split"] == "train"]
        self.role_validation = [row for row in role if row["split"] == "validation"]
        self.tool_train = [row for row in tool if row["split"] == "train"]
        self.tool_validation = [row for row in tool if row["split"] == "validation"]
        self.fit_history = preprocessing.build_history_index(self.role_train + self.role_validation)
        self.role_static = preprocessing.StaticPreprocessor().fit(self.role_train, self.fit_history)
        self.tool_static = preprocessing.StaticPreprocessor().fit(self.tool_train, self.fit_history)
        self.role_seq = models_seq.shared_seq_prep(self.role_train, self.fit_history)
        self.tool_seq = models_seq.shared_seq_prep(self.tool_train, self.fit_history)
        self.role_vocab = models_seq.seq_vocab_sizes(self.role_seq, self.role_train, self.fit_history)
        self.tool_vocab = models_seq.seq_vocab_sizes(self.tool_seq, self.tool_train, self.fit_history)
        self.models: list[tuple[torch.nn.Module, torch.nn.Module]] = []
        self.checkpoint_paths = checkpoint_paths
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        for path in checkpoint_paths:
            state = torch.load(path, map_location="cpu", weights_only=False)
            role_model = models_seq.SequenceRNN(
                self.role_vocab,
                self.role_static.dimension,
                models_seq.DEFAULT_HYPERPARAMETERS["hidden"],
                models_seq.DEFAULT_HYPERPARAMETERS["dropout"],
                len(ROLE_LABELS),
                cell="gru",
            ).to(self.device)
            tool_model = models_seq.SequenceRNN(
                self.tool_vocab,
                self.tool_static.dimension,
                models_seq.DEFAULT_HYPERPARAMETERS["hidden"],
                models_seq.DEFAULT_HYPERPARAMETERS["dropout"],
                len(FAMILY_LABELS),
                cell="gru",
            ).to(self.device)
            role_model.load_state_dict(state["role_state_dict"])
            tool_model.load_state_dict(state["family_state_dict"])
            role_model.eval()
            tool_model.eval()
            self.models.append((role_model, tool_model))

    def _encode(self, sample: dict[str, Any], history_index: dict[str, list[dict[str, Any]]], head: str) -> dict[str, torch.Tensor]:
        if head == "role":
            static_encoder, sequence_encoder = self.role_static, self.role_seq
        else:
            static_encoder, sequence_encoder = self.tool_static, self.tool_seq
        static = static_encoder.transform(sample, history_index)
        tokens, mask, lengths, _ = sequence_encoder.transform(sample, history_index)
        return {
            "static": torch.from_numpy(static).to(self.device),
            "tokens": torch.from_numpy(tokens).to(self.device),
            "lengths": torch.from_numpy(lengths).to(self.device),
            "mask": torch.from_numpy(mask).to(self.device),
        }

    def predict_one(self, sample: dict[str, Any], history_index: dict[str, list[dict[str, Any]]], head: str) -> np.ndarray:
        return self.predict_model_rows(None, [sample], history_index, head)[0]

    def _encode_rows(self, rows: list[dict[str, Any]], history_index: dict[str, list[dict[str, Any]]], head: str) -> dict[str, torch.Tensor]:
        encoded = [self._encode(row, history_index, head) for row in rows]
        return {key: torch.cat([item[key] for item in encoded], dim=0) for key in encoded[0]}

    def predict_model_rows(
        self,
        model: torch.nn.Module | None,
        rows: list[dict[str, Any]],
        history_index: dict[str, list[dict[str, Any]]],
        head: str,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Predict in the same batch size as the official B05 writer.

        The CUDA packed-GRU kernel can differ slightly between batch size one
        and the official batch size.  Holdout recomputation therefore uses
        this path, while synthetic rollouts still call ``predict_one``.
        """
        if not rows:
            return np.empty((0, len(ROLE_LABELS if head == "role" else FAMILY_LABELS)), dtype=np.float32)
        encoded = self._encode_rows(rows, history_index, head)
        models = [pair[0] if head == "role" else pair[1] for pair in self.models]
        if model is not None:
            models = [model]
        outputs: list[np.ndarray] = []
        for selected in models:
            chunks: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(rows), batch_size):
                    batch = {key: value[start : start + batch_size] for key, value in encoded.items()}
                    chunks.append(torch.softmax(selected(batch), dim=1).cpu().numpy())
            outputs.append(np.concatenate(chunks, axis=0))
        result = np.mean(np.asarray(outputs), axis=0)
        finite_probability(result)
        return result

    def predict_rows(self, rows: list[dict[str, Any]], history_index: dict[str, list[dict[str, Any]]], head: str) -> np.ndarray:
        return self.predict_model_rows(None, rows, history_index, head)

    def _predict_one_legacy(self, sample: dict[str, Any], history_index: dict[str, list[dict[str, Any]]], head: str) -> np.ndarray:
        batch = self._encode(sample, history_index, head)
        values: list[np.ndarray] = []
        with torch.no_grad():
            for role_model, tool_model in self.models:
                model = role_model if head == "role" else tool_model
                values.append(torch.softmax(model(batch), dim=1).cpu().numpy()[0])
        result = np.mean(np.asarray(values), axis=0)
        finite_probability(result.reshape(1, -1))
        return result


def actual_history_index(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        index[str(row["run_id"])].append(row)
    for values in index.values():
        values.sort(key=lambda row: int(row.get("event_index", 0)))
    return dict(index)


def prototype_map(templates: list[dict[str, Any]]) -> dict[str, dict[tuple[str, str | None], dict[str, Any]]]:
    result: dict[str, dict[tuple[str, str | None], dict[str, Any]]] = {}
    for template in templates:
        grouped: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
        for node in template.get("nodes") or []:
            role = node_role(node)
            if role is None:
                continue
            grouped[(role, node_family(node) if role == "execute" else None)].append(node)
        result[str(template["template_id"])] = {}
        for key, rows in grouped.items():
            first = sorted(rows, key=lambda row: (str(row.get("model_id")), int(row.get("sequence_index", 0))))[0]
            result[str(template["template_id"])][key] = {
                "model_id": first.get("model_id"),
                "execution_lane": first.get("execution_lane"),
                "raw_action": first.get("raw_action"),
                "prototype_source": "template_mode_tiebreak_first",
            }
    return result


def synthetic_role(base: dict[str, Any], run_id: str, event_index: int, role: str, raw_action: str) -> dict[str, Any]:
    sample = clean_sample(base)
    sample.update({
        "run_id": run_id,
        "event_index": event_index,
        "cutoff_event_index": event_index,
        "current_role": role,
        "current_raw_action": raw_action,
        "split": "validation",
        "source_event_id": f"synthetic:{run_id}:{event_index}",
    })
    return sample


def synthetic_tool(base: dict[str, Any], run_id: str, event_index: int, role: str) -> dict[str, Any]:
    sample = copy.deepcopy(base)
    for key in ("target_raw_action", "target_source_event_id", "family_label", "target_event_index"):
        sample.pop(key, None)
    sample.update({
        "run_id": run_id,
        "cutoff_event_index": event_index,
        "position": event_index,
        "current_role": role,
        "prefix_id": f"synthetic:{run_id}:semantic:{event_index}",
        "split": "validation",
    })
    return sample


def predicted_raw_action(role: str, family: str | None) -> str:
    if role == "plan":
        return "planner.generate"
    if role == "aggregate":
        return "generalist.generate"
    if role == "execute":
        return FAMILY_TO_RAW.get(family or "other", "other")
    if role == "terminate":
        return "end"
    return "unknown"


def prototype_for(proto: dict[tuple[str, str | None], dict[str, Any]], role: str, family: str | None) -> dict[str, Any]:
    value = proto.get((role, family if role == "execute" else None))
    if value is not None:
        return value
    value = proto.get((role, None))
    if value is not None:
        return {**value, "prototype_source": "template_role_fallback"}
    return {"model_id": "unknown", "execution_lane": "unknown", "raw_action": predicted_raw_action(role, family), "prototype_source": "unsupported"}


def expand_scenarios(
    runner: B05Runner,
    base: dict[str, Any],
    actual_history: list[dict[str, Any]],
    actual_tool_by_cutoff: dict[int, dict[str, Any]],
    prototypes: dict[tuple[str, str | None], dict[str, Any]],
    horizon: int,
    beam_size: int,
    *,
    collect_horizons: tuple[int, ...] | None = None,
    initial_role_prob: np.ndarray | None = None,
    initial_family_prob: np.ndarray | None = None,
) -> list[dict[str, Any]] | dict[int, list[dict[str, Any]]]:
    real_run = str(base["run_id"])
    synthetic_run = f"synthetic:{real_run}:{base.get('event_index', 0)}"
    prefix = [copy.deepcopy(row) for row in actual_history if int(row.get("event_index", 0)) < int(base.get("event_index", 0))]
    initial_current = clean_sample(base)
    initial_current["run_id"] = synthetic_run
    initial_current["split"] = "validation"
    base_index = int(base.get("event_index", 0))
    beams: list[dict[str, Any]] = [{"probability": 1.0, "steps": [], "history": prefix}]
    snapshots: dict[int, list[dict[str, Any]]] = {}

    def normalized(current_beams: list[dict[str, Any]]) -> list[dict[str, Any]]:
        total = sum(float(item["probability"]) for item in current_beams) or 1.0
        return [
            {
                "scenario_id": f"s{index:02d}",
                "scenario_probability": float(item["probability"]) / total,
                "steps": item["steps"],
                "synthetic_rollout": True,
            }
            for index, item in enumerate(current_beams)
        ]

    for offset in range(1, horizon + 1):
        expanded: list[dict[str, Any]] = []
        for beam in beams:
            history_index = {synthetic_run: beam["history"]}
            current_index = base_index + offset - 1
            if offset == 1:
                current = copy.deepcopy(initial_current)
            else:
                current = synthetic_role(
                    base,
                    synthetic_run,
                    current_index,
                    str(beam.get("last_role") or "plan"),
                    str(beam.get("last_raw_action") or "planner.generate"),
                )
            current["cutoff_event_index"] = current_index
            role_prob = initial_role_prob if offset == 1 and initial_role_prob is not None else runner.predict_one(current, history_index, "role")
            for role_item in top_probabilities(role_prob, ROLE_LABELS, min(beam_size, len(ROLE_LABELS))):
                role = role_item["label"]
                family_items = [{"label": None, "probability": 1.0, "source": "role_only"}]
                if role == "execute":
                    tool_base = actual_tool_by_cutoff.get(current_index)
                    if tool_base is None:
                        tool_candidates = [row for row in getattr(runner, "tool_train", []) if row.get("current_role") in {"plan", "aggregate"}]
                        if not tool_candidates:
                            continue
                        tool_base = tool_candidates[0]
                    if offset == 1 and initial_family_prob is not None:
                        family_prob = initial_family_prob
                    else:
                        tool_sample = synthetic_tool(tool_base, synthetic_run, current_index, str(current.get("current_role")))
                        family_prob = runner.predict_one(tool_sample, history_index, "family")
                    family_items = [
                        {"label": item["label"], "probability": item["probability"], "source": "B05_family"}
                        for item in top_probabilities(family_prob, FAMILY_LABELS, min(beam_size, len(FAMILY_LABELS)))
                    ]
                for family_item in family_items:
                    family = family_item["label"]
                    probability = float(beam["probability"]) * float(role_item["probability"]) * float(family_item["probability"])
                    proto = prototype_for(prototypes, role, family)
                    step = {
                        "step_offset": offset,
                        "role": role,
                        "action_family": family,
                        "raw_action": predicted_raw_action(role, family),
                        "model_id": proto["model_id"],
                        "execution_lane": proto["execution_lane"],
                        "prototype_source": proto["prototype_source"],
                        "role_probability": float(role_item["probability"]),
                        "family_probability": float(family_item["probability"]),
                    }
                    next_history = [copy.deepcopy(row) for row in beam["history"]]
                    next_history.append({
                        "event_index": current_index,
                        "current_role": current.get("current_role"),
                        "current_raw_action": current.get("current_raw_action"),
                        "status": "success",
                    })
                    next_history.append({
                        "event_index": current_index + 1,
                        "current_role": role,
                        "current_raw_action": step["raw_action"],
                        "status": "predicted",
                    })
                    expanded.append({
                        "probability": probability,
                        "steps": beam["steps"] + [step],
                        "history": next_history,
                        "last_role": role,
                        "last_raw_action": step["raw_action"],
                    })
        if not expanded:
            break
        expanded.sort(key=lambda item: (-item["probability"], json.dumps(item["steps"], sort_keys=True)))
        beams = expanded[:beam_size]
        if collect_horizons and offset in collect_horizons:
            snapshots[offset] = normalized(beams)
        if all(item.get("last_role") == "terminate" for item in beams):
            break
    if collect_horizons is not None:
        final_snapshot = normalized(beams)
        for requested in collect_horizons:
            snapshots.setdefault(requested, final_snapshot)
        return snapshots
    return normalized(beams)


def write_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def recompute_holdout(runner: B05Runner) -> dict[str, Any]:
    report: dict[str, Any] = {"seeds": [], "tolerance": 1e-5}
    for seed, checkpoint in zip((11, 22, 33), runner.checkpoint_paths):
        role_model, tool_model = runner.models[(11, 22, 33).index(seed)]
        role_rows = runner.role_validation
        tool_rows = runner.tool_validation
        saved = read_jsonl(runner.dataset_dir.parent / "evaluation" / "B05" / f"seed_{seed}" / "predictions.jsonl")
        saved_role = {str(row["source_event_id"]): np.asarray(row["probabilities"], dtype=np.float32) for row in saved if row.get("head") == "role"}
        saved_tool = {str(row["prefix_id"]): np.asarray(row["probabilities"], dtype=np.float32) for row in saved if row.get("head") == "family"}
        # The runner holds the averaged ensemble.  Recompute is a separate
        # structural check; exact seed equality is verified below by direct
        # single-model evaluation using the loaded state.
        role_batch = runner.predict_model_rows(role_model, role_rows, runner.fit_history, "role")
        tool_batch = runner.predict_model_rows(tool_model, tool_rows, runner.fit_history, "family")
        role_diffs = [float(np.max(np.abs(pred - saved_role[str(row["source_event_id"])]))) for row, pred in zip(role_rows, role_batch) if str(row["source_event_id"]) in saved_role]
        tool_diffs = [float(np.max(np.abs(pred - saved_tool[str(row["prefix_id"])]))) for row, pred in zip(tool_rows, tool_batch) if str(row["prefix_id"]) in saved_tool]
        report["seeds"].append({
            "seed": seed,
            "checkpoint": str(checkpoint),
            "role_rows_compared": len(role_diffs),
            "family_rows_compared": len(tool_diffs),
            "role_max_abs_diff": max(role_diffs, default=None),
            "family_max_abs_diff": max(tool_diffs, default=None),
            "pass": max(role_diffs, default=0.0) <= report["tolerance"] and max(tool_diffs, default=0.0) <= report["tolerance"],
        })
    report["pass"] = all(row["pass"] for row in report["seeds"])
    return report


def leakage_audit(runner: B05Runner, role_rows: list[dict[str, Any]], tool_rows: list[dict[str, Any]], role_index: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for row in role_rows[:20]:
        changed = clean_sample(row)
        changed["next_role"] = "terminate" if row.get("next_role") != "terminate" else "plan"
        changed["target_source_event_id"] = "leakage-test"
        rid = str(row["run_id"])
        a = runner.predict_one(row, role_index, "role")
        b = runner.predict_one(changed, role_index, "role")
        checks.append({"head": "role", "source_event_id": row.get("source_event_id"), "max_abs_diff": float(np.max(np.abs(a - b)))})
    for row in tool_rows[:20]:
        changed = copy.deepcopy(row)
        changed["family_label"] = "detect"
        changed["target_raw_action"] = "leakage-test"
        changed["target_source_event_id"] = "leakage-test"
        a = runner.predict_one(row, role_index, "family")
        b = runner.predict_one(changed, role_index, "family")
        checks.append({"head": "family", "prefix_id": row.get("prefix_id"), "max_abs_diff": float(np.max(np.abs(a - b)))})
    return {
        "schema_version": "scheduling-future-leakage-audit-v1",
        "label_mutation_checks": len(checks),
        "max_label_mutation_diff": max((row["max_abs_diff"] for row in checks), default=0.0),
        "pass": max((row["max_abs_diff"] for row in checks), default=0.0) <= 1e-7,
        "input_contract": {
            "future_events_excluded": True,
            "ground_truth_excluded": True,
            "answer_text_used_as_feature": False,
            "video_id_used_as_feature": False,
            "target_labels_used_as_feature": False,
        },
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/processed/scheduling_future_v1_20260812")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--limit-nodes", type=int, default=0, help="Smoke-test limit; 0 means all 9366 template nodes")
    args = parser.parse_args()
    output_root = args.output_root
    if not output_root.is_dir():
        raise RuntimeError(f"E0 output root does not exist: {output_root}")
    dataset_dir = ROOT / "results/processed/behavior_nn_v1_r2/final_holdout/dataset"
    candidate_dir = ROOT / "results/processed/behavior_nn_v1_r1v5_candidate/data"
    checkpoint_paths = [
        ROOT / "results/processed/behavior_nn_v1_r2/final_holdout/evaluation/B05/seed_11/checkpoint.pt",
        ROOT / "results/processed/behavior_nn_v1_r2/final_holdout/evaluation/B05/seed_22/checkpoint.pt",
        ROOT / "results/processed/behavior_nn_v1_r2/final_holdout/evaluation/B05/seed_33/checkpoint.pt",
    ]
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    runner = B05Runner(dataset_dir, checkpoint_paths, args.device)
    candidate_role = read_jsonl(candidate_dir / "role_event_samples.jsonl")
    candidate_tool = read_jsonl(candidate_dir / "semantic_tool_samples.jsonl")
    role_index = actual_history_index(candidate_role)
    role_by_source = {str(row["source_event_id"]): row for row in candidate_role}
    tool_by_run_cutoff = {(str(row["run_id"]), int(row["cutoff_event_index"])): row for row in candidate_tool}
    tool_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tool_cutoffs_by_run: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in candidate_tool:
        run_id = str(row["run_id"])
        tool_by_run[run_id].append(row)
        tool_cutoffs_by_run[run_id][int(row["cutoff_event_index"])] = row
    templates = read_jsonl(ROOT / "results/processed/workload_v0_2_smoke_fixed_20260812/job_templates.jsonl")
    prototypes = prototype_map(templates)

    holdout_report = recompute_holdout(runner)
    if not holdout_report["pass"]:
        raise RuntimeError("B05 holdout recomputation failed: " + json.dumps(holdout_report, ensure_ascii=False))
    leakage_report = leakage_audit(runner, candidate_role, candidate_tool, role_index)
    if not leakage_report["pass"]:
        raise RuntimeError("B05 leakage audit failed: " + json.dumps(leakage_report, ensure_ascii=False))

    # Encode the canonical candidate rows once in official-size batches.  The
    # synthetic future steps remain causal one-row calls, but the repeated
    # observed-prefix work is no longer paid once per template node.
    role_probabilities = runner.predict_rows(candidate_role, role_index, "role")
    family_probabilities = runner.predict_rows(candidate_tool, role_index, "family")
    role_prob_by_source = {str(row["source_event_id"]): role_probabilities[index] for index, row in enumerate(candidate_role)}
    family_prob_by_prefix = {str(row["prefix_id"]): family_probabilities[index] for index, row in enumerate(candidate_tool)}

    expected_nodes = 9366 if args.limit_nodes <= 0 else args.limit_nodes
    node_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for template in templates:
        proto = prototypes[str(template["template_id"])]
        for node in template.get("nodes") or []:
            if len(node_rows) >= expected_nodes:
                break
            source_id = str((node.get("source_event_ids") or [""])[0])
            role_row = role_by_source.get(source_id)
            if role_row is None:
                missing.append(source_id)
                continue
            actual_history = role_index.get(str(role_row["run_id"]), [])
            cutoff = int(role_row.get("event_index", 0))
            tool_sample = tool_by_run_cutoff.get((str(role_row["run_id"]), cutoff))
            if tool_sample is None:
                tool_sample = tool_by_run.get(str(role_row["run_id"]), [None])[0]
            role_prob = role_prob_by_source.get(source_id)
            if role_prob is None:
                role_prob = runner.predict_one(role_row, role_index, "role")
            family_prob = None
            family_source = "unavailable"
            if tool_sample is not None:
                family_prob = family_prob_by_prefix.get(str(tool_sample.get("prefix_id")))
                if family_prob is None:
                    family_prob = runner.predict_one(tool_sample, role_index, "family")
                family_source = "observed_tool_prefix" if tool_by_run_cutoff.get((str(role_row["run_id"]), cutoff)) is not None else "same_run_tool_template"
            all_horizons = expand_scenarios(
                runner,
                role_row,
                actual_history,
                tool_cutoffs_by_run.get(str(role_row["run_id"]), {}),
                proto,
                5,
                args.beam_size,
                collect_horizons=(1, 3, 5),
                initial_role_prob=role_prob,
                initial_family_prob=family_prob,
            )
            h1 = all_horizons[1]
            h3 = all_horizons[3]
            h5 = all_horizons[5]
            prefix_ids = [str(row.get("source_event_id")) for row in actual_history if int(row.get("event_index", 0)) < cutoff]
            node_rows.append({
                "schema_version": "scheduling-future-b05-node-v1",
                "template_id": template["template_id"],
                "video_id": template.get("video_id"),
                "baseline": template.get("baseline"),
                "model_stack_id": template.get("model_stack_id"),
                "node_id": node.get("node_id"),
                "source_event_id": source_id,
                "sequence_index": node.get("sequence_index"),
                "prefix_hash": canonical_hash(prefix_ids),
                "current_role": role_row.get("current_role"),
                "current_raw_action": role_row.get("current_raw_action"),
                "role_top": top_probabilities(role_prob, ROLE_LABELS, len(ROLE_LABELS)),
                "family_top": top_probabilities(family_prob, FAMILY_LABELS, len(FAMILY_LABELS)) if family_prob is not None else [],
                "family_source": family_source,
                "future_h1": h1,
                "future_h3": h3,
                "future_h5": h5,
                "input_contract": {
                    "future_events_excluded": True,
                    "target_labels_excluded": True,
                    "observed_resource_targets_excluded": True,
                    "synthetic_rollout": True,
                },
            })
            if len(node_rows) % 100 == 0:
                print(json.dumps({"progress_nodes": len(node_rows), "target_nodes": expected_nodes}), flush=True)
    if missing:
        raise RuntimeError(f"source-event coverage failed: missing={len(missing)}")
    if len(node_rows) != expected_nodes:
        raise RuntimeError(f"expected {expected_nodes} node rows, got {len(node_rows)}")

    artifact_dir = output_root / "prediction_artifacts"
    audit_dir = output_root / "leakage_audit"
    artifact_dir.mkdir(exist_ok=True)
    audit_dir.mkdir(exist_ok=True)
    h1_rows = [{**row, "future_h3": None, "future_h5": None} for row in node_rows]
    h3_rows = [{**row, "role_top": None, "family_top": None, "future_h1": None, "future_h5": None} for row in node_rows]
    h5_rows = [{**row, "role_top": None, "family_top": None, "future_h1": None, "future_h3": None} for row in node_rows]
    outputs = {
        "h1": {"path": str(artifact_dir / "b05_node_h1.jsonl.gz"), "rows": write_gzip_jsonl(artifact_dir / "b05_node_h1.jsonl.gz", h1_rows)},
        "h3": {"path": str(artifact_dir / "b05_future_h3.jsonl.gz"), "rows": write_gzip_jsonl(artifact_dir / "b05_future_h3.jsonl.gz", h3_rows)},
        "h5": {"path": str(artifact_dir / "b05_future_h5.jsonl.gz"), "rows": write_gzip_jsonl(artifact_dir / "b05_future_h5.jsonl.gz", h5_rows)},
    }
    (audit_dir / "b05_recompute_report.json").write_text(json.dumps(holdout_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (audit_dir / "b05_leakage_report.json").write_text(json.dumps(leakage_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "scheduling-future-b05-manifest-v1",
        "model_id": "B05",
        "seeds": [11, 22, 33],
        "beam_size": args.beam_size,
        "horizons": [1, 3, 5],
        "device": str(args.device),
        "candidate_role_rows": len(candidate_role),
        "candidate_tool_rows": len(candidate_tool),
        "templates": len(templates),
        "nodes": len(node_rows),
        "coverage_denominator": 9366,
        "source_event_coverage": len(node_rows) / 9366.0,
        "fit_role_rows": len(runner.role_train),
        "fit_tool_rows": len(runner.tool_train),
        "checkpoints": [{"path": str(path), "sha256": sha256(path)} for path in checkpoint_paths],
        "outputs": outputs,
        "holdout_recompute": holdout_report,
        "leakage_audit": leakage_report,
        "input_contract": {
            "fit_split": "train_only",
            "target_labels_used_as_features": False,
            "future_events_used_as_features": False,
            "video_id_used_as_feature": False,
            "resource_truth_used_as_feature": False,
        },
    }
    (artifact_dir / "b05_artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"nodes": len(node_rows), "coverage": len(node_rows) / 9366.0, "outputs": outputs, "holdout_recompute_pass": holdout_report["pass"], "leakage_pass": leakage_report["pass"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
