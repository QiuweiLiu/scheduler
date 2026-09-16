#!/usr/bin/env python3
"""Pack the frozen J predictor (J3:seed11) into scheduler-facing future artifacts.

Outputs (per anchor row):
- j_future_h1/h3/h5.jsonl.gz : legacy-style event scenarios (1 scenario, prob 1.0)
  with predicted role/action_family/model_id (+probabilities) per step and an
  additional per-step ``resource`` block from the joint resource head.
- j_future_h5_layers.jsonl.gz : the repaired layer-H5 sidecar
  (validated with tracing.scheduling.future_topology.validate_layer_scenarios).

The packer is inference-only: it never fits, never reads test, and records the
checkpoint hash + contract flags in the manifest. Contract smoke runs on
P_dev validation anchors first; S_* inference needs a separate anchor builder.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts import j_series_common as common  # noqa: E402
from scripts import j_series_train_eval as jte  # noqa: E402

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"torch is required: {exc}")

from tracing.scheduling.future_topology import (  # noqa: E402
    LAYER_H5_SCHEMA_VERSION,
    project_event_scenarios_to_unary_layers,
    validate_layer_scenarios,
)

TAUS = common.TARGET_TAUS


def read_rows(path: Path) -> List[Dict[str, Any]]:
    return list(common.read_jsonl_gz(path))


def build_raw_action_map(train_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Counter]:
    mapping: Dict[str, Counter] = defaultdict(Counter)
    for row in train_rows:
        for slot in row["future"]:
            key = str(slot.get("action_family"))
            mapping[key][str(slot.get("raw_action"))] += 1
    return mapping


def mode(map_obj: Mapping[str, Counter], key: str) -> str | None:
    counter = map_obj.get(key)
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def lane_for(model_class: str) -> str:
    if model_class.startswith("cpu") or model_class in ("finish_argument",):
        return "cpu"
    return "gpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--anchors-file", type=Path, default=None, help="optional inference-only anchors (model_input + metadata); skips the dataset split")
    parser.add_argument("--min-steps", type=int, default=0, help="shape-matched control: emit at least this many steps per anchor (default 0 = predicted length)")
    parser.add_argument("--emit-distributions", action="store_true", help="C2/survival support: emit per-step model_probabilities and per-row length_probabilities")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    ctx = jte.make_ctx(args.config, None)
    spec = config["candidate"]["primary"]
    variant, seed = spec["variant"], spec["seed"]
    source_root = Path(config["candidate"]["source_run_root"])
    checkpoint_path = source_root / "runs" / variant / f"seed{seed}" / "checkpoint.pt"
    model, payload = jte.load_checkpoint(checkpoint_path, ctx, variant)
    model.eval()

    split_path = Path(config["output_root"]) / f"j_{args.split}.jsonl.gz"
    if args.anchors_file is not None:
        raw_rows = read_rows(args.anchors_file)
        rows = []
        for row in raw_rows:
            rows.append({**row, "future": [], "bounded_future_length": 0, "termination": 0})
        total_label = f"anchors:{args.anchors_file}"
    else:
        rows = read_rows(split_path)
        total_label = f"split:{args.split}"
    if args.limit:
        rows = rows[: args.limit]
    train_rows = read_rows(Path(config["output_root"]) / "j_train.jsonl.gz")
    raw_action_map = build_raw_action_map(train_rows)

    vocab = ctx.vocabs.slot
    role_values = vocab["role"].values
    family_values = vocab["action_family"].values
    model_values = vocab["model_class"].values

    arrays = common.encode_rows(rows, ctx.vocabs, ctx.horizon)
    out_rows: List[Dict[str, Any]] = []
    total_nodes = 0
    n_scenarios = 0
    with torch.no_grad():
        for start in range(0, len(rows), ctx.batch_size):
            indices = np.arange(start, min(start + ctx.batch_size, len(rows)), dtype=np.int64)
            batch = jte.to_torch_batch(arrays, indices, ctx.device)
            o = jte.forward(model, batch, variant, ctx)
            length = o["structure"]["length_logits"].argmax(-1).cpu().numpy()
            term = torch.sigmoid(o["structure"]["termination_logit"]).cpu().numpy()
            role_p = torch.softmax(o["attributes"]["role"], -1).cpu().numpy()
            fam_p = torch.softmax(o["attributes"]["action_family"], -1).cpu().numpy()
            model_p = torch.softmax(o["attributes"]["model_class"], -1).cpu().numpy()
            merged = torch.sigmoid(o["attributes"]["merged"]).cpu().numpy()
            retry = torch.sigmoid(o["attributes"]["retry"]).cpu().numpy()
            rt = torch.expm1(o["resource"]["runtime_log_quantiles"]).clamp(min=0.0).cpu().numpy()
            ld = torch.expm1(o["resource"]["load_dur_log_quantiles"]).clamp(min=0.0).cpu().numpy()
            occ = torch.sigmoid(o["resource"]["load_occ_logit"]).cpu().numpy()

            length_p = torch.softmax(o["structure"]["length_logits"], -1).cpu().numpy()
            for i, idx in enumerate(indices):
                row = rows[int(idx)]
                L = int(length[i])
                emit = max(L, int(args.min_steps)) if args.min_steps else L
                steps: List[Dict[str, Any]] = []
                for t in range(emit):
                    role_id = int(role_p[i, t].argmax())
                    fam_id = int(fam_p[i, t].argmax())
                    model_id = int(model_p[i, t].argmax())
                    role_name = role_values[role_id - 2] if role_id >= 2 else None
                    family_name = family_values[fam_id - 2] if fam_id >= 2 else None
                    model_name = model_values[model_id - 2] if model_id >= 2 else None
                    steps.append(
                        {
                            "step_offset": t + 1,
                            "role": role_name,
                            "role_probability": float(role_p[i, t, role_id]),
                            "action_family": family_name,
                            "family_probability": float(fam_p[i, t, fam_id]),
                            "model_id": model_name,
                            "model_probability": float(model_p[i, t, model_id]),
                            **(
                                {"model_probabilities": {model_values[k - 2]: float(model_p[i, t, k]) for k in range(2, len(model_values) + 2) if float(model_p[i, t, k]) >= 0.01}}
                                if args.emit_distributions
                                else {}
                            ),
                            "execution_lane": lane_for(str(model_name or "")),
                            "raw_action": mode(raw_action_map, str(family_name)),
                            "merged_nested_call": bool(merged[i, t] > 0.5),
                            "merged_nested_call_probability": float(merged[i, t]),
                            "is_retry": bool(retry[i, t] > 0.5),
                            "is_retry_probability": float(retry[i, t]),
                            "prototype_source": f"j_series_joint_predictor_{variant}_seed{seed}",
                            "resource": {
                                "runtime_ms_quantiles": {f"p{int(tau * 100)}": float(rt[i, t, k]) for k, tau in enumerate(TAUS)},
                                "load_occurrence_probability": float(occ[i, t]),
                                "load_duration_ms_quantiles": {f"p{int(tau * 100)}": float(ld[i, t, k]) for k, tau in enumerate(TAUS)},
                            },
                        }
                    )
                scenario = {
                    "scenario_id": "j-pred-1",
                    "scenario_probability": 1.0,
                    "synthetic_rollout": False,
                    "topology_source": f"j_series_joint_predictor_{variant}_seed{seed}",
                    "steps": steps,
                }
                total_nodes += len(steps)
                n_scenarios += 1
                stack = row["model_input"]["stack_context"]
                current = row["model_input"]["current_node"]
                out_rows.append(
                    {
                        "schema_version": "j-series-future-predictor-v1",
                        "template_id": None,
                        "node_id": row["current_node_id"],
                        "video_id": row["video_id"],
                        "run_id": row["run_id"],
                        "prefix_hash": row.get("prefix_hash"),
                        "sequence_index": None,
                        "source_event_id": None,
                        "baseline": stack.get("baseline"),
                        "model_stack_id": stack.get("model_stack_id"),
                        "current_role": current.get("role"),
                        "current_raw_action": current.get("raw_action"),
                        "role_top": steps[0]["role"] if steps else None,
                        "family_top": steps[0]["action_family"] if steps else None,
                        "family_source": f"j_series_joint_predictor_{variant}_seed{seed}",
                        "termination_probability": float(term[i]),
                        "predicted_future_length": L,
                        **({"length_probabilities": [float(x) for x in length_p[i]]} if args.emit_distributions else {}),
                        "input_contract": {
                            "fit_split": "P_dev_train_only",
                            "future_events_excluded": True,
                            "target_labels_excluded": True,
                            "observed_resource_targets_excluded": True,
                            "formal_topology_predictor": True,
                            "model": f"{variant}:seed{seed}",
                            "checkpoint_sha256": common.sha256_file(checkpoint_path),
                        },
                        **{f"future_h{n}": [{**scenario, "steps": steps[:n]}] for n in sorted({1, 3, 5, int(ctx.horizon)})},
                    }
                )

    # layer sidecar via the scheduler's own projection + validator
    horizons = sorted({1, 3, 5, int(ctx.horizon)})
    layer_rows = []
    node_counts = []
    for row in out_rows:
        scenarios = project_event_scenarios_to_unary_layers(row[f"future_h{int(ctx.horizon)}"], int(ctx.horizon))
        validate_layer_scenarios(scenarios, int(ctx.horizon))
        node_counts.extend([sum(len(layer["nodes"]) for layer in sc["layers"]) for sc in scenarios])
        layer_rows.append(
            {
                "schema_version": LAYER_H5_SCHEMA_VERSION,
                "template_id": row["template_id"],
                "node_id": row["node_id"],
                "future_h5_layers": scenarios,
                "input_contract": row["input_contract"],
            }
        )

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    written = {}
    for n in horizons:
        name = f"j_future_h{n}.jsonl.gz"
        key = f"future_h{n}"
        path = out_root / name
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
            for row in out_rows:
                handle.write(json.dumps({**{k: v for k, v in row.items() if not k.startswith("future_h")}, key: row[key]}, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        written[name] = common.sha256_file(path)
    layers_path = out_root / "j_future_h5_layers.jsonl.gz"
    with gzip.open(layers_path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in layer_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    written["j_future_h5_layers.jsonl.gz"] = common.sha256_file(layers_path)

    manifest = {
        "schema_version": "j-series-future-predictor-manifest-v1",
        "horizon": int(ctx.horizon),
        "source": total_label,
        "min_steps": int(args.min_steps),
        "split": args.split,
        "rows": len(out_rows),
        "scenarios": n_scenarios,
        "predicted_node_count": {
            "mean": total_nodes / max(1, n_scenarios),
            "min": int(min(node_counts)) if node_counts else 0,
            "max": int(max(node_counts)) if node_counts else 0,
        },
        "model": f"{variant}:seed{seed}",
        "checkpoint_sha256": common.sha256_file(checkpoint_path),
        "layer_schema_version": LAYER_H5_SCHEMA_VERSION,
        "layers_validated": True,
        "output_files": written,
        "notes": [
            "single-scenario deterministic chain (probability 1.0); no DAG width claimed",
            "resource block added per step (runtime p50/p90/p95, load occurrence prob, load duration p50/p90/p95)",
            "raw_action is the train-mode mapping given predicted action_family; execution_lane heuristic",
            "S_* inference requires a separate anchor feature builder (not part of this packer)",
        ],
    }
    (out_root / "j_predictor_artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=1)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
