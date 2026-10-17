#!/usr/bin/env python3
"""H5-v1 targeted features (advisor step 2): rebuild the 113 sampled states,
extract decision-time-visible future-contention features, join with the cached
labels by (state, action).  No hidden suffix / future arrivals are used.

New features (per candidate action (node, gpu)):
  next3_not_resident / next3_coldload_ms / next3_resident_bytes  -- the
    candidate's own predicted next 1-3 steps vs the target GPU's residency;
  comp_models / comp_overlap / comp_not_resident_any  -- competing ready
    requests' predicted model demand (cache/GPU contention);
  terminates_within3 -- predicted chain ends within 3 steps.
Cold-load cost comes from the frozen TRAIN profile (model -> load_p50), not
from the zeroed load_occurrence_probability fields.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import milp_v0_tests as T  # noqa: E402
import milp_v0_performance as P  # noqa: E402
from tracing.analysis.des_rollout import build_planning_episode, run_planning  # noqa: E402
from tracing.analysis.profile_contract import apply_real_workload_profile_contract  # noqa: E402
from tracing.analysis.workload_v02_simulator import estimate, read_jsonl, simulate_episode  # noqa: E402

FORMAL = ROOT / "experiments/EXP-20261014_burst64_v1/high_contention"
DATASET = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/av_v1_dataset.json"
OUT = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/av_v2_dataset.json"


def main() -> int:
    config_path = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/serial_prefetch_v1_extension.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _serial, templates, _n, stats, artifacts = P.load_split("confirm", 1, config_path)
    tm = {k: replace(v, split="train") for k, v in templates.items()}
    rows = json.loads(DATASET.read_text(encoding="utf-8"))

    # frozen train-only model -> load_p50 profile
    model_load: dict[str, float] = {}
    for t in templates.values():
        if t.split != "train":
            continue
        for nd in t.nodes:
            if nd.lane == "gpu" and nd.model_id:
                row = estimate(nd, stats)
                model_load.setdefault(str(nd.model_id), float(row["load_p50_ms"] or 0.0))
    print(f"profile: {len(model_load)} models, load_ms sample:",
          {k: round(v) for k, v in list(model_load.items())[:4]})

    by_state: dict[str, list[dict]] = {}
    for r in rows:
        by_state.setdefault(r["state"], []).append(r)
    print(f"states to rebuild: {len(by_state)}")

    train_eps = read_jsonl(FORMAL / "workload_burst64_r150_b15_train.jsonl")
    dev_eps = read_jsonl(FORMAL / "workload_burst64_r150_b15_validation.jsonl")
    for row in train_eps + dev_eps:
        row["split"] = "train"
    tm, train_eps = apply_real_workload_profile_contract(tm, train_eps, config)
    tm, dev_eps = apply_real_workload_profile_contract(tm, dev_eps, config)

    out_rows = []
    done = 0
    for state_id, srows in sorted(by_state.items()):
        partition, rest = state_id.split(":", 1)
        ep_id, frac_s = rest.split("@")
        frac = float(frac_s)
        eps = train_eps if partition == "train" else dev_eps
        ep = next((e for e in eps if e["episode_id"] == ep_id), None)
        if ep is None:
            print("missing episode:", state_id)
            continue
        summary, _ev = simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                                        future_artifacts=artifacts, future_horizon=5,
                                        extension_config=config,
                                        policy_context={"activation": {}}, collect_events=False)
        at = round(frac * float(summary["makespan_ms"]), 1)
        capture = {"at_ms_list": [at], "snapshots": []}
        simulate_episode(ep, tm, "pdrs_resident", train_stats=stats,
                         future_artifacts=artifacts, future_horizon=5,
                         extension_config=config,
                         policy_context={"activation": {}, "des_capture": capture},
                         collect_events=False)
        snaps = capture.get("snapshots") or []
        if not snaps:
            continue
        snap = snaps[0]
        jobs_meta = [{"job_instance_id": j["job_instance_id"], "template_id": j["template_id"]}
                     for j in ep["jobs"]]
        obs = T.obs_from_snapshot(snap, jobs_meta, tm, stats, artifacts)
        if abs(float(snap["now_ms"]) - float(srows[0]["now_ms"])) > 1e-6:
            print("timestamp drift, skipping:", state_id)
            continue

        # competitor demand: other ready GPU anchors' predicted next-3 models
        others_models: set[str] = set()
        for it in obs.ready:
            if it.lane != "gpu":
                continue
            for s in obs.chains.get(it.node_id, ())[:3]:
                if s.model_id:
                    others_models.add(str(s.model_id))

        for r in srows:
            node, gpu = str(r["action"][0]), int(r["action"][1])
            it = next((x for x in obs.ready if str(x.node_id) == node), None)
            gpu_row = obs.gpus[gpu] if gpu < len(obs.gpus) else None
            if it is None or gpu_row is None:
                continue
            chain = obs.chains.get(it.node_id, ())
            next3 = [s for s in chain[:3] if s.model_id]
            n3_models = [str(s.model_id) for s in next3]
            resident = gpu_row.resident
            not_res = [m for m in n3_models if m not in resident]
            any_res = set()
            for g in obs.gpus:
                any_res.update(str(m) for m in g.resident)
            feat = {
                "next3_not_resident": float(len(set(not_res))),
                "next3_coldload_ms": float(sum(model_load.get(m, 0.0) for m in set(not_res))),
                "next3_resident_bytes": float(sum(float(resident.get(m, 0.0)) for m in set(n3_models) if m in resident)),
                "comp_models": float(len(others_models)),
                "comp_overlap": float(len(set(n3_models) & others_models)),
                "comp_not_resident_any": float(len(others_models - any_res)),
                "terminates_within3": 1.0 if len(chain) <= 3 else 0.0,
            }
            out_rows.append({**r, **feat})
        done += 1
        if done % 20 == 0:
            print(f"  rebuilt {done}/{len(by_state)} states, rows={len(out_rows)}")

    OUT.write_text(json.dumps(out_rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"rows: {len(out_rows)} | states rebuilt: {done} | evidence: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
