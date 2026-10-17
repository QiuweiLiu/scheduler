#!/usr/bin/env python3
"""True-H5 oracle features + prediction-quality audit (advisor 2026-10-16).

Rebuilds the 113 sampled states and, per (state, action):
  - TRUE-H5 features from the episode's real template chain (next 1-3 nodes:
    models, compute, cold-load vs target-GPU residency, termination) -- this is
    an OFFLINE INFORMATION ORACLE, never a deployable input;
  - corrected predicted-H5 features: same quantities from the frozen predicted
    chains, with the competitor set now EXCLUDING the candidate itself;
  - audit rows: predicted vs true next-3 model-set match, termination match,
    duration error; per-state improvable flag (from cached labels).
Joins everything with the cached labels into dataset v3.
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
DATASET = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/av_v2_dataset.json"
OUT = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/av_v3_dataset.json"


def main() -> int:
    config_path = ROOT / "experiments/EXP-20261012_milp_v0/artifacts/serial_prefetch_v1_extension.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _serial, templates, _n, stats, artifacts = P.load_split("confirm", 1, config_path)
    tm = {k: replace(v, split="train") for k, v in templates.items()}
    rows = json.loads(DATASET.read_text(encoding="utf-8"))

    model_load: dict[str, float] = {}
    for t in templates.values():
        if t.split != "train":
            continue
        for nd in t.nodes:
            if nd.lane == "gpu" and nd.model_id:
                row = estimate(nd, stats)
                model_load.setdefault(str(nd.model_id), float(row["load_p50_ms"] or 0.0))

    by_state: dict[str, list[dict]] = {}
    for r in rows:
        by_state.setdefault(r["state"], []).append(r)
    print(f"states: {len(by_state)} | rows: {len(rows)}")

    train_eps = read_jsonl(FORMAL / "workload_burst64_r150_b15_train.jsonl")
    dev_eps = read_jsonl(FORMAL / "workload_burst64_r150_b15_validation.jsonl")
    for row in train_eps + dev_eps:
        row["split"] = "train"
    tm, train_eps = apply_real_workload_profile_contract(tm, train_eps, config)
    tm, dev_eps = apply_real_workload_profile_contract(tm, dev_eps, config)

    out_rows = []
    audit = []
    done = 0
    for state_id, srows in sorted(by_state.items()):
        partition, rest = state_id.split(":", 1)
        ep_id, frac_s = rest.split("@")
        frac = float(frac_s)
        eps = train_eps if partition == "train" else dev_eps
        ep = next((e for e in eps if e["episode_id"] == ep_id), None)
        if ep is None:
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
            continue

        # true chains per ready GPU node (from the real template)
        def true_next(it, k=3):
            node = tm[it.node_id.split(":")[0]].by_id[it.node_id]
            tpl = tm[it.node_id.split(":")[0]]
            nxt = sorted([nd for nd in tpl.nodes if nd.sequence_index > node.sequence_index],
                         key=lambda nd: nd.sequence_index)
            return nxt[:k], len(nxt)

        true_chains = {}
        for it in obs.ready:
            if it.lane == "gpu":
                true_chains[str(it.node_id)] = true_next(it)

        # competitor sets EXCLUDING the candidate (fix from advisor review)
        others_true: set[str] = set()
        others_pred: set[str] = set()
        for it in obs.ready:
            if it.lane != "gpu":
                continue
            nxt_t, _ = true_chains[str(it.node_id)]
            for nd in nxt_t:
                if nd.model_id:
                    others_true.add(str(nd.model_id))
            for s in obs.chains.get(it.node_id, ())[:3]:
                if s.model_id:
                    others_pred.add(str(s.model_id))

        # per-state improvable flag (cached labels)
        top5 = [r["label_dj"] for r in srows if r.get("in_top5")]
        improvable = bool(top5) and min(top5) < -1e-9

        for r in srows:
            node, gpu = str(r["action"][0]), int(r["action"][1])
            it = next((x for x in obs.ready if str(x.node_id) == node), None)
            gpu_row = obs.gpus[gpu] if gpu < len(obs.gpus) else None
            if it is None or gpu_row is None:
                continue
            resident = gpu_row.resident
            # TRUE next 1-3
            nxt_t, rem_len = true_chains[node]
            t_models = [str(nd.model_id) for nd in nxt_t if nd.model_id]
            t_not_res = [m for m in set(t_models) if m not in resident]
            t_compute = sum(max(0.1, float(estimate(nd, stats)["runtime_p50_ms"])
                                - float(estimate(nd, stats)["load_p50_ms"] or 0.0)) for nd in nxt_t if nd.lane == "gpu")
            # predicted next 1-3 (frozen)
            chain = obs.chains.get(node, ())
            p_models = [str(s.model_id) for s in chain[:3] if s.model_id]
            p_compute = sum(max(0.1, s.runtime_ms - s.load_ms) for s in chain[:3] if s.lane == "gpu")
            others_pred_wo_self = others_pred - set(p_models)  # exclude candidate's own models too (cache view)
            others_true_wo_self = others_true - set(t_models)
            feat = {
                "t_next3_not_resident": float(len(set(t_not_res))),
                "t_next3_coldload_ms": float(sum(model_load.get(m, 0.0) for m in set(t_not_res))),
                "t_next3_compute_ms": float(t_compute),
                "t_terminates_within3": 1.0 if rem_len <= 3 else 0.0,
                "t_comp_models": float(len(others_true_wo_self)),
                "t_comp_overlap": float(len(set(t_models) & others_true_wo_self)),
                "p_comp_models_fixed": float(len(others_pred_wo_self)),
                "p_comp_overlap_fixed": float(len(set(p_models) & others_pred_wo_self)),
            }
            out_rows.append({**r, **feat})
            audit.append({
                "state": state_id, "improvable": improvable,
                "model_set_match": len(set(t_models) & set(p_models)) == len(set(t_models) | set(p_models)) if (t_models or p_models) else True,
                "term_match": (rem_len <= 3) == (len(chain) <= 3),
                "dur_abs_err_ms": abs(t_compute - p_compute),
                "true_dur_ms": t_compute, "pred_dur_ms": p_compute,
            })
        done += 1
        if done % 30 == 0:
            print(f"  {done}/{len(by_state)} states, rows={len(out_rows)}")

    OUT.write_text(json.dumps(out_rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # audit summary
    imp = [a for a in audit if a["improvable"]]
    non = [a for a in audit if not a["improvable"]]
    def rate(rows_, key):
        return round(sum(1 for a in rows_ if a[key]) / max(1, len(rows_)), 3)
    import statistics
    summary = {
        "rows": len(audit), "improvable_rows": len(imp), "non_improvable_rows": len(non),
        "model_set_match_improvable": rate(imp, "model_set_match"),
        "model_set_match_non": rate(non, "model_set_match"),
        "term_match_improvable": rate(imp, "term_match"),
        "term_match_non": rate(non, "term_match"),
        "dur_err_improvable_mean": round(statistics.fmean([a["dur_abs_err_ms"] for a in imp]), 1) if imp else None,
        "dur_err_non_mean": round(statistics.fmean([a["dur_abs_err_ms"] for a in non]), 1) if non else None,
        "true_dur_mean": round(statistics.fmean([a["true_dur_ms"] for a in audit]), 1),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    (OUT.parent / "av_v3_prediction_audit.json").write_text(
        json.dumps({"summary": summary, "rows": audit}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("evidence:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
