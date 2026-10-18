#!/usr/bin/env python3
"""Low-dimensional arrival-summary analysis (advisor 2026-10-16, approved).

Zero new DES: rebuilds the frozen arrival sequences for the 8 P2 states'
scenarios (arrival-count anchor + t+500ms rule, from the frozen workload),
reconciles against donor_meta, computes pre-registered summaries, and runs
BOTH evaluations:
  A) In-sample Summary-Oracle VOI (group-optimal within summary groups; report
     group sizes / singleton rate / window coverage / retention);
  B) Leave-One-Scenario-Out decision gain vs I0 (same-summary-group best from
     the OTHER scenarios, >=2 matches required else fallback I0).
Summaries: S1 exact counts (10/30/60s) + coarse buckets 0/1/2-3/>=4;
S2 per-GPU-model demand counts (10/30/60s) + binary mask; S3 fixed 60s window
in two 30s buckets (model-demand combination encoding).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

P2 = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/gate1a_p2_expanded.json"
WORKLOAD = ROOT / "experiments/EXP-20261014_burst64_v1/high_contention/workload_burst64_r150_b15_train.jsonl"
TEMPLATES = ROOT / "results/processed/r7_workload_v041_ontology_no_run_container/job_templates_r7_v041.jsonl"
OUT = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/arrival_summary_analysis_expanded.json"
DONORS = [0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19]
WINDOWS = [10000.0, 30000.0, 60000.0]


def coarse(n: int) -> str:
    return "0" if n == 0 else ("1" if n == 1 else ("2-3" if n <= 3 else ">=4"))


def main() -> int:
    p2 = json.loads(P2.read_text(encoding="utf-8"))
    eps = [json.loads(l) for l in WORKLOAD.read_text(encoding="utf-8").strip().split("\n")]
    # template -> set of gpu models
    tpl_models = {}
    for line in TEMPLATES.read_text(encoding="utf-8").strip().split("\n"):
        row = json.loads(line)
        models = {str(n.get("model_id")) for n in (row.get("nodes") or [])
                  if n.get("execution_lane") == "gpu" and n.get("model_id")}
        if models:
            tpl_models[str(row["template_id"])] = models
    ep_by_id = {e["episode_id"]: e for e in eps}

    results = []
    for st in p2["states"]:
        if "table" not in st:
            continue
        base_id, frac_s = st["state"].split("@")
        base = ep_by_id[base_id]
        now = float(st["now_ms"])
        # rebuild scenarios: orig + donors (arrival-count anchor + 500ms rule)
        scen_arrivals = {}
        keep = [j for j in base["jobs"] if float(j["arrival_ms"]) <= now]
        k_arr = len(keep)
        scen_arrivals["orig"] = [(float(j["arrival_ms"]), str(j["template_id"]))
                                 for j in base["jobs"] if float(j["arrival_ms"]) > now]
        for d in DONORS:
            donor = eps[d]
            ds = sorted(donor["jobs"], key=lambda j: float(j["arrival_ms"]))
            if len(ds) <= k_arr:
                continue
            donor_t = float(ds[k_arr - 1]["arrival_ms"])
            fut = ds[k_arr:]
            shift = 0.0
            if fut:
                fr = float(fut[0]["arrival_ms"]) - donor_t
                if fr < 500.0:
                    shift = 500.0 - fr
            scen_arrivals[f"d{d}"] = [(now + shift + (float(j["arrival_ms"]) - donor_t), str(j["template_id"]))
                                      for j in fut]
        # reconcile with donor_meta
        recon = []
        for sname, arr in scen_arrivals.items():
            meta = (st.get("donor_meta") or {}).get(sname)
            if meta and (meta["n_new_jobs"] != len(arr)
                         or abs(meta["first_gap_ms"] - round(arr[0][0] - now, 1)) > 1.0):
                recon.append(f"{sname}: meta {meta['n_new_jobs']}/{meta['first_gap_ms']} vs rebuilt {len(arr)}/{round(arr[0][0]-now,1)}")
        # summaries per scenario
        def summaries(arr):
            out = {}
            for w in WINDOWS:
                in_w = [(t, tid) for t, tid in arr if t - now <= w]
                out[f"S1_exact_{int(w/1000)}s"] = str(len(in_w))
                models = [m for _t, tid in in_w for m in tpl_models.get(tid, set())]
                out[f"S2_counts_{int(w/1000)}s"] = "|".join(sorted(f"{m}:{models.count(m)}" for m in set(models)))
                out[f"S2_mask_{int(w/1000)}s"] = "|".join(sorted(set(models)))
            in60 = [(t, tid) for t, tid in arr if t - now <= 60000.0]
            out["S1_coarse_60s"] = coarse(len(in60))
            b1 = sorted({m for t, tid in in60 if t - now <= 30000.0 for m in tpl_models.get(tid, set())})
            b2 = sorted({m for t, tid in in60 if t - now > 30000.0 for m in tpl_models.get(tid, set())})
            out["S3_buckets_60s"] = "|".join(b1) + "//" + "|".join(b2)
            out["coverage_60s"] = str(len(in60))
            return out
        sc_sum = {s: summaries(a) for s, a in scen_arrivals.items()}
        keys = sorted(next(iter(sc_sum.values())).keys())
        common = set.intersection(*[set(st["paired_deltas"][s].keys()) for s in scen_arrivals
                                    if s in st["paired_deltas"]])
        # evaluation per summary key
        per_key = {}
        for key in keys:
            groups = {}
            for s in scen_arrivals:
                if s not in st["paired_deltas"]:
                    continue
                g = sc_sum[s][key]
                groups.setdefault(g, []).append(s)
            # A: in-sample summary-oracle
            cost_a = []
            for s in scen_arrivals:
                if s not in st["paired_deltas"]:
                    continue
                g = sc_sum[s][key]
                members = groups[g]
                best = min(common, key=lambda a: statistics.fmean(st["paired_deltas"][m][a] for m in members))
                cost_a.append(st["paired_deltas"][s][best])
            voi_full = st["voi_ms"]
            i0 = statistics.fmean([st["paired_deltas"][s][st["best_fixed_action"]] for s in st["paired_deltas"]])
            voi_a = i0 - statistics.fmean(cost_a)
            # B: leave-one-scenario-out
            gains, covered, switched = [], 0, 0
            for s in scen_arrivals:
                if s not in st["paired_deltas"]:
                    continue
                train = [x for x in scen_arrivals if x != s and x in st["paired_deltas"]]
                if not train:
                    continue
                i0_act = min(common, key=lambda a: statistics.fmean(st["paired_deltas"][x][a] for x in train))
                g = sc_sum[s][key]
                match = [x for x in train if sc_sum[x][key] == g]
                if len(match) >= 2:
                    act = min(common, key=lambda a: statistics.fmean(st["paired_deltas"][x][a] for x in match))
                    covered += 1
                    if act != i0_act:
                        switched += 1
                else:
                    act = i0_act
                gains.append(st["paired_deltas"][s][i0_act] - st["paired_deltas"][s][act])
            per_key[key] = {
                "n_groups": len(groups), "sizes": sorted((len(v) for v in groups.values()), reverse=True),
                "singletons": sum(1 for v in groups.values() if len(v) == 1),
                "in_sample_voi_ms": round(voi_a, 1),
                "retention": round(voi_a / voi_full, 3) if abs(voi_full) > 1e-9 else None,
                "loo_mean_gain_ms": round(statistics.fmean(gains), 1) if gains else None,
                "loo_coverage": round(covered / max(1, len(gains)), 2),
                "loo_switches": switched,
                "loo_worst_ms": round(min(gains), 1) if gains else None,
            }
        results.append({"state": st["state"], "stratum": st["stratum"],
                        "voi_full_ms": st["voi_ms"], "n_scenarios": len(scen_arrivals),
                        "reconcile_issues": recon, "summaries": per_key})

    report = {"schema": "arrival-summary-analysis-v1", "states": results}
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{'state':>18} {'summary':>16} {'groups':>6} {'singl':>5} {'inVOI':>7} {'retain':>6} {'LOOgain':>8} {'cov':>4}")
    for st in results:
        for key in ("S1_exact_60s", "S1_coarse_60s", "S2_counts_60s", "S3_buckets_60s"):
            v = st["summaries"].get(key)
            if v:
                print(f"{st['state'][-12:]:>18} {key:>16} {v['n_groups']:>6} {v['singletons']:>5} "
                      f"{v['in_sample_voi_ms']:>7} {str(v['retention']):>6} {str(v['loo_mean_gain_ms']):>8} {v['loo_coverage']:>4}")
    print("evidence:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
