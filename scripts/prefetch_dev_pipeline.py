"""Dev pipeline for exp G: probe states -> high-budget labels -> convergence ->
distilled value function -> frozen artifact.

Phases (docs/research/2026-10-10_exp_g_prefetch_value_methods.md):
  A probe   mainline trajectory on dev episodes; snapshot decision states that
            have real prefetch opportunities (bounded per episode)
  B labels  per state: no-prefetch + ALL feasible (model, device) actions,
            rollout over the SAME 256 sampled scenarios; benefit at scenario
            prefixes 8/16/32/64/128/256
  C convergence  argmax agreement of prefix-N vs N=256; freeze smallest N*
            with agreement >= 0.90 (disclosed criterion)
  D fit     linear least squares on the frozen interpretable features
            (train/val split by episode); save distilled_benefit_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from tracing.analysis.hermes_methods import model_load_estimate_ms  # noqa: E402
from tracing.analysis.prefetch_value_methods import (  # noqa: E402
    FEATURE_NAMES,
    DeviceState,
    FrontierItem,
    benefit_feature_table,
    build_scenarios,
    candidate_feature_vector,
    distilled_score,
    rollout_cost,
)
from tracing.analysis.profile_contract import apply_real_workload_profile_contract  # noqa: E402
from tracing.analysis.workload_v02_simulator import (  # noqa: E402
    load_resource_v2_overlay,
    load_templates,
    read_jsonl,
    simulate_episode,
    train_resource_stats,
)

PROJECTION = (ROOT / "results/processed/r7_workload_v041_ontology_no_run_container"
              / "job_templates_r7_v041.jsonl")
EPISODES_FILE = (ROOT / "results/processed/r7_workload_v03_no_run_container"
                 / "episodes/workload_validation_r7_v03.jsonl")
SPLIT_MANIFEST = ROOT / "data/manifests/validation_split_dev700_confirm300.json"
EXTENSION_CONFIG = (ROOT / "experiments/EXP-20261004_real_strata_calibration_v1"
                    / "artifacts/substrate_extension_v7.json")
BASE_ARTIFACTS = ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"
F0_ARTIFACTS = ROOT / "outputs/resource_v2_artifacts/f0_seed11"
ART = ROOT / "experiments/EXP-20261010_prefetch_value_methods_v1/artifacts"
PROBE_POLICY = "pdrs_resident_probe"
PREFIX_GRID = (8, 16, 32, 64, 128, 256)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r_squared(y: Sequence[float], yhat: Sequence[float]) -> float:
    y = list(y)
    mean = sum(y) / len(y)
    ss_tot = sum((v - mean) ** 2 for v in y)
    ss_res = sum((v - h) ** 2 for v, h in zip(y, yhat))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--probe-cap", type=int, default=6)
    parser.add_argument("--scenarios", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--tag", default="v1")
    args = parser.parse_args()

    extension_config = json.loads(EXTENSION_CONFIG.read_text(encoding="utf-8"))
    templates = load_templates(PROJECTION, topology_view="causal_v3")
    split_manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    by_id = {str(e.get("episode_id")): e for e in read_jsonl(EPISODES_FILE)}
    dev_episodes = [by_id[eid] for eid in sorted(set(split_manifest["development"])) if eid in by_id][:args.episodes]
    templates, dev_episodes = apply_real_workload_profile_contract(templates, dev_episodes, extension_config)
    stats = train_resource_stats({k: v for k, v in templates.items() if v.split == "train"})
    pack, preflight = load_resource_v2_overlay(BASE_ARTIFACTS, F0_ARTIFACTS)

    def memory_mb_of(model: str) -> float:
        for template in templates.values():
            for node in template.nodes:
                if node.model_id == model and node.resident_model_mb is not None:
                    return float(node.resident_model_mb)
        return 0.0

    load_ms_of = lambda model: float(model_load_estimate_ms(stats, model) or 0.0)

    # ---------- Phase A: probe ----------
    started = time.time()
    states: List[Dict[str, Any]] = []
    for episode in dev_episodes:
        context: Dict[str, Any] = {"activation": {}, "probe_states": [], "probe_cap": args.probe_cap}
        summary, _ = simulate_episode(
            episode, templates, PROBE_POLICY, train_stats=stats, policy_context=context,
            future_artifacts=pack, future_horizon=5, extension_config=extension_config,
            collect_events=False)
        if int(summary.get("failed_jobs") or 0) != 0:
            raise AssertionError("probe episode %s failed" % episode.get("episode_id"))
        states.extend(context["probe_states"])
    print("phase A: %d states from %d dev episodes (%.1f s)"
          % (len(states), len(dev_episodes), time.time() - started))

    # ---------- Phase B: high-budget labels ----------
    label_rows: List[Dict[str, Any]] = []
    scenario_cache: Dict[int, Any] = {}
    for state_index, state in enumerate(states):
        items = []
        skip = False
        for entry in state["items"]:
            row = pack.get(str(entry["node_id"]))
            if not row or not (row.get("future_h5") or []):
                skip = True
                break
            items.append(FrontierItem(key=str(entry["node_id"]), row=row,
                                      current_model=str(entry["current_model"]),
                                      current_remaining_ms=float(entry["current_remaining_ms"]),
                                      current_mb=float(entry["current_mb"])))
        if skip or not items:
            continue
        devices = [DeviceState(capacity_mb=float(d["capacity_mb"]),
                               resident={str(m): float(mb) for m, mb in (d["resident"] or {}).items()},
                               busy_until=float(d["busy_until"]))
                   for d in state["devices"]]
        now = float(state["time_ms"])
        scenarios = build_scenarios(items, count=args.scenarios, seed=args.seed)
        none_costs = [rollout_cost(items, sc, devices, None, load_ms_of, memory_mb_of, now)
                      for sc in scenarios]
        action_rows: List[Dict[str, Any]] = []
        table = benefit_feature_table(items, load_ms_of)
        for model, gpu_index, interference in state["candidates"]:
            costs = [rollout_cost(items, sc, devices, (model, int(gpu_index)),
                                  load_ms_of, memory_mb_of, now) for sc in scenarios]
            deltas = {}
            for n in PREFIX_GRID:
                none_mean = sum(none_costs[:n]) / n
                action_mean = sum(costs[:n]) / n
                deltas[str(n)] = none_mean - action_mean
            features = candidate_feature_vector(
                table.get(model, {}), model, int(gpu_index), devices,
                memory_mb_of, float(interference), load_ms_of)
            action_rows.append({
                "model": model, "gpu": int(gpu_index),
                "delta": deltas, "features": features,
            })
        label_rows.append({
            "episode_id": state["episode_id"], "state_index": state_index,
            "time_ms": now, "n_actions": len(action_rows), "actions": action_rows,
        })
    print("phase B: %d states labeled, %d action rows (%.1f s)"
          % (len(label_rows), sum(row["n_actions"] for row in label_rows), time.time() - started))

    # ---------- Phase C: convergence ----------
    def chosen_at(row: Dict[str, Any], n: int) -> Tuple[str, float]:
        best = ("__none__", 0.0)
        for action in row["actions"]:
            value = action["delta"][str(n)]
            label = "%s@%d" % (action["model"], action["gpu"])
            if value > best[1] + 1e-9:
                best = (label, value)
        return best

    convergence: Dict[str, Any] = {}
    for n in PREFIX_GRID:
        agree = 0
        total = 0
        abs_err = []
        for row in label_rows:
            if n == args.scenarios:
                continue
            for action in row["actions"]:
                abs_err.append(abs(action["delta"][str(n)] - action["delta"][str(args.scenarios)]))
            total += 1
            if chosen_at(row, n)[0] == chosen_at(row, args.scenarios)[0]:
                agree += 1
        if total:
            convergence[str(n)] = {
                "choice_agreement": agree / total,
                "mean_abs_delta_error_ms": (sum(abs_err) / len(abs_err)) if abs_err else 0.0,
            }
    n_star = args.scenarios
    for n in PREFIX_GRID:
        if n == args.scenarios:
            break
        if convergence[str(n)]["choice_agreement"] >= 0.90:
            n_star = n
            break
    convergence["n_star"] = n_star
    print("phase C: convergence %s -> n_star=%d"
          % (json.dumps({k: (round(v["choice_agreement"], 3) if isinstance(v, dict) else v)
                         for k, v in convergence.items()}), n_star))

    # ---------- Phase D: fit distilled value function ----------
    episode_ids = sorted({row["episode_id"] for row in label_rows})
    cut = max(1, int(len(episode_ids) * args.train_frac))
    train_eps = set(episode_ids[:cut])
    X_train, y_train, X_val, y_val = [], [], [], []
    val_rows = []
    for row in label_rows:
        in_train = row["episode_id"] in train_eps
        for action in row["actions"]:
            vector = [float(action["features"][name]) for name in FEATURE_NAMES]
            label = float(action["delta"][str(args.scenarios)])
            if in_train:
                X_train.append(vector)
                y_train.append(label)
            else:
                X_val.append(vector)
                y_val.append(label)
    Xt = np.asarray(X_train, dtype=float)
    yt = np.asarray(y_train, dtype=float)
    # frozen fitting protocol: z-score features + small ridge, then rescale the
    # coefficients back to raw-feature space so online scoring is unchanged
    mu = Xt.mean(axis=0)
    sd = Xt.std(axis=0)
    sd[sd == 0.0] = 1.0
    Xts = (Xt - mu) / sd
    ridge = 1e-3
    beta_std = np.linalg.solve(Xts.T @ Xts + ridge * np.eye(Xts.shape[1]), Xts.T @ yt)
    beta_vec = beta_std / sd
    beta = {name: float(value) for name, value in zip(FEATURE_NAMES, beta_vec)}
    train_r2 = r_squared(y_train, Xt @ beta_vec)
    val_r2 = None
    if X_val:
        Xv = np.asarray(X_val, dtype=float)
        yv = np.asarray(y_val, dtype=float)
        val_r2 = r_squared(y_val, Xv @ beta_vec)
    # action-agreement on val: value function argmax vs label argmax at n_star
    agree_val = 0
    total_val = 0
    for row in label_rows:
        if row["episode_id"] in train_eps:
            continue
        best_label = chosen_at(row, args.scenarios)[0]
        best_value = ("__none__", 0.0)
        for action in row["actions"]:
            score = distilled_score(action["features"], beta)
            if score > best_value[1] + 1e-9:
                best_value = ("%s@%d" % (action["model"], action["gpu"]), score)
        total_val += 1
        if best_value[0] == best_label:
            agree_val += 1
    val_agreement = (agree_val / total_val) if total_val else None
    print("phase D: train n=%d R2=%.3f | val n=%d R2=%s | val action agreement=%s"
          % (len(y_train), train_r2, len(y_val),
             ("%.3f" % val_r2) if val_r2 is not None else "n/a",
             ("%.3f" % val_agreement) if val_agreement is not None else "n/a"))
    print("beta:", json.dumps({k: round(v, 6) for k, v in beta.items()}))
    label_values = [a["delta"][str(args.scenarios)] for row in label_rows for a in row["actions"]]
    label_stats = {
        "n": len(label_values),
        "mean": float(np.mean(label_values)),
        "std": float(np.std(label_values)),
        "p05": float(np.percentile(label_values, 5)),
        "p95": float(np.percentile(label_values, 95)),
        "positive_frac": float(np.mean(np.asarray(label_values) > 0.0)),
        "states_with_positive_best": sum(
            1 for row in label_rows
            if max([a["delta"][str(args.scenarios)] for a in row["actions"]] + [0.0]) > 0.0),
        "states_total": len(label_rows),
    }
    print("label stats:", json.dumps(label_stats))

    ART.mkdir(parents=True, exist_ok=True)
    (ART / ("probe_states_%s.json" % args.tag)).write_text(
        json.dumps(states, ensure_ascii=False) + "\n", encoding="utf-8")
    (ART / ("action_value_labels_%s.json" % args.tag)).write_text(
        json.dumps({"prefix_grid": list(PREFIX_GRID), "rows": label_rows}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (ART / ("convergence_%s.json" % args.tag)).write_text(
        json.dumps(convergence, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    distilled = {
        "version": "distilled-benefit-v1",
        "feature_names": list(FEATURE_NAMES),
        "beta": beta,
        "n_scenarios": int(n_star),
        "max_scenarios": args.scenarios,
        "scenario_seed": args.seed,
        "probe_episodes": len(dev_episodes),
        "states": len(label_rows),
        "train_rows": len(y_train),
        "val_rows": len(y_val),
        "train_r2": train_r2,
        "val_r2": val_r2,
        "val_action_agreement": val_agreement,
        "convergence": convergence,
        "label_stats": label_stats,
        "j_train": "n/a (scenario sampling from frozen predictor marginals; no truth read)",
        "j_train_file_sha256": sha256_file(EPISODES_FILE),
    }
    (ART / ("distilled_benefit_%s.json" % args.tag)).write_text(
        json.dumps(distilled, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote artifacts under %s (%.1f s total)" % (ART.relative_to(ROOT), time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
