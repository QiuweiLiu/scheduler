#!/usr/bin/env python3
"""Colocation P3 re-measurement runner (advisor-approved design, 2026-10-16).

Identifies the dynamic co-residency semantics (piecewise recovery / sticky
persistence / one-time overhead) via OFFSET-START conditions on an RTX 4080-class
GPU.  This script is the full runner; it MUST first pass its CPU dry-run
protocol self-test (--dry-run) before any GPU session.

Protocol (per advisor ruling):
  * two workers (one per model), multi-process deployment;
  * COMMON start barrier on a shared host MONOTONIC clock (time.monotonic_ns);
  * per-round timestamps: request recv, barrier release, generate start
    (post pre-sync), generate end (post post-sync); optional progress samples;
  * conditions: T0 sync start (fit), T1 early late-join, T2 late late-join,
    T3 no-overlap negative control (T1/T2/T3 = validation, never fit);
  * solo baselines measured under the SAME instrumentation;
  * per-round JSONL logs; assertions + fail-fast (barrier miss, OOM, worker
    error); budget guard aborts cleanly; no silent sample dropping.

Usage:
  python scripts/real_strata_colocation_remeasure.py --dry-run --out /tmp/p3_dry
  python scripts/real_strata_colocation_remeasure.py --gpu --out DIR --reps 5 --budget-min 90
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "8B": "Qwen/Qwen3-VL-8B-Instruct",
    "4B": "Qwen/Qwen3-4B",
    "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
}

# advisor-approved pair set and pre-registered offsets (ms)
PAIRS = {
    "P-1": {"a": ("4B", "planner"), "b": ("8B", "answer_generation"), "t1_ms": 100, "t2_ms": 400},
    "P-2": {"a": ("8B", "planner"), "b": ("4B", "planner"), "t1_ms": 50, "t2_ms": 300},
    "P-3": {"a": ("8B", "planner"), "b": ("3B", "videotool_spatial"), "t1_ms": 100, "t2_ms": 400},
}
# extra exploratory (survivor-identity reversal) — NOT part of the main test
P2_EXPLORATORY_T2_MS = 1000

BARRIER_LEAD_NS = 400_000_000  # 400ms lead for barrier scheduling
BARRIER_TOL_NS = 50_000_000    # barrier miss if release > target + 50ms

WORKER_SRC = r'''
import argparse, json, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print(json.dumps({"ready": True, "model": args.model, "dry_run": True}), flush=True)
        dur = {"planner": 1.2, "answer_generation": 0.15, "videotool_spatial": 0.8}
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            if req.get("op") == "quit":
                break
            t_recv = time.monotonic_ns()
            barrier = int(req["barrier_ns"])
            while time.monotonic_ns() < barrier:
                time.sleep(0.0005)
            t_barrier = time.monotonic_ns()
            t0 = time.monotonic_ns()
            secs = dur.get(req["tier"], 0.5)
            steps = 8
            progress = []
            for i in range(steps):
                time.sleep(secs / steps)
                progress.append([i + 1, time.monotonic_ns()])
            t1 = time.monotonic_ns()
            print(json.dumps({"tag": req.get("tag"), "ms": round((t1 - t0) / 1e6, 1),
                              "t_recv_ns": t_recv, "t_barrier_ns": t_barrier,
                              "t_gen_start_ns": t0, "t_gen_end_ns": t1,
                              "progress": progress}), flush=True)
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    try:
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    except (ValueError, RuntimeError):
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    import torch as _t
    print(json.dumps({"ready": True, "model": args.model,
                      "device": _t.cuda.get_device_name(0),
                      "total_mb": _t.cuda.get_device_properties(0).total_memory / 2**20,
                      "torch": _t.__version__}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("op") == "quit":
            break
        t_recv = time.monotonic_ns()
        barrier = int(req["barrier_ns"])
        while time.monotonic_ns() < barrier:
            time.sleep(0.0005)
        t_barrier = time.monotonic_ns()
        _t.cuda.synchronize()
        t0 = time.monotonic_ns()
        inputs, n_out = req["_inputs"], req["_n_out"] if False else (None, None)  # placeholder; inputs built below
        raise SystemExit("GPU worker input construction must be wired before a GPU session")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


class Worker:
    def __init__(self, model_key: str, dry_run: bool):
        cmd = [sys.executable, __file__, "--worker", MODELS[model_key]]
        if dry_run:
            cmd.append("--dry-run")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        line = self.proc.stdout.readline()
        try:
            msg = json.loads(line)
            assert msg.get("ready"), f"worker {model_key} not ready: {line[:200]}"
        except Exception:
            self.proc.kill()
            raise
        self.info = msg

    def send(self, req: dict) -> None:
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()

    def recv(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read()[:400] if self.proc.stderr else ""
            raise RuntimeError(f"worker died: {err}")
        return json.loads(line)

    def close(self) -> None:
        try:
            self.send({"op": "quit"})
            self.proc.wait(timeout=15)
        except Exception:
            self.proc.kill()


def run_round(wa: Worker, wb: Worker, tag: str, tier_a: str, tier_b: str,
              offset_b_ms: float, logs: list, budget_ns: int, t_start: int) -> dict:
    """One co-located round: A starts at barrier, B at barrier+offset (T0: offset=0)."""
    barrier = time.monotonic_ns() + BARRIER_LEAD_NS
    send_a = time.monotonic_ns()
    wa.send({"op": "go", "tag": tag + "a", "tier": tier_a, "barrier_ns": barrier})
    send_b = time.monotonic_ns()
    wb.send({"op": "go", "tag": tag + "b", "tier": tier_b,
             "barrier_ns": barrier + int(offset_b_ms * 1e6)})
    ra = wa.recv()
    rb = wb.recv()
    if time.monotonic_ns() - t_start > budget_ns:
        raise TimeoutError("budget exceeded")
    rec = {"tag": tag, "offset_b_ms": offset_b_ms, "send_a_ns": send_a, "send_b_ns": send_b,
           "barrier_ns": barrier, "a": ra, "b": rb}
    rec["barrier_miss_a_ns"] = ra["t_barrier_ns"] - barrier
    rec["barrier_miss_b_ns"] = rb["t_barrier_ns"] - (barrier + int(offset_b_ms * 1e6))
    overlap_start = max(ra["t_gen_start_ns"], rb["t_gen_start_ns"])
    overlap_end = min(ra["t_gen_end_ns"], rb["t_gen_end_ns"])
    rec["overlap_ns"] = max(0, overlap_end - overlap_start)
    rec["overlap_frac_a"] = rec["overlap_ns"] / max(1, ra["t_gen_end_ns"] - ra["t_gen_start_ns"])
    rec["overlap_frac_b"] = rec["overlap_ns"] / max(1, rb["t_gen_end_ns"] - rb["t_gen_start_ns"])
    rec["first_finisher"] = "a" if ra["t_gen_end_ns"] <= rb["t_gen_end_ns"] else "b"
    rec["barrier_ok"] = (abs(rec["barrier_miss_a_ns"]) <= BARRIER_TOL_NS
                         and abs(rec["barrier_miss_b_ns"]) <= BARRIER_TOL_NS)
    logs.append(rec)
    return rec


def run_solo(w: Worker, tag: str, tier: str, logs: list) -> dict:
    barrier = time.monotonic_ns() + BARRIER_LEAD_NS
    w.send({"op": "go", "tag": tag, "tier": tier, "barrier_ns": barrier})
    r = w.recv()
    rec = {"tag": tag, "solo": True, "result": r}
    logs.append(rec)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--model", help=argparse.SUPPRESS)
    ap.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--gpu", action="store_true", help="run the real GPU session")
    ap.add_argument("--out", type=Path, required=False)
    ap.add_argument("--pairs", default="P-1,P-2,P-3")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--budget-min", type=float, default=90.0)
    args = ap.parse_args()

    if args.worker:
        # delegate to embedded worker source (subprocess mode)
        ns = {"__name__": "__main__", "__file__": __file__}
        code = compile(WORKER_SRC, __file__, "exec")
        sys.argv = [__file__, "--model", args.model] + (["--dry-run"] if args.dry_run else [])
        exec(code, ns)
        return 0

    if not args.dry_run and not args.gpu:
        print("refusing to run: pass --dry-run (CPU protocol self-test) or --gpu (real session)")
        return 2
    out = args.out or (Path("/tmp") / f"p3_{'dry' if args.dry_run else 'gpu'}")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "rounds.jsonl"
    budget_ns = int(args.budget_min * 60 * 1e9)
    t_start = time.monotonic_ns()
    logs: list = []
    failures: list = []

    for pair_id in args.pairs.split(","):
        spec = PAIRS[pair_id.strip()]
        (ma, ta), (mb, tb) = spec["a"], spec["b"]
        wa, wb = None, None
        try:
            wa = Worker(ma, args.dry_run)
            wb = Worker(mb, args.dry_run)
            env = {"pair": pair_id, "a": {"model": ma, "tier": ta}, "b": {"model": mb, "tier": tb},
                   "worker_a": wa.info, "worker_b": wb.info}
            run_solo(wa, f"{pair_id}_solo_a", ta, logs)
            run_solo(wb, f"{pair_id}_solo_b", tb, logs)
            conditions = [("T0", 0.0), ("T1", spec["t1_ms"]), ("T2", spec["t2_ms"]),
                          ("T3", None)]  # T3 handled separately (no overlap)
            for cond, off in conditions:
                for rep in range(args.reps):
                    if cond == "T3":
                        # no-overlap negative control: B starts after A finishes
                        rec_solo = run_solo(wa, f"{pair_id}_T3_pre", ta, logs)
                        a_end = rec_solo["result"]["t_gen_end_ns"]
                        gap_ms = 50.0
                        barrier = a_end + int(gap_ms * 1e6)
                        wb.send({"op": "go", "tag": f"{pair_id}_T3_{rep}b", "tier": tb, "barrier_ns": barrier})
                        rb = wb.recv()
                        rec = {"tag": f"{pair_id}_T3_{rep}", "condition": "T3", "b": rb, "no_overlap": True,
                               "gap_ms": gap_ms}
                        logs.append(rec)
                    else:
                        rec = run_round(wa, wb, f"{pair_id}_{cond}_{rep}", ta, tb, off, logs, budget_ns, t_start)
                        rec["condition"] = cond
                        if not rec["barrier_ok"]:
                            failures.append(f"{rec['tag']}: barrier miss")
                    logs[-1].update({"pair": pair_id, "env": env} if "pair" not in logs[-1] else {})
        except TimeoutError as e:
            failures.append(f"{pair_id}: {e}")
            break
        except Exception as e:  # noqa: BLE001
            failures.append(f"{pair_id}: {type(e).__name__}: {e}")
        finally:
            if wa:
                wa.close()
            if wb:
                wb.close()
            with log_path.open("w", encoding="utf-8") as f:
                for rec in logs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {"schema": "colocation-p3-rounds", "mode": "dry-run" if args.dry_run else "gpu",
               "n_rounds": len(logs), "failures": failures, "elapsed_s": round((time.monotonic_ns() - t_start) / 1e9, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print("logs:", log_path)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
