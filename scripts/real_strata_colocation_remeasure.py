#!/usr/bin/env python3
"""Colocation P3 re-measurement runner — GPU-wired (advisor-approved design).

Identifies the dynamic co-residency semantics (piecewise recovery / sticky
persistence / one-time overhead) via OFFSET-START conditions on an RTX
4080-class GPU.  Same protocol as the CPU dry-run self-test; the GPU worker
path is now fully wired (probe construction copied from the original
real_strata_calibration.py so strata match the frozen DES medians).

Protocol (per advisor ruling 2026-10-16):
  * two workers (one per model), multi-process deployment, same strata/frame;
  * COMMON start barrier on host MONOTONIC clock; per-round timestamps:
    request recv, barrier release, generate start (post pre-sync), generate
    end (post post-sync); per-token progress timestamps (LogitsProcessor);
  * conditions: T0 sync start (fit), T1/T2 late joins (validation), T3
    no-overlap negative control; warm-up round per tier before measurements;
  * solo baselines under the SAME instrumentation;
  * per-round JSONL + env/versions; fail-fast (barrier miss, worker death,
    OOM); budget guard; no silent sample dropping.

Usage (remote, vllm_env):
  python real_strata_colocation_remeasure.py --gpu --out /root/autodl-tmp/p3_measure --pairs P-1 --reps 1 --conditions T0   # smoke
  python real_strata_colocation_remeasure.py --gpu --out /root/autodl-tmp/p3_measure --reps 5 --budget-min 90
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MODELS = {
    "8B": "/root/autodl-tmp/Qwen3-VL-8B-Instruct",
    "4B": "/root/autodl-tmp/scheduler/Qwen3-4B",
    "3B": "/root/autodl-tmp/scheduler/Qwen2.5-VL-3B-Instruct",
}
FRAME = "/root/autodl-tmp/pilot_frames/frame_00.jpg"

PAIRS = {
    "P-1": {"a": ("4B", "planner"), "b": ("8B", "answer_generation"), "t1_ms": 100, "t2_ms": 400},
    "P-2": {"a": ("8B", "planner"), "b": ("4B", "planner"), "t1_ms": 50, "t2_ms": 300},
    "P-3": {"a": ("8B", "planner"), "b": ("3B", "videotool_spatial"), "t1_ms": 100, "t2_ms": 400},
}
BARRIER_LEAD_NS = 500_000_000
BARRIER_TOL_NS = 50_000_000
P2_EXPLORATORY_T2_MS = 1000  # not mixed into the main test

STRATA = {
    "8B": {"planner": (1055, 47, 0), "videotool_spatial": (264, 50, 1), "answer_generation": (574, 2, 0)},
    "4B": {"planner": (1072, 36, 0)},
    "3B": {"answer_generation": (442, 3, 0), "videotool_spatial": (353, 35, 1)},
}
FILLER = ("The purpose of this measurement is to characterize steady-state GPU behaviour "
          "under a fixed representative workload for scheduling research. ")

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
    from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                              AutoProcessor, AutoTokenizer, LogitsProcessor,
                              LogitsProcessorList)
    import transformers as _tf

    name = args.model
    path = MODELS[name]
    try:
        proc = AutoProcessor.from_pretrained(path)
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
        vision = True
    except (ValueError, RuntimeError) as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
        proc = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
        vision = False
    print(json.dumps({"ready": True, "model": name, "path": path,
                      "device": torch.cuda.get_device_name(0),
                      "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1),
                      "torch": torch.__version__, "transformers": _tf.__version__,
                      "vision": vision}), flush=True)

    cache = {}

    def _tok_of(p):
        return getattr(p, "tokenizer", p)

    def text_prompt(tok, n_tokens):
        t = _tok_of(tok)
        text = "Summarize the following notes in one sentence.\n\n"
        while len(t(text)["input_ids"]) < n_tokens:
            text += FILLER
        return t.decode(t(text)["input_ids"][:n_tokens])

    def probe_for(tier):
        if tier not in cache:
            n_in, n_out, n_img = STRATA[name][tier]
            if n_img:
                from PIL import Image
                img = Image.open(FRAME).convert("RGB")
                text_tokens = n_in
                inputs = None
                for _ in range(4):
                    text = text_prompt(proc, max(1, text_tokens))
                    content = [{"type": "image"} for _ in range(n_img)] + [{"type": "text", "text": text}]
                    prompt = proc.apply_chat_template([{"role": "user", "content": content}],
                                                      tokenize=False, add_generation_prompt=True)
                    inputs = proc(text=[prompt], images=[img], return_tensors="pt")
                    total = int(inputs["input_ids"].shape[1])
                    if total == n_in:
                        break
                    text_tokens += n_in - total
                inputs = inputs.to("cuda:0")
            else:
                text = text_prompt(proc, n_in)
                prompt = proc.apply_chat_template([{"role": "user", "content": text}],
                                                  tokenize=False, add_generation_prompt=True)
                inputs = proc(text=[prompt], return_tensors="pt").to("cuda:0")
            cache[tier] = (inputs, n_out)
        return cache[tier]

    class _Stamp(LogitsProcessor):
        def __init__(self):
            self.times = []

        def __call__(self, input_ids, scores):
            self.times.append(time.monotonic_ns())
            return scores

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
        inputs, n_out = probe_for(str(req["tier"]))
        torch.cuda.synchronize()
        t0 = time.monotonic_ns()
        stamp = _Stamp()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=n_out, min_new_tokens=n_out, do_sample=False,
                           logits_processor=LogitsProcessorList([stamp]))
        torch.cuda.synchronize()
        t1 = time.monotonic_ns()
        print(json.dumps({"tag": req.get("tag"), "ms": round((t1 - t0) / 1e6, 1),
                          "t_recv_ns": t_recv, "t_barrier_ns": t_barrier,
                          "t_gen_start_ns": t0, "t_gen_end_ns": t1,
                          "progress": stamp.times}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


class Worker:
    def __init__(self, model_key: str, dry_run: bool):
        cmd = [sys.executable, __file__, "--worker", model_key]
        if dry_run:
            cmd.append("--dry-run")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        line = self.proc.stdout.readline()
        try:
            msg = json.loads(line)
            assert msg.get("ready"), f"worker {model_key} not ready: {line[:200]}"
        except Exception:
            err = ""
            try:
                err = self.proc.stderr.read()[:600]
            except Exception:
                pass
            self.proc.kill()
            raise RuntimeError(f"worker {model_key} failed to start: {line[:200]} {err}")
        self.info = msg

    def send(self, req: dict) -> None:
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()

    def recv(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read()[:600] if self.proc.stderr else ""
            raise RuntimeError(f"worker died: {err}")
        return json.loads(line)

    def close(self) -> None:
        try:
            self.send({"op": "quit"})
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()


def _barrier_result(req: dict, barrier: int, offset_ns: int, rec: dict) -> dict:
    return {"send_ns": None, "barrier_ns": barrier, "result": rec,
            "barrier_miss_ns": rec["t_barrier_ns"] - (barrier + offset_ns)}


def run_round(wa: Worker, wb: Worker, tag: str, tier_a: str, tier_b: str, offset_b_ms: float,
              logs: list, budget_ns: int, t_start: int, env: dict) -> dict:
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
           "barrier_ns": barrier, "a": ra, "b": rb, "env": env}
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


def run_solo(w: Worker, tag: str, tier: str, logs: list, env: dict, warm: bool = False) -> dict:
    barrier = time.monotonic_ns() + 100_000_000
    w.send({"op": "go", "tag": tag, "tier": tier, "barrier_ns": barrier})
    r = w.recv()
    rec = {"tag": tag, "solo": True, "warm": warm, "env": env, "result": r}
    logs.append(rec)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--model", help=argparse.SUPPRESS)
    ap.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", type=Path, required=False)
    ap.add_argument("--pairs", default="P-1,P-2,P-3")
    ap.add_argument("--conditions", default="T0,T1,T2,T3")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--budget-min", type=float, default=90.0)
    ap.add_argument("--interleave", action="store_true",
                    help="drift-robust: per-round solo baselines + all conditions together")
    args = ap.parse_args()

    if args.worker:
        sys.argv = [__file__, "--model", args.worker] + (["--dry-run"] if args.dry_run else [])
        exec(compile(WORKER_SRC, __file__, "exec"), globals())
        return 0

    if not args.dry_run and not args.gpu:
        print("refusing to run: pass --dry-run (CPU self-test) or --gpu (real session)")
        return 2
    out = args.out or (Path("/tmp") / f"p3_{'dry' if args.dry_run else 'gpu'}")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "rounds.jsonl"
    budget_ns = int(args.budget_min * 60 * 1e9)
    t_start = time.monotonic_ns()
    logs: list = []
    failures: list = []
    envs: dict = {}
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    for pair_id in args.pairs.split(","):
        spec = PAIRS[pair_id.strip()]
        (ma, ta), (mb, tb) = spec["a"], spec["b"]
        wa, wb = None, None
        try:
            wa = Worker(ma, args.dry_run)
            wb = Worker(mb, args.dry_run)
            env = {"pair": pair_id, "a": {"model": ma, "tier": ta}, "b": {"model": mb, "tier": tb},
                   "worker_a": wa.info, "worker_b": wb.info}
            envs[pair_id] = env
            run_solo(wa, f"{pair_id}_warm_a", ta, logs, env, warm=True)
            run_solo(wb, f"{pair_id}_warm_b", tb, logs, env, warm=True)
            plan = []
            if "T0" in conditions:
                plan.append(("T0", 0.0))
            if "T1" in conditions:
                plan.append(("T1", spec["t1_ms"]))
            if "T2" in conditions:
                plan.append(("T2", spec["t2_ms"]))
            if args.interleave:
                # drift-robust: every round carries its own solo baselines and all
                # conditions, so the machine's slow drift is shared across conditions
                # (mirrors the original campaign's interleaved design).
                for rep in range(args.reps):
                    run_solo(wa, f"{pair_id}_r{rep}_solo_a", ta, logs, env)
                    run_solo(wb, f"{pair_id}_r{rep}_solo_b", tb, logs, env)
                    for cond, off in plan:
                        rec = run_round(wa, wb, f"{pair_id}_{cond}_r{rep}", ta, tb, off,
                                        logs, budget_ns, t_start, env)
                        rec["condition"] = cond
                        if not rec["barrier_ok"]:
                            failures.append(f"{rec['tag']}: barrier miss")
                    if "T3" in conditions:
                        rec_a = run_solo(wa, f"{pair_id}_T3_r{rep}_a_solo", ta, logs, env)
                        a_end = rec_a["result"]["t_gen_end_ns"]
                        barrier = a_end + int(50.0 * 1e6)
                        wb.send({"op": "go", "tag": f"{pair_id}_T3_r{rep}b", "tier": tb, "barrier_ns": barrier})
                        rb = wb.recv()
                        rec = {"tag": f"{pair_id}_T3_r{rep}", "condition": "T3", "no_overlap": True,
                               "gap_ms": 50.0, "b": rb, "a_solo_ms": rec_a["result"]["ms"], "env": env}
                        logs.append(rec)
            else:
                run_solo(wa, f"{pair_id}_solo_a", ta, logs, env)
                run_solo(wb, f"{pair_id}_solo_b", tb, logs, env)
                for cond, off in plan:
                    for rep in range(args.reps):
                        rec = run_round(wa, wb, f"{pair_id}_{cond}_{rep}", ta, tb, off,
                                        logs, budget_ns, t_start, env)
                        rec["condition"] = cond
                        if not rec["barrier_ok"]:
                            failures.append(f"{rec['tag']}: barrier miss")
                if "T3" in conditions:
                    for rep in range(args.reps):
                        rec_a = run_solo(wa, f"{pair_id}_T3_{rep}_a_solo", ta, logs, env)
                        a_end = rec_a["result"]["t_gen_end_ns"]
                        barrier = a_end + int(50.0 * 1e6)
                        wb.send({"op": "go", "tag": f"{pair_id}_T3_{rep}b", "tier": tb, "barrier_ns": barrier})
                        rb = wb.recv()
                        rec = {"tag": f"{pair_id}_T3_{rep}", "condition": "T3", "no_overlap": True,
                               "gap_ms": 50.0, "b": rb, "a_solo_ms": rec_a["result"]["ms"], "env": env}
                        logs.append(rec)
        except TimeoutError as e:
            failures.append(f"{pair_id}: {e}")
            break
        except Exception as e:  # noqa: BLE001
            failures.append(f"{pair_id}: {type(e).__name__}: {str(e)[:300]}")
        finally:
            if wa:
                wa.close()
            if wb:
                wb.close()
            with log_path.open("w", encoding="utf-8") as f:
                for rec in logs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {"schema": "colocation-p3-rounds", "mode": "dry-run" if args.dry_run else "gpu",
               "n_rounds": len(logs), "failures": failures, "envs": envs,
               "elapsed_s": round((time.monotonic_ns() - t_start) / 1e9, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "envs"}, ensure_ascii=False))
    print("logs:", log_path)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
