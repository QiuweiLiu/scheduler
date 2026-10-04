#!/usr/bin/env python
"""Real-strata full-table calibration (2026-10-04).

Measures the multi-process co-location table and the load-interference table at
the EXACT strata the frozen DES uses (the model|role token medians frozen in
request_preemption_profile_v1.json), so every combination the frozen workload
can produce has a measured cell:

  Part A: co-location, 11 cells = all stratum combinations of the cross-model
          pairs (8B|4B, 8B|3B, 4B|3B).
  Part B: load interference, 12 cells = each infer stratum x each OTHER load
          model.  Same-model loads are structurally unreachable (a running
          model is resident, so it is never loaded again) and are not measured.

Protocol (frozen, mirrors mp_calibration.py): 1 warmup + 3 rounds, alternating
solo order, medians.  Worker reuse: one worker per model per part, with the
probe rebuilt per requested tier, so the bf16 model load is paid once per model
per part instead of once per cell; the per-cell math is unchanged.

Usage:
  python real_strata_calibration.py --part A --only 4Bplanner --rounds 1 --warmup 0   # smoke
  python real_strata_calibration.py --part A,B --rounds 3 --warmup 1                  # full
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

OUT = Path("/root/autodl-tmp/scheduler_measurements/real_strata")
MODELS = {
    "Qwen3-VL-8B": "/root/autodl-tmp/Qwen3-VL-8B-Instruct",
    "Qwen3-4B": "/root/autodl-tmp/scheduler/Qwen3-4B",
    "Qwen2.5-VL-3B": "/root/autodl-tmp/scheduler/Qwen2.5-VL-3B-Instruct",
}
SHORT = {"Qwen3-VL-8B": "8B", "Qwen3-4B": "4B", "Qwen2.5-VL-3B": "3B"}
FRAME = "/root/autodl-tmp/pilot_frames/frame_00.jpg"
# exact DES medians (request_preemption_profile_v1 layers); the two spatial
# strata carry one image in the real replay (token prior image_count p50=1) and
# their input median already INCLUDES the image tokens, so the probe calibrates
# the text so that (text + image) tokens == in_p50.
STRATA = {
    "Qwen3-VL-8B": {"planner": (1055, 47, 0), "videotool_spatial": (264, 50, 1),
                    "answer_generation": (574, 2, 0)},
    "Qwen3-4B": {"planner": (1072, 36, 0)},
    "Qwen2.5-VL-3B": {"answer_generation": (442, 3, 0), "videotool_spatial": (353, 35, 1)},
}
PAIRS_A = [("Qwen3-VL-8B", "Qwen3-4B"), ("Qwen3-VL-8B", "Qwen2.5-VL-3B"),
           ("Qwen3-4B", "Qwen2.5-VL-3B")]

FILLER = ("The purpose of this measurement is to characterize steady-state GPU behaviour "
          "under a fixed representative workload for scheduling research. ")


def _tok_of(proc):
    return getattr(proc, "tokenizer", proc)


def text_prompt(tok, n_tokens: int) -> str:
    t = _tok_of(tok)
    text = "Summarize the following notes in one sentence.\n\n"
    while len(t(text)["input_ids"]) < n_tokens:
        text += FILLER
    return t.decode(t(text)["input_ids"][:n_tokens])


def load_qwen(path):
    from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                              AutoProcessor, AutoTokenizer)
    try:
        proc = AutoProcessor.from_pretrained(path)
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
        return proc, model, True
    except (ValueError, RuntimeError) as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
        return tok, model, False


# ---------------- worker（推理，按请求携带 tier） ----------------

def worker_main(args) -> int:
    name = args.worker
    proc, model, vision = load_qwen(MODELS[name])
    cache: dict[str, tuple] = {}

    def probe_for(tier: str):
        if tier not in cache:
            n_in, n_out, n_img = STRATA[name][tier]
            if n_img:
                from PIL import Image
                img = Image.open(FRAME).convert("RGB")
                text_tokens = n_in
                inputs = None
                for _ in range(4):
                    text = text_prompt(proc, max(1, text_tokens))
                    content = [{"type": "image"} for _ in range(n_img)] + \
                              [{"type": "text", "text": text}]
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
                msgs = [{"role": "user", "content": text}]
                prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = proc(text=[prompt], return_tensors="pt").to("cuda:0")
            cache[tier] = (inputs, n_out)
        return cache[tier]

    with torch.no_grad():
        for tier in STRATA[name]:
            inputs, n_out = probe_for(tier)
            model.generate(**inputs, max_new_tokens=n_out, min_new_tokens=n_out, do_sample=False)
    print(json.dumps({"ready": True, "model": name, "tiers": sorted(STRATA[name])}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("op") == "quit":
            break
        inputs, n_out = probe_for(str(req["tier"]))
        t0w = time.time()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=n_out, min_new_tokens=n_out, do_sample=False)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        print(json.dumps({"tag": req.get("tag"), "ms": round(ms, 1),
                          "wall_start": t0w, "wall_end": time.time()}), flush=True)
    return 0


class Worker:
    def __init__(self, model: str):
        self.proc = subprocess.Popen([sys.executable, __file__, "--worker", model],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        line = self.proc.stdout.readline()
        try:
            msg = json.loads(line)
            assert msg.get("ready"), f"worker {model} not ready: {line[:200]}"
        except Exception:
            self.proc.kill()
            raise

    def go(self, tag: str, tier: str) -> None:
        self.proc.stdin.write(json.dumps({"op": "go", "tag": tag, "tier": tier}) + "\n")
        self.proc.stdin.flush()

    def recv(self) -> dict:
        return json.loads(self.proc.stdout.readline())

    def run(self, tag: str, tier: str) -> float:
        self.go(tag, tier)
        return self.recv()["ms"]

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=15)
        except Exception:  # noqa: BLE001
            self.proc.kill()


LOADER_SRC = r'''
import json, sys, time
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
import torch
path = sys.argv[1]
print(json.dumps({"ready": True}), flush=True)
t0 = time.perf_counter()
try:
    m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda:0")
except (ValueError, RuntimeError):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda:0")
torch.cuda.synchronize()
print(json.dumps({"load_ms": round((time.perf_counter()-t0)*1000, 1)}), flush=True)
'''


def _spawn_loader(model: str):
    lp = subprocess.Popen([sys.executable, "-c", LOADER_SRC, MODELS[model]],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    lp.stdout.readline()
    return lp


def _finish_loader(lp) -> dict:
    rest = lp.stdout.read()
    lp.wait(timeout=600)
    out = {"wall_end": time.time()}
    for line in rest.strip().splitlines():
        try:
            d = json.loads(line)
            if "load_ms" in d:
                out["load_ms"] = d["load_ms"]
        except Exception:  # noqa: BLE001
            pass
    return out


def _load_once(model: str):
    return _finish_loader(_spawn_loader(model)).get("load_ms")


def _save(root: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "real_strata_calibration.json").write_text(
        json.dumps(root, ensure_ascii=False, indent=1))


def _done(root: dict, part: str, cell_id: str) -> bool:
    return any(c.get("cell_id") == cell_id and "error" not in c for c in root["cells"][part])


def run_part_a(root: dict, rounds: int, warmup: int, only: str) -> None:
    for m1, m2 in PAIRS_A:
        pair_id = f"{SHORT[m1]}|{SHORT[m2]}"
        combos = [(t1, t2) for t1 in sorted(STRATA[m1]) for t2 in sorted(STRATA[m2])]
        pending = [c for c in combos
                   if not (only and only not in f"{SHORT[m1]}|{c[0]}|{SHORT[m2]}|{c[1]}")
                   and not _done(root, "A", f"A_{SHORT[m1]}[{c[0]}]|{SHORT[m2]}[{c[1]}]")]
        if not pending:
            continue
        print(f"\n##### Part A pair {pair_id}: {len(pending)} cells #####", flush=True)
        subprocess.run(["pkill", "-f", "real_strata_calibration.py --worker"], capture_output=True)
        time.sleep(1)
        wa = wb = None
        try:
            wa = Worker(m1)
            wb = Worker(m2)
            for t1, t2 in pending:
                cell_id = f"A_{SHORT[m1]}[{t1}]|{SHORT[m2]}[{t2}]"
                print(f"\n===== {cell_id} =====", flush=True)
                t_cell = time.time()
                try:
                    for _ in range(warmup):
                        wa.run("warm", t1)
                        wb.run("warm", t2)
                        wa.run("warmc", t1)
                        wb.run("warmc", t2)
                    ra, rb, ca, cb = [], [], [], []
                    for r in range(rounds):
                        order = ["a", "b"] if r % 2 == 0 else ["b", "a"]
                        for which in order:
                            if which == "a":
                                ra.append(wa.run(f"r{r}", t1))
                            else:
                                rb.append(wb.run(f"r{r}", t2))
                        wa.go(f"r{r}c", t1)
                        wb.go(f"r{r}c", t2)
                        ca.append(wa.recv()["ms"])
                        cb.append(wb.recv()["ms"])
                    cell = {
                        "cell_id": cell_id,
                        "model_a": m1, "tier_a": t1, "in_a": STRATA[m1][t1][0], "out_a": STRATA[m1][t1][1], "images_a": STRATA[m1][t1][2],
                        "model_b": m2, "tier_b": t2, "in_b": STRATA[m2][t2][0], "out_b": STRATA[m2][t2][1], "images_b": STRATA[m2][t2][2],
                        "solo_a_ms": ra, "solo_b_ms": rb, "coloc_a_ms": ca, "coloc_b_ms": cb,
                        "slowdown_a": round(statistics.median(ca) / statistics.median(ra), 4),
                        "slowdown_b": round(statistics.median(cb) / statistics.median(rb), 4),
                        "elapsed_s": round(time.time() - t_cell, 1),
                    }
                    print(f"  solo=({statistics.median(ra):.0f},{statistics.median(rb):.0f}) "
                          f"slow=({cell['slowdown_a']},{cell['slowdown_b']})", flush=True)
                except Exception as e:  # noqa: BLE001
                    cell = {"cell_id": cell_id, "error": str(e)[:300]}
                    print(f"  ERROR {str(e)[:200]}", flush=True)
                root["cells"]["A"].append(cell)
                _save(root)
                import gc
                gc.collect(); torch.cuda.empty_cache()
        finally:
            if wa: wa.close()
            if wb: wb.close()


def run_part_b(root: dict, rounds: int, warmup: int, only: str) -> None:
    for mi in sorted(STRATA, key=lambda m: -len(STRATA[m])):
        load_models = [m for m in MODELS if m != mi]
        pending = []
        for ti in sorted(STRATA[mi]):
            for lm in load_models:
                cell_id = f"B_{SHORT[mi]}[{ti}]<-{SHORT[lm]}"
                if only and only not in cell_id:
                    continue
                if not _done(root, "B", cell_id):
                    pending.append((ti, lm, cell_id))
        if not pending:
            continue
        print(f"\n##### Part B infer {SHORT[mi]}: {len(pending)} cells #####", flush=True)
        subprocess.run(["pkill", "-f", "real_strata_calibration.py --worker"], capture_output=True)
        time.sleep(1)
        w = None
        try:
            w = Worker(mi)
            for ti, lm, cell_id in pending:
                print(f"\n===== {cell_id} =====", flush=True)
                t_cell = time.time()
                try:
                    for _ in range(warmup):
                        w.run("warm", ti)
                    solo_infer = [w.run(f"s{i}", ti) for i in range(rounds)]
                    solo_load = [_load_once(lm) for _ in range(rounds)]
                    ov_infer, ov_load = [], []
                    for i in range(rounds + warmup):
                        lp = _spawn_loader(lm)
                        w.go(f"o{i}", ti)
                        res = w.recv()
                        info = _finish_loader(lp)
                        if i >= warmup:
                            ov_infer.append(res["ms"])
                            ov_load.append(info.get("load_ms"))
                    ok_load = [x for x in solo_load + ov_load if x]
                    cell = {
                        "cell_id": cell_id,
                        "infer_model": mi, "infer_tier": ti,
                        "infer_in": STRATA[mi][ti][0], "infer_out": STRATA[mi][ti][1], "infer_images": STRATA[mi][ti][2],
                        "load_model": lm,
                        "solo_infer_ms": solo_infer, "solo_load_ms": solo_load,
                        "overlap_infer_ms": ov_infer, "overlap_load_ms": ov_load,
                        "extra_ms": round(statistics.median(ov_infer) - statistics.median(solo_infer), 1),
                        "dilation": round(statistics.median([x for x in ov_load if x]) /
                                          statistics.median([x for x in solo_load if x]), 4) if ok_load else None,
                        "elapsed_s": round(time.time() - t_cell, 1),
                    }
                    print(f"  solo_infer={statistics.median(solo_infer):.0f} "
                          f"overlap={statistics.median(ov_infer):.0f} extra={cell['extra_ms']:.0f}ms "
                          f"dil={cell['dilation']}", flush=True)
                except Exception as e:  # noqa: BLE001
                    cell = {"cell_id": cell_id, "error": str(e)[:300]}
                    print(f"  ERROR {str(e)[:200]}", flush=True)
                root["cells"]["B"].append(cell)
                _save(root)
                import gc
                gc.collect(); torch.cuda.empty_cache()
        finally:
            if w: w.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="A,B")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--worker", default="")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    if args.worker:
        return worker_main(args)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "real_strata_calibration.json"
    if dest.exists():
        root = json.loads(dest.read_text())
    else:
        root = {
            "schema": "real-strata-calibration-v1",
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1),
            "protocol": {"rounds": args.rounds, "warmup": args.warmup,
                         "worker_reuse": "one worker per model per part; probe rebuilt per tier",
                         "source": "mirrors mp_calibration.py (1 warmup + 3 rounds, alternating order, medians)"},
            "strata": {m: {t: {"in": v[0], "out": v[1], "images": v[2]} for t, v in d.items()}
                       for m, d in STRATA.items()},
            "started_at": time.time(),
            "cells": {"A": [], "B": []},
        }
    root["protocol"] = {"rounds": args.rounds, "warmup": args.warmup,
                        "worker_reuse": "one worker per model per part; probe rebuilt per tier",
                        "source": "mirrors mp_calibration.py (1 warmup + 3 rounds, alternating order, medians)"}
    if "A" in args.part:
        run_part_a(root, args.rounds, args.warmup, args.only)
    if "B" in args.part:
        run_part_b(root, args.rounds, args.warmup, args.only)
    root["finished_at"] = time.time()
    _save(root)
    n_a = len([c for c in root["cells"]["A"] if "error" not in c])
    n_b = len([c for c in root["cells"]["B"] if "error" not in c])
    print(f"\nWROTE {dest}  A={n_a}/11  B={n_b}/12", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
