#!/usr/bin/env python
"""P1：打断-恢复往返验证（抢占的恢复代价 R_m 是否 = F2 prefill 预测）。

三种跑法对比（同一 prompt）：
  ① full        : prefill(L) + N 个 decode step
  ② interrupted : prefill(L) + k 个 step → 【丢 KV】→ 重算(L+k) → 继续 N-k 个 step
  ③ theory      : ②的"重算"段应 = F2 prefill 曲线(intercept + rate×(L+k))
判据：t_rebuild − theory 的残差（编排开销，预期 <100ms）；总时长差 ≈ t_rebuild。

用法：python p1_resume_roundtrip.py --reps 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

OUT = Path("/root/autodl-tmp/scheduler_measurements/resume_roundtrip_3b")
MODELS = {
    "Qwen3-4B": "/root/autodl-tmp/scheduler/Qwen3-4B",
    "Qwen3-VL-8B": "/root/autodl-tmp/Qwen3-VL-8B-Instruct",
    "Qwen2.5-VL-3B": "/root/autodl-tmp/scheduler/Qwen2.5-VL-3B-Instruct",
}
F2 = {
    "Qwen3-4B": (8.1, 0.11457),
    "Qwen3-VL-8B": (9.8, 0.19001),
    "Qwen2.5-VL-3B": (10.2, 0.07529),  # F2 artifact (EXP-20260929_substrate_phase_profile_v1)
}  # (intercept ms, ms/token)
# P1-5: 3B representative resume cells + one 4B control cell (same protocol) to
# separate model effects from instance-to-instance GPU differences.
CONFIGS = [("Qwen2.5-VL-3B", 1000, 48, 20), ("Qwen2.5-VL-3B", 1000, 48, 40), ("Qwen3-4B", 1000, 48, 20)]
FILLER = ("The purpose of this measurement is to characterize steady-state GPU behaviour "
          "under a fixed representative workload for scheduling research. ")


def text_prompt(tok, n_tokens: int) -> str:
    t = getattr(tok, "tokenizer", tok)
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
        return proc, model
    except (ValueError, RuntimeError) as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
        return tok, model


def _ev():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def decode_step(model, tokid, past):
    a, b = _ev()
    a.record()
    with torch.no_grad():
        o = model(input_ids=tokid, past_key_values=past, use_cache=True)
    b.record()
    torch.cuda.synchronize()
    return o, a.elapsed_time(b)


def full_run(model, inputs, n):
    a, b = _ev()
    a.record()
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    b.record()
    torch.cuda.synchronize()
    t_pre = a.elapsed_time(b)
    past = out.past_key_values
    tokid = out.logits[:, -1].argmax(-1, keepdim=True)
    del out
    total = t_pre
    for _ in range(n):
        o, dt = decode_step(model, tokid, past)
        total += dt
        past = o.past_key_values
        tokid = o.logits[:, -1].argmax(-1, keepdim=True)
        del o
    del past, tokid
    torch.cuda.empty_cache()
    return {"t_total": total}


def interrupted_run(model, inputs, k, n):
    a, b = _ev()
    a.record()
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    b.record()
    torch.cuda.synchronize()
    t_pre = a.elapsed_time(b)
    past = out.past_key_values
    tokid = out.logits[:, -1].argmax(-1, keepdim=True)
    del out
    fed = []
    t1 = t_pre
    for _ in range(k):
        fed.append(tokid)
        o, dt = decode_step(model, tokid, past)
        t1 += dt
        past = o.past_key_values
        tokid = o.logits[:, -1].argmax(-1, keepdim=True)
        del o
    # 模拟抢占：丢弃 KV
    del past
    torch.cuda.empty_cache()
    # 重算（resume rebuild）
    rebuild_ids = torch.cat([inputs["input_ids"]] + fed, dim=1)
    attn = torch.ones_like(rebuild_ids)
    a, b = _ev()
    a.record()
    with torch.no_grad():
        out2 = model(input_ids=rebuild_ids, attention_mask=attn, use_cache=True)
    b.record()
    torch.cuda.synchronize()
    t_rebuild = a.elapsed_time(b)
    past = out2.past_key_values
    tokid = out2.logits[:, -1].argmax(-1, keepdim=True)
    del out2
    t2 = 0.0
    for _ in range(n - k):
        o, dt = decode_step(model, tokid, past)
        t2 += dt
        past = o.past_key_values
        tokid = o.logits[:, -1].argmax(-1, keepdim=True)
        del o
    del past, tokid
    torch.cuda.empty_cache()
    return {"t_prefill": round(t_pre, 2), "t_phase1": round(t1, 2),
            "t_rebuild": round(t_rebuild, 2), "t_continue": round(t2, 2),
            "t_total": round(t1 + t_rebuild + t2, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "p1_resume.json"
    res: dict = {"config": {"reps": args.reps, "gpu": torch.cuda.get_device_name(0)}, "cells": []}
    if dest.exists():
        try:
            res = json.loads(dest.read_text())
        except Exception:  # noqa: BLE001
            pass

    by_model: dict[str, list] = {}
    for cfg in CONFIGS:
        if args.only and args.only not in cfg[0]:
            continue
        by_model.setdefault(cfg[0], []).append(cfg)

    for model_name, cfgs in by_model.items():
        print(f"\n########## 加载 {model_name} ##########", flush=True)
        proc, model = load_qwen(MODELS[model_name])
        for cfg in cfgs:
            _, L, N, k = cfg
            cell_id = f"{model_name}|L{L}|N{N}|k{k}"
            if any(c.get("cell_id") == cell_id and "error" not in c for c in res["cells"]):
                print(f"skip (done): {cell_id}", flush=True)
                continue
            print(f"\n===== {cell_id} =====", flush=True)
            cell: dict = {"cell_id": cell_id, "model": model_name, "L": L, "N": N, "k": k}
            try:
                text = text_prompt(proc, L)
                msgs = [{"role": "user", "content": text}]
                prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = proc(text=[prompt], return_tensors="pt").to("cuda:0")
                cell["in_tokens"] = int(inputs["input_ids"].shape[1])
                # 预热
                full_run(model, inputs, N)
                interrupted_run(model, inputs, k, N)
                fulls, ints = [], []
                for r in range(args.reps):
                    if r % 2 == 0:
                        fulls.append(full_run(model, inputs, N))
                        ints.append(interrupted_run(model, inputs, k, N))
                    else:
                        ints.append(interrupted_run(model, inputs, k, N))
                        fulls.append(full_run(model, inputs, N))
                t_full = statistics.median(x["t_total"] for x in fulls)
                t_int = statistics.median(x["t_total"] for x in ints)
                t_reb = statistics.median(x["t_rebuild"] for x in ints)
                icpt, rate = F2[model_name]
                n_ctx = cell["in_tokens"] + k
                pred = icpt + rate * n_ctx
                cell.update({
                    "full_total_ms": round(t_full, 2),
                    "interrupted_total_ms": round(t_int, 2),
                    "total_delta_ms": round(t_int - t_full, 2),
                    "t_rebuild_ms": round(t_reb, 2),
                    "f2_pred_ms": round(pred, 2),
                    "residual_ms": round(t_reb - pred, 2),
                    "residual_pct": round(100 * (t_reb - pred) / pred, 1),
                    "rebuild_runs": [x["t_rebuild"] for x in ints],
                    "n_ctx": n_ctx,
                })
                print(f"  full={t_full:.0f}ms interrupted={t_int:.0f}ms delta={t_int-t_full:.0f}ms | "
                      f"rebuild={t_reb:.1f}ms vs F2 pred={pred:.1f}ms residual={t_reb-pred:+.1f}ms", flush=True)
            except Exception as e:  # noqa: BLE001
                cell["error"] = str(e)[:300]
                print(f"  ERROR {str(e)[:200]}", flush=True)
            res["cells"].append(cell)
            dest.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        del model, proc
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    print("WROTE", dest, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
