#!/usr/bin/env python3
"""LLM 零样本角色预测(冷启动对照,本地 Qwen3-4B)

输入:role_dataset 的 test 子集(默认 200 行)
prompt:任务描述 + 已执行角色序列 + 当前状态 → 预测下一步角色
评估:Top-1 准确率(含按类混淆)
"""
import argparse
import json
import random
from collections import defaultdict, Counter

ROLES = ["init", "plan", "execute", "aggregate", "terminate", "verify"]
ROLE_DESC = {
    "init": "初始化(读取输入/视频/任务信息)",
    "plan": "规划(LLM 决定下一步做什么)",
    "execute": "执行(调用视觉工具/检测/问答)",
    "aggregate": "汇总(生成答案/报告)",
    "terminate": "结束",
    "verify": "验证(检查结果,如失败会重试)",
}


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def build_prompt(run_roles, r):
    seq_str = " → ".join(run_roles)
    resident = "模型已驻留(热调用)" if r.get("model_resident_before") else "模型需加载(冷调用)"
    sys = ("你是一个视频问答 Agent 的执行流程预测器。流程角色包括:"
           + ", ".join(f"{k}({v})" for k, v in ROLE_DESC.items())
           + "。根据已执行的步骤序列和当前状态,预测下一步的角色。只输出角色名,不要其他内容。")
    user = (f"已执行步骤序列: [{seq_str}]\n"
            f"当前模型: {r['model_id']};{resident};"
            f"baseline: {r['baseline']}\n"
            f"请预测下一步流程角色(从 {', '.join(ROLES)} 中选择):")
    return sys, user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-path", default="/root/autodl-tmp/scheduler/Qwen3-4B")
    ap.add_argument("--n-sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_jsonl(args.dataset)
    test_rows = [r for r in rows if r["split"] == "test"]
    rng = random.Random(args.seed)
    sample = rng.sample(test_rows, min(args.n_sample, len(test_rows)))
    print(f"test 总数 {len(test_rows)}, 采样 {len(sample)} 行")

    # 每 run 的完整角色序列
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    run_seqs = {}
    for rid, rs in by_run.items():
        rs_sorted = sorted(rs, key=lambda x: x["event_index"])
        run_seqs[rid] = [x["role"] for x in rs_sorted]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"加载模型 {args.model_path} ...")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)
    model.eval()

    correct = 0
    conf = Counter()
    per_class = defaultdict(lambda: {"n": 0, "hit": 0})
    with torch.no_grad():
        for r in sample:
            seq = run_seqs[r["run_id"]]
            pos = r["event_index"]
            done = seq[: pos + 1]
            sys_p, user_p = build_prompt(done, r)
            msgs = [{"role": "system", "content": sys_p},
                    {"role": "user", "content": user_p}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tok(prompt, return_tensors="pt").to("cuda")
            out = model.generate(**inp, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = None
            for role in ROLES:
                if role in text.lower():
                    pred = role
                    break
            if pred is None:
                pred = "unknown"
            conf[pred] += 1
            per_class[r["next_role"]]["n"] += 1
            if pred == r["next_role"]:
                correct += 1
                per_class[r["next_role"]]["hit"] += 1

    acc = correct / len(sample)
    print(f"\n零样本 Top-1 准确率: {acc:.4f} ({correct}/{len(sample)})")
    print("预测分布:", dict(conf))
    print("按类命中:")
    for c, d in sorted(per_class.items()):
        print(f"  {c:12s} n={d['n']:5d} 命中={d['hit']:5d} acc={d['hit']/d['n']:.3f}")

    result = {"model": args.model_path, "n": len(sample), "top1": round(acc, 4),
              "pred_dist": dict(conf),
              "per_class": {c: {"n": d["n"], "hit": d["hit"]} for c, d in per_class.items()}}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入: {args.out}")


if __name__ == "__main__":
    main()
