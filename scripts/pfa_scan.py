#!/usr/bin/env python3
"""PFA 权衡扫描:alpha x 字母表(原始 11 / 工具族 6)完整图状态数对比。

用法:
  python3 scripts/pfa_scan.py PREFIX_FILE [--out OUT]
"""
import argparse
import json
import math
from collections import Counter, defaultdict

ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.5]

# 工具族映射(11 种 raw -> 6 族)
TOOL_FAMILY = {
    "frame-selector": "select_frames",
    "image-grid-selector": "select_frames",
    "image-qa": "visual_qa",
    "image-grid-qa": "visual_qa",
    "patch-zoomer": "visual_qa",
    "temporal-grounding": "temporal_ops",
    "temporal-qa": "temporal_ops",
    "summarization-tool": "summarize",
    "answer": "answer",
    "yolo-tracker": "detect",
    "__END__": "__END__",
}


def load_rows(path):
    return [json.loads(l) for l in open(path)]


def extract_sequences(rows, planner=None, baseline=None, family=False):
    by_run = defaultdict(list)
    for r in rows:
        if planner and r.get("planner_model_id") != planner:
            continue
        if baseline and r.get("baseline") != baseline:
            continue
        by_run[r["run_id"]].append(r)
    seqs = []
    for rid, rs in by_run.items():
        rs_sorted = sorted(rs, key=lambda x: x["position"])
        seen = set()
        seq = []
        for r in rs_sorted:
            if r["position"] in seen:
                continue
            seen.add(r["position"])
            a = str(r.get("target_next_raw_action") or "__MISSING__")
            if family:
                a = TOOL_FAMILY.get(a, a)
            seq.append(a)
        if seq and seq[-1] == "__END__":
            seq = seq[:-1]
        seqs.append(seq)
    return seqs


class Node:
    __slots__ = ("id", "depth", "trans", "term", "children")

    def __init__(self, nid, depth):
        self.id = nid
        self.depth = depth
        self.trans = Counter()
        self.term = 0
        self.children = {}

    def count(self):
        return sum(self.trans.values()) + self.term


def hoeffding_bound(n1, n2, alpha):
    eps = math.sqrt(0.5 * math.log(2.0 / alpha))
    return eps * (1.0 / math.sqrt(n1) + 1.0 / math.sqrt(n2))


def compatible(n1, n2, alpha):
    c1, c2 = n1.count(), n2.count()
    if c1 == 0 or c2 == 0:
        return True
    bound = hoeffding_bound(c1, c2, alpha)
    for s in set(n1.trans) | set(n2.trans):
        p1 = n1.trans.get(s, 0) / c1
        p2 = n2.trans.get(s, 0) / c2
        if abs(p1 - p2) > bound:
            return False
    return abs(n1.term / c1 - n2.term / c2) <= bound


def fold(red, blue, nodes):
    for s, c in blue.trans.items():
        red.trans[s] += c
    red.term += blue.term
    for s, bc in blue.children.items():
        rc = red.children.get(s)
        if rc is None:
            rc = Node(len(nodes), red.depth + 1)
            nodes.append(rc)
            red.children[s] = rc
        fold(rc, bc, nodes)


def build_prefix_tree(seqs, nodes):
    root = Node(0, 0)
    nodes.append(root)
    for seq in seqs:
        cur = root
        for sym in seq:
            cur.trans[sym] += 1
            if sym not in cur.children:
                n = Node(len(nodes), cur.depth + 1)
                nodes.append(n)
                cur.children[sym] = n
            cur = cur.children[sym]
        cur.term += 1
    return root


def learn(seqs, alpha):
    nodes = []
    root = build_prefix_tree(seqs, nodes)
    red = [root]
    blue = list(root.children.values())
    while blue:
        b = blue.pop(0)
        merged = False
        for r in red:
            if r.depth == b.depth and compatible(r, b, alpha):
                fold(r, b, nodes)
                merged = True
                break
        if not merged:
            red.append(b)
            for c in b.children.values():
                blue.append(c)
    return root, nodes


def all_reachable(root):
    out = []
    seen = set()

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        out.append(n)
        for c in n.children.values():
            visit(c)

    visit(root)
    return out


def coverage(seqs, root):
    ok = 0
    for seq in seqs:
        cur = root
        good = True
        for sym in seq:
            nxt = cur.children.get(sym)
            if nxt is None:
                good = False
                break
            cur = nxt
        if good and cur.term > 0:
            ok += 1
    return ok, len(seqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix_file")
    ap.add_argument("--out", default=None)
    ap.add_argument("--planner", default="Qwen3-4B")
    ap.add_argument("--baseline", default="langgraph_react")
    args = ap.parse_args()

    rows = load_rows(args.prefix_file)
    raw_seqs = extract_sequences(rows, planner=args.planner, baseline=args.baseline, family=False)
    fam_seqs = extract_sequences(rows, planner=args.planner, baseline=args.baseline, family=True)
    print(f"组: planner={args.planner}, baseline={args.baseline}, runs={len(raw_seqs)}")
    print(f"原始字母表序列样例: {len(set(s for seq in raw_seqs for s in seq))} 种符号")
    print(f"工具族字母表: {len(set(s for seq in fam_seqs for s in seq))} 种符号\n")

    header = f"{'alpha':>6} | {'字母表':>9} | {'状态数':>6} | {'转移数':>6} | {'覆盖率':>7} | {'每状态均样本':>9} | {'大状态(>=20)':>10}"
    print(header)
    print("-" * len(header))

    results = []
    for alpha in ALPHAS:
        for fam, seqs in ((False, raw_seqs), (True, fam_seqs)):
            root, nodes = learn(seqs, alpha)
            reach = all_reachable(root)
            n_trans = sum(len(n.trans) for n in reach)
            ok, tot = coverage(seqs, root)
            total_count = sum(n.count() for n in reach)
            per_state = total_count / len(reach) if reach else 0
            big = sum(1 for n in reach if n.count() >= 20)
            label = "族6" if fam else "原始11"
            print(f"{alpha:>6} | {label:>9} | {len(reach):>6} | {n_trans:>6} | {ok/tot:>7.1%} | {per_state:>9.1f} | {big:>10}")
            results.append({"alpha": alpha, "alphabet": label, "n_states": len(reach),
                            "n_transitions": n_trans, "coverage": ok / tot,
                            "per_state_mean": round(per_state, 1), "big_states_ge20": big})

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入: {args.out}")


if __name__ == "__main__":
    main()
