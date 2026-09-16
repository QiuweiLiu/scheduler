#!/usr/bin/env python3
"""PFA 试点 v2:Blue-Fringe ALERGIA 概率自动机学习(修正版,含子节点折叠)

输入:prefix_samples(v0_9),过滤 (planner_model_id, baseline) 组
序列:每个 run 按 position 排序的 target_next_raw_action(含 __END__ 终止)
输出:状态图(ASCII + JSON)、正则模板、覆盖率
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict

random.seed(42)


def load_rows(path):
    return [json.loads(l) for l in open(path)]


def extract_sequences(rows, planner=None, baseline=None):
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
            a = r.get("target_next_raw_action")
            seq.append(str(a) if a is not None else "__MISSING__")
        seqs.append(seq)
    return seqs


class Node:
    __slots__ = ("id", "depth", "trans", "term", "children")

    def __init__(self, nid, depth):
        self.id = nid
        self.depth = depth
        self.trans = Counter()
        self.term = 0
        self.children = {}   # symbol -> Node

    def count(self):
        return sum(self.trans.values()) + self.term


def build_prefix_tree(seqs):
    root = Node(0, 0)
    for seq in seqs:
        cur = root
        for sym in seq:
            cur.trans[sym] += 1
            if sym not in cur.children:
                cur.children[sym] = Node(len(cur.children) + 1, cur.depth + 1)
            cur = cur.children[sym]
        cur.term += 1
    return root


def hoeffding_bound(n1, n2, alpha=0.05):
    eps = math.sqrt(0.5 * math.log(2.0 / alpha))
    return eps * (1.0 / math.sqrt(n1) + 1.0 / math.sqrt(n2))


def compatible(n1, n2, alpha=0.05):
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


def fold(red, blue):
    """把 blue 折叠进 red:计数相加 + 递归合并子节点。"""
    for s, c in blue.trans.items():
        red.trans[s] += c
    red.term += blue.term
    for s, bc in blue.children.items():
        rc = red.children.get(s)
        if rc is None:
            rc = Node(-1, red.depth + 1)
            red.children[s] = rc
        fold(rc, bc)


def learn_blue_fringe(seqs, alpha=0.05):
    """标准 Blue-Fringe ALERGIA。"""
    root = build_prefix_tree(seqs)

    def collect_all(n, acc):
        acc.append(n)
        for c in n.children.values():
            collect_all(c, acc)

    all_nodes = []
    collect_all(root, all_nodes)

    red = [root]
    red_ids = {id(root)}
    blue = sorted(root.children.values(), key=lambda n: -n.count())

    def try_merge(b, red_set):
        for r in red_set:
            if r.depth == b.depth and compatible(r, b, alpha):
                fold(r, b)
                return True
        return False

    while blue:
        b = blue.pop(0)
        if try_merge(b, red):
            continue
        red.append(b)
        red_ids.add(id(b))
        for c in b.children.values():
            blue.append(c)
        blue.sort(key=lambda n: -n.count())

    return root, red


def extract_graph(red):
    """状态图:node_id -> {depth, count, term_p, trans, children}"""
    by_id = {}
    for n in red:
        total = n.count()
        trans = sorted(((s, c / total) for s, c in n.trans.items()), key=lambda x: -x[1])
        by_id[id(n)] = {
            "id": len(by_id), "depth": n.depth, "count": total,
            "term_p": n.term / total if total else 0.0,
            "trans": trans,
            "children": {s: id(c) for s, c in n.children.items()},
            "_node": n,
        }
    return by_id


def coverage(seqs, graph, root):
    """沿图走:有子节点转移就走;最终停在终止节点(term_p>0)。"""
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


def render_ascii(graph, topk=4):
    lines = []
    for g in sorted(graph.values(), key=lambda x: (x["depth"], x["id"])):
        trans_str = ", ".join(f"{s}:{p:.2f}" for s, p in g["trans"][:topk])
        term = f" 终止{g['term_p']:.2f}" if g["term_p"] > 0 else ""
        lines.append(f"  S{g['id']} (深{g['depth']}, 样本{g['count']}{term})")
        lines.append(f"      -> {trans_str}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix_file")
    ap.add_argument("--planner", default="Qwen3-4B")
    ap.add_argument("--baseline", default="langgraph_react")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_rows(args.prefix_file)
    seqs = extract_sequences(rows, planner=args.planner, baseline=args.baseline)
    print(f"组: planner={args.planner}, baseline={args.baseline}")
    print(f"runs: {len(seqs)}")
    lens = Counter(len(s) for s in seqs)
    print(f"序列长度分布: {dict(sorted(lens.items()))}")
    for i, s in enumerate(seqs[:5]):
        print(f"  样例{i}: {' → '.join(s)}")

    root, red = learn_blue_fringe(seqs, alpha=args.alpha)
    graph = extract_graph(red)
    # 用 graph 里的 id 重建 children 引用(指向 graph id)
    id_map = {id(n): g["id"] for n, g in graph.items()}
    n_trans = 0
    for g in graph.values():
        g["children"] = {s: id_map.get(c, -1) for s, c in g["children"].items()}
        n_trans += len(g["trans"])

    ok, tot = coverage(seqs, graph, root)
    print(f"\n合并后状态数: {len(graph)}, 转移数: {n_trans}")
    print(f"覆盖率(结构+终止): {ok}/{tot} = {ok/tot:.1%}")
    print("\n=== 状态图 ===")
    print(render_ascii(graph))

    result = {"planner": args.planner, "baseline": args.baseline,
              "runs": len(seqs), "alpha": args.alpha,
              "n_states": len(graph), "n_transitions": n_trans,
              "coverage": {"ok": ok, "total": tot, "ratio": ok / tot},
              "graph": {g["id"]: {k: v for k, v in g.items() if k != "_node"}
                        for g in graph.values()}}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入: {args.out}")


if __name__ == "__main__":
    main()
