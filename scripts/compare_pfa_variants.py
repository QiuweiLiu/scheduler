#!/usr/bin/env python3
"""定位 PFA 合并差异(自包含版,每个配置独立状态空间)。

4 种配置交叉:
  blue 顺序:  fifo(标准 BFS) / count(降序, v1 风格)
  终止处理:  end_symbol(__END__ 作为符号) / term_event(去符号)
完整图统计:从 root 可达的全部节点(v1 的 21 状态是"只画红色根节点"的残缺图假象)。
"""
import json
import math
from collections import Counter, defaultdict


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


def build_prefix_tree(seqs, end_symbol):
    root = Node(0, 0)
    nodes = [root]
    for seq in seqs:
        cur = root
        syms = list(seq)
        if end_symbol:
            syms = syms + ["__END__"]
        for sym in syms:
            cur.trans[sym] += 1
            if sym not in cur.children:
                n = Node(len(nodes), cur.depth + 1)
                nodes.append(n)
                cur.children[sym] = n
            cur = cur.children[sym]
        if not end_symbol:
            cur.term += 1
        else:
            cur.term += 1  # __END__ 之后序列结束,终止落在 __END__ 节点
    return root, nodes


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


def learn(seqs, alpha, blue_mode, end_symbol):
    root, nodes = build_prefix_tree(seqs, end_symbol)
    red = [root]
    blue = list(root.children.values())

    def sort_blue():
        if blue_mode == "count":
            blue.sort(key=lambda n: -n.count())

    sort_blue()
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
            sort_blue()
    return root


def _all(root):
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


def main():
    rows = load_rows("results/processed/neural_prefix_dataset_v0_9_options_sourcefix_20260805/prefixes_visual_context.jsonl")
    seqs_raw = extract_sequences(rows, planner="Qwen3-4B", baseline="langgraph_react")

    configs = [
        ("fifo + end_symbol",  "fifo", True),
        ("fifo + term_event",  "fifo", False),
        ("count + end_symbol", "count", True),
        ("count + term_event", "count", False),
    ]
    for label, blue_mode, end_symbol in configs:
        root = learn(seqs_raw, 0.05, blue_mode, end_symbol)
        nodes = _all(root)
        by_depth = Counter(n.depth for n in nodes)
        n_trans = sum(len(n.trans) for n in nodes)
        print(f"{label:24s} 完整状态数={len(nodes):5d}  转移数={n_trans:5d}  按深度={dict(sorted(by_depth.items()))}")


if __name__ == "__main__":
    main()
