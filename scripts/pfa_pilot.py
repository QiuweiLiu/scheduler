#!/usr/bin/env python3
"""PFA 试点 v2:标准 ALERGIA(修正 4 处差异,与 v1 对比)

v1 -> v2 修正:
1. Blue 处理顺序:count 降序 -> 标准 BFS 深度序(FIFO)
2. fold 创建的新子节点登记进全局 registry,JSON 图完整
3. 终止符:__END__ 不再作为普通符号;序列末尾节点 term+=1(标准终止事件)
4. (打分机制属于 FlexFringe/EDSM,标准 ALERGIA 本无,不做)
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
        # 终止符从序列中去掉(__END__ 作为终止事件,不进字母表)
        if seq and seq[-1] == "__END__":
            seq = seq[:-1]
        seqs.append(seq)
    return seqs


registry = []   # 全局节点登记(含 fold 创建的子节点)


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
    registry.append(root)
    for seq in seqs:
        cur = root
        for sym in seq:
            cur.trans[sym] += 1
            if sym not in cur.children:
                n = Node(len(registry), cur.depth + 1)
                registry.append(n)
                cur.children[sym] = n
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
    """把 blue 折叠进 red:计数相加 + 递归合并子节点(创建的新节点登记进 registry)。"""
    for s, c in blue.trans.items():
        red.trans[s] += c
    red.term += blue.term
    for s, bc in blue.children.items():
        rc = red.children.get(s)
        if rc is None:
            rc = Node(len(registry), red.depth + 1)
            registry.append(rc)
            red.children[s] = rc
        fold(rc, bc)


def learn_blue_fringe(seqs, alpha=0.05):
    """标准 Blue-Fringe ALERGIA:blue 队列按 BFS 深度序(FIFO),无排序启发式。"""
    root = build_prefix_tree(seqs)
    red = [root]
    blue = list(root.children.values())
    while blue:
        b = blue.pop(0)
        merged = False
        for r in red:
            if r.depth == b.depth and compatible(r, b, alpha):
                fold(r, b)
                merged = True
                break
        if not merged:
            red.append(b)
            for c in b.children.values():
                blue.append(c)
    return root


def extract_graph(root):
    """状态图:从 root 可达的所有节点(registry 登记,children 映射完整)。"""
    graph = {}
    id_map = {}

    def visit(n):
        if id(n) in id_map:
            return
        gid = len(graph)
        id_map[id(n)] = gid
        total = n.count()
        trans = sorted(((s, c / total) for s, c in n.trans.items()), key=lambda x: -x[1])
        graph[gid] = {"id": gid, "depth": n.depth, "count": total,
                      "term_p": n.term / total if total else 0.0,
                      "trans": trans, "children": {}, "_node": n}
        for s, c in n.children.items():
            visit(c)

    visit(root)
    for g in graph.values():
        g["children"] = {s: id_map[id(c)] for s, c in g["_node"].children.items()}
    return graph


def coverage(seqs, graph, root):
    """沿图走:序列(无终止符)走完且最后节点 term>0 即覆盖。"""
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
    print(f"序列长度分布(不含终止符): {dict(sorted(lens.items()))}")
    for i, s in enumerate(seqs[:5]):
        print(f"  样例{i}: {' → '.join(s)} → __END__")

    root = learn_blue_fringe(seqs, alpha=args.alpha)
    graph = extract_graph(root)
    n_trans = sum(len(g["trans"]) for g in graph.values())

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
