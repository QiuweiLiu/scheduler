#!/usr/bin/env python3
"""行为预测器对比实验(对应远端 docs/predictor_selection_plan_20260805.md)

模型:B1 条件计数(统计)/ B2 LightGBM / B3 MLP / B4 GRU / B4b LSTM
目标:抽象角色 3 类(主)+ 细粒度 7 类(对照)
split:prefix 自带 train/validation/test(视频级 48/8/8 口径)
指标:Top-1 / Top-3 / NLL / ECE
"""
import argparse
import json
import math
from collections import Counter, defaultdict
import numpy as np

ACTIVITY_TO_ROLE = {
    "sample_seek": "execute", "spatial_qa": "execute", "temporal_qa": "execute",
    "object_detection": "execute", "summarize": "execute",
    "answer": "aggregate", "__END__": "terminate",
}
ROLES = ["execute", "aggregate", "terminate"]
FAMILY = {
    "frame-selector": "select_frames", "image-grid-selector": "select_frames",
    "image-qa": "visual_qa", "image-grid-qa": "visual_qa", "patch-zoomer": "visual_qa",
    "temporal-grounding": "temporal_ops", "temporal-qa": "temporal_ops",
    "summarization-tool": "summarize", "answer": "answer",
    "yolo-tracker": "detect", "__END__": "__END__",
}


def load_rows(path):
    return [json.loads(l) for l in open(path)]


def qcut(vals, n=3):
    vals = sorted(v for v in vals if v is not None)
    edges = []
    for i in range(1, n):
        edges.append(vals[min(len(vals) - 1, int(i / n * len(vals)))])
    edges = sorted(set(edges))
    def b(v):
        if v is None:
            return "missing"
        for i, e in enumerate(edges):
            if v <= e:
                return f"bin{i}"
        return f"bin{len(edges)}"
    return b


def extract_features(rows):
    def ev(d):
        return d.get("coverage_ratio") if isinstance(d, dict) else None
    def pr(d):
        return d.get("progress_ratio") if isinstance(d, dict) else None
    cov_b = qcut([ev((r.get("state_features") or {}).get("evidence")) for r in rows])
    prog_b = qcut([pr((r.get("state_features") or {}).get("prefix")) for r in rows])
    raw_counts = Counter()
    for r in rows:
        cat = (r.get("features") or {}).get("categorical") or {}
        t = cat.get("raw_prefix_tail2")
        if t:
            raw_counts[t] += 1
    top_raw = [k for k, _ in raw_counts.most_common(20)]
    raw_map = {k: i for i, k in enumerate(top_raw)}

    feats, y_role, y_fine, seqs = [], [], [], []
    for r in rows:
        ts = r.get("task_structure") or {}
        sf = r.get("state_features") or {}
        cat = (r.get("features") or {}).get("categorical") or {}
        cov = cov_b(((sf.get("evidence") or {}).get("coverage_ratio")) if isinstance(sf.get("evidence"), dict) else None)
        prog = prog_b(((sf.get("prefix") or {}).get("progress_ratio")) if isinstance(sf.get("prefix"), dict) else None)
        rawt = cat.get("raw_prefix_tail2")
        # 首步族
        first_raw = None
        for a in (r.get("prefix_raw_actions") or []):
            if a not in ("__START__", "__END__"):
                first_raw = a
                break
        fam = FAMILY.get(first_raw, "unknown") if first_raw else "none"
        feats.append({
            "planner": r.get("planner_model_id") or "unknown",
            "baseline": r.get("baseline") or "unknown",
            "position": r.get("position") or 0,
            "question_type": (ts.get("question_type") if isinstance(ts, dict) else None) or "unknown",
            "domain": (ts.get("domain") if isinstance(ts, dict) else None) or "unknown",
            "coverage": cov, "progress": prog,
            "raw_tail2": raw_map.get(rawt, len(raw_map)) if rawt else len(raw_map),
            "family": fam,
            "video_id": r.get("video_id"), "split": r.get("split"),
        })
        tgt = r.get("target_next_activity")
        y_fine.append(tgt if tgt else "unknown")
        y_role.append(ACTIVITY_TO_ROLE.get(tgt, "other"))
        seqs.append([ACTIVITY_TO_ROLE.get(a, "other") for a in (r.get("prefix_activities") or [])])
    return feats, y_role, y_fine, seqs


def encode_cat(values, all_vals):
    m = {v: i for i, v in enumerate(sorted(set(all_vals)))}
    return [m.get(v, m.get("unknown", 0)) for v in values]


def evaluate(y_true, probs, labels):
    idx = {l: i for i, l in enumerate(labels)}
    top1 = top3 = nll = 0
    for t, p in zip(y_true, probs):
        order = sorted(range(len(p)), key=lambda i: -p[i])
        if labels[order[0]] == t:
            top1 += 1
        if t in {labels[i] for i in order[:3]}:
            top3 += 1
        nll += -math.log(max(p[idx.get(t, 0)], 1e-9))
    n = len(y_true)
    return {"top1": round(top1 / n, 4), "top3": round(top3 / n, 4), "nll": round(nll / n, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_rows(args.prefix)
    feats, y_role, y_fine, seqs = extract_features(rows)
    print(f"prefix 行: {len(feats)}")

    split_idx = {"train": [], "validation": [], "test": []}
    for i, f in enumerate(feats):
        s = f["split"]
        if s in split_idx:
            split_idx[s].append(i)
    print(f"split: train={len(split_idx['train'])} val={len(split_idx['validation'])} test={len(split_idx['test'])}")

    results = {"plan_doc": "docs/predictor_selection_plan_20260805.md", "targets": {}}

    # 特征编码(train 拟合)
    tr_i = split_idx["train"]
    all_f = [feats[i] for i in tr_i]
    cat_cols = {
        "planner": encode_cat([f["planner"] for f in feats], [f["planner"] for f in all_f]),
        "baseline": encode_cat([f["baseline"] for f in feats], [f["baseline"] for f in all_f]),
        "question_type": encode_cat([f["question_type"] for f in feats], [f["question_type"] for f in all_f]),
        "domain": encode_cat([f["domain"] for f in feats], [f["domain"] for f in all_f]),
        "coverage": encode_cat([f["coverage"] for f in feats], [f["coverage"] for f in all_f]),
        "progress": encode_cat([f["progress"] for f in feats], [f["progress"] for f in all_f]),
        "raw_tail2": [f["raw_tail2"] for f in feats],
        "family": encode_cat([f["family"] for f in feats], [f["family"] for f in all_f]),
    }
    n_pos = max(f["position"] for f in feats) + 1
    X = np.array([[cat_cols["planner"][i], cat_cols["baseline"][i], feats[i]["position"],
                   cat_cols["question_type"][i], cat_cols["domain"][i],
                   cat_cols["coverage"][i], cat_cols["progress"][i],
                   cat_cols["raw_tail2"][i], cat_cols["family"][i]]
                  for i in range(len(feats))], dtype=np.float32)

    def run_target(name, y):
        labels = sorted(set(y))
        ytr = [y[i] for i in split_idx["train"]]
        yva = [y[i] for i in split_idx["validation"]]
        yte = [y[i] for i in split_idx["test"]]
        Xtr, Xva, Xte = X[split_idx["train"]], X[split_idx["validation"]], X[split_idx["test"]]
        lmap = {l: i for i, l in enumerate(labels)}
        ytr_i = np.array([lmap[v] for v in ytr])

        def proba_to_metric(y_ev, probs):
            return evaluate(y_ev, probs.tolist(), labels)

        tgt = {"labels": labels, "models": {}}

        # B1: 条件计数(planner+baseline+position)
        ctx_counts = defaultdict(Counter)
        for i in split_idx["train"]:
            ctx = (int(X[i, 0]), int(X[i, 1]), int(X[i, 2]))
            ctx_counts[ctx][y[i]] += 1
        ctx_prior = {c: {l: v / sum(vals.values()) for l, v in vals.items()}
                     for c, vals in ctx_counts.items()}
        probs_b1 = []
        for i in split_idx["test"]:
            ctx = (int(X[i, 0]), int(X[i, 1]), int(X[i, 2]))
            p = ctx_prior.get(ctx, {})
            probs_b1.append([p.get(l, 1e-6) for l in labels])
        tgt["models"]["B1 条件计数"] = proba_to_metric(yte, np.array(probs_b1))
        print(f"[{name}] B1 条件计数: {tgt['models']['B1 条件计数']}")

        # B2: LightGBM
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                               verbose=-1, random_state=42)
        m.fit(Xtr, ytr_i)
        tgt["models"]["B2 LightGBM"] = proba_to_metric(yte, m.predict_proba(Xte))
        print(f"[{name}] B2 LightGBM: {tgt['models']['B2 LightGBM']}")

        # B3: MLP
        import torch
        import torch.nn as nn
        torch.manual_seed(42)
        ncls = len(labels)
        net = nn.Sequential(nn.Linear(9, 128), nn.ReLU(), nn.Dropout(0.2),
                            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, ncls))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr_i).long()
        xv = torch.from_numpy(Xva)
        yv = torch.from_numpy(np.array([lmap[v] for v in yva])).long()
        best_acc, best_state = -1, None
        for ep in range(80):
            net.train(); opt.zero_grad()
            loss = nn.functional.cross_entropy(net(xt), yt)
            loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                acc = (net(xv).argmax(1) == yv).float().mean().item()
            if acc > best_acc:
                best_acc, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            probs = torch.softmax(net(torch.from_numpy(Xte)), dim=1).numpy()
        tgt["models"]["B3 MLP"] = proba_to_metric(yte, probs)
        print(f"[{name}] B3 MLP: {tgt['models']['B3 MLP']} (val_acc={best_acc:.3f})")

        # B4/B4b: GRU / LSTM(序列输入)
        seq_id = {}
        sall = [a for s in seqs for a in s]
        for a in sorted(set(sall)):
            seq_id.setdefault(a, len(seq_id))
        max_len = max(len(s) for s in seqs)
        pad_id = len(seq_id)
        def pad(s):
            ids = [seq_id[a] for a in s]
            ids = ids[-max_len:]
            return ids + [pad_id] * (max_len - len(ids))
        S = np.array([pad(s) for s in seqs], dtype=np.int64)
        Str, Sva, Ste = S[split_idx["train"]], S[split_idx["validation"]], S[split_idx["test"]]
        Slen = np.array([min(len(s), max_len) for s in seqs], dtype=np.int64)

        def run_rnn(kind):
            torch.manual_seed(42)
            class RNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.emb = nn.Embedding(pad_id + 1, 32, padding_idx=pad_id)
                    self.rnn = nn.GRU(32, 64, batch_first=True) if kind == "gru" else nn.LSTM(32, 64, batch_first=True)
                    self.head = nn.Sequential(nn.Linear(64 + 9, 64), nn.ReLU(), nn.Linear(64, ncls))
                def forward(self, s, x):
                    e = self.emb(s)
                    out, h = self.rnn(e)
                    if isinstance(h, tuple):
                        h = h[0]
                    hh = h[-1]
                    return self.head(torch.cat([hh, x], dim=1))
            model = RNN()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            st = torch.from_numpy(Str); slt = torch.from_numpy(Slen[split_idx["train"]])
            xtt = torch.from_numpy(Xtr)
            sv = torch.from_numpy(Sva); slv = torch.from_numpy(Slen[split_idx["validation"]])
            xvv = torch.from_numpy(Xva)
            best_acc, best_state = -1, None
            for ep in range(40):
                model.train(); opt.zero_grad()
                loss = nn.functional.cross_entropy(model(st, xtt), yt)
                loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    acc = (model(sv, xvv).argmax(1) == yv).float().mean().item()
                if acc > best_acc:
                    best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                probs = torch.softmax(model(torch.from_numpy(Ste), torch.from_numpy(Xte)), dim=1).numpy()
            return proba_to_metric(yte, probs), best_acc

        for kind in ("gru", "lstm"):
            mtr, vacc = run_rnn(kind)
            tgt["models"][f"B4 {kind.upper()}"] = mtr
            print(f"[{name}] B4 {kind.upper()}: {mtr} (val_acc={vacc:.3f})")

        return tgt

    results["targets"]["role3"] = run_target("角色3类", y_role)
    results["targets"]["fine7"] = run_target("细粒度7类", y_fine)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入: {args.out}")


if __name__ == "__main__":
    main()
