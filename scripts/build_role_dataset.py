#!/usr/bin/env python3
"""角色序列数据集构建(5 类:init/plan/execute/aggregate/terminate,verify 预留)

流程:
1. 提取 expansion 472 runs 的 compute 事件(字段对齐 core)
2. 合并 core(768 runs)+ expansion
3. 冷热状态推导(per run model 首次=冷/再次=热;yolo 恒冷)+ yolo_batch=8 补全
4. 每 run 按 event_index 排序 -> 角色序列;每行目标 = 下一角色,末行 = terminate
5. 视频级重划分 train 240 / val 30 / test 30(seed=42)
输出:role_dataset_v0_1.jsonl + split_manifest_v0_1.json
"""
import argparse
import json
import random
from collections import defaultdict, Counter

NODE_TO_ROLE = {
    "run_control": "init", "planner": "plan",
    "videotool_spatial": "execute", "videotool_temporal": "execute",
    "videotool_generalist": "execute", "answer_generation": "aggregate",
}
ROLES = ["init", "plan", "execute", "aggregate", "terminate"]
YOLO_MODELS = {"yolo11x.pt"}


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def extract_expansion(expansion_root):
    import glob
    events = []
    for trace_path in sorted(glob.glob(f"{expansion_root}/*/trace.jsonl")):
        run_dir = trace_path.rsplit("/", 2)[1]
        base = run_dir.rsplit("_", 1)[0]
        baseline = "star" if run_dir.endswith("_star_" + run_dir.split("_")[-1]) else None
        # baseline 从目录名判断:目录形如 <video>_<baseline>_<ts>
        for b in ("star", "langgraph_react"):
            if f"_{b}_" in run_dir:
                baseline = b
                break
        rows = load_jsonl(trace_path)
        task_id = rows[0].get("task_id") or ""
        # video_id 从 task_id 解析:videomme300_<video>_<qid>_r01
        video_id = None
        parts = task_id.split("_")
        if len(parts) >= 3 and parts[0] == "videomme300":
            video_id = parts[1]
        inp = rows[0].get("input") or {}
        if not baseline:
            baseline = inp.get("baseline", "unknown")
        for ei, r in enumerate(rows):
            res = r.get("resource") or {}
            rt = res.get("runtime_ms")
            if rt is None:
                rt = res.get("local_runtime_ms")
            events.append({
                "event_index": ei,
                "event_type": r.get("event_type", "?"),
                "node_type": r.get("node_type", "?"),
                "model_id": r.get("model_id", "?"),
                "raw_action": r.get("action"),
                "retry_of": r.get("retry_of"),
                "status": r.get("status"),
                "runtime_ms": float(rt) if rt is not None else None,
                "peak_allocated_mb": res.get("peak_allocated_mb"),
                "load_ms": res.get("load_ms"),
                "queue_ms": res.get("queue_ms"),
                "api_wait_ms": res.get("api_wait_ms"),
                "gpu_model": res.get("gpu_model"),
                "baseline": baseline,
                "video_id": video_id or run_dir,
                "run_id": r.get("run_id") or run_dir,
                "source": "expansion",
            })
    return events


def derive_resident(events_by_run):
    """per run: model 首次出现 = 冷(False),再次 = 热(True);yolo 恒冷。"""
    for rid, evs in events_by_run.items():
        seen = set()
        for e in evs:
            m = e["model_id"]
            if m in YOLO_MODELS:
                e["model_resident_before"] = False
            elif m not in seen:
                e["model_resident_before"] = False
                seen.add(m)
            else:
                e["model_resident_before"] = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-compute", required=True)
    ap.add_argument("--expansion-root", required=True)
    ap.add_argument("--prefix", required=True, help="prefix 数据(question_type 视频级 join 来源)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    core = load_jsonl(args.core_compute)
    for e in core:
        e["source"] = "core"
    exp = extract_expansion(args.expansion_root)
    print(f"core 事件: {len(core)}, expansion 事件: {len(exp)}")
    all_events = core + exp

    # question_type:视频级 join prefix(取非 unknown 多数)
    prefixes = load_jsonl(args.prefix)
    qt_by_video = defaultdict(Counter)
    for p in prefixes:
        ts = p.get("task_structure") or {}
        qt = (ts.get("question_type") if isinstance(ts, dict) else None) or "unknown"
        qt_by_video[p["video_id"]][qt] += 1
    video_qt = {}
    for v, c in qt_by_video.items():
        non_unknown = Counter({k: n for k, n in c.items() if k != "unknown"})
        video_qt[v] = (non_unknown if non_unknown else c).most_common(1)[0][0]
    print(f"question_type 视频级映射: {len(video_qt)} 视频")

    # 补 yolo_batch(yolo 事件 = 8)+ frame_count(core 从 input_scale 解析)
    for e in all_events:
        if e.get("model_id") in YOLO_MODELS:
            e["yolo_batch"] = 8
        else:
            e["yolo_batch"] = None
        fc = None
        if e.get("source") == "core" and isinstance(e.get("input_scale"), dict):
            fc = e["input_scale"].get("frame_count")
        e["frame_count"] = fc

    # 冷热推导
    by_run = defaultdict(list)
    for e in all_events:
        by_run[e["run_id"]].append(e)
    for evs in by_run.values():
        evs.sort(key=lambda x: x.get("event_index", 0))
    derive_resident(by_run)
    cold = sum(1 for e in all_events if e.get("model_resident_before") is False)
    hot = sum(1 for e in all_events if e.get("model_resident_before") is True)
    print(f"冷热推导: 冷={cold} 热={hot} 未推导={len(all_events)-cold-hot}")

    # 角色序列 + 下一角色标签
    rows = []
    video_set = set()
    for rid, evs in sorted(by_run.items()):
        roles = [NODE_TO_ROLE.get(e.get("node_type"), "other") for e in evs]
        for i, e in enumerate(evs):
            nxt = roles[i + 1] if i + 1 < len(roles) else "terminate"
            rows.append({
                "run_id": rid, "event_index": i,
                "role": roles[i], "next_role": nxt,
                "model_id": e.get("model_id"), "baseline": e.get("baseline"),
                "video_id": e.get("video_id"), "event_type": e.get("event_type"),
                "model_resident_before": e.get("model_resident_before"),
                "yolo_batch": e.get("yolo_batch"),
                "runtime_ms": e.get("runtime_ms"),
                "peak_allocated_mb": e.get("peak_allocated_mb"),
                "load_ms": e.get("load_ms"),
                "frame_count": e.get("frame_count"),
                "raw_action": e.get("raw_action") or e.get("activity"),
                "question_type": video_qt.get(e.get("video_id"), "unknown"),
                "source": e.get("source"),
            })
            video_set.add(e.get("video_id"))
    print(f"数据集行数: {len(rows)}, 视频数: {len(video_set)}")
    print("角色分布:", dict(Counter(r["role"] for r in rows)))
    print("下一角色分布:", dict(Counter(r["next_role"] for r in rows)))

    # 重划分:视频级 240/30/30
    rng = random.Random(args.seed)
    videos = sorted(video_set)
    rng.shuffle(videos)
    n_tr, n_va = 240, 30
    split_map = {}
    for i, v in enumerate(videos):
        if i < n_tr:
            split_map[v] = "train"
        elif i < n_tr + n_va:
            split_map[v] = "validation"
        else:
            split_map[v] = "test"
    for r in rows:
        r["split"] = split_map[r["video_id"]]
    print("split 视频数:", dict(Counter(split_map.values())))
    print("split 行数:", dict(Counter(r["split"] for r in rows)))

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"schema": "role-dataset-v0.1", "seed": args.seed,
                "split": {v: s for v, s in split_map.items()},
                "n_train": n_tr, "n_validation": n_va,
                "roles": ROLES, "n_rows": len(rows), "n_videos": len(video_set)}
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"写入: {args.out} + {args.manifest}")


if __name__ == "__main__":
    main()
