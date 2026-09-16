#!/usr/bin/env python3
"""R1-seq_module:统一序列模块(正式产物,供 R2 接入 GRU/LSTM/BiLSTM/Transformer/residual)

- 从 canonical 视图构建事件序列 tokens/lengths/mask
- token 约定:PAD=0,非 execute 事件=1,execute 族=2..7
- 空 prefix → len=0(模型需支持,见 r1c 测试)
- 两种轴:完整事件轴 / execute-only 压缩轴
"""
import json
from collections import defaultdict

FAMILY = {"sample_seek": "select_frames", "frame-selector": "select_frames",
          "image-grid-selector": "select_frames", "spatial_qa": "visual_qa",
          "image-qa": "visual_qa", "image-grid-qa": "visual_qa", "patch-zoomer": "visual_qa",
          "temporal-qa": "temporal_ops", "temporal-grounding": "temporal_ops",
          "summarize": "summarize", "summarization-tool": "summarize",
          "object_detection": "detect", "yolo-tracker": "detect"}
FAM_TOKEN = {"select_frames": 2, "visual_qa": 3, "temporal_ops": 4,
             "summarize": 5, "detect": 6, "other": 7}
PAD = 0
NON_EXECUTE = 1
MAX_LEN = 64


def load_rows(path):
    rows = [json.loads(l) for l in open(path)]
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    for rl in by_run.values():
        rl.sort(key=lambda x: x["event_index"])
    return by_run


def tok_of(r):
    if r["role"] == "execute":
        return FAM_TOKEN.get(FAMILY.get(r["raw_action"], "other"), 7)
    return NON_EXECUTE


def build_sequences(samples, by_run, axis="full", max_len=MAX_LEN):
    """返回 tokens 列表(list of list)、真实 lengths;max_len 超长只保留最早 max_len 个
    统一接口:cutoff_event_index(role 视图=event_index;tool 视图=target_event_index-1)"""
    seqs = []
    for s in samples:
        # 兼容两种视图的 cutoff 键
        tidx = s.get("cutoff_event_index")
        if tidx is None:
            tidx = s.get("target_event_index")
        if tidx is None:
            tidx = s.get("event_index")
        rl = [r for r in by_run.get(s["run_id"], []) if r["event_index"] < tidx]
        if axis == "execute":
            rl = [r for r in rl if r["role"] == "execute"]
        seqs.append([tok_of(r) for r in rl][-max_len:])
    lengths = [len(s) for s in seqs]
    return seqs, lengths
