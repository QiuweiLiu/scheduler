import json
import numpy as np
import torch
import sys
sys.path.insert(0, '/root/autodl-tmp/scheduler')
import torch.nn as nn

D = "/root/autodl-tmp/scheduler/"
rows = [json.loads(l) for l in open(D + "results/processed/role_dataset_v0_3.jsonl")]
FAMILY = {"sample_seek": "select_frames", "frame-selector": "select_frames",
          "image-grid-selector": "select_frames", "spatial_qa": "visual_qa",
          "image-qa": "visual_qa", "image-grid-qa": "visual_qa", "patch-zoomer": "visual_qa",
          "temporal-qa": "temporal_ops", "temporal-grounding": "temporal_ops",
          "summarize": "summarize", "summarization-tool": "summarize",
          "object_detection": "detect", "yolo-tracker": "detect"}
FAMS = ["select_frames", "visual_qa", "temporal_ops", "summarize", "detect", "other"]
fam = lambda raw: FAMILY.get(raw, "other")

# 复刻 retest 数据管线(最小版:只算 nf_te 的 select_frames 先验 + 重训 GRU 报分布)
from collections import defaultdict
by_run = defaultdict(list)
for r in rows: by_run[r["run_id"]].append(r)
for rs in by_run.values(): rs.sort(key=lambda x: x["event_index"])
seq_of = {}
next_fam = {}
for rid, rs in by_run.items():
    seq_of[rid] = [fam(r["raw_action"]) if r["role"] == "execute" else "non_execute" for r in rs]
    for k, r in enumerate(rs):
        if k + 1 < len(rs):
            a = seq_of[rid][k + 1]
            next_fam[id(r)] = fam(rs[k + 1]["raw_action"]) if a != "non_execute" else None
        else:
            next_fam[id(r)] = None
te = [r for r in rows if r["split"] == "test"]
nf_te = [r for r in te if next_fam[id(r)]]
from collections import Counter
c = Counter(next_fam[id(r)] for r in nf_te)
tot = sum(c.values())
print("test next_fam 分布:", dict(c))
print("select_frames 先验:", round(c["select_frames"] / tot, 4), "(n=", tot, ")")
