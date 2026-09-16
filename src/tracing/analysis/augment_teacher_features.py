#!/usr/bin/env python3
"""Attach train-only current-best predictor probabilities to neural prefix rows."""
from __future__ import annotations
import argparse, json, os
from collections import Counter
from pathlib import Path
from tracing.analysis import trace_predictor as tp

def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--raw-prefix", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--teacher-fit-prefix", default="", help="optional core-only prefix file used to fit the train-only teacher")
    args=ap.parse_args()
    rows=read_jsonl(Path(args.input))
    split_map=tp.load_split(Path(args.split_manifest))
    raw=tp.load_prefixes(Path(args.raw_prefix), split_map)
    raw_by={(r.run_id,r.position):r for r in raw}
    fit_raw=tp.load_prefixes(Path(args.teacher_fit_prefix), split_map) if args.teacher_fit_prefix else raw
    train_raw=[r for r in fit_raw if r.split=="train"]
    labels=sorted({r.target for r in train_raw})
    model=tp.PlannerAwareTracePredictor(labels).fit(train_raw)
    missing=[]; out=[]
    for row in rows:
        base=raw_by.get((row.get("run_id"), int(row.get("position",0))))
        if base is None:
            missing.append((row.get("run_id"),row.get("position"))); continue
        probs=model.predict(base)
        enriched=dict(row)
        enriched["teacher_probs"]=[float(probs.get(label,0.0)) for label in labels]
        enriched["teacher_labels"]=labels
        enriched["teacher_contract"]={"model":"planner_state_count_interpolation_v04","fit_split":"train","uses_target_or_future":False}
        out.append(enriched)
    if missing:
        raise RuntimeError(f"missing raw rows: {len(missing)} first={missing[:3]}")
    write_jsonl(Path(args.output),out)
    meta={"schema_version":"neural-prefix-teacher-v0.1","input":args.input,"output":args.output,"rows":len(out),"labels":labels,"split_counts":dict(Counter(r["split"] for r in out)),"raw_rows":len(raw),"teacher_fit_prefix":args.teacher_fit_prefix or args.raw_prefix,"raw_train_rows":len(train_raw),"missing":len(missing),"leakage_contract":{"teacher_fit_split":"train_only","target_fields_not_used_for_teacher_features":True,"test_video_used_only_for_prediction_input":True}}
    Path(args.metadata).parent.mkdir(parents=True,exist_ok=True)
    Path(args.metadata).write_text(json.dumps(meta,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
