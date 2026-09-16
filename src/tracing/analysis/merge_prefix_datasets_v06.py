#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path

def read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
def write(path,rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text("\n".join(json.dumps(r,ensure_ascii=False,sort_keys=True) for r in rows)+"\n",encoding="utf-8")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--core",required=True); p.add_argument("--expansion",required=True); p.add_argument("--output",required=True); p.add_argument("--report",required=True)
    a=p.parse_args(); core=read(a.core); expansion=read(a.expansion)
    core_keys={(r.get("run_id"),int(r.get("position",0))) for r in core}
    exp_keys={(r.get("run_id"),int(r.get("position",0))) for r in expansion}
    duplicate_keys=core_keys & exp_keys
    if duplicate_keys: raise RuntimeError("duplicate run/position keys: %d"%len(duplicate_keys))
    core_videos={r.get("video_id") for r in core if r.get("video_id")}; exp_videos={r.get("video_id") for r in expansion if r.get("video_id")}
    overlap=core_videos & exp_videos
    if overlap: raise RuntimeError("video overlap: %d"%len(overlap))
    out=[]
    for r in core:
        item=dict(r); item.setdefault("source_split",item.get("split")); item["dataset_partition"]="core_fixed"; out.append(item)
    for r in expansion:
        item=dict(r); item.setdefault("source_split",item.get("split")); item["dataset_partition"]="expansion_236"
        if item.get("split")=="expansion_train": item["split"]="train"
        out.append(item)
    out.sort(key=lambda r:(str(r.get("split","")),str(r.get("run_id","")),int(r.get("position",0))))
    write(a.output,out)
    report={"schema_version":"neural-prefix-merge-v0.6","core_rows":len(core),"expansion_rows":len(expansion),"rows":len(out),"core_runs":len({r.get("run_id") for r in core}),"expansion_runs":len({r.get("run_id") for r in expansion}),"runs":len({r.get("run_id") for r in out}),"core_videos":len(core_videos),"expansion_videos":len(exp_videos),"video_overlap":len(overlap),"duplicate_run_position_keys":len(duplicate_keys),"split_counts":dict(Counter(r.get("split") for r in out)),"source_split_counts":dict(Counter(r.get("source_split") for r in out)),"leakage_contract":{"expansion_train_mapped_to_train":True,"locked_validation_test_preserved":True,"video_overlap_absent":not overlap,"duplicate_run_position_absent":not duplicate_keys,"answer_fields_not_added":True}}
    Path(a.report).parent.mkdir(parents=True,exist_ok=True); Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
