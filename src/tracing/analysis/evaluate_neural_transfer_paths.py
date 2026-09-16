#!/usr/bin/env python3
"""Evaluate autoregressive H=3/5 paths for the locked transfer ensemble."""
from __future__ import annotations
import argparse, copy, json, math, sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
import torch
sys.path.insert(0, "/root/autodl-tmp/scheduler")
from tracing.analysis import neural_trace_predictor as n
from tracing.analysis import trace_predictor as tp

def path_metrics(rows, predictor, alpha):
    labels=predictor["labels"]; models=predictor["models"]; artifacts=predictor["artifacts"]; device=predictor["device"]; edges=predictor["edges"]; baseline=predictor["baseline"]; raw_by=predictor["raw_by"]
    hits=Counter(); counts=Counter()
    for row in rows:
        raw_work=raw_by[(row["run_id"],int(row["position"]))]
        neural_work={**row,"prefix_activities":list(row["prefix_activities"]),"features":{"categorical":dict(row["features"]["categorical"]),"numeric":dict(row["features"]["numeric"]),"teacher_probs":list(row.get("teacher_probs") or [])}}
        predicted=[]
        for _ in n.HORIZONS:
            item=n.TraceDataset([neural_work],artifacts)[0]; batch=n.collate([item])
            tensors={key:value.to(device) for key,value in batch.items() if key in {"tokens","lengths","cats","numeric","teacher"}}
            logps=[]
            for model in models:
                with torch.no_grad():
                    logps.append(torch.log_softmax(model(**tensors)["1"][0],-1))
            neural_prob=torch.stack(logps).logsumexp(0).sub(math.log(len(logps))).exp().cpu()
            base_prob=baseline.predict(raw_work)
            previous=raw_work.prefix[-1] if raw_work.prefix else n.TOKEN_START
            allowed=list(edges.get(previous,Counter()).keys()) or labels
            allowed=[candidate for candidate in allowed if candidate in labels] or labels
            choice=max(allowed,key=lambda candidate: alpha*float(neural_prob[artifacts["label_vocab"][candidate]])+(1-alpha)*base_prob[candidate])
            predicted.append(choice)
            sf=copy.deepcopy(dict(raw_work.state_features)); px=copy.deepcopy(dict(sf.get("prefix") or {})); actions=list(px.get("last_actions") or list(raw_work.prefix)); px["last_actions"]=(actions+[choice])[-3:]; sf["prefix"]=px
            raw_work=replace(raw_work,prefix=tuple(list(raw_work.prefix)+[choice]),raw_prefix=tuple(list(raw_work.raw_prefix)+[choice]),position=raw_work.position+1,state_features=sf)
            neural_work["prefix_activities"].append(choice); neural_work["features"]["categorical"]["last_activity"]=choice; neural_work["features"]["numeric"]["position"]=float(neural_work["features"]["numeric"].get("position",0.0))+1.0; neural_work["teacher_probs"]=[base_prob[label] for label in labels]
        for k in (1,3,5):
            targets=[row.get(f"target_h{h}") for h in range(1,k+1)]
            if all(target in labels for target in targets):
                counts[str(k)]+=1; hits[str(k)]+=predicted[:k]==targets
    return {"path":{f"prefix_hit@{k}":hits[str(k)]/max(counts[str(k)],1) for k in (1,3,5)},"path_n":dict(counts)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True); ap.add_argument("--output",required=True); ap.add_argument("--checkpoint-dir",required=True); ap.add_argument("--raw-prefix",required=True); ap.add_argument("--split-manifest",required=True); ap.add_argument("--alpha",type=float,required=True)
    args=ap.parse_args(); root=Path("/root/autodl-tmp/scheduler"); rows=n.read_jsonl(Path(args.input)); artifacts=n.build_vocab(rows); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); cats=[len(artifacts["cat_vocabs"][k]) for k in n.CAT_KEYS]; models=[]
    for seed in (11,22,33):
        payload=torch.load(Path(args.checkpoint_dir)/f"seed_{seed}"/"checkpoint_best.pt",map_location="cpu"); model=n.MultiHorizonGRU(len(artifacts["token_vocab"]),cats,len(n.NUM_KEYS),len(artifacts["label_vocab"]),int(artifacts.get("teacher_dim",0)),1.0); model.load_state_dict(payload["state_dict"]); model.to(device); model.eval(); models.append(model)
    labels=[label for label,_ in sorted(artifacts["label_vocab"].items(),key=lambda pair:pair[1])]; split_map=tp.load_split(Path(args.split_manifest)); raw=tp.load_prefixes(Path(args.raw_prefix),split_map); raw_by={(r.run_id,r.position):r for r in raw}; baseline=tp.PlannerAwareTracePredictor(labels).fit([r for r in raw if r.split=="train"]); edges,_=n.transition_graph([r for r in rows if r["split"]=="train"])
    predictor={"labels":labels,"models":models,"artifacts":artifacts,"device":device,"edges":edges,"baseline":baseline,"raw_by":raw_by}
    report={"schema_version":"neural-transfer-path-comparison-v0.1","alpha":args.alpha,"selection_split":"validation","test_used_for_alpha":False,"device":str(device),"seeds":[11,22,33],"validation":{},"test":{}}
    for split in ("validation","test"):
        subset=[row for row in rows if row["split"]==split]; report[split]={"selected_gated_ensemble":path_metrics(subset,predictor,args.alpha),"neural_ensemble":path_metrics(subset,predictor,1.0),"teacher_only":path_metrics(subset,predictor,0.0)}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
