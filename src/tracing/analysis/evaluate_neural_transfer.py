#!/usr/bin/env python3
"""Fair comparison of public-transfer ensemble against the locked v0.4 predictor."""
from __future__ import annotations
import argparse, hashlib, json, math, random, sys
from collections import defaultdict
from pathlib import Path
import torch
sys.path.insert(0, "/root/autodl-tmp/scheduler")
from tracing.analysis import neural_trace_predictor as n
from tracing.analysis import trace_predictor as tp

def metrics(items, alpha):
    if not items:
        return {"n": 0, "top1": 0.0, "top3": 0.0, "mrr": 0.0, "nll": 0.0}
    labels = items[0]["labels"]; t1=t3=mrr=nll=0.0
    for item in items:
        p={label: alpha*item["neural"][j] + (1.0-alpha)*item["baseline"][label] for j,label in enumerate(labels)}
        ranked=sorted(p, key=lambda x:(-p[x], x))
        target=item["target"]
        rank=ranked.index(target)+1 if target in p else len(labels)+1
        t1 += rank == 1; t3 += rank <= 3; mrr += 1.0/max(rank,1); nll -= math.log(max(p.get(target,1e-12),1e-12))
    count=len(items)
    return {"n":count,"top1":t1/count,"top3":t3/count,"mrr":mrr/count,"nll":nll/count}

def bootstrap_delta(items, alpha, seed, rounds):
    by_video=defaultdict(list)
    for item in items: by_video[item["video_id"]].append(item)
    videos=sorted(by_video)
    rng=random.Random(seed); values=[]
    for _ in range(rounds):
        sampled=[video for _ in videos for video in [rng.choice(videos)]]
        sample=[item for video in sampled for item in by_video[video]]
        selected=metrics(sample,alpha); base=metrics(sample,0.0)
        values.append((selected["top1"]-base["top1"], selected["nll"]-base["nll"]))
    top1_values=sorted(x[0] for x in values)
    nll_values=sorted(x[1] for x in values)
    lo=int(0.025*len(values)); hi=int(0.975*len(values))-1
    return {
        "top1_delta": {"mean":sum(top1_values)/len(top1_values), "ci95": [top1_values[lo],top1_values[hi]]},
        "nll_delta": {"mean":sum(nll_values)/len(nll_values), "ci95": [nll_values[lo],nll_values[hi]]},
        "videos":len(videos), "rounds":rounds
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--raw-prefix", required=True)
    ap.add_argument("--split-manifest", required=True)
    args=ap.parse_args()
    root=Path("/root/autodl-tmp/scheduler")
    rows=n.read_jsonl(Path(args.input)); artifacts=n.build_vocab(rows)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cat_sizes=[len(artifacts["cat_vocabs"][key]) for key in n.CAT_KEYS]
    all_logits=[]
    for seed in (11,22,33):
        ckpt_path=Path(args.checkpoint_dir)/f"seed_{seed}"/"checkpoint_best.pt"
        payload=torch.load(ckpt_path,map_location="cpu")
        model=n.MultiHorizonGRU(len(artifacts["token_vocab"]),cat_sizes,len(n.NUM_KEYS),len(artifacts["label_vocab"]),int(artifacts.get("teacher_dim",0)),1.0)
        model.load_state_dict(payload["state_dict"]); model.to(device)
        all_logits.append(n.collect_logits(model,rows,artifacts,device,128))
        del model
    labels=[label for label,_ in sorted(artifacts["label_vocab"].items(),key=lambda pair:pair[1])]
    split_map=tp.load_split(Path(args.split_manifest))
    raw=tp.load_prefixes(Path(args.raw_prefix),split_map)
    raw_by={(row.run_id,row.position):row for row in raw}
    teacher=tp.PlannerAwareTracePredictor(labels).fit([row for row in raw if row.split=="train"])
    items=[]
    for index,row in enumerate(rows):
        raw_row=raw_by[(row["run_id"],int(row["position"]))]
        averaged=torch.stack([torch.log_softmax(seed_logits[index][1],-1) for seed_logits in all_logits]).logsumexp(0)-math.log(len(all_logits))
        items.append({"video_id":row["video_id"],"split":row["split"],"target":row["target_h1"],"labels":labels,"neural":[float(value) for value in torch.exp(averaged)],"baseline":teacher.predict(raw_row)})
    validation=[item for item in items if item["split"]=="validation"]
    test=[item for item in items if item["split"]=="test"]
    base_validation=metrics(validation,0.0)
    candidates=[]
    for step in range(21):
        alpha=step/20.0
        current=metrics(validation,alpha)
        if current["top1"]+1e-12 >= base_validation["top1"] and current["nll"] < base_validation["nll"]-1e-12:
            candidates.append((alpha,current))
    selected_alpha=candidates[0][0] if candidates else 0.0
    report={"schema_version":"neural-transfer-comparison-v0.1","input":args.input,"checkpoint_dir":args.checkpoint_dir,"device":str(device),"seeds":[11,22,33],"selection_rule":"smallest alpha on fixed grid {0,.05,...,1} whose validation Top-1 is not below teacher-only and whose validation NLL is strictly lower","selection":{"validation_teacher":base_validation,"eligible":[{"alpha":a,"metrics":m} for a,m in candidates],"selected_alpha":selected_alpha,"validation_selected":metrics(validation,selected_alpha)},"test":{"teacher_only":metrics(test,0.0),"neural_ensemble":metrics(test,1.0),"selected_gated_ensemble":metrics(test,selected_alpha)},"bootstrap_video":{"selected_vs_teacher":bootstrap_delta(test,selected_alpha,20260804,5000)},"leakage_contract":{"teacher_fit_split":"train_only","alpha_selection_split":"validation_only","test_used_for_selection":False,"video_bootstrap_group":"video_id"}}
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
