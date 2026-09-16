#!/usr/bin/env python3
"""Multimodal causal path predictor v0.3."""
from __future__ import annotations
import argparse,json,random
from collections import Counter
from pathlib import Path
from typing import Any
import torch
from torch import Tensor,nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tracing.analysis import neural_trace_predictor as n
from tracing.analysis import neural_multimodal_predictor_v02 as m
HORIZONS=m.HORIZONS
def seed_everything(seed:int)->None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
class Model(nn.Module):
    def __init__(self,artifacts:dict[str,Any],adj:Tensor,hidden:int=256):
        super().__init__()
        self.backbone=m.V2Predictor(artifacts,adj,"gru")
        self.labels=len(artifacts["label_vocab"]); self.bos=self.labels; self.hidden=hidden
        self.emb=nn.Embedding(self.labels+1,96); self.pos=nn.Embedding(len(HORIZONS),16)
        self.dec=nn.GRU(112,hidden,num_layers=2,dropout=.1,batch_first=True)
        self.init=nn.Sequential(nn.Linear(256,2*hidden),nn.Tanh()); self.head=nn.Linear(hidden,self.labels)
        self.idx={k:i for i,k in enumerate(m.NUM_KEYS)}
    def context(self,b:dict[str,Tensor])->Tensor:
        q=self.backbone
        sem=q.encode_sem(b["sem"],b["sem_lengths"])
        raw=m.encode_sequence({"embedding":q.raw_embedding,"encoder":q.raw_encoder},b["raw"],b["raw_lengths"])
        text=m.encode_sequence({"embedding":q.text_embedding,"encoder":q.text_encoder},b["text"],b["text_lengths"])
        cat=torch.cat([e(b["cats"][:,i]) for i,e in enumerate(q.cat_embeddings)],-1); x=b["numeric"]
        vis=x[:,[self.idx[k] for k in m.VISUAL_KEYS]]; task=x[:,[self.idx[k] for k in m.TASK_KEYS]]; res=x[:,[self.idx[k] for k in m.RESOURCE_KEYS]]
        return q.fusion(torch.cat([sem,raw,text,q.visual_branch(vis),q.task_branch(task),q.resource_branch(res),q.static_branch(cat),q.teacher_proj(b["teacher"])],-1))
    def init_state(self,c:Tensor)->Tensor:
        return self.init(c).view(-1,2,self.hidden).transpose(0,1).contiguous()
    def step(self,previous:Tensor,position:int,state:Tensor)->tuple[Tensor,Tensor]:
        p=self.pos.weight[position].unsqueeze(0).expand(previous.shape[0],-1)
        y,state=self.dec(torch.cat([self.emb(previous),p],-1).unsqueeze(1),state)
        return self.head(y[:,0]),state
    def teacher(self,b:dict[str,Tensor],ratio:float)->Tensor:
        c=self.context(b); state=self.init_state(c); previous=torch.full((b["sem"].shape[0],),self.bos,dtype=torch.long,device=c.device); out=[]
        for pos,_ in enumerate(HORIZONS):
            logits,state=self.step(previous,pos,state); out.append(logits); gold=b["targets"][:,pos]; valid=gold>=0
            use=(torch.rand(previous.shape[0],device=c.device)<ratio)&valid; previous=torch.where(use,gold,logits.argmax(-1))
        return torch.stack(out,1)
def tensors(batch:dict[str,Tensor],device:torch.device)->dict[str,Tensor]:
    keys=("sem","sem_lengths","raw","raw_lengths","text","text_lengths","cats","numeric","teacher","targets")
    return {k:batch[k].to(device) for k in keys}
def allowed(edges:dict[str,Counter[str]],previous:str,labels:list[str])->list[str]:
    values=[x for x in edges.get(previous,Counter()).keys() if x in labels]; return values or labels
def decode(model:Model,row:dict[str,Any],a:dict[str,Any],edges:dict[str,Counter[str]],labels:list[str],device:torch.device,width:int)->list[str]:
    b=m.collate([m.V2Dataset([row],a)[0]]); x=tensors(b,device); model.eval()
    with torch.no_grad(): state=model.init_state(model.context(x))
    lid={x:i for i,x in enumerate(labels)}; start=(row.get("prefix_activities") or [n.TOKEN_START])[-1]; beams=[(0.,[],model.bos,state)]
    for pos,_ in enumerate(HORIZONS):
        cand=[]
        for score,path,previous,hidden in beams:
            prev_activity=path[-1] if path else start
            with torch.no_grad():
                logits,next_state=model.step(torch.tensor([previous],device=device),pos,hidden); lp=torch.log_softmax(logits[0],-1)
            rank=sorted(allowed(edges,prev_activity,labels),key=lambda z:float(lp[lid[z]]),reverse=True)
            for label in rank[:max(width,1)]:
                idx=lid[label]; cand.append((score+float(lp[idx]),path+[label],idx,next_state.detach()))
        if not cand: break
        cand.sort(key=lambda z:(-z[0],z[1])); beams=cand[:max(width,1)]
    return beams[0][1] if beams else []
def evaluate(model:Model,rows:list[dict[str,Any]],a:dict[str,Any],edges:dict[str,Counter[str]],labels:list[str],device:torch.device,width:int)->dict[str,Any]:
    hit,count=Counter(),Counter()
    for row in rows:
        pred=decode(model,row,a,edges,labels,device,width)
        for h in (1,3,5):
            target=[row.get(f"target_h{i}") for i in range(1,h+1)]
            if all(v in a["label_vocab"] for v in target): count[str(h)]+=1; hit[str(h)]+=int(pred[:h]==target)
    return {"path":{f"prefix_hit@{h}":hit[str(h)]/max(count[str(h)],1) for h in (1,3,5)},"path_n":dict(count)}
def loss(model:Model,batch:dict[str,Tensor],device:torch.device,weight:Tensor,ratio:float)->Tensor:
    x=tensors(batch,device); logits=model.teacher(x,ratio); target=batch["targets"].to(device); ls=[]
    for p,h in enumerate(HORIZONS):
        mask=target[:,p]>=0
        if bool(mask.any()):
            ls.append({1:.75,2:.9,3:1.,4:1.1,5:1.2}[h]*F.cross_entropy(logits[mask,p],target[mask,p],weight=weight,label_smoothing=.02))
    return sum(ls)/max(len(ls),1)
def run_seed(args:argparse.Namespace,seed:int,rows:list[dict[str,Any]],a:dict[str,Any],device:torch.device)->dict[str,Any]:
    seed_everything(seed); tr=[r for r in rows if r.get("split")=="train"]; va=[r for r in rows if r.get("split")=="validation"]; te=[r for r in rows if r.get("split")=="test"]
    edges,_=n.transition_graph(tr); labels=[x for x,_ in sorted(a["label_vocab"].items(),key=lambda z:z[1])]; model=Model(a,m.build_graph(tr,a)).to(device)
    loader=DataLoader(m.V2Dataset(tr,a),batch_size=args.batch_size,shuffle=True,collate_fn=m.collate); weight=n.class_weights(tr,a["label_vocab"]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best=-1.; best_epoch=0; best_state=None; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.; ratio=max(args.min_teacher_forcing,args.teacher_forcing-(epoch-1)*args.teacher_forcing_decay)
        for batch in loader:
            opt.zero_grad(set_to_none=True); l=loss(model,batch,device,weight,ratio); l.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); total+=float(l.detach())
        val=evaluate(model,va,a,edges,labels,device,args.beam_width); score=args.h3_weight*val["path"]["prefix_hit@3"]+args.h5_weight*val["path"]["prefix_hit@5"]
        rec={"epoch":epoch,"loss":total/max(len(loader),1),"teacher_forcing":ratio,"validation":val,"selection_score":score}; history.append(rec); print(json.dumps({"seed":seed,**rec},ensure_ascii=False),flush=True)
        if score>best: best,best_epoch=score,epoch; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch-best_epoch>=args.patience: break
    if best_state is None: raise RuntimeError("no checkpoint")
    model.load_state_dict(best_state); val=evaluate(model,va,a,edges,labels,device,args.beam_width); test=evaluate(model,te,a,edges,labels,device,args.beam_width)
    out=Path(args.output_dir)/f"seed_{seed}"; out.mkdir(parents=True,exist_ok=True); torch.save({"schema_version":"neural-multimodal-seq2seq-v0.3","state_dict":best_state,"artifacts":a,"seed":seed},out/"checkpoint_best.pt")
    report={"schema_version":"neural-multimodal-seq2seq-report-v0.3","seed":seed,"input":args.input,"rows":{"train":len(tr),"validation":len(va),"test":len(te)},"best_epoch":best_epoch,"best_validation_selection_score":best,"validation":val,"test":test,"history":history,"beam_width":args.beam_width,"leakage_contract":{"future_events_excluded":True,"ground_truth_excluded":True,"video_id_used_as_feature":False,"remaining_steps_used_as_input":False,"teacher_fit_split":"train_only","graph_fit_split":"train_only"}}
    (out/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); return report
def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--input",default="results/processed/neural_prefix_dataset_v0_5_resource_multimodal/prefixes_teacher.jsonl"); p.add_argument("--output-dir",default="results/processed/neural_multimodal_seq2seq_v0_3")
    p.add_argument("--epochs",type=int,default=20); p.add_argument("--patience",type=int,default=5); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--lr",type=float,default=8e-4); p.add_argument("--teacher-forcing",type=float,default=.95); p.add_argument("--teacher-forcing-decay",type=float,default=.05); p.add_argument("--min-teacher-forcing",type=float,default=.35); p.add_argument("--beam-width",type=int,default=8); p.add_argument("--h3-weight",type=float,default=.45); p.add_argument("--h5-weight",type=float,default=.55); p.add_argument("--seeds",default="11"); p.add_argument("--device",default="auto")
    args=p.parse_args(); rows=m.read_jsonl(Path(args.input)); a=m.build_artifacts(rows); device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else args.device)
    reports=[run_seed(args,int(s),rows,a,device) for s in args.seeds.split(",") if s.strip()]; out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    summary={"schema_version":"neural-multimodal-seq2seq-summary-v0.3","input":args.input,"output_dir":args.output_dir,"device":str(device),"reports":[{"seed":r["seed"],"best_epoch":r["best_epoch"],"validation":r["validation"],"test":r["test"]} for r in reports],"config":vars(args)}
    (out/"metrics_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
