#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,json,math,random,re
from collections import Counter,defaultdict
from pathlib import Path
from typing import Any
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from torch import Tensor,nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader,Dataset
from tracing.analysis import neural_trace_predictor as n

HORIZONS=n.HORIZONS
SEM_START=n.TOKEN_START
SEM_UNK=n.TOKEN_UNK
RAW_PAD='__RAW_PAD__'; RAW_START='__RAW_START__'; RAW_UNK='__RAW_UNK__'
TEXT_PAD='__TEXT_PAD__'; TEXT_START='__TEXT_START__'; TEXT_UNK='__TEXT_UNK__'
CAT_KEYS=tuple(n.CAT_KEYS)+('raw_last_action','raw_prefix_tail2','raw_prefix_tail3','compute_last_model_id','compute_last_node_type','compute_last_event_type')
EXTRA_NUM_KEYS=('evidence_confidence_mean','visual_evidence_density','raw_action_count','raw_action_unique_count','question_word_count','option_text_chars_total','task_text_chars_total',
'compute_event_count','compute_api_call_count','compute_action_count','compute_success_rate','compute_oom_count','compute_retry_count','compute_peak_allocated_mb','compute_peak_reserved_mb','compute_runtime_ms','compute_load_ms','compute_queue_ms','compute_input_frame_count','compute_qwen_image_count','compute_yolo_batch','compute_model_switches')
NUM_KEYS=tuple(n.NUM_KEYS)+EXTRA_NUM_KEYS
VISUAL_KEYS=('coverage_ratio','frames_seen','observed_interval_count','ocr_chars','temporal_relation_count','object_count','modality_count','evidence_confidence_mean','visual_evidence_density')
TASK_KEYS=('question_chars','question_tokens','option_count','option_chars_mean','duration_s','fps','question_word_count','option_text_chars_total','task_text_chars_total')
RESOURCE_KEYS=('retry_count','error_count','api_wait_ms','local_runtime_ms','compute_event_count','compute_api_call_count','compute_action_count','compute_success_rate','compute_oom_count','compute_retry_count','compute_peak_allocated_mb','compute_peak_reserved_mb','compute_runtime_ms','compute_load_ms','compute_queue_ms','compute_input_frame_count','compute_qwen_image_count','compute_yolo_batch','compute_model_switches')

def seed_everything(seed:int)->None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def read_jsonl(path:Path)->list[dict[str,Any]]:
    with path.open(encoding='utf-8') as f: return [json.loads(line) for line in f if line.strip()]

def word_tokens(value:str)->list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?",value.lower())

def remaining_label(row:dict[str,Any])->int:
    value=row.get('remaining_steps')
    try: return min(max(int(value),0),8)
    except (TypeError,ValueError):
        values=[h for h in HORIZONS if row.get(f'target_h{h}')]
        return max(values,default=1)

def build_artifacts(rows:list[dict[str,Any]])->dict[str,Any]:
    train=[r for r in rows if r.get('split')=='train']
    base=n.build_vocab(rows)
    raw_values={RAW_START}; text_values={TEXT_START}
    cat_values={key:set() for key in CAT_KEYS}
    for row in train:
        raw_values.update(str(x) for x in (row.get('prefix_raw_actions') or []))
        txt=row.get('features',{}).get('text',{})
        if isinstance(txt,dict):
            text_values.update(word_tokens(str(txt.get('question',''))))
            for option in txt.get('options') or []: text_values.update(word_tokens(str(option)))
        categorical=row.get('features',{}).get('categorical',{})
        for key in CAT_KEYS: cat_values[key].add(str(categorical.get(key,'unknown') or 'unknown'))
    raw_vocab={RAW_PAD:0,RAW_START:1,RAW_UNK:2}
    for value in sorted(raw_values):
        if value not in raw_vocab: raw_vocab[value]=len(raw_vocab)
    text_vocab={TEXT_PAD:0,TEXT_START:1,TEXT_UNK:2}
    for value in sorted(text_values):
        if value not in text_vocab: text_vocab[value]=len(text_vocab)
    cat_vocabs={}
    for key in CAT_KEYS:
        values={'__UNK__'}|cat_values[key]; vocab={'__UNK__':0}
        for value in sorted(values):
            if value not in vocab: vocab[value]=len(vocab)
        cat_vocabs[key]=vocab
    numeric_stats={}
    for key in NUM_KEYS:
        values=[]
        for row in train:
            try: value=float(row.get('features',{}).get('numeric',{}).get(key,0.0) or 0.0)
            except (TypeError,ValueError): value=0.0
            values.append(value if math.isfinite(value) else 0.0)
        mean=sum(values)/max(len(values),1)
        variance=sum((x-mean)**2 for x in values)/max(len(values),1)
        numeric_stats[key]={'mean':mean,'std':math.sqrt(variance) or 1.0}
    return {'token_vocab':base['token_vocab'],'label_vocab':base['label_vocab'],'cat_vocabs':cat_vocabs,
            'numeric_stats':numeric_stats,'raw_vocab':raw_vocab,'text_vocab':text_vocab}

class V2Dataset(Dataset):
    def __init__(self,rows,artifacts):
        self.rows=rows; self.a=artifacts
    def __len__(self): return len(self.rows)
    def sid(self,v): return int(self.a['token_vocab'].get(v,self.a['token_vocab'].get(SEM_UNK,2)))
    def rid(self,v): return int(self.a['raw_vocab'].get(v,self.a['raw_vocab'][RAW_UNK]))
    def tid(self,v): return int(self.a['text_vocab'].get(v,self.a['text_vocab'][TEXT_UNK]))
    def __getitem__(self,index):
        row=self.rows[index]
        sem=[SEM_START]+[str(x) for x in (row.get('prefix_activities') or [])]
        raw=[RAW_START]+[str(x) for x in (row.get('prefix_raw_actions') or [])]
        text=row.get('features',{}).get('text',{})
        question=str(text.get('question','')) if isinstance(text,dict) else ''
        options=text.get('options') if isinstance(text,dict) else []
        words=[TEXT_START]+word_tokens(question+' '+' '.join(str(x) for x in (options or [])))
        cats=row.get('features',{}).get('categorical',{})
        cat_ids=[self.a['cat_vocabs'][key].get(str(cats.get(key,'unknown') or 'unknown'),0) for key in CAT_KEYS]
        nums=[]
        numeric=row.get('features',{}).get('numeric',{})
        for key in NUM_KEYS:
            try: value=float(numeric.get(key,0.0) or 0.0)
            except (TypeError,ValueError): value=0.0
            stats=self.a['numeric_stats'][key]
            nums.append((value-stats['mean'])/stats['std'])
        targets=[]
        for h in HORIZONS:
            target=row.get(f'target_h{h}')
            targets.append(self.a['label_vocab'].get(target,-1) if target else -1)
        teacher_vec=[0.0]*len(self.a['label_vocab'])
        teacher_labels=row.get('teacher_labels') or []
        teacher_probs=row.get('teacher_probs') or []
        for label,prob in zip(teacher_labels,teacher_probs):
            if label in self.a['label_vocab']:
                teacher_vec[self.a['label_vocab'][label]]=float(prob)
        return {'index':index,'sem':torch.tensor([self.sid(v) for v in sem],dtype=torch.long),
            'raw':torch.tensor([self.rid(v) for v in raw],dtype=torch.long),
            'text':torch.tensor([self.tid(v) for v in words],dtype=torch.long),
            'cats':torch.tensor(cat_ids,dtype=torch.long),'numeric':torch.tensor(nums,dtype=torch.float32),
            'teacher':torch.tensor(teacher_vec,dtype=torch.float32),
            'targets':torch.tensor(targets,dtype=torch.long),'remaining_length':remaining_label(row),
            'sem_length':len(sem),'raw_length':len(raw),'text_length':len(words)}

def collate(batch):
    def pad(key,length_key):
        max_len=max(x[length_key] for x in batch)
        out=torch.zeros((len(batch),max_len),dtype=torch.long)
        lengths=[]
        for i,item in enumerate(batch):
            value=item[key]; out[i,:value.numel()]=value; lengths.append(item[length_key])
        return out,torch.tensor(lengths,dtype=torch.long)
    sem,sem_lengths=pad('sem','sem_length'); raw,raw_lengths=pad('raw','raw_length'); text,text_lengths=pad('text','text_length')
    return {'index':torch.tensor([x['index'] for x in batch],dtype=torch.long),'sem':sem,'sem_lengths':sem_lengths,
            'raw':raw,'raw_lengths':raw_lengths,'text':text,'text_lengths':text_lengths,
            'cats':torch.stack([x['cats'] for x in batch]),'numeric':torch.stack([x['numeric'] for x in batch]),
            'teacher':torch.stack([x['teacher'] for x in batch]),
            'targets':torch.stack([x['targets'] for x in batch]),
            'remaining_length':torch.tensor([x['remaining_length'] for x in batch],dtype=torch.long)}

def build_graph(rows,artifacts):
    size=len(artifacts['token_vocab']); adj=torch.zeros((size,size),dtype=torch.float32)
    for row in rows:
        prev=(row.get('prefix_activities') or [SEM_START])[-1]; target=row.get('target_h1')
        if not target: continue
        i=artifacts['token_vocab'].get(prev,artifacts['token_vocab'][SEM_UNK]); j=artifacts['token_vocab'].get(target,artifacts['token_vocab'][SEM_UNK])
        adj[i,j]+=1
    adj+=torch.eye(size); return adj/adj.sum(dim=1,keepdim=True).clamp_min(1.0)

def encode_sequence(module,tokens,lengths):
    embedded=module['embedding'](tokens)
    packed=pack_padded_sequence(embedded,lengths.cpu(),batch_first=True,enforce_sorted=False)
    _,hidden=module['encoder'](packed)
    if isinstance(hidden,tuple): hidden=hidden[0]
    return hidden[-1]

class V2Predictor(nn.Module):
    def __init__(self,artifacts,graph_adj,kind):
        super().__init__()
        if kind not in {'gru','lstm','gnn'}: raise ValueError(kind)
        self.kind=kind
        token_count=len(artifacts['token_vocab'])
        self.sem_embedding=nn.Embedding(token_count,96,padding_idx=0)
        if kind=='gnn':
            self.node_embedding=nn.Embedding(token_count,128)
            self.graph_self=nn.Linear(128,128); self.graph_neigh=nn.Linear(128,128)
            self.register_buffer('graph_adj',graph_adj)
        else:
            RNN=nn.GRU if kind=='gru' else nn.LSTM
            self.sem_encoder=RNN(96,128,num_layers=2,dropout=0.1,batch_first=True)
        self.raw_embedding=nn.Embedding(len(artifacts['raw_vocab']),64,padding_idx=0)
        self.raw_encoder=nn.GRU(64,96,num_layers=1,batch_first=True)
        self.text_embedding=nn.Embedding(len(artifacts['text_vocab']),64,padding_idx=0)
        self.text_encoder=nn.GRU(64,96,num_layers=1,batch_first=True)
        self.cat_embeddings=nn.ModuleList(nn.Embedding(len(artifacts['cat_vocabs'][key]),12) for key in CAT_KEYS)
        self.visual_branch=nn.Sequential(nn.Linear(len(VISUAL_KEYS),64),nn.GELU(),nn.Dropout(0.1))
        self.task_branch=nn.Sequential(nn.Linear(len(TASK_KEYS),64),nn.GELU(),nn.Dropout(0.1))
        self.resource_branch=nn.Sequential(nn.Linear(len(RESOURCE_KEYS),64),nn.GELU(),nn.Dropout(0.1))
        self.static_branch=nn.Sequential(nn.Linear(12*len(CAT_KEYS),128),nn.GELU(),nn.Dropout(0.1))
        self.teacher_proj=nn.Sequential(nn.Linear(len(artifacts['label_vocab']),64),nn.GELU(),nn.Dropout(0.1))
        self.fusion=nn.Sequential(nn.Linear(128+96+96+64+64+64+128+64,256),nn.GELU(),nn.Dropout(0.1))
        labels=len(artifacts['label_vocab'])
        self.heads=nn.ModuleDict({str(h):nn.Linear(256,labels) for h in HORIZONS})
        self.length_head=nn.Linear(256,9)
        self.idx={key:i for i,key in enumerate(NUM_KEYS)}
    def encode_sem(self,tokens,lengths):
        if self.kind!='gnn':
            embedded=self.sem_embedding(tokens); packed=pack_padded_sequence(embedded,lengths.cpu(),batch_first=True,enforce_sorted=False)
            _,hidden=self.sem_encoder(packed)
            if isinstance(hidden,tuple): hidden=hidden[0]
            return hidden[-1]
        nodes=self.node_embedding.weight
        for _ in range(2):
            nodes=torch.tanh(self.graph_self(nodes)+self.graph_neigh(self.graph_adj@nodes))
        ids=tokens.clamp(0,nodes.shape[0]-1); states=nodes[ids]
        mask=(torch.arange(tokens.shape[1],device=tokens.device)[None,:]<lengths[:,None]).float().unsqueeze(-1)
        return (states*mask).sum(1)/lengths.float().clamp_min(1).unsqueeze(-1)
    def forward(self,sem,sem_lengths,raw,raw_lengths,text,text_lengths,cats,numeric,teacher):
        sem_state=self.encode_sem(sem,sem_lengths)
        raw_state=encode_sequence({'embedding':self.raw_embedding,'encoder':self.raw_encoder},raw,raw_lengths)
        text_state=encode_sequence({'embedding':self.text_embedding,'encoder':self.text_encoder},text,text_lengths)
        cat=torch.cat([emb(cats[:,i]) for i,emb in enumerate(self.cat_embeddings)],dim=-1)
        visual=numeric[:,[self.idx[k] for k in VISUAL_KEYS]]
        task=numeric[:,[self.idx[k] for k in TASK_KEYS]]
        resource=numeric[:,[self.idx[k] for k in RESOURCE_KEYS]]
        static=self.static_branch(cat)
        teacher_state=self.teacher_proj(teacher)
        fused=self.fusion(torch.cat([sem_state,raw_state,text_state,self.visual_branch(visual),self.task_branch(task),self.resource_branch(resource),static,teacher_state],dim=-1))
        return {str(h):self.heads[str(h)](fused) for h in HORIZONS}|{'length':self.length_head(fused)}

def beam_decode(start,log_probs,edges,labels,beam_size=5):
    label_to_idx={label:i for i,label in enumerate(labels)}
    paths=[(0.0,[],start)]
    for h in HORIZONS:
        next_paths=[]
        for score,path,previous in paths:
            allowed=list(edges.get(previous,Counter()).keys()) or labels
            for candidate in allowed:
                if candidate not in label_to_idx: continue
                next_paths.append((score+float(log_probs[h][label_to_idx[candidate]]),path+[candidate],candidate))
        next_paths.sort(key=lambda x:x[0],reverse=True); paths=next_paths[:64]
        if not paths: break
    paths.sort(key=lambda x:x[0],reverse=True)
    return [p for _,p,_ in paths[:beam_size]]

def autoregressive_decode(model,row,artifacts,edges,device):
    labels=[x for x,_ in sorted(artifacts['label_vocab'].items(),key=lambda z:z[1])]
    working=dict(row)
    working['prefix_activities']=list(row.get('prefix_activities') or [])
    working['prefix_raw_actions']=list(row.get('prefix_raw_actions') or [])
    features=row.get('features') or {}
    working['features']={'categorical':dict(features.get('categorical') or {}),
                         'numeric':dict(features.get('numeric') or {}),
                         'text':copy.deepcopy(features.get('text') or {})}
    predicted=[]
    for _ in HORIZONS:
        item=V2Dataset([working],artifacts)[0]
        batch=collate([item])
        tensors={k:batch[k].to(device) for k in ('sem','sem_lengths','raw','raw_lengths','text','text_lengths','cats','numeric','teacher')}
        with torch.no_grad():
            log_probs=torch.log_softmax(model(**tensors)['1'][0],dim=-1).cpu()
        previous=working['prefix_activities'][-1] if working['prefix_activities'] else SEM_START
        allowed=list(edges.get(previous,Counter()).keys()) or labels
        allowed=[x for x in allowed if x in artifacts['label_vocab']] or labels
        choice=max(allowed,key=lambda x:float(log_probs[artifacts['label_vocab'][x]]))
        predicted.append(choice)
        working['prefix_activities'].append(choice)
        working['prefix_raw_actions'].append(choice)
        working['features']['categorical']['last_activity']=choice
        working['features']['numeric']['position']=float(working['features']['numeric'].get('position',0.0))+1.0
        working['features']['numeric']['raw_action_count']=float(working['features']['numeric'].get('raw_action_count',0.0))+1.0
        if choice=='__END__': break
    return predicted

def evaluate(model,rows,artifacts,edges,device,batch_size,details_path=None):
    loader=DataLoader(V2Dataset(rows,artifacts),batch_size=batch_size,shuffle=False,collate_fn=collate)
    labels=[x for x,_ in sorted(artifacts['label_vocab'].items(),key=lambda z:z[1])]
    hits=Counter(); counts=Counter(); ar_hits=Counter(); ar_counts=Counter(); direct_hits=Counter(); direct_counts=Counter(); length_hits=0; details=[]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tensors={k:batch[k].to(device) for k in ('sem','sem_lengths','raw','raw_lengths','text','text_lengths','cats','numeric','teacher')}
            output=model(**tensors)
            for off,index in enumerate(batch['index'].tolist()):
                row=rows[int(index)]
                log_probs={h:torch.log_softmax(output[str(h)][off].detach().cpu(),dim=-1) for h in HORIZONS}
                start=(row.get('prefix_activities') or [SEM_START])[-1]
                paths=beam_decode(start,log_probs,edges,labels,5); beam_predicted=paths[0] if paths else []; predicted_ar=autoregressive_decode(model,row,artifacts,edges,device); predicted=beam_predicted
                for h in (1,3,5):
                    target=[row.get(f'target_h{i}') for i in range(1,h+1)]
                    if all(x in artifacts['label_vocab'] for x in target):
                        counts[str(h)]+=1; hits[str(h)]+=int(predicted[:h]==target); ar_counts[str(h)]+=1; ar_hits[str(h)]+=int(predicted_ar[:h]==target)
                for h in HORIZONS:
                    target=row.get(f'target_h{h}')
                    if target in artifacts['label_vocab']:
                        direct_counts[str(h)]+=1
                        direct_hits[str(h)]+=int(labels[int(output[str(h)][off].argmax())]==target)
                pred_len=int(output['length'][off].argmax().item()); true_len=remaining_label(row); length_hits+=int(pred_len==true_len)
                if details_path is not None:
                    top={}
                    for h in HORIZONS:
                        probs=torch.softmax(output[str(h)][off].detach().cpu(),dim=-1); vals,inds=probs.topk(min(3,probs.numel()))
                        top[str(h)]={'candidates':[{'activity':labels[int(i)],'probability':float(v)} for v,i in zip(vals,inds)],
                                     'entropy':float(-(probs*probs.clamp_min(1e-8).log()).sum()),
                                     'margin':float(vals[0]-vals[1]) if len(vals)>1 else float(vals[0])}
                    len_probs=torch.softmax(output['length'][off].detach().cpu(),dim=-1)
                    details.append({'run_id':row['run_id'],'position':row['position'],'prefix_activities':row.get('prefix_activities',[]),
                        'predicted_path':predicted,'path_candidates':paths,'autoregressive_path':predicted_ar,'predicted_remaining_steps':pred_len,
                        'true_remaining_steps_for_audit':true_len,'horizon_predictions':top,
                        'length_entropy':float(-(len_probs*len_probs.clamp_min(1e-8).log()).sum())})
    metrics={'path':{f'prefix_hit@{h}':hits[str(h)]/max(counts[str(h)],1) for h in (1,3,5)},
        'path_n':dict(counts),'autoregressive_path':{'path':{f'prefix_hit@{h}':ar_hits[str(h)]/max(ar_counts[str(h)],1) for h in (1,3,5)},'path_n':dict(ar_counts)},'direct_top1':{str(h):direct_hits[str(h)]/max(direct_counts[str(h)],1) for h in HORIZONS},
        'length_exact':length_hits/max(len(rows),1),'length_n':len(rows)}
    if details_path is not None:
        details_path.parent.mkdir(parents=True,exist_ok=True)
        details_path.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in details),encoding='utf-8')
    return metrics

def loss_for_batch(model,batch,device,class_weight):
    tensors={k:batch[k].to(device) for k in ('sem','sem_lengths','raw','raw_lengths','text','text_lengths','cats','numeric','teacher')}
    output=model(**tensors); losses=[]
    for h in HORIZONS:
        target=batch['targets'][:,h-1].to(device); mask=target>=0
        if bool(mask.any()):
            weight=1.0 if h==1 else (0.85 if h==2 else (0.7 if h==3 else 0.6))
            losses.append(weight*F.cross_entropy(output[str(h)][mask],target[mask],weight=class_weight,label_smoothing=0.03))
    length_target=batch['remaining_length'].to(device)
    losses.append(0.2*F.cross_entropy(output['length'],length_target.clamp(0,8)))
    return sum(losses)/max(len(losses),1)

def train_one(args,kind):
    rows=read_jsonl(Path(args.input)); artifacts=build_artifacts(rows)
    train=[r for r in rows if r.get('split')=='train']; val=[r for r in rows if r.get('split')=='validation']; test=[r for r in rows if r.get('split')=='test']
    edges,_=n.transition_graph(train); graph_adj=build_graph(train,artifacts)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else args.device)
    seed_everything(args.seed); model=V2Predictor(artifacts,graph_adj,kind).to(device)
    loader=DataLoader(V2Dataset(train,artifacts),batch_size=args.batch_size,shuffle=True,collate_fn=collate)
    weights=n.class_weights(train,artifacts['label_vocab']).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best_score=-float('inf'); best_epoch=0; best_state=None; history=[]
    out_dir=Path(args.output_dir)/kind; out_dir.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.0
        for batch in loader:
            opt.zero_grad(set_to_none=True); loss=loss_for_batch(model,batch,device,weights); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=float(loss.detach())
        val_metrics=evaluate(model,val,artifacts,edges,device,args.batch_size)
        score=0.45*val_metrics['path']['prefix_hit@3']+0.55*val_metrics['path']['prefix_hit@5']
        record={'epoch':epoch,'loss':total/max(len(loader),1),'validation':val_metrics,'selection_score':score}; history.append(record)
        print(json.dumps({'model':kind,'seed':args.seed,**record},ensure_ascii=False),flush=True)
        if score>best_score:
            best_score=score; best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch-best_epoch>=args.patience: break
    if best_state is None: raise RuntimeError('no checkpoint')
    model.load_state_dict(best_state)
    val_metrics=evaluate(model,val,artifacts,edges,device,args.batch_size)
    test_metrics=evaluate(model,test,artifacts,edges,device,args.batch_size,out_dir/'test_predictions.jsonl')
    report={'schema_version':'multimodal-predictor-v0.2','model':kind,'seed':args.seed,'input':args.input,'best_epoch':best_epoch,
        'best_validation_selection_score':best_score,'rows':{'train':len(train),'validation':len(val),'test':len(test)},
        'validation':val_metrics,'test':test_metrics,
        'feature_branches':{'semantic_prefix':True,'raw_action_prefix':True,'task_text':True,'structured_visual_evidence':list(VISUAL_KEYS),'task_numeric':list(TASK_KEYS),'resource_prefix':list(RESOURCE_KEYS),'teacher_prior':True},
        'leakage_contract':{'future_events_excluded':True,'ground_truth_excluded':True,'video_id_used_as_feature':False,
            'remaining_steps_used_as_input':False,'remaining_steps_used_as_supervision_only':True,'compute_events_strictly_before_target':True,'teacher_fit_split':'train_only'},
        'history':history}
    torch.save({'schema_version':'multimodal-predictor-checkpoint-v0.2','state_dict':best_state,'artifacts':artifacts,'model':kind,'seed':args.seed},out_dir/'checkpoint_best.pt')
    (out_dir/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'model':kind,'final':{'validation':val_metrics,'test':test_metrics}},ensure_ascii=False))
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='/root/autodl-tmp/scheduler/results/processed/neural_prefix_dataset_v0_2/prefixes.jsonl')
    parser.add_argument('--output-dir',default='/root/autodl-tmp/scheduler/results/processed/multimodal_predictor_v0_2')
    parser.add_argument('--model',choices=('gru','lstm','gnn','all'),default='all')
    parser.add_argument('--seed',type=int,default=11); parser.add_argument('--epochs',type=int,default=25)
    parser.add_argument('--patience',type=int,default=5); parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--lr',type=float,default=1e-3); parser.add_argument('--device',default='auto')
    args=parser.parse_args()
    kinds=('gru','lstm','gnn') if args.model=='all' else (args.model,)
    reports=[train_one(args,kind) for kind in kinds]
    summary={'schema_version':'multimodal-predictor-summary-v0.2','reports':reports,'config':vars(args)}
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    (Path(args.output_dir)/'metrics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
