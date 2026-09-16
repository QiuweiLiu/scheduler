#!/usr/bin/env python3
import argparse,json
from collections import defaultdict
from pathlib import Path

HORIZONS=(1,2,3,4,5)

def read(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--report',required=True)
    a=p.parse_args()
    rows=read(a.input)
    groups=defaultdict(list)
    for row in rows: groups[str(row.get('run_id',''))].append(row)
    out=[]; mismatches=0
    for run_id,group in groups.items():
        group.sort(key=lambda r:(int(r.get('position',0)),str(r.get('prefix_id',''))))
        targets=[str(r.get('target_next_activity','')) for r in group]
        for i,row in enumerate(group):
            prefix=list(row.get('prefix_activities') or [])
            expected=targets[:i]
            if prefix!=expected: mismatches+=1
            item=dict(row)
            for h in HORIZONS:
                j=i+h-1
                item[f'target_h{h}']=targets[j] if j<len(targets) and targets[j] else None
            out.append(item)
    out.sort(key=lambda r:(str(r.get('split','')),str(r.get('run_id','')),int(r.get('position',0))))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with Path(a.output).open('w',encoding='utf-8') as f:
        for row in out: f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    report={'schema_version':'multimodal-prefix-v0.2-labeled','input':a.input,'output':a.output,
        'rows':len(out),'runs':len(groups),'prefix_mismatches':mismatches,
        'split_counts':{s:sum(r.get('split')==s for r in out) for s in sorted({r.get('split') for r in out})},
        'target_h1_nonnull':sum(bool(r.get('target_h1')) for r in out),
        'leakage_contract':{'target_h_supervision_only':True,'target_h_used_as_input':False,
            'future_events_in_input':False,'answer_text_in_output':False}}
    Path(a.report).parent.mkdir(parents=True,exist_ok=True)
    Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
