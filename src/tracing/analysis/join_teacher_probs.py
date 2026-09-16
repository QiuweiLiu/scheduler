#!/usr/bin/env python3
import argparse,json
from pathlib import Path

def read(path):
    with Path(path).open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]

def key(row):
    return (str(row.get('run_id','')),int(row.get('position',0)))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True);p.add_argument('--teacher',required=True);p.add_argument('--output',required=True);p.add_argument('--report',required=True)
    a=p.parse_args()
    rows=read(a.input); teachers=read(a.teacher)
    mapping={key(r):r for r in teachers}
    output=[]; matched=0; mismatched=0; bad=0
    for row in rows:
        item=dict(row); t=mapping.get(key(row))
        if t is None: item['teacher_probs']=None; item['teacher_labels']=None
        else:
            labels=t.get('teacher_labels'); probs=t.get('teacher_probs')
            if isinstance(labels,list) and isinstance(probs,list) and len(labels)==len(probs):
                item['teacher_labels']=labels; item['teacher_probs']=[float(x) for x in probs]; matched+=1
            else:
                item['teacher_probs']=None; item['teacher_labels']=None; bad+=1
        output.append(item)
        if 'answer' in item or 'answers' in item: mismatched+=1
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with Path(a.output).open('w',encoding='utf-8') as f:
        for row in output: f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    report={'schema_version':'multimodal-prefix-v0.2-teacher-join','input':a.input,'teacher':a.teacher,'output':a.output,
        'rows':len(output),'teacher_rows':len(teachers),'matched_rows':matched,'bad_teacher_rows':bad,
        'answer_keys_in_output':mismatched,'teacher_contract':'fit_train_only_and_prefix_only',
        'leakage_contract':{'teacher_fit_split':'train_only','future_events_excluded':True,
            'ground_truth_excluded':True,'remaining_steps_used_as_input':False,'teacher_used_as_input':True}}
    Path(a.report).parent.mkdir(parents=True,exist_ok=True)
    Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
