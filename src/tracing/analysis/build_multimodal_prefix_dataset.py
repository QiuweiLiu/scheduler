#!/usr/bin/env python3
import argparse,json,math,re
from collections import defaultdict
from pathlib import Path

def read_jsonl(path):
    rows=[]
    with path.open(encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            if line.strip():
                value=json.loads(line)
                if not isinstance(value,dict): raise ValueError(f'{path}:{n}: expected object')
                rows.append(value)
    return rows

def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')

def text(value,default='unknown'):
    if value is None: return default
    value=str(value).strip()
    return value if value else default

def number(value,default=0.0):
    try:
        value=float(value)
        return value if math.isfinite(value) else default
    except (TypeError,ValueError): return default

def qid_from_base_task(value):
    value=text(value,'')
    return value.rsplit('_',1)[-1] if '_' in value else ''

def parse_source_index(value):
    match=re.search(r':(?:action|run):(\d+)$',text(value,''))
    return int(match.group(1)) - 1 if match else None

def load_sources(paths):
    by_pair={}; by_video=defaultdict(list)
    for path in paths:
        if not path.exists(): continue
        for row in read_jsonl(path):
            vid=row.get('video_id') or row.get('videoID')
            item=dict(row); item.pop('answer',None); item.pop('answers',None)
            pair=(text(vid,''),text(row.get('question_id'),''))
            if pair[0]:
                by_pair[pair]=item; by_video[pair[0]].append(item)
    return by_pair,dict(by_video)

def source_for(raw,by_pair,by_video):
    vid=text(raw.get('video_id'),'')
    exact=by_pair.get((vid,qid_from_base_task(raw.get('base_task_id'))))
    if exact is not None: return exact
    candidates=by_video.get(vid,[])
    return candidates[0] if candidates else None

def load_compute(path):
    grouped=defaultdict(list)
    if path.exists():
        for row in read_jsonl(path):
            run_id=text(row.get('run_id'),'')
            if run_id: grouped[run_id].append(row)
    for run_id in grouped:
        grouped[run_id].sort(key=lambda x:(number(x.get('source_event_index'),number(x.get('event_index'))),number(x.get('event_index'))))
    return dict(grouped)

def aggregate_compute(events,cutoff):
    selected=[]
    for event in events:
        index=int(number(event.get('source_event_index'),number(event.get('event_index'))))
        if cutoff is None or index<cutoff: selected.append(event)
    selected.sort(key=lambda x:(number(x.get('source_event_index'),number(x.get('event_index'))),number(x.get('event_index'))))
    api=sum(text(e.get('event_type'),'')=='api_call' for e in selected)
    actions=sum(text(e.get('event_type'),'')=='action' for e in selected)
    success=sum(text(e.get('status'),'').lower() in {'success','ok','completed'} for e in selected)
    oom=sum('oom' in text(e.get('status'),'').lower() or 'out of memory' in text(e.get('status'),'').lower() for e in selected)
    retries=sum(e.get('retry_of') not in (None,'',False) for e in selected)
    peaks=[number(e.get('peak_allocated_mb')) for e in selected if e.get('peak_allocated_mb') is not None]
    reserved=[number(e.get('peak_reserved_mb')) for e in selected if e.get('peak_reserved_mb') is not None]
    runtime=[number(e.get('runtime_ms')) for e in selected if e.get('runtime_ms') is not None]
    loads=[number(e.get('load_ms')) for e in selected if e.get('load_ms') is not None]
    queues=[number(e.get('queue_ms')) for e in selected if e.get('queue_ms') is not None]
    frame_counts=[]; qwen_images=[]; yolo_batches=[]
    for e in selected:
        scale=e.get('input_scale')
        if isinstance(scale,dict):
            if scale.get('frame_count') is not None: frame_counts.append(number(scale.get('frame_count')))
            if scale.get('qwen_image_count') is not None: qwen_images.append(number(scale.get('qwen_image_count')))
            if scale.get('yolo_batch') is not None: yolo_batches.append(number(scale.get('yolo_batch')))
    model_switches=0; last_model=''
    for e in selected:
        model=text(e.get('model_id'),'')
        if model and last_model and model!=last_model: model_switches+=1
        if model: last_model=model
    numeric={
        'compute_event_count':float(len(selected)),'compute_api_call_count':float(api),
        'compute_action_count':float(actions),'compute_success_rate':float(success/len(selected)) if selected else 0.0,
        'compute_oom_count':float(oom),'compute_retry_count':float(retries),
        'compute_peak_allocated_mb':max(peaks,default=0.0),'compute_peak_reserved_mb':max(reserved,default=0.0),
        'compute_runtime_ms':sum(runtime),'compute_load_ms':sum(loads),'compute_queue_ms':sum(queues),
        'compute_input_frame_count':sum(frame_counts),'compute_qwen_image_count':sum(qwen_images),
        'compute_yolo_batch':max(yolo_batches,default=0.0),'compute_model_switches':float(model_switches)}
    categorical={'compute_last_model_id':last_model or 'none',
        'compute_last_node_type':text(selected[-1].get('node_type'),'none') if selected else 'none',
        'compute_last_event_type':text(selected[-1].get('event_type'),'none') if selected else 'none'}
    return numeric,categorical,len(selected)

def enrich(raw,source,events):
    row=dict(raw)
    state=raw.get('state_features') if isinstance(raw.get('state_features'),dict) else {}
    evidence=state.get('evidence') if isinstance(state.get('evidence'),dict) else {}
    video=state.get('video') if isinstance(state.get('video'),dict) else {}
    prefix=[text(x,'unknown') for x in (raw.get('prefix_activities') or [])]
    raw_prefix=[text(x,'unknown') for x in (raw.get('prefix_raw_actions') or [])]
    cutoff=parse_source_index(raw.get('target_source_event_id'))
    compute_numeric,compute_categorical,compute_count=aggregate_compute(events,cutoff)
    options=source.get('options') if source else []
    if isinstance(options,str): option_list=[x.strip() for x in options.splitlines() if x.strip()]
    elif isinstance(options,list): option_list=[text(x,'') for x in options if text(x,'')]
    else: option_list=[]
    task=raw.get('task_structure') if isinstance(raw.get('task_structure'),dict) else {}
    if not task and isinstance(state.get('task'),dict): task=state.get('task')
    question=text(source.get('question'),'') if source else ''
    # Preserve structured task context in the text stream without using the answer.
    task_meta=[]
    for key in ('question_type','temporal_scope','answer_type','domain','sub_category'):
        value=task.get(key)
        if value not in (None,'',[]): task_meta.append(f"{key}={text(value)}")
    if source:
        for key in ('task_type','official_task_type'):
            value=source.get(key)
            if value not in (None,'',[]): task_meta.append(f"{key}={text(value)}")
    if task_meta:
        question=('['+';'.join(task_meta)+'] '+question).strip()
    option_chars=sum(len(x) for x in option_list)
    numeric=dict((raw.get('features') or {}).get('numeric') or {})
    numeric.update({'question_chars':float(len(question)),
        'question_tokens':float(len(re.findall(r"[A-Za-z0-9']+",question))),
        'option_count':float(len(option_list)),
        'option_chars_mean':float(option_chars/max(len(option_list),1)),
        'duration_s':number(video.get('duration_s')),
        'fps':number(video.get('fps')),
        'evidence_confidence_mean':number(evidence.get('confidence_mean')),
        'visual_evidence_density':number(evidence.get('frames_seen'))/max(number(video.get('duration_s')),1.0),
        'raw_action_count':float(len(raw_prefix)),'raw_action_unique_count':float(len(set(raw_prefix))),
        'question_word_count':float(len(re.findall(r"[A-Za-z0-9']+",question))),
        'option_text_chars_total':float(option_chars),'task_text_chars_total':float(len(question)+option_chars),**compute_numeric})
    categorical=dict((raw.get('features') or {}).get('categorical') or {})
    task_categorical={}
    for key in ('question_type','temporal_scope','answer_type','domain','sub_category'):
        value=task.get(key)
        if value not in (None,'',[]): task_categorical[key]=text(value)
    modalities=task.get('required_modalities')
    if isinstance(modalities,list): task_categorical['required_modalities']='|'.join(text(x) for x in modalities)
    elif modalities not in (None,''): task_categorical['required_modalities']=text(modalities)
    categorical.update({**task_categorical,
        'raw_last_action':raw_prefix[-1] if raw_prefix else '__START__',
        'raw_prefix_tail2':'|'.join(raw_prefix[-2:]) if raw_prefix else '__START__',
        'raw_prefix_tail3':'|'.join(raw_prefix[-3:]) if raw_prefix else '__START__',**compute_categorical})
    row['features']={'categorical':categorical,'numeric':numeric,
        'text':{'question':question,'options':option_list,'answer_excluded':True},
        'evidence':{'coverage_ratio':number(evidence.get('coverage_ratio')),
            'frames_seen':number(evidence.get('frames_seen')),
            'observed_interval_count':number(evidence.get('observed_interval_count')),
            'modality_counts':evidence.get('modality_counts') if isinstance(evidence.get('modality_counts'),dict) else {},
            'object_counts':evidence.get('object_counts') if isinstance(evidence.get('object_counts'),dict) else {}},
        'compute_prefix':{'cutoff_source_event_index':cutoff,'included_event_count':compute_count,'future_events_excluded':True}}
    row.pop('answer',None); row.pop('answers',None)
    row['multimodal_enrichment_version']='v0.2'
    row['input_contract']={'question_and_options_allowed':True,'answer_text_used_as_feature':False,
        'future_events_excluded':True,'ground_truth_excluded':True,'remaining_steps_excluded':True,
        'remaining_runtime_excluded':True,'video_id_used_as_feature':False,
        'compute_events_cutoff_strictly_before_target':True}
    return row,{'source_joined':source is not None,'compute_events_included':compute_count,'cutoff':cutoff}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--report',required=True); parser.add_argument('--compute',required=True)
    parser.add_argument('--source',action='append',required=True)
    args=parser.parse_args()
    raw_rows=read_jsonl(Path(args.input))
    by_pair,by_video=load_sources([Path(x) for x in args.source])
    computes=load_compute(Path(args.compute)); output=[]; joined=0; compute_rows=0; answer_keys=0; target_excluded=0
    for raw in raw_rows:
        row,audit=enrich(raw,source_for(raw,by_pair,by_video),computes.get(text(raw.get('run_id'),''),[]))
        output.append(row); joined+=int(audit['source_joined']); compute_rows+=int(audit['compute_events_included']>0)
        answer_keys+=int('answer' in row or 'answers' in row); target_excluded+=int(audit['cutoff'] is not None)
    write_jsonl(Path(args.output),output)
    numeric_keys=sorted([key for key in (output[0].get('features',{}).get('numeric',{}) if output else {}) if key.startswith('compute_')])
    report={'schema_version':'multimodal-prefix-v0.2-report','input':args.input,'output':args.output,'rows':len(output),
        'source_rows_indexed':sum(len(v) for v in by_video.values()),'source_joined_rows':joined,
        'compute_joined_rows':compute_rows,'rows_with_target_cutoff':target_excluded,
        'leakage_audit':{'answer_keys_in_output':answer_keys,'future_events_included_in_input':False,
            'ground_truth_included_in_input':False,'remaining_steps_used_as_input':False,
            'remaining_runtime_used_as_input':False,'video_id_used_as_model_feature':False,
            'compute_events_strictly_before_target':True},
        'feature_groups':{'structured_visual_evidence':['coverage_ratio','frames_seen','observed_interval_count',
            'ocr_chars','temporal_relation_count','object_count','modality_count','evidence_confidence_mean','visual_evidence_density'],
            'task_text':['question','options','question_word_count','option_text_chars_total','task_text_chars_total'],
            'observed_action_history':['prefix_activities','prefix_raw_actions','raw_action_count','raw_action_unique_count'],
            'resource_prefix':numeric_keys}}
    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
