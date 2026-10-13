"""Closed-world full-plan replay (T10b-synth v2).

Builds a synthetic DES world containing ONLY the plan's nodes (anchors + mapped
chain steps, linear, aligned truth) with all GPUs free and the snapshot's
residents, then forces the plan's action sequence and compares the executed
order / placement / load presence / start / finish against the MILP plan.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import milp_v0_tests as T  # noqa: E402
from tracing.analysis.des_rollout import run_planning  # noqa: E402
from tracing.analysis.workload_v02_simulator import Node  # noqa: E402
from tracing.scheduling.milp_v0 import _build_nodes, solve_v0a  # noqa: E402


def synth_plan_world(obs, dump, nodes, node_map, ep, templates):
    """Build (episode, templates) containing only the plan's nodes."""

    vals = dump['values']
    idx = dump['index']
    origin = float(dump.get('time_origin', 0.0))  # relative time origin (advisor P1)
    plan_info = {}
    for i in dump['gpu_nodes']:
        node = nodes[i]
        for g in range(dump['gpu_count']):
            if vals[idx['x0'] + dump['gpu_nodes'].index(i) * dump['gpu_count'] + g] >= 0.5:
                plan_info[node.key] = {
                    'gpu': g, 'a': float(vals[idx['a0'] + i]) + origin,
                    'c': float(vals[idx['c0'] + i]) + origin,
                    'z': sum(vals[idx['z0'] + dump['gpu_nodes'].index(i) * dump['gpu_count'] + gg]
                             for gg in range(dump['gpu_count'])),
                }
    # group plan nodes by anchor (job)
    groups = {}
    for key, info in plan_info.items():
        anchor = key[5:].rsplit(':', 1)[0] if key.startswith('pred:') else key
        groups.setdefault(anchor, []).append(key)
    # map virtual keys -> real node ids; build one synthetic template per anchor
    synth_templates = {}
    jobs = []
    for anchor_key, keys in groups.items():
        keys = sorted(keys, key=lambda k: 0 if not k.startswith('pred:') else int(k.rsplit(':', 1)[1]))
        steps = obs.chains.get(anchor_key) or ()
        max_step = max((int(k.rsplit(':', 1)[1]) for k in keys if k.startswith('pred:')), default=0)
        owner_job = None
        for job in ep['jobs']:
            tpl = templates[str(job['template_id'])]
            if anchor_key in tpl.by_id:
                owner_job = job
                break
        if owner_job is None:
            continue
        base_tpl = templates[str(owner_job['template_id'])]
        new_nodes = []
        # anchor first
        anchor_node = base_tpl.by_id[anchor_key]
        anchor_item = next(r for r in obs.ready if r.node_id == anchor_key)
        new_nodes.append(replace(anchor_node, runtime_ms=anchor_item.runtime_p50_ms,
                                 load_ms=anchor_item.load_p50_ms,
                                 workspace_peak_mb=anchor_item.memory_p95_mb))
        # full chain positions 1..max_step, CPU steps PRESERVED (GPU->CPU->GPU)
        for k in range(1, max_step + 1):
            rid = node_map.get(f'pred:{anchor_key}:{k}')
            if rid is None:
                continue
            node = base_tpl.by_id[rid]
            step = steps[k - 1] if k - 1 < len(steps) else None
            if step is not None:
                node = replace(node, model_id=step.model_id, lane=step.lane,
                               runtime_ms=step.runtime_ms, load_ms=step.load_ms,
                               workspace_peak_mb=step.memory_mb,
                               resident_model_mb=obs.resident_mb_by_model.get(step.model_id))
            new_nodes.append(node)
        # linear rewire
        wired = []
        for pos, node in enumerate(new_nodes):
            preds = (new_nodes[pos - 1].node_id,) if pos > 0 else ()
            succs = (new_nodes[pos + 1].node_id,) if pos + 1 < len(new_nodes) else ()
            wired.append(replace(node, predecessors=preds, successors=succs))
        synth_templates['synth_' + anchor_key] = replace(
            base_tpl, template_id='synth_' + anchor_key, nodes=tuple(wired),
            by_id={n.node_id: n for n in wired})
        jobs.append({'job_instance_id': 'synth_' + anchor_key, 'template_id': 'synth_' + anchor_key,
                     'arrival_ms': 0.0, 'service_class': owner_job.get('service_class', 'normal')})
    injected = {
        'schema_version': 'des-state-v1', 'episode_id': 'synth_plan',
        'now_ms': float(obs.now_ms), 'sequence': 0, 'rr_cursor': 0,
        'jobs': [],  # keep build_jobs defaults (all nodes pending)

        'gpus': [{'index': g.index, 'capacity_mb': g.capacity_mb,
                  'resident': dict(g.resident), 'active_tasks': [], 'prefetch_pending': [],
                  'prefetched_models': [], 'used_prefetched_models': [], 'evicted_models_seen': [],
                  'evictions': 0, 'prefetch_count': 0, 'prefetch_load_ms': 0.0,
                  'wasted_prefetches': 0, 'cold_loads_on_demand': 0, 'evicted_then_reloaded': 0,
                  'peak_memory_mb': 0.0} for g in obs.gpus],
        'finish_entries': [], 'ready_items': [],
        'arrivals_pending': [[0.0, i] for i in range(len(jobs))],
        'active_jobs': [], 'wait_logged': [],
    }
    episode = {
        'episode_id': 'synth_plan', 'gpu_topology_mb': [g.capacity_mb for g in obs.gpus],
        'initial_residency_hint': [], 'jobs': jobs, 'split': 'train',
        'injected_state': injected,
    }
    all_templates = dict(templates)
    all_templates.update(synth_templates)
    return episode, all_templates, plan_info


def run_case(ep, jobs_meta, snap, templates, stats, artifacts, serial):
    obs = T.obs_from_snapshot(snap, jobs_meta, templates, stats, artifacts)
    if not [r for r in obs.ready if r.lane == 'gpu']:
        return None
    dump = {}
    res = solve_v0a(obs, time_limit_s=0.5, debug_dump=dump)
    if res.fallback:
        return {'fallback': True, 'reason': res.reason}
    vals = dump['values']
    idx = dump['index']
    nodes = _build_nodes(obs)
    # virtual -> real mapping (positional)
    snap_jobs = {str(r['job_instance_id']): r for r in snap.get('jobs') or []}
    node_map = {}
    for item in obs.ready:
        node_map[item.node_id] = item.node_id
        owner_job = None
        for job in ep['jobs']:
            tpl = templates[str(job['template_id'])]
            if item.node_id in tpl.by_id:
                owner_job = job
                break
        tpl = templates[str(owner_job['template_id'])]
        states_row = (snap_jobs.get(str(owner_job['job_instance_id'])) or {}).get('node_state') or {}
        anchor_pos = -1
        for pos, node in enumerate(tpl.nodes):
            if states_row.get(node.node_id) != 'pending':
                anchor_pos = pos
        if anchor_pos < 0:
            anchor_pos = 0
        pending = [n.node_id for n in tpl.nodes[anchor_pos + 1:] if states_row.get(n.node_id) == 'pending']
        for k in range(len(obs.chains.get(item.node_id, ()))):
            if k < len(pending):
                node_map[f'pred:{item.node_id}:{k + 1}'] = pending[k]
    synth_ep, synth_templates, plan_info = synth_plan_world(obs, dump, nodes, node_map, ep, templates)
    # action sequence in plan order (mapped)
    planned = sorted(((info['a'], key, info['gpu']) for key, info in plan_info.items()))
    seq = [{'node_key': node_map[key], 'gpu_index': g} for _a, key, g in planned if key in node_map]
    _s, events, ctx = run_planning(
        synth_ep, synth_templates, future_artifacts=artifacts, train_stats=stats,
        extension_config=serial, policy='des_rollout', forced_action=None,
        collect_events=True, extra_context={'des_seq': seq, 'des_seq_pos': 0})
    act = ctx.get('activation') or {}
    forced = int(act.get('des_seq_forced', 0))
    waits = int(act.get('des_seq_wait', 0))
    observed = [(str(e.get('node_id')), int(e.get('gpu_index'))) for e in events
                if e.get('event_type') == 'node_dispatch']
    seq_pairs = [(a['node_key'], a['gpu_index']) for a in seq]
    order_ok = observed == seq_pairs
    # timing check: executed start/finish vs the plan
    timing = []
    for key, g in seq_pairs:
        info = plan_info.get(key) or plan_info.get(next((k for k, v in node_map.items() if v == key), None))
        start = finish = None
        for e in events:
            if str(e.get('node_id')) == key and e.get('event_type') == 'node_start' and start is None:
                start = float(e.get('time_ms'))
            if str(e.get('node_id')) == key and e.get('event_type') == 'node_finish' and finish is None:
                finish = float(e.get('time_ms'))
        if info is None or start is None or finish is None:
            timing.append({'node': key[-20:], 'ok': False, 'reason': 'missing'})
            continue
        t0 = float(obs.now_ms)
        tol = 0.01
        start_ok = abs((start - t0) - (info['a'] - t0)) <= tol
        finish_ok = abs((finish - t0) - (info['c'] - t0)) <= tol
        dur_ok = abs((finish - start) - (info['c'] - info['a'])) <= tol
        ok = start_ok and finish_ok and dur_ok
        timing.append({'node': key[-20:], 'ok': ok, 'start_ok': start_ok, 'finish_ok': finish_ok,
                       'dur_ok': dur_ok, 'start': start, 'plan_a': info['a'],
                       'finish': finish, 'plan_c': info['c']})
    timing_ok = all(t.get('ok') for t in timing)
    passed = forced == len(seq) and order_ok and timing_ok and len(seq) > 0
    return {'pass': passed, 'plan_nodes': len(seq), 'forced': forced, 'waits': waits,
            'order_ok': order_ok, 'timing_ok': timing_ok, 'timing': timing}


def main():
    serial, templates, episodes, stats, artifacts = T.load_fixtures(3)
    states = T.capture_states(episodes, templates, stats, artifacts, serial, cap=3)
    ok = 0
    checked = 0
    for ep, jobs_meta, snap in states:
        r = run_case(ep, jobs_meta, snap, templates, stats, artifacts, serial)
        if r is None or r.get('fallback'):
            print(f"{ep['episode_id']}@{snap['now_ms']:.0f}: fallback {r and r.get('reason')}")
            continue
        checked += 1
        if r['pass']:
            ok += 1
        print(f"{ep['episode_id']}@{snap['now_ms']:.0f}: pass={r['pass']} nodes={r['plan_nodes']} "
              f"forced={r['forced']} waits={r['waits']} order={r['order_ok']} timing={r['timing_ok']}")
        for t in r['timing']:
            if not t.get('ok'):
                print('   timing fail:', t)
    print(f'closed-world replay: {ok}/{checked}')


if __name__ == '__main__':
    main()
