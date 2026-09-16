"""Evaluate all 5 PPO seeds on S_val."""
import json, statistics, torch
from pathlib import Path
from torch import nn
from tracing.analysis.workload_v02_simulator import (load_future_artifacts, load_templates, read_jsonl, simulate_episode, train_resource_stats)

FEATURE_DIM = 14
class BcActor(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x)

class Critic(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()

import hashlib

def evaluate(actor, critic, episodes, templates, artifacts, train_stats, seed):
    rows = []
    actor.eval()
    with torch.no_grad():
        for ep in episodes:
            ctx = {'model': actor, 'mode': 'eval', 'trajectory': [], 'critic': critic}
            s, _ = simulate_episode(ep, templates, 'bc_h5', future_artifacts=artifacts, train_stats=train_stats, collect_events=False, rl_context=ctx)
            s = dict(s)
            s.update({'seed': seed, 'policy': 'ppo_h5', 'decision_count': len(ctx['trajectory'])})
            rows.append(s)
    return rows

templates = load_templates(Path('results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl'))
episodes = read_jsonl(Path('results/processed/r7_workload_20260817/episodes/workload_validation_r7.jsonl'))[:200]
artifacts = load_future_artifacts(Path('results/processed/r7_scheduling_future_20260817/prediction_artifacts'))
train_stats = train_resource_stats(templates)

for seed in [42, 43, 44, 45, 46]:
    actor = BcActor()
    ckpt = torch.load(f'results/processed/r8_bc/p4_ppo_seed{seed}_20260820/checkpoint.pt', weights_only=True, map_location='cpu')
    actor.load_state_dict(ckpt['actor_state_dict'])
    critic = Critic()
    critic.load_state_dict(ckpt['critic_state_dict'])
    rows = evaluate(actor, critic, episodes, templates, artifacts, train_stats, seed)
    outdir = Path(f'results/processed/r8_bc/p4_ppo_seed{seed}_20260820')
    (outdir / 'validation_results.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + '\n', encoding='utf-8')
    completions = [float(r['mean_completion_ms']) for r in rows if r.get('mean_completion_ms') is not None]
    failed = sum(int(r.get('failed_jobs') or 0) for r in rows)
    report = {'schema_version': 'r8-bc-ppo-v0.1', 'status': 'passed' if failed == 0 else 'failed', 'seed': seed, 'validation_episodes': len(rows), 'validation_mean_completion_ms': statistics.fmean(completions) if completions else None, 'validation_failed_jobs': failed, 'checkpoint_sha256': sha256(outdir / 'checkpoint.pt')}
    (outdir / 'ppo_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
