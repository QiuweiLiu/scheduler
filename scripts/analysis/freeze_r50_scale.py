import json
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\scheduler")

# dev700 episode ids
dev_ids = {
    line.strip()
    for line in (ROOT / "data/manifests/validation_split_dev700_ids.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
}

episodes = []
for line in (ROOT / "results/processed/r7_workload_20260817/episodes/workload_validation_r7.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        row = json.loads(line)
        if str(row["episode_id"]) in dev_ids:
            episodes.append(row)

templates = {}
for line in (ROOT / "results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        row = json.loads(line)
        templates[str(row["template_id"])] = row

node_ids = set()
for episode in episodes:
    for job in episode.get("jobs") or []:
        template = templates.get(str(job.get("template_id")))
        if not template:
            continue
        for node in template.get("nodes") or []:
            node_ids.add(str(node.get("node_id")))

print("dev episodes:", len(episodes), "node ids referenced:", len(node_ids))

import gzip

p95_totals = []
p50_totals = []
ratios = []
path = ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts/b05_future_h5.jsonl.gz"
with gzip.open(path, "rt", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("node_id")) not in node_ids:
            continue
        steps = (row.get("future_h5") or [{}])[0].get("steps") or []
        s95 = 0.0
        s50 = 0.0
        for step in steps[:5]:
            q = (step.get("resource") or {}).get("runtime_ms_quantiles") or {}
            p95 = q.get("p95") or 0.0
            p50 = q.get("p50") or 0.0
            s95 += float(p95)
            s50 += float(p50)
        if s50 > 0.0:
            p95_totals.append(s95)
            p50_totals.append(s50)
            ratios.append(s95 / s50)

k_ratio_of_means = float(np.mean(p95_totals) / np.mean(p50_totals))
k_mean_of_ratios = float(np.mean(ratios))
print("nodes with positive p50:", len(p50_totals))
print("mean sum_p95:", round(float(np.mean(p95_totals)), 1))
print("mean sum_p50:", round(float(np.mean(p50_totals)), 1))
print("k (ratio of means):", round(k_ratio_of_means, 4))
print("k (mean of ratios):", round(k_mean_of_ratios, 4))
print("ratio percentiles:", {p: round(float(np.percentile(ratios, p)), 3) for p in (10, 25, 50, 75, 90)})

out = ROOT / "experiments/EXP-20260911_forecast_aware_scheduling/phase11_r50_scale.json"
out.write_text(
    json.dumps(
        {
            "frozen_on": "dev700 nodes (validation_split_dev700_ids.txt)",
            "nodes": len(p50_totals),
            "mean_sum_p95": float(np.mean(p95_totals)),
            "mean_sum_p50": float(np.mean(p50_totals)),
            "k_ratio_of_means": k_ratio_of_means,
            "k_mean_of_ratios": k_mean_of_ratios,
            "ratio_percentiles": {str(p): float(np.percentile(ratios, p)) for p in (10, 25, 50, 75, 90)},
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
    newline="\n",
)
print("frozen scale written:", out)
