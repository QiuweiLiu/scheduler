#!/usr/bin/env python3
"""Report result tables (CSV + Markdown) from the frozen experiment artifacts.

Outputs into outputs/report_materials/tables/.
Run: python3 scripts/make_report_tables.py
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_materials" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

MT = ROOT / "experiments/EXP-20261004_main_table_comparison_v1/artifacts/main_table_tail_v1.json"
FCFS = ROOT / "experiments/EXP-20261004_main_table_comparison_v1/artifacts/fcfs_tail_v1.json"
RES = ROOT / "experiments/EXP-20261005_residency_comparison_v1/artifacts/residency_comparison_v1.json"
RES95 = ROOT / "experiments/EXP-20261005_residency_comparison_v1/artifacts/residency_p95_v1.json"
PRE = ROOT / "experiments/EXP-20261005_residency_comparison_v1/artifacts/residency_preempt_v1.json"

METRICS = ("mean_completion_ms", "p95_completion_ms", "deadline_miss_rate", "makespan_ms")


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def ci(deltas, n=2000, seed=11):
    rng = random.Random(seed)
    m = len(deltas)
    means = sorted(sum(deltas[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def write(name: str, header: list[str], rows: list[list[str]], note: str = "") -> None:
    lines = []
    if note:
        lines.append(f"> {note}\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n")
    (OUT / f"{name}.csv").write_text(
        ",".join(header) + "\n" + "\n".join(",".join(c.replace(",", "") for c in r) for r in rows) + "\n"
    )
    print("wrote", name)


def main() -> None:
    mt = load(MT)
    fcfs = load(FCFS)
    ref = mt["reference_summary"]["metrics"]
    # ---- main table ----
    header = ["arm", "mean_delta_ms", "mean_ci_lo", "mean_ci_hi", "p95_delta_ms", "p95_ci_lo",
              "p95_ci_hi", "miss_delta", "makespan_delta_ms"]
    rows = []
    for arm, r in mt["results"].items():
        m = r["metrics"]
        rows.append([arm, f"{m['mean_completion_ms']['delta_point']:+.1f}",
                     f"{m['mean_completion_ms']['delta_ci95'][0]:+.1f}",
                     f"{m['mean_completion_ms']['delta_ci95'][1]:+.1f}",
                     f"{m['p95_completion_ms']['delta_point']:+.1f}",
                     f"{m['p95_completion_ms']['delta_ci95'][0]:+.1f}",
                     f"{m['p95_completion_ms']['delta_ci95'][1]:+.1f}",
                     f"{m['deadline_miss_rate']['delta_point']:+.4f}",
                     f"{m['makespan_ms']['delta_point']:+.1f}"])
    f0mean = ref["mean_completion_ms"]["mean"]
    fdeltas = [a - b for a, b in zip(fcfs["metrics"]["mean_completion_ms"]["episode_values"],
                                     mt["reference_summary"]["metrics"]["mean_completion_ms"]["episode_values"])]
    flo, fhi = ci(fdeltas)
    rows.append(["fcfs", f"{statistics.fmean(fdeltas):+.1f}", f"{flo:+.1f}", f"{fhi:+.1f}",
                 f"{fcfs['metrics']['p95_completion_ms']['mean'] - ref['p95_completion_ms']['mean']:+.1f}",
                 "", "", "", ""])
    write("main_table_vs_F0", header, rows,
          f"Frozen confirm300; delta vs F0 (F0 mean = {f0mean:.1f} ms). Positive = worse. "
          "F0 p95 = 130879.3 ms. Baselines run under their original action interfaces.")
    # ---- residency arms vs F0 ----
    rows = []
    for src, tag in ((load(RES), "v1"), (load(RES95), "p95"), (load(PRE), "preempt")):
        for arm, r in src["results"].items():
            m = r["metrics"]
            rows.append([arm, f"{m['mean_completion_ms']['delta_point']:+.1f}",
                         f"{m['mean_completion_ms']['delta_ci95'][0]:+.1f}",
                         f"{m['mean_completion_ms']['delta_ci95'][1]:+.1f}",
                         f"{m['p95_completion_ms']['delta_point']:+.1f}",
                         f"{m['p95_completion_ms']['delta_ci95'][0]:+.1f}",
                         f"{m['p95_completion_ms']['delta_ci95'][1]:+.1f}", tag])
    write("residency_arms_vs_F0",
          ["arm", "mean_delta_ms", "mean_ci_lo", "mean_ci_hi", "p95_delta_ms", "p95_ci_lo",
           "p95_ci_hi", "run"], rows,
          "All arms share the frozen F0 ordering; delta vs F0 (positive = worse). "
          "The main line is pdrs_resident.")
    # ---- mechanism ----
    res = load(RES)
    refm = res["reference_summary"]["info"]["mechanism"]
    rows = []
    for arm, r in res["results"].items():
        mm = r["mechanism"]
        rows.append([arm,
                     str(mm["gpu_evictions"]),
                     f"{(mm['gpu_evictions'] - refm['gpu_evictions']) / refm['gpu_evictions'] * 100:+.1f}%",
                     str(mm["cold_loads_on_demand"]),
                     f"{(mm['cold_loads_on_demand'] - refm['cold_loads_on_demand']) / refm['cold_loads_on_demand'] * 100:+.1f}%",
                     str(mm["evicted_then_reloaded"]),
                     f"{(mm['evicted_then_reloaded'] - refm['evicted_then_reloaded']) / refm['evicted_then_reloaded'] * 100:+.1f}%",
                     str(mm.get("prefetch_count", 0)), str(mm.get("used_prefetches", 0))])
    write("mechanism_counters",
          ["arm", "evictions", "evict_pct", "cold_loads", "cold_pct", "reload", "reload_pct",
           "prefetch", "prefetch_used"], rows,
          "300-episode totals; percentage vs the F0 reference (evictions 8940 / cold 9462 / reload 7672).")
    # ---- system level vs main line ----
    res_main = load(RES)["results"]["pdrs_resident"]["metrics"]
    header = ["baseline", "mean_delta_vs_main_ms", "mean_pct", "p95_delta_vs_main_ms", "p95_pct",
              "miss_delta", "makespan_delta_ms"]
    rows = []
    for arm, r in mt["results"].items():
        row = [arm]
        for metric in METRICS:
            a = r["metrics"][metric]["episode_values"]
            b = res_main[metric]["episode_values"]
            deltas = [x - y for x, y in zip(a, b)]
            p = statistics.fmean(deltas)
            base = statistics.fmean(b)
            if metric in ("mean_completion_ms", "p95_completion_ms"):
                row += [f"{p:+.1f}", f"{p / base * 100:+.2f}%"]
            elif metric == "deadline_miss_rate":
                row.append(f"{p:+.4f}")
            else:
                row.append(f"{p:+.1f}")
        rows.append(row)
    fc = fcfs["metrics"]
    row = ["fcfs"]
    for metric in METRICS:
        a = fc[metric]["episode_values"]
        b = res_main[metric]["episode_values"]
        deltas = [x - y for x, y in zip(a, b)]
        p = statistics.fmean(deltas)
        base = statistics.fmean(b)
        if metric in ("mean_completion_ms", "p95_completion_ms"):
            row += [f"{p:+.1f}", f"{p / base * 100:+.2f}%"]
        elif metric == "deadline_miss_rate":
            row.append(f"{p:+.4f}")
        else:
            row.append(f"{p:+.1f}")
    rows.append(row)
    write("system_level_vs_main_line", header, rows,
          "Paired per-episode comparison of each frozen baseline vs the main line "
          "(pdrs_resident = F0 + residency actions). Cross-interface: baselines do not carry "
          "the residency actions; use for end-to-end performance only, not mechanism attribution.")
    print("done")


if __name__ == "__main__":
    main()
