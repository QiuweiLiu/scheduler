#!/usr/bin/env python3
"""Report/paper figures (data figures) from the frozen experiment artifacts.

Outputs SVG + PNG into outputs/report_figures/.
Run with the `print` conda env:
  MPLCONFIGDIR=/tmp/mpl-cache conda run -n print python scripts/make_report_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_materials" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_TAIL = ROOT / "experiments/EXP-20261004_main_table_comparison_v1/artifacts/main_table_tail_v1.json"
FCFS_TAIL = ROOT / "experiments/EXP-20261004_main_table_comparison_v1/artifacts/fcfs_tail_v1.json"
PDRS_V2 = ROOT / "experiments/EXP-20261005_pdrs_comparison_v1/artifacts/pdrs_comparison_v2.json"
PDRS_V3 = ROOT / "experiments/EXP-20261005_pdrs_comparison_v1/artifacts/pdrs_comparison_v3.json"
RES_V1 = ROOT / "experiments/EXP-20261005_residency_comparison_v1/artifacts/residency_comparison_v1.json"
PREEMPT = ROOT / "experiments/EXP-20261005_residency_comparison_v1/artifacts/residency_preempt_v1.json"

BLUE = "#2b5d8a"
TEAL = "#2f8f83"
RED = "#b3544a"
GRAY = "#6b6b6b"
AMBER = "#c98a2e"
LIGHT = "#dfe6ee"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(fig, name: str) -> None:
    svg = OUT / f"{name}.svg"
    png = OUT / f"{name}.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png.relative_to(ROOT), "+ svg")


def fig3_main_table_forest() -> None:
    mt = load(MAIN_TAIL)
    fcfs = load(FCFS_TAIL)
    rows = []
    for arm, r in mt["results"].items():
        e = r["metrics"]["mean_completion_ms"]
        rows.append((arm, e["delta_point"], e["delta_ci95"][0], e["delta_ci95"][1]))
    fcfs_e = fcfs["metrics"]["mean_completion_ms"]
    rows.append(("fcfs", fcfs_e["mean"] - mt["reference_summary"]["metrics"]["mean_completion_ms"]["mean"],
                 fcfs_e["mean"] - mt["reference_summary"]["metrics"]["mean_completion_ms"]["mean"] - 0, 0))
    # fcfs CI from paired run vs F0 reference episode values
    ref = mt["reference_summary"]["metrics"]["mean_completion_ms"]["episode_values"]
    deltas = [a - b for a, b in zip(fcfs_e["episode_values"], ref)]
    import random
    import statistics
    rng = random.Random(11)
    n = len(deltas)
    boot = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    fcfs_ci = (boot[50], boot[1949])
    rows[-1] = ("fcfs", statistics.fmean(deltas), fcfs_ci[0], fcfs_ci[1])
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ys = np.arange(len(rows))[::-1]
    for y, (arm, point, lo, hi) in zip(ys, rows):
        color = RED if lo > 0 else GRAY
        ax.plot([lo, hi], [y, y], color=color, lw=2.0, solid_capstyle="round")
        ax.plot([point], [y], "o", color=color, ms=7)
        ax.text(hi + 90, y, f"+{point:,.0f} ms", va="center", fontsize=9.5, color="#333333")
    ax.axvline(0, color=BLUE, lw=1.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_xlabel("Mean completion-time difference vs. F0 (ms)  —  positive = worse", fontsize=10)
    ax.set_title("Main table (frozen confirm300): F0 significantly beats all five adapted baselines\n"
                 "and FCFS on mean completion time (paired bootstrap 95% CI)",
                 fontsize=11.5)
    ax.text(0.01, 0.13,
            "p95 note: Parrot beats F0 by 3.15 s (significant); others tie or worse.\n"
            "Same action interface (original action space), frozen design.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.8, color=GRAY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    save(fig, "fig3_main_table_mean_forest")


def fig4_design_space() -> None:
    v2 = load(PDRS_V2)
    v3 = load(PDRS_V3)
    res = load(RES_V1)
    pre = load(PREEMPT)

    def delta(d, arm, metric):
        return d["results"][arm]["metrics"][metric]["delta_point"]

    rows = [
        ("PDRS ordering (mean functional)", delta(v2, "pdrs_p", "mean_completion_ms"),
         delta(v2, "pdrs_p", "p95_completion_ms")),
        ("PDRS ordering (p95 functional)", delta(v3, "pdrs_p95", "mean_completion_ms"),
         delta(v3, "pdrs_p95", "p95_completion_ms")),
        ("Oracle future (true identity)", delta(res, "pdrs_resident_oracle", "mean_completion_ms"),
         delta(res, "pdrs_resident_oracle", "p95_completion_ms")),
        ("Preemption (residency + SRPT)", delta(pre, "pdrs_preempt", "mean_completion_ms"),
         delta(pre, "pdrs_preempt", "p95_completion_ms")),
        ("Residency actions (main line)", delta(res, "pdrs_resident", "mean_completion_ms"),
         delta(res, "pdrs_resident", "p95_completion_ms")),
    ]
    values = np.array([[r[1], r[2]] for r in rows])
    vmax = float(np.abs(values).max())

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["mean completion", "p95 completion"], fontsize=11)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, f"{v:+,.0f}", ha="center", va="center", fontsize=10.5,
                    color="white" if abs(v) > vmax * 0.55 else "#1a1a1a")
    ax.set_title("Design-space pruning: richer prediction and destructive actions do not help\n"
                 "(Δ vs. F0 in ms; blue = better, red = worse)", fontsize=11.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Δ vs. F0 (ms)")
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    save(fig, "fig4_design_space_pruning")


def fig5_residency() -> None:
    res = load(RES_V1)
    ref = res["reference_summary"]["info"]["mechanism"]
    m = res["results"]["pdrs_resident"]["metrics"]
    mech = res["results"]["pdrs_resident"]["mechanism"]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    items = [("mean completion", "mean_completion_ms"),
             ("p95 completion", "p95_completion_ms"),
             ("deadline miss", "deadline_miss_rate"),
             ("makespan", "makespan_ms")]
    ys = np.arange(len(items))[::-1]
    for y, (label, key) in zip(ys, items):
        e = m[key]
        lo, hi = e["delta_ci95"]
        point = e["delta_point"]
        ax.plot([lo, hi], [y, y], color=BLUE, lw=2.2, solid_capstyle="round")
        ax.plot([point], [y], "o", color=BLUE, ms=7)
        if key == "deadline_miss_rate":
            ax.text(point - 250, y, f"{point:+.4f}", va="center", ha="right",
                    fontsize=9.5, color="#333")
        else:
            ax.text(hi + 60, y, f"{point:+,.0f}", va="center", fontsize=9.5, color="#333")
    ax.axvline(0, color=GRAY, lw=1.2)
    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items], fontsize=10.5)
    ax.set_xlabel("Δ vs. F0 (ms; miss rate in absolute fraction)", fontsize=9.5)
    ax.set_title("Residency actions vs. F0 (main line, pdrs_resident)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)

    ax2 = axes[1]
    ev = (mech["gpu_evictions"] - ref["gpu_evictions"]) / ref["gpu_evictions"] * 100
    cold = (mech["cold_loads_on_demand"] - ref["cold_loads_on_demand"]) / ref["cold_loads_on_demand"] * 100
    reload_ = (mech["evicted_then_reloaded"] - ref["evicted_then_reloaded"]) / ref["evicted_then_reloaded"] * 100
    hit = mech["used_prefetches"] / mech["prefetch_count"] * 100
    labels = ["evictions", "demand\ncold loads", "evict→reload\ncycles", "prefetch\nhit rate"]
    vals = [ev, cold, reload_, hit]
    colors = [TEAL, TEAL, TEAL, AMBER]
    bars = ax2.bar(np.arange(4), vals, color=colors, width=0.62)
    for i, (x, v) in enumerate(zip(np.arange(4), vals)):
        txt = f"{v:+.1f}%" if i < 3 else f"{v:.1f}%"
        ax2.text(x, v + (2 if v >= 0 else -5), txt, ha="center", fontsize=10.5)
    ax2.axhline(0, color=GRAY, lw=1.0)
    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylim(min(vals) * 1.25, max(vals) * 1.35)
    ax2.set_ylabel("relative change vs. F0 (%)", fontsize=9.5)
    ax2.set_title("Mechanism counters (300 episodes)", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(0.0, -0.30, "hit rate is absolute (F0 has no prefetch); other bars: relative change vs. F0",
             transform=ax2.transAxes, fontsize=8, color=GRAY)
    save(fig, "fig5_residency_result_and_mechanism")


def fig6_attribution() -> None:
    res = load(RES_V1)
    arms = [("f0point\n(point belief)", "f0point_resident"),
            ("pdrs\n(distributional)", "pdrs_resident"),
            ("shuffled\nbelief", "pdrs_resident_shuffle"),
            ("oracle\n(true identity)", "pdrs_resident_oracle")]
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    x = np.arange(len(arms))
    w = 0.36
    for offset, (metric, color, label) in zip(
        (-w / 2, w / 2),
        (("mean_completion_ms", BLUE, "mean"), ("p95_completion_ms", TEAL, "p95")),
    ):
        points, los, his = [], [], []
        for _, arm in arms:
            e = res["results"][arm]["metrics"][metric]
            points.append(e["delta_point"])
            los.append(e["delta_point"] - e["delta_ci95"][0])
            his.append(e["delta_ci95"][1] - e["delta_point"])
        ax.bar(x + offset, points, w, color=color, label=label,
               yerr=[los, his], capsize=3, error_kw={"lw": 1.2, "ecolor": "#444444"})
    ax.axhline(0, color=GRAY, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([a[0] for a in arms], fontsize=10)
    ax.set_ylabel("Δ vs. F0 (ms)", fontsize=10)
    ax.set_title("Prediction causality decomposition (residency actions in all arms)\n"
                 "mean: all beliefs win (shuffled included) — actions drive the mean;\n"
                 "p95: real beliefs win significantly, shuffled does not — instance-specificity shows in the tail",
                 fontsize=10.6)
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    save(fig, "fig6_attribution")


def fig7_tail_story() -> None:
    mt = load(MAIN_TAIL)
    res = load(RES_V1)
    pre = load(PREEMPT)
    rows = [
        ("Parrot App-FIFO (fair ordering)", mt["results"]["parrot_appfifo"]["metrics"]["p95_completion_ms"], GRAY),
        ("Residency actions (main line)", res["results"]["pdrs_resident"]["metrics"]["p95_completion_ms"], TEAL),
        ("Residency + f0point belief", res["results"]["f0point_resident"]["metrics"]["p95_completion_ms"], TEAL),
        ("Oracle future + actions", res["results"]["pdrs_resident_oracle"]["metrics"]["p95_completion_ms"], TEAL),
        ("No actions (pdrs_p, SRPT-ish ordering)", res["results"]["pdrs_p"]["metrics"]["p95_completion_ms"], RED),
        ("Preemption (residency + SRPT)", pre["results"]["pdrs_preempt"]["metrics"]["p95_completion_ms"], RED),
    ]
    rows.sort(key=lambda r: r[1]["delta_point"])
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ys = np.arange(len(rows))[::-1]
    for y, (label, e, color) in zip(ys, rows):
        lo, hi = e["delta_ci95"]
        ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="round")
        ax.plot([e["delta_point"]], [y], "o", color=color, ms=7)
        ax.text(e["delta_point"], y + 0.32, f"{e['delta_point']:+,.0f} ms", ha="center",
                fontsize=9, color="#333")
    ax.axvline(0, color=BLUE, lw=1.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_ylim(-0.75, len(rows) - 0.05)
    ax.set_xlabel("p95 completion-time difference vs. F0 (ms)  —  negative = better", fontsize=10)
    ax.set_title("Tail latency is governed by starvation/fairness, not only by prediction\n"
                 "(p95 Δ vs. F0; residency improves the tail without delaying victims)",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    save(fig, "fig7_tail_story")



if __name__ == "__main__":
    fig3_main_table_forest()
    fig4_design_space()
    fig5_residency()
    fig6_attribution()
    fig7_tail_story()
    print("done")
