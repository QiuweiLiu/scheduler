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
from matplotlib import font_manager  # noqa: E402

_CJK_FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
font_manager.fontManager.addfont(_CJK_FONT)
matplotlib.rcParams["font.sans-serif"] = ["Heiti TC", "Arial", "DejaVu Sans"]
matplotlib.rcParams["font.family"] = "sans-serif"
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
J_TEST = ROOT / "experiments/EXP-20260911_p9d_j_series_joint_resource/test_eval.json"
TRUTH_RANK = ROOT / "experiments/EXP-20260921_histres_causal_input_v1/artifacts/pack_vs_truth_ranking.json"

BLUE = "#2b5d8a"
TEAL = "#2f8f83"
RED = "#b3544a"
GRAY = "#6b6b6b"
AMBER = "#c98a2e"
LIGHT = "#dfe6ee"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(fig, name: str) -> None:
    name = name + "_zh"
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
    ax.set_xlabel("相对 F0 的平均完成时间差(ms)  —  正 = 更差", fontsize=10)
    ax.set_title("主表(frozen confirm300):F0 在平均完成时间上显著优于全部五条适配基线\n"
                 "与 FCFS(逐集配对 bootstrap 95% CI)",
                 fontsize=11.5)
    ax.text(0.01, 0.13,
            "p95 注:Parrot 比 F0 好 3.15 s(显著);其余持平或更差。\n"
            "同一动作接口(原动作空间),设计冻结。",
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
        ("PDRS 排序(均值口径)", delta(v2, "pdrs_p", "mean_completion_ms"),
         delta(v2, "pdrs_p", "p95_completion_ms")),
        ("PDRS 排序(p95 口径)", delta(v3, "pdrs_p95", "mean_completion_ms"),
         delta(v3, "pdrs_p95", "p95_completion_ms")),
        ("开天眼未来(真身份)", delta(res, "pdrs_resident_oracle", "mean_completion_ms"),
         delta(res, "pdrs_resident_oracle", "p95_completion_ms")),
        ("驻留包+抢占(SRPT 风格)", delta(pre, "pdrs_preempt", "mean_completion_ms"),
         delta(pre, "pdrs_preempt", "p95_completion_ms")),
        ("驻留动作(主线)", delta(res, "pdrs_resident", "mean_completion_ms"),
         delta(res, "pdrs_resident", "p95_completion_ms")),
    ]
    values = np.array([[r[1], r[2]] for r in rows])
    vmax = float(np.abs(values).max())

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["平均完成时间", "p95 完成时间"], fontsize=11)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, f"{v:+,.0f}", ha="center", va="center", fontsize=10.5,
                    color="white" if abs(v) > vmax * 0.55 else "#1a1a1a")
    ax.set_title("设计空间裁剪:更丰富的预测与破坏性动作都没有帮助\n"
                 "(Δ vs. F0,ms;蓝 = 更好,红 = 更差)", fontsize=11.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Δ vs. F0(ms)")
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
    items = [("平均完成时间", "mean_completion_ms"),
             ("p95 完成时间", "p95_completion_ms"),
             ("违约率", "deadline_miss_rate"),
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
    ax.set_xlabel("Δ vs. F0(ms;违约率为绝对比例)", fontsize=9.5)
    ax.set_title("驻留动作 vs. F0(主线 pdrs_resident)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)

    ax2 = axes[1]
    ev = (mech["gpu_evictions"] - ref["gpu_evictions"]) / ref["gpu_evictions"] * 100
    cold = (mech["cold_loads_on_demand"] - ref["cold_loads_on_demand"]) / ref["cold_loads_on_demand"] * 100
    reload_ = (mech["evicted_then_reloaded"] - ref["evicted_then_reloaded"]) / ref["evicted_then_reloaded"] * 100
    hit = mech["used_prefetches"] / mech["prefetch_count"] * 100
    labels = ["驱逐", "按需\n冷加载", "驱逐后\n重载", "预取\n命中率"]
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
    ax2.set_ylabel("相对 F0 变化(%)", fontsize=9.5)
    ax2.set_title("机制计数(300 集合计)", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(0.0, -0.30, "命中率为绝对值(F0 无预取);其余为相对 F0 变化",
             transform=ax2.transAxes, fontsize=8, color=GRAY)
    save(fig, "fig5_residency_result_and_mechanism")


def fig6_attribution() -> None:
    res = load(RES_V1)
    arms = [("f0point\n(点消费)", "f0point_resident"),
            ("pdrs\n(分布消费)", "pdrs_resident"),
            ("打乱\n信念", "pdrs_resident_shuffle"),
            ("oracle\n(真身份)", "pdrs_resident_oracle")]
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    x = np.arange(len(arms))
    w = 0.36
    for offset, (metric, color, label) in zip(
        (-w / 2, w / 2),
        (("mean_completion_ms", BLUE, "均值"), ("p95_completion_ms", TEAL, "p95")),
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
    ax.set_ylabel("Δ vs. F0(ms)", fontsize=10)
    ax.set_title("预测因果拆解(所有臂均含驻留动作)\n"
                 "均值:所有信念都赢(含打乱)→ 均值收益来自动作;\n"
                 "p95:真实信念显著赢,打乱不赢 → 实例特异性只在尾部",
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
        ("Parrot App-FIFO(公平排序)", mt["results"]["parrot_appfifo"]["metrics"]["p95_completion_ms"], GRAY),
        ("驻留动作(主线)", res["results"]["pdrs_resident"]["metrics"]["p95_completion_ms"], TEAL),
        ("驻留 + f0point 信念", res["results"]["f0point_resident"]["metrics"]["p95_completion_ms"], TEAL),
        ("开天眼 + 动作", res["results"]["pdrs_resident_oracle"]["metrics"]["p95_completion_ms"], TEAL),
        ("无动作(pdrs_p,SRPT 式排序)", res["results"]["pdrs_p"]["metrics"]["p95_completion_ms"], RED),
        ("驻留包+抢占(显式 SRPT)", pre["results"]["pdrs_preempt"]["metrics"]["p95_completion_ms"], RED),
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
    ax.set_xlabel("相对 F0 的 p95 完成时间差(ms)  —  负 = 更好", fontsize=10)
    ax.set_title("尾部由饥饿/公平主导,而非只由预测决定\n"
                 "(p95 Δ vs. F0;驻留动作改善尾部且不拖延 victim)",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    save(fig, "fig7_tail_story")


def fig13_predictor() -> None:
    jt = load(J_TEST)
    tr = load(TRUTH_RANK)
    seeds = ["11", "22", "33"]
    variants = [("J0", RED), ("J1", RED), ("J2", BLUE), ("J3", TEAL)]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})

    ax = axes[0]
    ys = np.arange(len(variants))[::-1]
    ymin, ymax = -0.55, 3.62
    ax.set_ylim(ymin, ymax)
    all_lo, all_hi = [], []
    for y, (v, color) in zip(ys, variants):
        for k, s in enumerate(seeds):
            e = jt["per_seed"][s][v]["runtime_delta"]
            off = (k - 1) * 0.21
            ax.plot([e["ci_lower"], e["ci_upper"]], [y + off, y + off],
                    color=color, lw=1.1, alpha=0.55, solid_capstyle="round")
            ax.plot([e["delta_mean"]], [y + off], "o", color=color, ms=3.2, alpha=0.85)
            all_lo.append(e["ci_lower"])
            all_hi.append(e["ci_upper"])
    xmin, xmax = min(all_lo), max(all_hi)
    ax.set_xlim(xmin - 18, xmax + 165)
    for y, (v, color) in zip(ys, variants):
        m = float(np.mean([jt["per_seed"][s][v]["runtime_delta"]["delta_mean"] for s in seeds]))
        ax.text(xmax + 155, y, f"{m:+.0f}", va="center", ha="right", fontsize=9.5, color="#333")
    ax.text(xmax + 155, ymax - 0.30, "均值 Δ", va="center", ha="right", fontsize=8.5, color=GRAY)
    ax.axvline(0, color=GRAY, lw=1.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([v for v, _ in variants], fontsize=10.5)
    ax.set_xlabel("相对参照 B1 的 RuntimeQScore 差(ms;负 = 更好)", fontsize=9.5)
    ax.set_title("资源头训练(test 仅消费一次):\n"
                 "J2/J3 在 3/3 seeds 上改善 runtime 预测;J0/J1 显著更差",
                 fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)

    ax2 = axes[1]
    packs = [("F0\n(部署)", "F0_seed11", AMBER),
             ("J3\n(前代)", "J3 (A0)", BLUE),
             ("R1b", "R1b (A1)", GRAY),
             ("R3a-U", "R3a-U (A2)", GRAY)]
    xs = np.arange(len(packs))
    vals = [tr["agreement_with_truth"][k]["spearman"] for _, k, _ in packs]
    ax2.bar(xs, vals, color=[c for _, _, c in packs], width=0.62)
    for x, v in zip(xs, vals):
        ax2.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=10.5)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([p for p, _, _ in packs], fontsize=9.5)
    ax2.set_ylim(0, 0.78)
    ax2.set_ylabel("与真值排序的 Spearman", fontsize=9.5)
    ax2.set_title("部署选型:真值参照排序质量\n"
                  "(7,663 锚点;F0 = 首个超过前代 J3 的候选包)", fontsize=10.5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color=LIGHT, lw=0.7)
    ax2.set_axisbelow(True)

    fig.text(0.012, -0.035,
             "注:J 系列 test 仅消费一次;J2/J3 因 load-duration 非劣失败(+3~+7%)未获 Core GO;"
             "遥测消融(F1−F0)CI 跨 0,未采纳;真值走查为固定空驻留代理,四包公平。",
             fontsize=8, color=GRAY)
    save(fig, "fig13_predictor_quality")


if __name__ == "__main__":
    fig3_main_table_forest()
    fig4_design_space()
    fig5_residency()
    fig6_attribution()
    fig7_tail_story()
    fig13_predictor()
    print("done")
