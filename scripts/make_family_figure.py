#!/usr/bin/env python3
"""fig16: the main line vs three comparison families (delta vs F0, confirm300).

  A  basic baselines (no future information, no complex actions)
  B  future-information methods (belief / ordering controls)
  C  complex-action variants (residency / preemption ladder)

Outputs SVG + PNG (zh + en) into outputs/report_materials/figures/.
Run with the `print` conda env:
  MPLCONFIGDIR=/tmp/mpl-cache conda run -n print python scripts/make_family_figure.py
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402

_CJK_FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
font_manager.fontManager.addfont(_CJK_FONT)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_materials" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MT = ROOT / "experiments/EXP-20261004_main_table_comparison_v1/artifacts/main_table_tail_v1.json"
FCFS = ROOT / "experiments/EXP-20261004_main_table_comparison_v1/artifacts/fcfs_tail_v1.json"
RES = ROOT / "experiments/EXP-20261005_residency_comparison_v1/artifacts/residency_comparison_v1.json"
PRE = ROOT / "experiments/EXP-20261005_residency_comparison_v1/artifacts/residency_preempt_v1.json"
PDRS2 = ROOT / "experiments/EXP-20261005_pdrs_comparison_v1/artifacts/pdrs_comparison_v2.json"
PDRS3 = ROOT / "experiments/EXP-20261005_pdrs_comparison_v1/artifacts/pdrs_comparison_v3.json"

GRAY, TEAL, AMBER, BLUE = "#8a97a6", "#2f8f83", "#c98a2e", "#2b5d8a"
LIGHT = "#e8eaed"
TXT = "#2b2f33"
GRAYTXT = "#6b6b6b"
GROUP_COLOR = {"A": GRAY, "B": TEAL, "C": AMBER}

T = {
    "zh": {
        "suptitle": "调度器总对比:主线(pdrs_resident)vs 基础基线 / 未来信息 / 复杂动作(Δ vs F0)",
        "panels": ["平均完成时间", "p95 完成时间"],
        "xlabel": "Δ vs F0(ms,负 = 更好)",
        "headers": {"A": "基础基线(无未来信息 / 无复杂动作)",
                     "B": "未来信息类(对照:信息质量与排序)",
                     "C": "复杂动作类(对照:驻留 / 抢占取舍)"},
        "main_tag": "主线",
        "rows": {
            "fcfs": "FCFS", "torpor": "Torpor(生命周期)", "qlm": "QLM(队列)",
            "parrot": "Parrot(App-FIFO)", "llmsched": "LLMSched", "hermes": "Hermes(Gittins)",
            "pdrs_p": "PDRS 排序(均值口径,无动作)", "pdrs_p95": "PDRS 排序(p95 口径,无动作)",
            "shuffle": "打乱信念 + 驻留", "f0point": "点信念(f0point)+ 驻留",
            "oracle": "开天眼(真值)+ 驻留", "f0": "F0 底座(无驻留动作)",
            "evict": "仅驱逐(无预取)", "main": "主线:分布信念 + 驱逐 + 预取",
            "preempt": "+ 抢占(SRPT)",
        },
        "foot": ("参照 F0(同 run 逐位复现):mean 69,788.9 / p95 130,879.3 ms;"
                 "CI = 300 集配对 bootstrap 95%(confirm300 冻结集)。"
                 "组色:灰 = 基础基线,青 = 未来信息类,琥珀 = 复杂动作类,蓝 = 主线。"
                 "Myopic(补充参考,仅均值)+1,491.4 [+945.4, +2,102.1] ms 未画入。"),
    },
    "en": {
        "suptitle": "Scheduler comparison — main line (pdrs_resident) vs basic / future-info / complex-action families (Δ vs F0)",
        "panels": ["Mean completion", "p95 completion"],
        "xlabel": "Δ vs F0 (ms, negative = better)",
        "headers": {"A": "Basic baselines (no future info / no complex actions)",
                     "B": "Future-information methods (belief / ordering controls)",
                     "C": "Complex-action variants (residency / preemption)"},
        "main_tag": "main line",
        "rows": {
            "fcfs": "FCFS", "torpor": "Torpor (lifecycle)", "qlm": "QLM (queue)",
            "parrot": "Parrot (App-FIFO)", "llmsched": "LLMSched", "hermes": "Hermes (Gittins)",
            "pdrs_p": "PDRS ordering (mean functional)", "pdrs_p95": "PDRS ordering (p95 functional)",
            "shuffle": "Shuffled belief + residency", "f0point": "Point belief (f0point) + residency",
            "oracle": "Oracle (true identity) + residency", "f0": "F0 base (no residency actions)",
            "evict": "Eviction only (no prefetch)", "main": "Main line: distributional belief + eviction + prefetch",
            "preempt": "+ Preemption (SRPT)",
        },
        "foot": ("Reference F0 (bit-identical across runs): mean 69,788.9 / p95 130,879.3 ms; "
                 "CIs = paired bootstrap 95% over 300 episodes (frozen confirm300). "
                 "Colors: gray = basic baselines, teal = future-info, amber = complex actions, blue = main line. "
                 "Myopic (supplementary, mean only) +1,491.4 [+945.4, +2,102.1] ms omitted."),
    },
}


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def build_rows():
    mt, fcfs = load(MT), load(FCFS)
    res, pre = load(RES), load(PRE)
    v2, v3 = load(PDRS2), load(PDRS3)
    ref = mt["reference_summary"]["metrics"]

    def delta(src, arm, metric):
        m = src["results"][arm]["metrics"][metric]
        return float(m["delta_point"]), tuple(float(x) for x in m["delta_ci95"])

    rng = random.Random(11)

    def boot_ci(deltas, B=2000):
        n = len(deltas)
        means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
        return means[50], means[1949]

    fcfs_stats = {}
    for metric in ("mean_completion_ms", "p95_completion_ms"):
        dd = [a - b for a, b in zip(fcfs["metrics"][metric]["episode_values"],
                                    ref[metric]["episode_values"])]
        fcfs_stats[metric] = (statistics.fmean(dd), boot_ci(dd))

    main_mean = delta(res, "pdrs_resident", "mean_completion_ms")
    main_p95 = delta(res, "pdrs_resident", "p95_completion_ms")
    return [
        ("A", "fcfs", fcfs_stats["mean_completion_ms"], fcfs_stats["p95_completion_ms"]),
        ("A", "torpor", delta(mt, "torpor_lifecycle", "mean_completion_ms"),
         delta(mt, "torpor_lifecycle", "p95_completion_ms")),
        ("A", "qlm", delta(mt, "qlm_queue", "mean_completion_ms"),
         delta(mt, "qlm_queue", "p95_completion_ms")),
        ("A", "parrot", delta(mt, "parrot_appfifo", "mean_completion_ms"),
         delta(mt, "parrot_appfifo", "p95_completion_ms")),
        ("A", "llmsched", delta(mt, "llmsched", "mean_completion_ms"),
         delta(mt, "llmsched", "p95_completion_ms")),
        ("A", "hermes", delta(mt, "hermes_gittins", "mean_completion_ms"),
         delta(mt, "hermes_gittins", "p95_completion_ms")),
        ("B", "pdrs_p", delta(v2, "pdrs_p", "mean_completion_ms"),
         delta(v2, "pdrs_p", "p95_completion_ms")),
        ("B", "pdrs_p95", delta(v3, "pdrs_p95", "mean_completion_ms"),
         delta(v3, "pdrs_p95", "p95_completion_ms")),
        ("B", "shuffle", delta(res, "pdrs_resident_shuffle", "mean_completion_ms"),
         delta(res, "pdrs_resident_shuffle", "p95_completion_ms")),
        ("B", "f0point", delta(res, "f0point_resident", "mean_completion_ms"),
         delta(res, "f0point_resident", "p95_completion_ms")),
        ("B", "oracle", delta(res, "pdrs_resident_oracle", "mean_completion_ms"),
         delta(res, "pdrs_resident_oracle", "p95_completion_ms")),
        ("C", "f0", (0.0, (0.0, 0.0)), (0.0, (0.0, 0.0))),
        ("C", "evict", delta(res, "pdrs_evict", "mean_completion_ms"),
         delta(res, "pdrs_evict", "p95_completion_ms")),
        ("C", "main", main_mean, main_p95),
        ("C", "preempt", delta(pre, "pdrs_preempt", "mean_completion_ms"),
         delta(pre, "pdrs_preempt", "p95_completion_ms")),
    ]


def render(lang: str):
    t = T[lang]
    rows = build_rows()
    if lang == "zh":
        plt.rcParams["font.sans-serif"] = ["Heiti TC", "Arial", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica Neue", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"

    # layout: sequential y positions with gaps + group headers
    entries = []
    y = 0.0
    for gi, group in enumerate(("A", "B", "C")):
        if gi > 0:
            y += 1.45
        entries.append(("header", group, y - 0.62))
        for g, key, mean, p95 in rows:
            if g == group:
                entries.append(("row", key, y))
                y += 1.0
    y_bottom = y

    fig = plt.figure(figsize=(12.2, 6.9))
    gs = fig.add_gridspec(1, 2, wspace=0.10)
    ax_mean = fig.add_subplot(gs[0, 0])
    ax_p95 = fig.add_subplot(gs[0, 1])
    plots = [
        (ax_mean, "mean", (-2300, 7500), 7300),
        (ax_p95, "p95", (-6000, 10700), 10450),
    ]

    main_val = {"mean": rows[-2][2][0], "p95": rows[-2][3][0]}

    for ax, metric, xlim, label_x in plots:
        for kind, key, yy in entries:
            if kind == "header":
                ax.text(xlim[0] + 60, yy, t["headers"][key], fontsize=9.2,
                        color=TXT, fontweight="bold")
                if key != "A":
                    ax.axhline(yy - 0.40, color=LIGHT, lw=0.9)
                continue
            group = next(g for g, k, _, _ in rows if k == key)
            color = GROUP_COLOR[group]
            point, (lo, hi) = (rows[[k for _, k, _, _ in rows].index(key)][2 if metric == "mean" else 3])
            highlight = key == "main"
            c = BLUE if highlight else color
            band = 0.42
            if highlight:
                ax.axhspan(yy - band, yy + band, color="#eef3f9", zorder=0)
            if key == "f0":
                ax.plot([0.0], [yy], "o", mfc="white", mec=color, ms=5.2, mew=1.3, zorder=3)
            else:
                ax.plot([lo, hi], [yy, yy], color=c, lw=1.9, alpha=0.55 if not highlight else 0.9,
                        solid_capstyle="round", zorder=2)
                ax.plot([point], [yy], "o", color=c, ms=5.6 if highlight else 4.6, zorder=3)
            label = "0" if key == "f0" else f"{point:+,.0f}"
            ax.text(label_x, yy, label, ha="right", va="center",
                    fontsize=8.2, color=c if highlight else GRAYTXT,
                    fontweight="bold" if highlight else "normal")
        # main line marker
        ax.axvline(main_val[metric], color=BLUE, lw=1.2, ls=(0, (4, 3)), alpha=0.75, zorder=1)
        ax.text(main_val[metric], -1.40, t["main_tag"], color=BLUE, fontsize=8.6,
                ha="center", va="bottom", clip_on=False)
        ax.set_xlim(*xlim)
        ax.set_ylim(y_bottom + 0.15, -1.35)
        ax.set_yticks([])
        ax.grid(axis="x", color=LIGHT, lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#9aa3ab")
        ax.spines["bottom"].set_linewidth(0.9)
        ax.tick_params(axis="x", labelsize=8.4)
        ax.set_title(t["panels"][0 if metric == "mean" else 1], fontsize=11, color=TXT,
                     pad=14, loc="left")

    # y tick labels on the left panel only
    labels = []
    for kind, key, yy in entries:
        labels.append((yy, t["headers"][key] if kind == "header" else t["rows"][key], kind, key))
    ax_mean.set_yticks([yy for yy, _, _, _ in labels])
    ax_mean.set_yticklabels(["" if kind == "header" else lab for _, lab, kind, _ in labels],
                            fontsize=8.6)
    ax_mean.tick_params(axis="y", length=0)
    for tick, (_, _, kind, key) in zip(ax_mean.get_yticklabels(), labels):
        if kind == "row" and key == "main":
            tick.set_color(BLUE); tick.set_fontweight("bold")

    fig.suptitle(t["suptitle"], fontsize=11.5, x=0.008, ha="left", y=1.015, color=TXT)
    ax_mean.set_xlabel(t["xlabel"], fontsize=9.0, color=TXT)
    ax_p95.set_xlabel(t["xlabel"], fontsize=9.0, color=TXT)
    fig.text(0.008, -0.045, t["foot"], fontsize=7.8, color=GRAYTXT)

    suffix = "_zh" if lang == "zh" else ""
    name = f"fig16_main_vs_families{suffix}"
    fig.savefig(OUT / f"{name}.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", format="png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


if __name__ == "__main__":
    render("zh")
    render("en")
    print("done")
