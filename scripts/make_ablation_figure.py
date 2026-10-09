#!/usr/bin/env python3
"""fig17: main-line ablations — action ablation and information (belief) ablation.

All arms are paired per episode against the main line (pdrs_resident), 300 frozen
confirm300 episodes. Bars + 95% bootstrap CI whiskers.

Outputs SVG + PNG (zh + en) into outputs/report_materials/figures/.
Run with the `print` conda env:
  MPLCONFIGDIR=/tmp/mpl-cache conda run -n print python scripts/make_ablation_figure.py
"""
from __future__ import annotations

import json
import random
import statistics
import textwrap
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

RES = ROOT / "experiments/EXP-20261005_residency_comparison_v1/artifacts/residency_comparison_v1.json"
PRE = ROOT / "experiments/EXP-20261005_residency_comparison_v1/artifacts/residency_preempt_v1.json"

GRAY, BLUE = "#8a97a6", "#2b5d8a"
LIGHT = "#e8eaed"
TXT = "#2b2f33"
GRAYTXT = "#6b6b6b"

T = {
    "zh": {
        "suptitle": "主线消融:动作消融与信息消融(Δ vs 主线,confirm300 冻结集)",
        "panels": ["平均完成时间", "p95 完成时间", "makespan"],
        "xlabel": "Δ vs 主线(ms;正 = 更差)",
        "headers": {"A": "动作消融(驻留动作的取舍)", "B": "信息消融(预测分布质量)"},
        "rows": {
            "f0": "主线(无驱逐、预取)",
            "evict": "主线(无预取)",
            "main_a": "主线(参考)", "preempt": "主线 + 抢占",
            "shuffle": "打乱预测分布 + 驻留",
            "f0point": "主线(概率分布估计改为点估计)",
            "main_b": "主线(参考)",
        },
        "tag": "主线",
        "foot": ("指标:平均完成时间 = confirm300(300 集)的平均完成时间;"
                 "p95 完成时间 = 完成时间的 95 分位数;makespan = 全部任务完成的总时长;"
                 "Δ vs 主线 = 各臂与主线的逐集配对差值(正 = 比主线差;CI = 95% bootstrap)。"),
    },
    "en": {
        "suptitle": "Main-line ablations — action and information variants (Δ vs main line, frozen confirm300)",
        "panels": ["Mean completion", "p95 completion", "Makespan"],
        "xlabel": "Δ vs main line (ms, positive = worse)",
        "headers": {"A": "Action ablation (residency choices)", "B": "Information ablation (predicted-distribution quality)"},
        "rows": {
            "f0": "Main line (no eviction/prefetch)",
            "evict": "Main line (no prefetch)",
            "main_a": "Main line (reference)", "preempt": "Main line + preemption",
            "shuffle": "Shuffled predicted distribution + residency",
            "f0point": "Main line (distribution estimate → point estimate)",
            "main_b": "Main line (reference)",
        },
        "tag": "main line",
        "foot": ("Metrics: mean completion = average completion time over the 300-episode confirm set; "
                 "p95 completion = 95th percentile of completion times; makespan = total span to finish all tasks; "
                 "Δ vs main line = per-episode paired difference (positive = worse; 95% bootstrap CI)."),
    },
}

ORDER = [
    ("A", "f0"), ("A", "evict"), ("A", "main_a"), ("A", "preempt"),
    ("B", "shuffle"), ("B", "f0point"), ("B", "main_b"),
]


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def build_rows():
    res, pre = load(RES), load(PRE)
    main = res["results"]["pdrs_resident"]["metrics"]
    ref_ep = res["reference_summary"]["episode_values"]
    rng = random.Random(11)

    def boot(dd, B=2000):
        n = len(dd)
        s = sorted(sum(dd[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
        return statistics.fmean(dd), (s[50], s[1949])

    metrics = ["mean_completion_ms", "p95_completion_ms", "makespan_ms"]
    arms = {
        "evict": res["results"]["pdrs_evict"],
        "shuffle": res["results"]["pdrs_resident_shuffle"],
        "f0point": res["results"]["f0point_resident"],
        "preempt": pre["results"]["pdrs_preempt"],
    }
    out = {}
    for key, arm in arms.items():
        stats = {}
        for metric in metrics:
            dd = [a - b for a, b in zip(arm["metrics"][metric]["episode_values"],
                                        main[metric]["episode_values"])]
            stats[metric] = boot(dd)
        out[key] = stats
    f0_stats = {}
    for metric in metrics:
        dd = [b - a for a, b in zip(main[metric]["episode_values"], ref_ep[metric])]
        f0_stats[metric] = boot(dd)
    out["f0"] = f0_stats
    zero = {metric: (0.0, (0.0, 0.0)) for metric in metrics}
    out["main_a"] = dict(zero)
    out["main_b"] = dict(zero)
    return out


def render(lang: str):
    t = T[lang]
    data = build_rows()
    if lang == "zh":
        plt.rcParams["font.sans-serif"] = ["Heiti TC", "Arial", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica Neue", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"

    # layout: group A rows at 0..3; header B + gap; group B rows at 5.4..8.4
    entries = [("header", "A", -0.85)]
    for i, (g, key) in enumerate(ORDER):
        y = i if i < 4 else 5.4 + (i - 4)
        entries.append(("row", key, y))
    entries.append(("header", "B", 4.55))
    y_bottom = 7.4

    fig = plt.figure(figsize=(12.6, 5.0))
    gs = fig.add_gridspec(1, 3, wspace=0.10, width_ratios=[1.12, 1.22, 1.0])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panels = list(zip(axes,
                      ["mean_completion_ms", "p95_completion_ms", "makespan_ms"],
                      [(-450, 2500), (-1050, 5500), (-950, 5800)],
                      [2350, 5330, 5630]))

    for ax, metric, xlim, label_x in panels:
        for kind, key, yy in entries:
            if kind == "header":
                if ax is axes[0]:
                    ax.text(xlim[0] + 55, yy, t["headers"][key], fontsize=9.0,
                            color=TXT, fontweight="bold")
                if key != "A":
                    ax.axhline(yy - 0.50, color=LIGHT, lw=0.9)
                continue
            highlight = key.startswith("main")
            color = BLUE if highlight else GRAY
            point, (lo, hi) = data[key][metric]
            if highlight:
                ax.axhspan(yy - 0.40, yy + 0.40, color="#eef3f9", zorder=0)
                ax.plot([0.0], [yy], "o", color=BLUE, ms=5.0, zorder=3)
            else:
                ax.barh(yy, point, height=0.52, color=color, alpha=0.75,
                        edgecolor="white", linewidth=0.6, zorder=2)
                ax.errorbar([point], [yy],
                            xerr=[[max(0.0, point - lo)], [max(0.0, hi - point)]],
                            fmt="none", ecolor="#4a5560", elinewidth=0.9,
                            capsize=2.4, capthick=0.9, zorder=3)
            label = "0" if highlight else f"{point:+,.0f}"
            ax.text(label_x, yy, label, ha="right", va="center", fontsize=8.0,
                    color=color if highlight else GRAYTXT,
                    fontweight="bold" if highlight else "normal")
        ax.axvline(0.0, color=BLUE, lw=1.2, ls=(0, (4, 3)), alpha=0.75, zorder=1)
        ax.text(0.0, -1.55, t["tag"], color=BLUE, fontsize=8.4,
                ha="center", va="bottom", clip_on=False)
        ax.set_xlim(*xlim)
        ax.set_ylim(y_bottom + 0.45, -1.75)
        ax.set_yticks([])
        ax.grid(axis="x", color=LIGHT, lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#9aa3ab")
        ax.spines["bottom"].set_linewidth(0.9)
        ax.tick_params(axis="x", labelsize=8.2)
        ax.set_xlabel(t["xlabel"], fontsize=8.4, color=TXT)
        ax.set_title(t["panels"][["mean_completion_ms", "p95_completion_ms",
                                  "makespan_ms"].index(metric)],
                     fontsize=10.2, color=TXT, pad=24, loc="left")

    axes[0].set_yticks([yy for e_kind, e_key, yy in entries if e_kind == "row"])
    axes[0].set_yticklabels([t["rows"][key] for kind, key, _ in entries if kind == "row"],
                            fontsize=8.3)
    axes[0].tick_params(axis="y", length=0)
    for tick, e in zip(axes[0].get_yticklabels(), [e for e in entries if e[0] == "row"]):
        if str(e[1]).startswith("main"):
            tick.set_color(BLUE); tick.set_fontweight("bold")

    fig.suptitle(t["suptitle"], fontsize=11.5, x=0.008, ha="left", y=1.03, color=TXT)
    width = 108 if lang == "zh" else 205
    foot = "\n".join(textwrap.wrap(t["foot"], width=width))
    fig.text(0.008, -0.055, foot, fontsize=7.6, color=GRAYTXT, va="top", linespacing=1.45)

    suffix = "_zh" if lang == "zh" else ""
    name = f"fig17_main_ablations{suffix}"
    fig.savefig(OUT / f"{name}.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", format="png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


if __name__ == "__main__":
    render("zh")
    render("en")
    print("done")
