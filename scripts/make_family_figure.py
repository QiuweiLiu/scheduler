#!/usr/bin/env python3
"""fig16: the main line vs the six adapted baselines (delta vs main line, confirm300).

One flat list; each baseline annotated with its method and source venue.
Reference row: the main line itself.

Outputs SVG + PNG (zh + en) into outputs/report_materials/figures/.
Run with the `print` conda env:
  MPLCONFIGDIR=/tmp/mpl-cache conda run -n print python scripts/make_family_figure.py
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

MT = ROOT / "experiments/EXP-20261004_main_table_comparison_v1/artifacts/main_table_tail_v1.json"
FCFS = ROOT / "experiments/EXP-20261004_main_table_comparison_v1/artifacts/fcfs_tail_v1.json"
RES = ROOT / "experiments/EXP-20261005_residency_comparison_v1/artifacts/residency_comparison_v1.json"

GRAY, BLUE = "#8a97a6", "#2b5d8a"
LIGHT = "#e8eaed"
TXT = "#2b2f33"
GRAYTXT = "#6b6b6b"

T = {
    "zh": {
        "suptitle": "调度器对比:主线 vs 六条适配基线(Δ vs 主线,confirm300 冻结集)",
        "panels": ["平均完成时间", "p95 完成时间", "makespan"],
        "xlabel": "Δ vs 主线(ms;正 = 更差)",
        "rows": {
            "fcfs": "FCFS:经典先到先服务",
            "torpor": "Torpor:生命周期管理 [ATC'25]",
            "qlm": "QLM:随机队列 + SLO [SoCC'24]",
            "parrot": "Parrot:应用级 FIFO [OSDI'24]",
            "llmsched": "LLMSched:贝叶斯信息获取 [ICDCS'25]",
            "hermes": "Hermes:PDGraph + Gittins + 预热 [TACO'26]",
            "main": "主线(参考)",
        },
        "tag": "主线",
        "foot": ("图示:六条适配基线相对主线(pdrs_resident = F0 排序 + PDRS 信念 + 最小 ΣV 驱逐 + 保守预取)的逐集配对差值"
                 "(三个面板:平均 / p95 / makespan;正 = 比主线差;"
                 "CI = 95% bootstrap,冻结 confirm300、跨运行配对;跨接口提示:基线不携带驻留动作,仅作端到端对比、不作机制归因;"
                 "Myopic(补充下限)未画)。基线来源:Parrot [OSDI '24],QLM [SoCC '24],LLMSched [ICDCS '25],"
                 "Hermes [TACO '26],Torpor [ATC '25],FCFS 为经典基线。"),
    },
    "en": {
        "suptitle": "Scheduler comparison — main line vs six adapted baselines (Δ vs main line, frozen confirm300)",
        "panels": ["Mean completion", "p95 completion", "Makespan"],
        "xlabel": "Δ vs main line (ms, positive = worse)",
        "rows": {
            "fcfs": "FCFS (classic)",
            "torpor": "Torpor: lifecycle [ATC'25]",
            "qlm": "QLM: stochastic queue + SLO [SoCC'24]",
            "parrot": "Parrot: application FIFO [OSDI'24]",
            "llmsched": "LLMSched: Bayesian info gain [ICDCS'25]",
            "hermes": "Hermes: PDGraph + Gittins + prewarm [TACO'26]",
            "main": "Main line (reference)",
        },
        "tag": "main line",
        "foot": ("What is plotted: per-episode paired deltas of the six adapted baselines against the main line "
                 "(pdrs_resident = F0 ordering + PDRS belief + minimal-ΣV eviction + conservative prefetch) "
                 "(mean / p95 / makespan; positive = worse; 95% bootstrap CIs; frozen confirm300, cross-run pairing; "
                 "cross-interface caveat: baselines do not carry the residency actions — end-to-end only, no mechanism attribution; "
                 "Myopic lower bound omitted). Baseline sources: Parrot [OSDI '24], QLM [SoCC '24], LLMSched [ICDCS '25], "
                 "Hermes [TACO '26], Torpor [ATC '25], FCFS classic."),
    },
}

ORDER = ["fcfs", "torpor", "qlm", "parrot", "llmsched", "hermes", "main"]


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def build_rows():
    mt, fcfs = load(MT), load(FCFS)
    res = load(RES)
    main_metrics = res["results"]["pdrs_resident"]["metrics"]
    rng = random.Random(11)

    def boot(dd, B=2000):
        n = len(dd)
        s = sorted(sum(dd[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
        return statistics.fmean(dd), (s[50], s[1949])

    sources = {
        "fcfs": fcfs,
        "torpor": mt["results"]["torpor_lifecycle"],
        "qlm": mt["results"]["qlm_queue"],
        "parrot": mt["results"]["parrot_appfifo"],
        "llmsched": mt["results"]["llmsched"],
        "hermes": mt["results"]["hermes_gittins"],
    }
    metrics = ["mean_completion_ms", "p95_completion_ms", "makespan_ms"]
    out = {}
    for key, src in sources.items():
        stats = {}
        for metric in metrics:
            dd = [a - b for a, b in zip(src["metrics"][metric]["episode_values"],
                                        main_metrics[metric]["episode_values"])]
            stats[metric] = boot(dd)
        out[key] = stats
    out["main"] = {metric: (0.0, (0.0, 0.0)) for metric in metrics}
    return out


def render(lang: str):
    t = T[lang]
    data = build_rows()
    if lang == "zh":
        plt.rcParams["font.sans-serif"] = ["Heiti TC", "Arial", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica Neue", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"

    entries = []
    y = 0.0
    for key in ORDER[:-1]:
        entries.append(("row", key, y))
        y += 1.0
    y += 0.95
    entries.append(("row", "main", y))
    y_bottom = y

    fig = plt.figure(figsize=(12.6, 4.5))
    gs = fig.add_gridspec(1, 3, wspace=0.10, width_ratios=[1.12, 1.22, 1.0])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panels = list(zip(axes,
                      ["mean_completion_ms", "p95_completion_ms", "makespan_ms"],
                      [(-700, 6700), (-2100, 10600), (-3300, 4400)],
                      [6500, 10400, 4200]))

    for ax, metric, xlim, label_x in panels:
        for kind, key, yy in entries:
            color = BLUE if key == "main" else GRAY
            point, (lo, hi) = data[key][metric]
            highlight = key == "main"
            if highlight:
                ax.axhspan(yy - 0.40, yy + 0.40, color="#eef3f9", zorder=0)
                ax.plot([0.0], [yy], "o", color=BLUE, ms=5.2, zorder=3)
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
        ax.text(0.0, -1.05, t["tag"], color=BLUE, fontsize=8.4,
                ha="center", va="bottom", clip_on=False)
        ax.set_xlim(*xlim)
        ax.set_ylim(y_bottom + 0.15, -0.95)
        ax.set_yticks([])
        ax.grid(axis="x", color=LIGHT, lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#9aa3ab")
        ax.spines["bottom"].set_linewidth(0.9)
        ax.tick_params(axis="x", labelsize=8.2)
        ax.set_xlabel(t["xlabel"], fontsize=8.4, color=TXT)
        ax.set_title(t["panels"][list(panels and [p[1] for p in panels]).index(metric)]
                     if False else t["panels"][["mean_completion_ms", "p95_completion_ms",
                                                "makespan_ms"].index(metric)],
                     fontsize=10.2, color=TXT, pad=14, loc="left")

    axes[0].set_yticks([yy for _, _, yy in entries])
    axes[0].set_yticklabels([t["rows"][key] for _, key, _ in entries], fontsize=8.3)
    axes[0].tick_params(axis="y", length=0)
    for tick, (_, key, _) in zip(axes[0].get_yticklabels(), entries):
        if key == "main":
            tick.set_color(BLUE); tick.set_fontweight("bold")

    fig.suptitle(t["suptitle"], fontsize=11.5, x=0.008, ha="left", y=1.03, color=TXT)
    width = 108 if lang == "zh" else 205
    foot = "\n".join(textwrap.wrap(t["foot"], width=width))
    fig.text(0.008, -0.06, foot, fontsize=7.6, color=GRAYTXT, va="top", linespacing=1.45)

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
