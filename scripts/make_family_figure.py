#!/usr/bin/env python3
"""fig16: the main line vs the adapted baselines (delta vs main line, confirm300).

Baselines grouped by capability (as adapted, per the frozen manifest):
  A  basic (no future info / no complex actions): FCFS, Parrot
  B  future information in ordering: QLM, LLMSched, Hermes
  C  complex actions (residency / loading): Torpor
Reference row: the main line itself.

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

GRAY, TEAL, AMBER, BLUE = "#8a97a6", "#2f8f83", "#c98a2e", "#2b5d8a"
LIGHT = "#e8eaed"
TXT = "#2b2f33"
GRAYTXT = "#6b6b6b"
GROUP_COLOR = {"A": GRAY, "B": TEAL, "C": AMBER}

T = {
    "zh": {
        "suptitle": "调度器对比:主线 vs 适配基线(按能力分组;Δ vs 主线,confirm300)",
        "panels": ["平均完成时间", "p95 完成时间", "makespan"],
        "xlabel": "Δ vs 主线(ms;正 = 更差)",
        "headers": {"A": "基础基线(无未来信息 / 无复杂动作)",
                     "B": "含未来信息(预测 / 信念参与排序)",
                     "C": "含复杂动作(主动驻留 / 加载管理)"},
        "rows": {
            "fcfs": "FCFS", "parrot": "Parrot(App-FIFO)",
            "qlm": "QLM(时长分布 + 场景)", "llmsched": "LLMSched(贝叶斯后验)",
            "hermes": "Hermes(PDGraph + Gittins;含在线预热)",
            "torpor": "Torpor(换入换出 + 驱逐 + 干扰感知)",
            "main": "主线(pdrs_resident;参考)",
        },
        "tag": "主线",
        "foot": ("参照 = 主线 pdrs_resident(F0 排序 + PDRS 信念 + 最小 ΣV 驱逐 + 保守预取);"
                 "Δ>0 = 比主线差。CI = 300 集逐集配对 bootstrap 95%(跨运行;同一冻结 confirm300,参照 F0 逐位复现)。"
                 "违约率:主线对全部 6 条更优(Δ 0.002–0.008);makespan:显著胜 Parrot/LLMSched/Hermes,"
                 "与 FCFS/QLM 打平(CI 含 0),显著负于 Torpor。"
                 "跨接口提示:基线不携带驻留动作,仅作端到端对比、不作机制归因。Myopic(补充下限)未画。"),
    },
    "en": {
        "suptitle": "Scheduler comparison — main line vs adapted baselines (grouped by capability; Δ vs main line, confirm300)",
        "panels": ["Mean completion", "p95 completion", "Makespan"],
        "xlabel": "Δ vs main line (ms, positive = worse)",
        "headers": {"A": "Basic baselines",
                     "B": "Future-information methods",
                     "C": "Complex-action methods"},
        "rows": {
            "fcfs": "FCFS", "parrot": "Parrot (App-FIFO)",
            "qlm": "QLM (duration dist. + scenarios)", "llmsched": "LLMSched (Bayesian posterior)",
            "hermes": "Hermes (PDGraph + Gittins; online prewarm)",
            "torpor": "Torpor (swap + eviction + interference-aware)",
            "main": "Main line (pdrs_resident; reference)",
        },
        "tag": "main line",
        "foot": ("Reference = the main line pdrs_resident (F0 ordering + PDRS belief + minimal-ΣV eviction + conservative prefetch); "
                 "Δ>0 means worse than the main line. CIs = paired bootstrap 95% over 300 episodes (cross-run; identical frozen confirm300, F0 reproduced bit-identically). "
                 "Deadline miss: main line better on all six (Δ 0.002–0.008); makespan: significant wins over Parrot/LLMSched/Hermes, ties FCFS/QLM (CI crosses 0), significant loss to Torpor. "
                 "Grouping: basic = no future info / no complex actions; future-info = prediction or belief in ordering; complex-action = active residency / loading management. "
                 "Cross-interface caveat: baselines do not carry the residency actions — end-to-end only, no mechanism attribution. Myopic (supplementary) omitted."),
    },
}

ORDER = [
    ("A", "fcfs"), ("A", "parrot"),
    ("B", "qlm"), ("B", "llmsched"), ("B", "hermes"),
    ("C", "torpor"),
    ("M", "main"),
]


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
        "parrot": mt["results"]["parrot_appfifo"],
        "qlm": mt["results"]["qlm_queue"],
        "llmsched": mt["results"]["llmsched"],
        "hermes": mt["results"]["hermes_gittins"],
        "torpor": mt["results"]["torpor_lifecycle"],
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
    for gi, group in enumerate(("A", "B", "C")):
        if gi > 0:
            y += 1.45
        entries.append(("header", group, y - 0.62))
        for g, key in ORDER:
            if g == group:
                entries.append(("row", key, y))
                y += 1.0
    y += 0.95
    entries.append(("row", "main", y))
    y_bottom = y

    fig = plt.figure(figsize=(12.8, 5.6))
    gs = fig.add_gridspec(1, 3, wspace=0.10, width_ratios=[1.12, 1.22, 1.0])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panels = list(zip(axes,
                      ["mean_completion_ms", "p95_completion_ms", "makespan_ms"],
                      [(-600, 7300), (-2000, 11500), (-3400, 4900)],
                      [7100, 11300, 4700]))

    for ax, metric, xlim, label_x in panels:
        for kind, key, yy in entries:
            if kind == "header":
                ax.text(xlim[0] + 60, yy, t["headers"][key], fontsize=8.8,
                        color=TXT, fontweight="bold")
                if key != "A":
                    ax.axhline(yy - 0.40, color=LIGHT, lw=0.9)
                continue
            group = next(g for g, k in ORDER if k == key)
            color = BLUE if key == "main" else GROUP_COLOR[group]
            point, (lo, hi) = data[key][metric]
            highlight = key == "main"
            if highlight:
                ax.axhspan(yy - 0.40, yy + 0.40, color="#eef3f9", zorder=0)
            if key == "main":
                ax.plot([0.0], [yy], "o", color=BLUE, ms=5.2, zorder=3)
            else:
                ax.plot([lo, hi], [yy, yy], color=color, lw=1.9, alpha=0.6,
                        solid_capstyle="round", zorder=2)
                ax.plot([point], [yy], "o", color=color, ms=4.6, zorder=3)
            label = "0" if highlight else f"{point:+,.0f}"
            ax.text(label_x, yy, label, ha="right", va="center", fontsize=8.0,
                    color=color if highlight else GRAYTXT,
                    fontweight="bold" if highlight else "normal")
        ax.axvline(0.0, color=BLUE, lw=1.2, ls=(0, (4, 3)), alpha=0.75, zorder=1)
        ax.text(0.0, -1.45, t["tag"], color=BLUE, fontsize=8.4,
                ha="center", va="bottom", clip_on=False)
        ax.set_xlim(*xlim)
        ax.set_ylim(y_bottom + 0.15, -1.35)
        ax.set_yticks([])
        ax.grid(axis="x", color=LIGHT, lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#9aa3ab")
        ax.spines["bottom"].set_linewidth(0.9)
        ax.tick_params(axis="x", labelsize=8.2)
        ax.set_xlabel(t["xlabel"], fontsize=8.4, color=TXT)

    # shared row labels on the first axes
    labels = []
    for kind, key, yy in entries:
        labels.append((yy, t["headers"][key] if kind == "header" else t["rows"][key], kind, key))
    axes[0].set_yticks([yy for yy, _, _, _ in labels])
    axes[0].set_yticklabels(["" if kind == "header" else lab for _, lab, kind, _ in labels],
                            fontsize=8.3)
    axes[0].tick_params(axis="y", length=0)
    for tick, (_, _, kind, key) in zip(axes[0].get_yticklabels(), labels):
        if kind == "row" and key == "main":
            tick.set_color(BLUE); tick.set_fontweight("bold")

    for ax, title in zip(axes, t["panels"]):
        ax.set_title(title, fontsize=10.2, color=TXT, pad=14, loc="left")

    fig.suptitle(t["suptitle"], fontsize=11.5, x=0.008, ha="left", y=1.03, color=TXT)
    fig.text(0.008, -0.075, t["foot"], fontsize=7.6, color=GRAYTXT)

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
