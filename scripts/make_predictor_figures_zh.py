#!/usr/bin/env python3
"""Predictor comparison figures (cross-family benchmarks) — Chinese report variants.

fig13: single-step family sweep (2026-08 benchmarks: next action-family + runtime)
fig14: multi-step structure prediction (P9d benchmark; statistical vs LightGBM vs causal GRU)

Outputs SVG + PNG into outputs/report_materials/figures/.
Run with the `print` conda env:
  MPLCONFIGDIR=/tmp/mpl-cache conda run -n print python scripts/make_predictor_figures.py
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_materials" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BENCH_ACTION = ROOT / "results/processed/benchmark_action_family_20260806.json"
BENCH_RESOURCE = ROOT / "results/processed/benchmark_resource_v07b_20260806.json"
P9D_EMPIRICAL = ROOT / "experiments/EXP-20260902_p9d_topology_empirical_baseline/metrics.json"
P9D_TABULAR = ROOT / "experiments/EXP-20260902_p9d_topology_tabular/metrics.json"
P9D_GRU = ROOT / "experiments/EXP-20260902_p9d_shared_causal_gru/metrics.json"

FAM_COLORS = {"stat": "#8a97a6", "tree": "#2f8f83", "knn": "#c98a2e", "nn": "#2b5d8a"}
STEM = "#d5dbe1"
GRID = "#e8eaed"
TXT = "#2b2f33"
GRAY = "#6b6b6b"

ACTION_FAMILY = {
    "01_全局先验": ("stat", "\u5168\u5c40\u5148\u9a8c"),
    "02_条件计数": ("stat", "\u6761\u4ef6\u8ba1\u6570"),
    "03_n-gram_w1": ("stat", "n-gram (w1)"),
    "03_n-gram_w2": ("stat", "n-gram (w2)"),
    "03_n-gram_w3": ("stat", "n-gram (w3)"),
    "04_逻辑回归": ("stat", "\u903b\u8f91\u56de\u5f52"),
    "05_朴素贝叶斯": ("stat", "\u6734\u7d20\u8d1d\u53f6\u65af"),
    "06_随机森林": ("tree", "\u968f\u673a\u68ee\u6797"),
    "07_LightGBM": ("tree", "LightGBM"),
    "08_XGBoost": ("tree", "XGBoost"),
    "09_MLP": ("nn", "MLP"),
    "10_CNN-1D": ("nn", "CNN-1D"),
    "11_GRU_embedding": ("nn", "GRU(\u5d4c\u5165)"),
    "11_GRU_onehot": ("nn", "GRU(one-hot)"),
    "12_BiLSTM": ("nn", "BiLSTM"),
    "13_Transformer-lite": ("nn", "Transformer"),
    "14_GNN": ("nn", "GNN"),
    "15_KNN": ("knn", "KNN"),
}
RUNTIME_FAMILY = {
    "R0_全局中位": ("stat", "\u5168\u5c40\u4e2d\u4f4d"),
    "R1_族x模型分位表": ("stat", "\u65cf\u00d7\u6a21\u578b\u5206\u4f4d\u8868"),
    "R2_分位表+backoff": ("stat", "\u5206\u4f4d\u8868+backoff"),
    "R7_roofline": ("stat", "roofline(\u7269\u7406)"),
    "R3_LightGBM": ("tree", "LightGBM"),
    "R4_XGBoost": ("tree", "XGBoost"),
    "R5_MLP": ("nn", "MLP"),
    "R6_ProD分布": ("nn", "ProD(\u5206\u5e03)"),
    "R8_KNN": ("knn", "KNN"),
}


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def save(fig, name: str) -> None:
    name = name + "_zh"
    svg = OUT / f"{name}.svg"
    png = OUT / f"{name}.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png.relative_to(ROOT), "+ svg")


def _lollipop(ax, items, xlim, ticks, fmt, label_dx, label_size=7.6):
    ys = np.arange(len(items))[::-1]
    for y, (label, value, fam) in zip(ys, items):
        color = FAM_COLORS[fam]
        ax.plot([xlim[0], value], [y, y], color=STEM, lw=1.1, zorder=1, solid_capstyle="round")
        ax.plot([value], [y], "o", color=color, ms=5.0, zorder=3)
        ax.text(value + label_dx, y, fmt.format(value), va="center", ha="left",
                fontsize=label_size, color=TXT)
    ax.set_yticks(ys)
    ax.set_yticklabels([item[0] for item in items], fontsize=8.2)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" if t < 1 else f"{t:.1f}" for t in ticks], fontsize=8.2)
    ax.set_ylim(-0.7, len(items) - 0.3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#9aa3ab")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig13_single_step() -> None:
    act = load(BENCH_ACTION)
    res = load(BENCH_RESOURCE)
    noise = res["noise_radius"]["rel_diff_median"]

    action_items = [(label, act["models"][key]["top1"], fam)
                    for key, (fam, label) in ACTION_FAMILY.items()
                    if key in act["models"]]
    action_items.sort(key=lambda t: t[1], reverse=True)
    runtime_items = [(label, res["runtime"][key]["rel_mae"], fam)
                     for key, (fam, label) in RUNTIME_FAMILY.items()
                     if key in res["runtime"]]
    runtime_items.sort(key=lambda t: t[1])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6), gridspec_kw={"width_ratios": [1.14, 1]})

    ax = axes[0]
    _lollipop(ax, action_items, xlim=(0.60, 0.895), ticks=[0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
              fmt="{:.3f}", label_dx=0.0045)
    ax.set_xlabel("top-1 \u51c6\u786e\u7387(\u8f74\u4ece 0.60 \u8d77)", fontsize=8.4, color=TXT)
    ax.set_title("A \u00b7 \u52a8\u4f5c\u65cf\u9884\u6d4b(\u5355\u6b65)top-1 \u51c6\u786e\u7387",
                 fontsize=10, color=TXT, pad=8, loc="left")

    ax = axes[1]
    _lollipop(ax, runtime_items, xlim=(0.0, 1.10), ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
              fmt="{:.1%}", label_dx=0.012)
    ax.axvline(noise, color="#b3544a", lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.set_ylim(-0.7, len(runtime_items) + 0.85)
    ax.text(noise, len(runtime_items) + 0.30, f"\u91cd\u590d\u6d4b\u91cf\u566a\u58f0 {noise:.3f}",
            va="bottom", ha="center", fontsize=7.2, color="#b3544a")
    ax.set_xlabel("\u76f8\u5bf9 MAE(\u4e2d\u4f4d;\u4f4e\u8005\u4f18)", fontsize=8.4, color=TXT)
    ax.set_title("B \u00b7 \u8fd0\u884c\u65f6\u9884\u6d4b(\u5355\u6b65)\u76f8\u5bf9 MAE",
                 fontsize=10, color=TXT, pad=8, loc="left")

    handles = [
        Line2D([], [], marker="o", ls="", color=FAM_COLORS["stat"], ms=5.5,
               label="\u7edf\u8ba1\u00b7\u6d45\u5c42"),
        Line2D([], [], marker="o", ls="", color=FAM_COLORS["tree"], ms=5.5,
               label="\u6811\u6a21\u578b(\u4f20\u7edf ML)"),
        Line2D([], [], marker="o", ls="", color=FAM_COLORS["knn"], ms=5.5,
               label="\u5b9e\u4f8b(KNN)"),
        Line2D([], [], marker="o", ls="", color=FAM_COLORS["nn"], ms=5.5,
               label="\u795e\u7ecf\u7f51\u7edc"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=8.2,
               bbox_to_anchor=(0.995, 1.035), handletextpad=0.35, columnspacing=1.1)
    fig.suptitle("\u9884\u6d4b\u5668\u8de8\u65cf\u57fa\u51c6(\u5355\u6b65\u4efb\u52a1;\u51bb\u7ed3\u8bc4\u4f30)",
                 fontsize=10.5, x=0.008, ha="left", y=1.045, color=TXT)
    fig.text(0.008, -0.035,
             "\u51bb\u7ed3\u57fa\u51c6(2026-08):\u52a8\u4f5c\u65cf 6,552 \u884c\u3001\u8fd0\u884c\u65f6 9,066 \u69fd\u4f4d\u5bf9\u3002"
             "\u5355\u6b65\u4efb\u52a1\u6811\u6a21\u578b\u9886\u5148 \u2192 \u65e9\u671f\u884c\u4e3a/\u8d44\u6e90\u9884\u6d4b\u91c7\u7528\u6811\u6a21\u578b;"
             "\u591a\u6b65\u8054\u5408\u7ed3\u6784\u9884\u6d4b\u6539\u7528\u56e0\u679c GRU(\u89c1\u56fe 14)\u3002",
             fontsize=7.8, color=GRAY)
    save(fig, "fig13_predictor_families_single_step")


def fig14_structure() -> None:
    emp = load(P9D_EMPIRICAL)["selected_metrics"]
    tab = load(P9D_TABULAR)["selected_metrics"]
    gru = load(P9D_GRU)["aggregate"]["topology_only"]

    metrics = [
        ("layer_count_mae", "\u5c42\u6570 MAE \u2193", True, [0.05, 0.1, 0.25, 0.5, 1, 2], (0.028, 5.5)),
        ("node_count_mae", "\u8282\u70b9\u6570 MAE \u2193", True, [0.25, 0.5, 1, 2, 5], (0.19, 9.0)),
        ("width_vector_mae", "\u5bbd\u5ea6\u5411\u91cf MAE \u2193", True, [0.05, 0.1, 0.2, 0.5, 1], (0.042, 2.3)),
        ("top1_exact_signature_coverage", "\u7cbe\u786e\u7b7e\u540d top-1 \u2191", False, [0.0, 0.15, 0.30, 0.45], (0.0, 0.60)),
    ]
    fams = [("\u7edf\u8ba1\u57fa\u7ebf\n(\u6761\u4ef6\u9891\u7387)", "#8a97a6"),
            ("LightGBM\n(\u8868\u683c ML)", "#2f8f83"),
            ("\u56e0\u679c GRU\n(\u672c\u65b9\u6cd5)", "#2b5d8a")]

    def series(src, split, key):
        if src is gru:
            e = src[split][key]
            return e["mean"], e["std"]
        return src[split][key], None

    def fmt(v):
        if v <= 0:
            return "0"
        return f"{v:.2f}" if v >= 1 else f"{v:.3f}"

    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.5))
    x = np.arange(3)
    w = 0.33
    for ax, (key, title, logy, ticks, ylim) in zip(axes, metrics):
        colors = [c for _, c in fams]
        vv = [series(s, "validation", key) for s in (emp, tab, gru)]
        vh = [series(s, "holdout", key) for s in (emp, tab, gru)]
        ax.bar(x - w / 2, [v[0] for v in vv], w, color=colors, alpha=0.28,
               edgecolor=colors, lw=0.9, hatch="////", zorder=2)
        ax.bar(x + w / 2, [v[0] for v in vh], w, color=colors, edgecolor="white",
               lw=0.6, zorder=2,
               yerr=[v[1] or 0 for v in vh],
               error_kw=dict(lw=0.9, capsize=2.2, ecolor="#3a3a3a", zorder=4))
        for xi, (val, err) in zip(x - w / 2, vv):
            ax.text(xi - 0.05, val * 1.10 if logy else val + 0.012, fmt(val),
                    ha="center", va="bottom", fontsize=6.8, color="#5a6672")
        for xi, (val, err) in zip(x + w / 2, vh):
            top = val + (err or 0)
            ax.text(xi + 0.05, top * 1.10 if logy else top + 0.020, fmt(val),
                    ha="center", va="bottom", fontsize=7.0, color="#1c1f23")
        if logy:
            ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=8.0)
        ax.minorticks_off()
        ax.set_title(title, fontsize=9.4, color=TXT, pad=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f for f, _ in fams], fontsize=7.6)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#9aa3ab")
        ax.spines[["left", "bottom"]].set_linewidth(0.8)

    handles = [Patch(facecolor="#b9c2cc", edgecolor="#8a97a6", hatch="////", alpha=0.6,
                     label="\u9a8c\u8bc1"),
               Patch(facecolor="#8a97a6", edgecolor="white", label="\u7559\u51fa(holdout)")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=8.4,
               bbox_to_anchor=(0.995, 1.045), handletextpad=0.4, columnspacing=1.2)
    fig.suptitle("\u591a\u6b65\u7ed3\u6784\u9884\u6d4b:\u7edf\u8ba1 vs \u8868\u683c ML vs \u56e0\u679c GRU",
                 fontsize=10.5, x=0.008, ha="left", y=1.045, color=TXT)
    fig.text(0.008, -0.055,
             "\u6307\u6807\u8bf4\u660e:\u5c42\u6570 / \u8282\u70b9\u6570 / \u5bbd\u5ea6\u5411\u91cf MAE = \u9884\u6d4b\u672a\u6765 DAG \u7ed3\u6784"
             "(\u5c42\u6570\u3001\u8282\u70b9\u6570\u3001\u5404\u5c42\u5bbd\u5ea6)\u4e0e\u771f\u503c\u7684\u5e73\u5747\u7edd\u5bf9\u8bef\u5dee,\u8d8a\u4f4e\u8d8a\u597d;"
             "\u7cbe\u786e\u7b7e\u540d top-1 = \u9884\u6d4b\u7ed3\u6784\u4e0e\u771f\u503c\u5b8c\u5168\u4e00\u81f4\u7684\u6bd4\u4f8b,\u8d8a\u9ad8\u8d8a\u597d\u3002",
             fontsize=7.8, color=GRAY)
    save(fig, "fig14_predictor_structure")


if __name__ == "__main__":
    fig13_single_step()
    fig14_structure()
    print("done")
