#!/usr/bin/env python3
"""Predictor comparison figures (cross-family benchmarks).

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
    "01_全局先验": ("stat", "Global prior"),
    "02_条件计数": ("stat", "Conditional count"),
    "03_n-gram_w1": ("stat", "n-gram (w1)"),
    "03_n-gram_w2": ("stat", "n-gram (w2)"),
    "03_n-gram_w3": ("stat", "n-gram (w3)"),
    "04_逻辑回归": ("stat", "Logistic regression"),
    "05_朴素贝叶斯": ("stat", "Naive Bayes"),
    "06_随机森林": ("tree", "Random forest"),
    "07_LightGBM": ("tree", "LightGBM"),
    "08_XGBoost": ("tree", "XGBoost"),
    "09_MLP": ("nn", "MLP"),
    "10_CNN-1D": ("nn", "CNN-1D"),
    "11_GRU_embedding": ("nn", "GRU (embedding)"),
    "11_GRU_onehot": ("nn", "GRU (one-hot)"),
    "12_BiLSTM": ("nn", "BiLSTM"),
    "13_Transformer-lite": ("nn", "Transformer"),
    "14_GNN": ("nn", "GNN"),
    "15_KNN": ("knn", "KNN"),
}
RUNTIME_FAMILY = {
    "R0_全局中位": ("stat", "Global median"),
    "R1_族x模型分位表": ("stat", "Group-quantile table"),
    "R2_分位表+backoff": ("stat", "Quantile + backoff"),
    "R7_roofline": ("stat", "Roofline (physical)"),
    "R3_LightGBM": ("tree", "LightGBM"),
    "R4_XGBoost": ("tree", "XGBoost"),
    "R5_MLP": ("nn", "MLP"),
    "R6_ProD分布": ("nn", "ProD (distr.)"),
    "R8_KNN": ("knn", "KNN"),
}


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def save(fig, name: str) -> None:
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
    ax.set_xlabel("top-1 accuracy   (axis starts at 0.60)", fontsize=8.4, color=TXT)
    ax.set_title("A · Next action family (single step) — top-1 accuracy",
                 fontsize=10, color=TXT, pad=8, loc="left")

    ax = axes[1]
    _lollipop(ax, runtime_items, xlim=(0.0, 1.10), ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
              fmt="{:.1%}", label_dx=0.012)
    ax.axvline(noise, color="#b3544a", lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.set_ylim(-0.7, len(runtime_items) + 0.85)
    ax.text(noise, len(runtime_items) + 0.30, f"repeat-measurement noise {noise:.3f}",
            va="bottom", ha="center", fontsize=7.2, color="#b3544a")
    ax.set_xlabel("relative MAE, median (lower is better)", fontsize=8.4, color=TXT)
    ax.set_title("B · Runtime (single step) — relative MAE",
                 fontsize=10, color=TXT, pad=8, loc="left")

    handles = [
        Line2D([], [], marker="o", ls="", color=FAM_COLORS["stat"], ms=5.5,
               label="Statistical / shallow"),
        Line2D([], [], marker="o", ls="", color=FAM_COLORS["tree"], ms=5.5,
               label="Tree (classic ML)"),
        Line2D([], [], marker="o", ls="", color=FAM_COLORS["knn"], ms=5.5,
               label="Instance (KNN)"),
        Line2D([], [], marker="o", ls="", color=FAM_COLORS["nn"], ms=5.5,
               label="Neural network"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=8.2,
               bbox_to_anchor=(0.995, 1.035), handletextpad=0.35, columnspacing=1.1)
    fig.suptitle("Cross-family predictor benchmarks \u2014 single-step tasks (frozen evaluation)",
                 fontsize=10.5, x=0.008, ha="left", y=1.045, color=TXT)
    fig.text(0.008, -0.035,
             "Frozen benchmarks (Aug-2026): 6,552 rows for action family; 9,066 slot pairs for runtime. "
             "Trees lead single-step tasks — early behavior/resource predictors adopted trees; "
             "the multi-step joint-structure target moved to a causal GRU (fig.14).",
             fontsize=7.8, color=GRAY)
    save(fig, "fig13_predictor_families_single_step")


def fig14_structure() -> None:
    emp = load(P9D_EMPIRICAL)["selected_metrics"]
    tab = load(P9D_TABULAR)["selected_metrics"]
    gru = load(P9D_GRU)["aggregate"]["topology_only"]

    metrics = [
        ("layer_count_mae", "Layer-count MAE ↓", True, [0.05, 0.1, 0.25, 0.5, 1, 2], (0.028, 5.5)),
        ("node_count_mae", "Node-count MAE ↓", True, [0.25, 0.5, 1, 2, 5], (0.19, 9.0)),
        ("width_vector_mae", "Width-vector MAE ↓", True, [0.05, 0.1, 0.2, 0.5, 1], (0.042, 2.3)),
        ("top1_exact_signature_coverage", "Exact signature top-1 ↑", False, [0.0, 0.15, 0.30, 0.45], (0.0, 0.60)),
    ]
    fams = [("Statistical\n(cond. frequency)", "#8a97a6"),
            ("LightGBM\n(tabular ML)", "#2f8f83"),
            ("Causal GRU\n(ours)", "#2b5d8a")]

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
                     label="validation"),
               Patch(facecolor="#8a97a6", edgecolor="white", label="holdout")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=8.4,
               bbox_to_anchor=(0.995, 1.045), handletextpad=0.4, columnspacing=1.2)
    fig.suptitle("Multi-step structure prediction — statistical vs. tabular-ML vs. causal GRU "
                 "(P9d benchmark; identical data & splits)",
                 fontsize=10.5, x=0.008, ha="left", y=1.045, color=TXT)
    fig.text(0.008, -0.055,
             "GRU: shared causal GRU, 3 seeds (mean±std over seeds; protocol-selected topology variant). "
             "Statistical baseline = conditional empirical frequency table; on holdout all condition keys are unseen → global fallback. "
             "P9d v1 anchors: 13,754 train / 2,029 validation / 1,520 test / 1,380 holdout; holdout read once after freezing.",
             fontsize=7.8, color=GRAY)
    save(fig, "fig14_predictor_structure")


if __name__ == "__main__":
    fig13_single_step()
    fig14_structure()
    print("done")
