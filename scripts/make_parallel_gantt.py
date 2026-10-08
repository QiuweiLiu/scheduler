#!/usr/bin/env python3
"""Parallel-scheduling Gantt: two GPU lanes, bars colored by MODEL, jobs labeled.

Window 0-140 s of the real trace validation_000013 (pdrs_resident).  The point:
multiple jobs run concurrently on the two GPUs with different models resident
(8B / 4B / 3B), including the 8B->4B switch at 60.5 s and the 3B prefetch at 112 s.

Run: MPLCONFIGDIR=/tmp/mpl-cache [FIG_LANG=en] conda run -n print python scripts/make_parallel_gantt.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402

font_manager.fontManager.addfont("/System/Library/Fonts/STHeiti Medium.ttc")
matplotlib.rcParams["font.sans-serif"] = ["Heiti TC", "Arial", "DejaVu Sans"]
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from residency_comparison import (  # noqa: E402
    BASE_ARTIFACTS, EPISODES_FILE, EXTENSION_CONFIG, F0_ARTIFACTS, FUTURE_HORIZON,
    PROJECTION, SPLIT_MANIFEST,
)
from tracing.analysis.profile_contract import apply_real_workload_profile_contract  # noqa: E402
from tracing.analysis.workload_v02_simulator import (  # noqa: E402
    load_resource_v2_overlay, load_templates, read_jsonl, simulate_episode,
    train_resource_stats,
)

OUT = ROOT / "outputs" / "report_materials" / "figures"
WINDOW_S = 140.0
LANG = os.environ.get("FIG_LANG", "zh")


def T(zh: str, en: str) -> str:
    return en if LANG == "en" else zh


MODEL_COLORS = {
    "Qwen3-VL-8B-Instruct": "#2b5d8a",
    "Qwen3-4B": "#2f8f83",
    "Qwen2.5-VL-3B-Instruct": "#c98a2e",
    "yolo11x.pt": "#b3544a",
}
MODEL_SHORT = {
    "Qwen3-VL-8B-Instruct": "8B",
    "Qwen3-4B": "4B",
    "Qwen2.5-VL-3B-Instruct": "3B",
    "yolo11x.pt": "YOLO",
}


def main() -> None:
    ext = json.loads(EXTENSION_CONFIG.read_text(encoding="utf-8"))
    templates = load_templates(PROJECTION, topology_view="causal_v3")
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    wanted = set(split["confirm"])
    by_id = {str(e.get("episode_id")): e for e in read_jsonl(EPISODES_FILE)}
    episodes = [by_id[eid] for eid in sorted(wanted) if eid in by_id]
    templates, episodes = apply_real_workload_profile_contract(templates, episodes, ext)
    stats = train_resource_stats({k: v for k, v in templates.items() if v.split == "train"})
    artifacts, _ = load_resource_v2_overlay(BASE_ARTIFACTS, F0_ARTIFACTS)
    episode = episodes[2]
    _summary, events = simulate_episode(
        episode, templates, "pdrs_resident", train_stats=stats,
        future_artifacts=artifacts, future_horizon=FUTURE_HORIZON,
        extension_config=ext, collect_events=True,
    )
    arrivals, starts, bars, prefetches = [], {}, [], []
    job_index: dict[str, int] = {}
    for event in events:
        kind = event.get("event_type")
        t = float(event.get("time_ms") or 0.0) / 1000.0
        job = str(event.get("job_instance_id") or "")
        if kind == "job_arrive" and not job.startswith("__prefetch__"):
            job_index.setdefault(job, len(job_index) + 1)
            arrivals.append((t, job_index[job]))
        elif kind == "node_start" and event.get("lane") == "gpu":
            starts[(job, str(event.get("node_id")))] = t
        elif kind == "node_finish" and event.get("lane") == "gpu":
            start = starts.get((job, str(event.get("node_id"))))
            if start is None or start > WINDOW_S:
                continue
            bars.append((start, min(t, WINDOW_S), int(event.get("gpu_index", 0)),
                         job_index.get(job, 0), str(event.get("model_id"))))
        elif kind == "prefetch_start":
            prefetches.append((t, float(event.get("load_ms") or 0.0),
                               int(event.get("gpu_index", 0)), str(event.get("model_id"))))

    fig, ax = plt.subplots(figsize=(13.6, 4.6))
    y_arr, y_gpu = 2.62, {0: 1.70, 1: 0.85}
    for y in (y_arr, y_gpu[0], y_gpu[1]):
        ax.axhspan(y - 0.30, y + 0.30, color="#f7f8fa", zorder=0)

    # co-located (overlapping) bars are split into stacked half-bars; others stay full height
    def overlaps_any(bar, others):
        s, f, g, _j, _m = bar
        return any(o[2] == g and o[0] < f - 1e-9 and o[1] > s + 1e-9 and o is not bar
                   for o in others)

    for bar in bars:
        start, finish, gpu, jidx, model = bar
        y = y_gpu[gpu]
        color = MODEL_COLORS.get(model, "#8a97a6")
        width = max(0.35, finish - start)
        co = overlaps_any(bar, bars)
        if not co:
            ax.add_patch(Rectangle((start, y - 0.15), width, 0.30, facecolor=color,
                                   edgecolor="white", lw=0.5, zorder=3))
            label_y, label_h = y, 0.30
        else:
            # smaller job index on the bottom half, larger on the top half
            bottom = all(not (o[2] == gpu and o[0] < finish - 1e-9 and o[1] > start + 1e-9)
                         or o[3] >= jidx for o in bars if o is not bar)
            if bottom:
                ax.add_patch(Rectangle((start, y - 0.155), width, 0.145, facecolor=color,
                                       edgecolor="white", lw=0.5, zorder=3))
                label_y = y - 0.0825
            else:
                ax.add_patch(Rectangle((start, y + 0.010), width, 0.145, facecolor=color,
                                       edgecolor="white", lw=0.5, zorder=3))
                label_y = y + 0.0825
            label_h = 0.145
        if width >= 3.2:
            ax.text(start + width / 2, label_y, f"J{jidx}", ha="center", va="center",
                    fontsize=7.8 if co else 8.5, color="white", fontweight="bold", zorder=4)

    for t, dur, gpu, model in prefetches:
        if t > WINDOW_S:
            continue
        y = y_gpu[gpu]
        ax.add_patch(Rectangle((t, y + 0.22), dur / 1000.0, 0.10, facecolor="#c98a2e",
                               edgecolor="#8a6a20", lw=0.5, alpha=0.95, zorder=4))

    for t, idx in arrivals:
        if t > WINDOW_S:
            continue
        ax.plot([t], [y_arr], marker="v", ms=7.5, color="#5a6672", zorder=6)
        ax.text(t, y_arr + 0.36, f"J{idx}", ha="center", va="bottom", fontsize=9,
                color="#5a6672", fontweight="bold", zorder=6)

    # annotations
    ax.annotate(T("J2 到达 → 派发到空闲 GPU1:两卡开始并行",
                  "J2 arrives → dispatched to idle GPU1: both GPUs busy"),
                xy=(24.2, y_gpu[1] + 0.2), xytext=(6, 0.28),
                fontsize=9, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#8a97a6", lw=1.0))
    ax.annotate(T("J4 到达 → 显存压力:驱逐 8B → 加载 4B(等待 4 s)",
                  "J4 arrives → memory pressure: evict 8B → load 4B (4 s wait)"),
                xy=(61.0, y_gpu[0] - 0.2), xytext=(56, 2.05),
                fontsize=9, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#8a97a6", lw=1.0))
    ax.annotate(T("两卡并行 · 不同模型:GPU0 跑 8B,GPU1 跑 4B",
                  "Both GPUs busy with DIFFERENT models: 8B on GPU0, 4B on GPU1"),
                xy=(87, y_gpu[0] + 0.2), xytext=(99, 2.32),
                fontsize=9, color="#2b5d8a",
                arrowprops=dict(arrowstyle="->", color="#8a97a6", lw=1.0))
    ax.annotate(T("空闲窗口 → 预取 3B:后续回答零等待",
                  "idle window → prefetch 3B: later answer has no wait"),
                xy=(114, y_gpu[1] + 0.34), xytext=(96, 0.28),
                fontsize=9, color="#8a6a20",
                arrowprops=dict(arrowstyle="->", color="#c98a2e", lw=1.0))

    ax.set_xlim(0, WINDOW_S)
    ax.set_ylim(0.25, 3.05)
    ax.set_yticks([y_arr, y_gpu[0], y_gpu[1]])
    ax.set_yticklabels([T("任务到达", "Job arrivals"), "GPU 0", "GPU 1"], fontsize=12)
    ax.set_xlabel(T("时间(s)· 真实仿真轨迹 validation_000013 · 每条 = 一个工作流阶段,颜色 = 模型",
                    "Time (s) · real trace validation_000013 · each bar = one workflow stage, color = model"),
                  fontsize=10.5)
    ax.grid(axis="x", color="#e6e9ee", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white")
               for c in (MODEL_COLORS["Qwen3-VL-8B-Instruct"], MODEL_COLORS["Qwen3-4B"],
                         MODEL_COLORS["Qwen2.5-VL-3B-Instruct"])]
    handles.append(plt.Line2D([0], [0], marker="v", color="#5a6672", ls="", ms=7.5))
    handles.append(Rectangle((0, 0), 1, 1, facecolor="#c98a2e", alpha=0.95))
    labels = [T("8B 模型", "model 8B"), T("4B 模型", "model 4B"), T("3B 模型", "model 3B"),
              T("任务到达", "job arrival"), T("预取", "prefetch")]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              fontsize=9.5, frameon=False, ncol=5)

    fig.tight_layout()
    suffix = "_zh" if LANG == "zh" else ""
    fig.savefig(OUT / f"fig10_parallel{suffix}.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUT / f"fig10_parallel{suffix}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig10_parallel{suffix}")


if __name__ == "__main__":
    main()
