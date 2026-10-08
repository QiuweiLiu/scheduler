#!/usr/bin/env python3
"""Scheduling timeline figure from a real simulation trace (episode validation_000013).

Shows, for the first ~140 s: job arrivals, per-GPU node execution (colored by stage
role), cold model loads (hatched), evictions, and the conservative prefetch.

Run: MPLCONFIGDIR=/tmp/mpl-cache conda run -n print python scripts/make_timeline_figure.py
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
EPISODE_INDEX = 2
LANG = os.environ.get("FIG_LANG", "zh")


def T(zh: str, en: str) -> str:
    return en if LANG == "en" else zh

ROLE_COLORS = {
    "planner": "#2b5d8a",
    "videotool_spatial": "#2f8f83",
    "answer_generation": "#7d6aa8",
}
ROLE_LABELS = {
    "planner": T("规划 Planner", "Planner"),
    "videotool_spatial": T("空间工具 Spatial", "Spatial tool"),
    "answer_generation": T("回答 Answer", "Answer"),
}
LOAD_FACE = "#d9dde3"
PREFETCH = "#c98a2e"
EVICT = "#b3544a"
ARRIVAL = "#5a6672"


def load_episode():
    ext = json.loads(EXTENSION_CONFIG.read_text(encoding="utf-8"))
    templates = load_templates(PROJECTION, topology_view="causal_v3")
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    wanted = set(split["confirm"])
    by_id = {str(e.get("episode_id")): e for e in read_jsonl(EPISODES_FILE)}
    episodes = [by_id[eid] for eid in sorted(wanted) if eid in by_id]
    templates, episodes = apply_real_workload_profile_contract(templates, episodes, ext)
    stats = train_resource_stats({k: v for k, v in templates.items() if v.split == "train"})
    artifacts, _ = load_resource_v2_overlay(BASE_ARTIFACTS, F0_ARTIFACTS)
    return ext, templates, stats, artifacts, episodes[EPISODE_INDEX]


def node_role_map(templates):
    mapping = {}
    for template in templates.values():
        for node in template.nodes:
            mapping[node.node_id] = node.role or ""
    return mapping


def main() -> None:
    ext, templates, stats, artifacts, episode = load_episode()
    summary, events = simulate_episode(
        episode, templates, "pdrs_resident", train_stats=stats,
        future_artifacts=artifacts, future_horizon=FUTURE_HORIZON,
        extension_config=ext, collect_events=True,
    )
    roles = node_role_map(templates)
    arrivals, starts, finishes, loads, evicts, prefetches = [], [], [], [], [], []
    job_order = {}
    for event in events:
        kind = event.get("event_type")
        t = float(event.get("time_ms") or 0.0) / 1000.0
        job = str(event.get("job_instance_id") or "")
        if kind == "job_arrive" and not job.startswith("__prefetch__"):
            job_order.setdefault(job, len(job_order) + 1)
            arrivals.append((t, job_order[job]))
        elif kind == "node_start" and event.get("lane") == "gpu":
            starts.append((t, job, str(event.get("node_id"))))
        elif kind == "node_finish" and event.get("lane") == "gpu":
            finishes.append((t, job, str(event.get("node_id")), int(event.get("gpu_index", 0))))
        elif kind == "model_load_start":
            loads.append((t, float(event.get("load_ms") or 0.0), int(event.get("gpu_index", 0)),
                          str(event.get("model_id")), job))
        elif kind == "model_evict":
            evicts.append((t, int(event.get("gpu_index", 0)), list(event.get("models") or [])))
        elif kind == "prefetch_start":
            prefetches.append((t, float(event.get("load_ms") or 0.0), int(event.get("gpu_index", 0)),
                               str(event.get("model_id"))))

    start_by = {(job, node): t for t, job, node in starts}
    fig, ax = plt.subplots(figsize=(13.2, 6.4))
    y_arr, y_gpu = 2.95, {0: 1.95, 1: 1.05}

    # row bands
    for y in (y_arr, y_gpu[0], y_gpu[1]):
        ax.axhspan(y - 0.42, y + 0.42, color="#f7f8fa", zorder=0)
    ax.axhline(y_arr - 0.55, color="#d9dde3", lw=0.8, zorder=0)

    # node bars (with cold-load hatch) + prefetch spans
    for t_finish, job, node, gpu in finishes:
        t_start = start_by.get((job, node))
        if t_start is None or t_start > WINDOW_S:
            continue
        t_finish = min(t_finish, WINDOW_S)
        role = roles.get(node, "")
        color = ROLE_COLORS.get(role, "#8a97a6")
        load = next(((lt, lm) for lt, lm, lg, _m, lj in loads if lj == job and abs(lt - t_start) < 1e-6), None)
        y = y_gpu[gpu]
        if load:
            lt, lm = load
            load_end = lt + lm / 1000.0
            ax.add_patch(Rectangle((lt, y - 0.22), max(0.0, load_end - lt), 0.44,
                                   facecolor=LOAD_FACE, edgecolor="#8a97a6", hatch="///",
                                   lw=0.6, zorder=3))
            ax.add_patch(Rectangle((load_end, y - 0.22), max(0.0, t_finish - load_end), 0.44,
                                   facecolor=color, edgecolor="white", lw=0.5, zorder=3))
        else:
            ax.add_patch(Rectangle((t_start, y - 0.22), max(0.0, t_finish - t_start), 0.44,
                                   facecolor=color, edgecolor="white", lw=0.5, zorder=3))

    for t, dur, gpu, model in prefetches:
        if t > WINDOW_S:
            continue
        y = y_gpu[gpu]
        ax.add_patch(Rectangle((t, y + 0.24), dur / 1000.0, 0.16, facecolor=PREFETCH,
                               edgecolor="#8a6a20", lw=0.6, alpha=0.9, zorder=4))
        ax.text(t + dur / 2000.0, y + 0.62, f"{T('预取', 'prefetch')} {model.split('-')[0]}", ha="center", va="bottom",
                fontsize=8, color="#8a6a20", zorder=5)

    for t, gpu, models in evicts:
        if t > WINDOW_S:
            continue
        y = y_gpu[gpu]
        ax.plot([t], [y], marker="x", ms=8, mew=2.2, color=EVICT, zorder=6)
        short = ",".join(m.replace("Qwen3-VL-8B-Instruct", "8B").replace("Qwen3-4B", "4B")
                         .replace("Qwen2.5-VL-3B-Instruct", "3B") for m in models)
        ax.text(t, y - 0.34, f"{T('驱逐', 'evict')} {short}", ha="center", va="top", fontsize=8, color=EVICT, zorder=6)

    seen_times: dict[float, int] = {}
    for t, idx in arrivals:
        if t > WINDOW_S:
            continue
        slot = seen_times.get(round(t, 3), 0)
        seen_times[round(t, 3)] = slot + 1
        ax.plot([t], [y_arr], marker="v", ms=8, color=ARRIVAL, zorder=6)
        ax.text(t + (3.0 * slot), y_arr + 0.52, f"J{idx}", ha="center", va="bottom",
                fontsize=9.5, color=ARRIVAL, fontweight="bold", zorder=6)

    ax.set_xlim(0, WINDOW_S)
    ax.set_ylim(0.35, 3.85)
    ax.set_yticks([y_arr, y_gpu[0], y_gpu[1]])
    ax.set_yticklabels([T("任务到达", "Job arrivals"), "GPU 0", "GPU 1"], fontsize=12)
    ax.set_xlabel(T("时间(s,真实仿真轨迹:validation_000013)",
                    "Time (s) \u2014 real simulation trace: validation_000013"), fontsize=11)
    ax.grid(axis="x", color="#e6e9ee", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    # legend
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white") for c in ROLE_COLORS.values()]
    handles.append(Rectangle((0, 0), 1, 1, facecolor=LOAD_FACE, edgecolor="#8a97a6", hatch="///"))
    handles.append(Rectangle((0, 0), 1, 1, facecolor=PREFETCH, alpha=0.9))
    handles.append(plt.Line2D([0], [0], marker="x", color=EVICT, ls="", ms=8, mew=2.2))
    handles.append(plt.Line2D([0], [0], marker="v", color=ARRIVAL, ls="", ms=8))
    labels = [ROLE_LABELS[k] for k in ROLE_COLORS] + [
        T("冷加载(模型装载)", "Cold model load"),
        T("预取", "Prefetch"),
        T("驱逐", "Eviction"),
        T("任务到达", "Job arrival"),
    ]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              fontsize=9.5, frameon=False, ncol=4)

    # key-action callouts
    ax.annotate(T("J4 到达 → 显存不足:\n驱逐 8B → 冷加载 4B(4.0 s)",
                  "J4 arrives \u2192 memory pressure:\nevict 8B \u2192 cold-load 4B (4.0 s)"),
                xy=(60.5, y_gpu[0] + 0.25), xytext=(66, 2.55),
                fontsize=9, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#8a97a6", lw=1.0))
    ax.annotate(T("空闲窗口 → 提前预取 3B(3.9 s)",
                  "idle window \u2192 prefetch 3B early (3.9 s)"),
                xy=(112.1, y_gpu[1] + 0.42), xytext=(84, 0.62),
                fontsize=9, color="#8a6a20",
                arrowprops=dict(arrowstyle="->", color="#c98a2e", lw=1.0))

    fig.tight_layout()
    suffix = "_zh" if LANG == "zh" else ""
    fig.savefig(OUT / f"fig10_timeline{suffix}.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUT / f"fig10_timeline{suffix}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig10_timeline{suffix}(.svg/.png)")


if __name__ == "__main__":
    main()
