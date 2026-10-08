#!/usr/bin/env python3
"""Timeline lanes: time axis on top, job arrivals / completions, then the two
GPUs' behavior over time (bars colored by model).

Real trace validation_000013 (pdrs_resident), first 140 s.
Run: MPLCONFIGDIR=/tmp/mpl-cache [FIG_LANG=en] conda run -n print python scripts/make_timeline_lanes.py
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
MODEL_LEGEND = {
    "Qwen3-VL-8B-Instruct": T("8B 模型", "model 8B"),
    "Qwen3-4B": T("4B 模型", "model 4B"),
    "Qwen2.5-VL-3B-Instruct": T("3B 模型", "model 3B"),
}
ROLE_SHORT = {
    "planner": T("规划", "plan"),
    "videotool_spatial": T("空间", "spatial"),
    "answer_generation": T("回答", "answer"),
    "videotool_temporal": T("时间", "temporal"),
    "videotool_generalist": T("通用", "gen"),
}
ARRIVAL = "#5a6672"
COMPLETE = "#4e7a3a"


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
    roles = {}
    for template in templates.values():
        for node in template.nodes:
            roles[node.node_id] = node.role or ""

    arrivals, completions, starts, bars = [], [], {}, []
    job_index: dict[str, int] = {}
    for event in events:
        kind = event.get("event_type")
        t = float(event.get("time_ms") or 0.0) / 1000.0
        job = str(event.get("job_instance_id") or "")
        if kind == "job_arrive" and not job.startswith("__prefetch__"):
            job_index.setdefault(job, len(job_index) + 1)
            arrivals.append((t, job_index[job]))
        elif kind == "job_finish" and not job.startswith("__prefetch__"):
            completions.append((t, job_index.get(job, 0)))
        elif kind == "node_start" and event.get("lane") == "gpu":
            starts[(job, str(event.get("node_id")))] = t
        elif kind == "node_finish" and event.get("lane") == "gpu":
            start = starts.get((job, str(event.get("node_id"))))
            if start is None or start > WINDOW_S:
                continue
            bars.append((start, min(t, WINDOW_S), int(event.get("gpu_index", 0)),
                         job_index.get(job, 0), str(event.get("model_id")),
                         roles.get(str(event.get("node_id")), "")))

    # merge consecutive bars of the same (gpu, job, role, model) with small gaps
    merged: list[list] = []
    for bar in sorted(bars):
        if merged:
            last = merged[-1]
            if (bar[2] == last[2] and bar[3] == last[3] and bar[4] == last[4]
                    and bar[5] == last[5] and bar[0] - last[1] <= 0.5):
                last[1] = max(last[1], bar[1])
                continue
        merged.append(list(bar))
    bars = [tuple(b) for b in merged]

    fig, ax = plt.subplots(figsize=(13.6, 5.0))
    y_arr, y_done, y_gpu = 4.30, 3.45, {0: 2.35, 1: 1.20}
    for y in (y_arr, y_done, y_gpu[0], y_gpu[1]):
        ax.axhspan(y - 0.34, y + 0.34, color="#f7f8fa", zorder=0)

    def overlaps_any(bar):
        s, f, g = bar[0], bar[1], bar[2]
        return any(o is not bar and o[2] == g and o[0] < f - 1e-9 and o[1] > s + 1e-9
                   for o in bars)

    for bar in bars:
        start, finish, gpu, jidx, model, role = bar
        y = y_gpu[gpu]
        color = MODEL_COLORS.get(model, "#8a97a6")
        width = max(0.4, finish - start)
        co = overlaps_any(bar)
        if co:
            bottom = all(not (o is not bar and o[2] == gpu and o[0] < finish - 1e-9
                              and o[1] > start + 1e-9) or o[3] >= jidx for o in bars)
            cy = y - 0.115 if bottom else y + 0.115
            ax.add_patch(Rectangle((start, cy - 0.10), width, 0.20, facecolor=color,
                                   edgecolor="white", lw=0.5, zorder=3))
        else:
            cy = y
            ax.add_patch(Rectangle((start, y - 0.22), width, 0.44, facecolor=color,
                                   edgecolor="white", lw=0.5, zorder=3))
        label = f"J{jidx}·{ROLE_SHORT.get(role, '')}"
        if width >= 5.5:
            ax.text(start + width / 2, cy, label, ha="center", va="center", fontsize=7.8,
                    color="white", fontweight="bold", zorder=4)
        elif width >= 3.0:
            ax.text(start + width / 2, cy, f"J{jidx}", ha="center", va="center", fontsize=7.2,
                    color="white", fontweight="bold", zorder=4)

    for t, idx in arrivals:
        if t > WINDOW_S:
            continue
        ax.plot([t], [y_arr], marker="v", ms=8, color=ARRIVAL, zorder=6)
        ax.text(t, y_arr + 0.40, f"J{idx}", ha="center", va="bottom", fontsize=9,
                color=ARRIVAL, fontweight="bold", zorder=6)

    used_times: list[float] = []
    for t, idx in completions:
        if t > WINDOW_S:
            continue
        slot = sum(1 for u in used_times if abs(u - t) < 12.0)
        used_times.append(t)
        ax.plot([t], [y_done], marker="P", ms=8, color=COMPLETE, zorder=6)
        ax.text(t, y_done - 0.42 - 0.30 * (slot % 2), f"J{idx} {T('完成', 'done')}",
                ha="center", va="top", fontsize=8.5, color=COMPLETE, zorder=6)

    ax.set_xlim(0, WINDOW_S)
    ax.set_ylim(0.55, 4.95)
    ax.set_yticks([y_arr, y_done, y_gpu[0], y_gpu[1]])
    ax.set_yticklabels([T("任务到达", "Job arrivals"), T("任务完成", "Job completions"),
                        "GPU 0", "GPU 1"], fontsize=12)
    # time axis on top
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel(T("时间(s)· 真实轨迹 validation_000013", "Time (s) · real trace validation_000013"),
                  fontsize=10.5, labelpad=8)
    ax.grid(axis="x", color="#e6e9ee", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("bottom", "right", "left"):
        ax.spines[side].set_visible(False)

    handles = [Rectangle((0, 0), 1, 1, facecolor=MODEL_COLORS[m], edgecolor="white")
               for m in ("Qwen3-VL-8B-Instruct", "Qwen3-4B", "Qwen2.5-VL-3B-Instruct")]
    handles.append(plt.Line2D([0], [0], marker="v", color=ARRIVAL, ls="", ms=8))
    handles.append(plt.Line2D([0], [0], marker="P", color=COMPLETE, ls="", ms=8))
    labels = [MODEL_LEGEND[m] for m in ("Qwen3-VL-8B-Instruct", "Qwen3-4B",
                                        "Qwen2.5-VL-3B-Instruct")]
    labels += [T("任务到达", "job arrival"), T("任务完成", "job completion")]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              fontsize=9.5, frameon=False, ncol=5)

    fig.tight_layout()
    suffix = "_zh" if LANG == "zh" else ""
    fig.savefig(OUT / f"fig12_lanes{suffix}.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUT / f"fig12_lanes{suffix}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig12_lanes{suffix}")


if __name__ == "__main__":
    main()
