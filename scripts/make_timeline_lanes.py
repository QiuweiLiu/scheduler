#!/usr/bin/env python3
"""Timeline lanes v2: time axis on top; one lane per JOB (agent behavior with
model+action labels); then the two GPU lanes (device behavior).

Real trace validation_000013 (pdrs_resident), first 140 s.  Consecutive stages of
the same (gpu, job, role, model) are merged into one block.
Run: MPLCONFIGDIR=/tmp/mpl-cache [FIG_LANG=en] conda run -n print python scripts/make_timeline_lanes.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
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
PX_PER_SECOND = 1245.0 / 140.0
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
ROLE_SHORT = {
    "planner": T("规划", "plan"),
    "videotool_spatial": T("空间选取", "spatial"),
    "answer_generation": T("回答", "answer"),
    "videotool_temporal": T("时间", "temporal"),
    "videotool_generalist": T("通用", "general"),
}


def label_seconds(text: str, fontsize: float, px_per_second: float) -> float:
    """Approximate width of a label in seconds (latin ~0.55em, CJK ~1.0em)."""
    px = 0.0
    for ch in text:
        if ch == "·":
            px += fontsize * 0.45
        elif ord(ch) > 0x2E80:
            px += fontsize
        else:
            px += fontsize * 0.55
    return px / px_per_second


def merge_bars(bars):
    groups = defaultdict(list)
    for bar in bars:
        groups[(bar[2], bar[3], bar[4], bar[5])].append(list(bar))
    merged = []
    for items in groups.values():
        items.sort()
        out: list[list] = []
        for it in items:
            if out and it[0] - out[-1][1] <= 0.5:
                out[-1][1] = max(out[-1][1], it[1])
            else:
                out.append(it)
        merged.extend(tuple(b) for b in out)
    return sorted(merged)


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

    starts, bars, job_index = {}, [], {}
    for event in events:
        kind = event.get("event_type")
        t = float(event.get("time_ms") or 0.0) / 1000.0
        job = str(event.get("job_instance_id") or "")
        if kind == "job_arrive" and not job.startswith("__prefetch__"):
            job_index.setdefault(job, len(job_index) + 1)
        elif kind == "node_start" and event.get("lane") == "gpu":
            starts[(job, str(event.get("node_id")))] = t
        elif kind == "node_finish" and event.get("lane") == "gpu":
            start = starts.get((job, str(event.get("node_id"))))
            if start is None or start > WINDOW_S:
                continue
            bars.append((start, min(t, WINDOW_S), int(event.get("gpu_index", 0)),
                         job_index.get(job, 0), str(event.get("model_id")),
                         roles.get(str(event.get("node_id")), "")))
    bars = merge_bars(bars)

    jobs_present = sorted({b[3] for b in bars})
    n_jobs = len(jobs_present)
    fig, ax = plt.subplots(figsize=(13.6, 7.6))
    # y layout: jobs on top (one lane each), then GPU 0 / GPU 1
    job_y = {j: (n_jobs - 1 - i) * 0.85 + 1.9 for i, j in enumerate(jobs_present)}
    y_gpu = {0: 1.05, 1: 0.15}

    for y in list(job_y.values()) + list(y_gpu.values()):
        ax.axhspan(y - 0.30, y + 0.30, color="#f7f8fa", zorder=0)

    def overlaps_any(bar):
        s, f, g = bar[0], bar[1], bar[2]
        return any(o is not bar and o[2] == g and o[0] < f - 1e-9 and o[1] > s + 1e-9
                   for o in bars)

    # job lanes: model + action labels
    for start, finish, gpu, jidx, model, role in bars:
        y = job_y[jidx]
        color = MODEL_COLORS.get(model, "#8a97a6")
        width = max(0.4, finish - start)
        ax.add_patch(Rectangle((start, y - 0.20), width, 0.40, facecolor=color,
                               edgecolor="white", lw=0.5, zorder=3))
        label = f"{MODEL_SHORT.get(model, '?')}·{ROLE_SHORT.get(role, '')}"
        need = label_seconds(label, 7.4, PX_PER_SECOND) + 1.0
        short = MODEL_SHORT.get(model, "?")
        if width >= need:
            ax.text(start + width / 2, y, label, ha="center", va="center", fontsize=7.4,
                    color="white", fontweight="bold", zorder=4)
        elif width >= label_seconds(short, 7.0, PX_PER_SECOND) + 0.3:
            ax.text(start + width / 2, y, short, ha="center",
                    va="center", fontsize=7.0, color="white", fontweight="bold", zorder=4)

    # GPU lanes: job + action labels; co-located bars split into half-bars
    for bar in bars:
        start, finish, gpu, jidx, model, role = bar
        y = y_gpu[gpu]
        color = MODEL_COLORS.get(model, "#8a97a6")
        width = max(0.4, finish - start)
        co = overlaps_any(bar)
        if co:
            bottom = all(not (o is not bar and o[2] == gpu and o[0] < finish - 1e-9
                              and o[1] > start + 1e-9) or o[3] >= jidx for o in bars)
            cy = y - 0.105 if bottom else y + 0.105
            ax.add_patch(Rectangle((start, cy - 0.09), width, 0.18, facecolor=color,
                                   edgecolor="white", lw=0.5, zorder=3))
        else:
            cy = y
            ax.add_patch(Rectangle((start, y - 0.20), width, 0.40, facecolor=color,
                                   edgecolor="white", lw=0.5, zorder=3))
        label = f"J{jidx}·{ROLE_SHORT.get(role, '')}"
        need = label_seconds(label, 6.8, PX_PER_SECOND) + 1.0
        short = f"J{jidx}"
        if width >= need:
            ax.text(start + width / 2, cy, label, ha="center", va="center", fontsize=6.8,
                    color="white", fontweight="bold", zorder=4)
        elif width >= label_seconds(short, 7.0, PX_PER_SECOND) + 0.3:
            ax.text(start + width / 2, cy, short, ha="center", va="center", fontsize=7.0,
                    color="white", fontweight="bold", zorder=4)

    ax.set_xlim(0, WINDOW_S)
    ax.set_ylim(-0.25, job_y[jobs_present[0]] + 0.55)
    ticks = [job_y[j] for j in jobs_present] + [y_gpu[0], y_gpu[1]]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"J{j}" for j in jobs_present] + ["GPU 0", "GPU 1"], fontsize=11.5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel(T("时间(s)· 真实轨迹 validation_000013 · 上:每个任务的行为(模型·动作);下:两张 GPU 的行为",
                    "Time (s) · real trace validation_000013 · top: per-job behavior (model·action); "
                    "bottom: per-GPU behavior"),
                  fontsize=10, labelpad=8)
    ax.grid(axis="x", color="#e6e9ee", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("bottom", "right", "left"):
        ax.spines[side].set_visible(False)

    handles = [Rectangle((0, 0), 1, 1, facecolor=MODEL_COLORS[m], edgecolor="white")
               for m in ("Qwen3-VL-8B-Instruct", "Qwen3-4B", "Qwen2.5-VL-3B-Instruct")]
    labels = [T("8B 模型", "model 8B"), T("4B 模型", "model 4B"), T("3B 模型", "model 3B")]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.015),
              fontsize=9.5, frameon=False, ncol=3)

    fig.tight_layout()
    suffix = "_zh" if LANG == "zh" else ""
    fig.savefig(OUT / f"fig12_lanes{suffix}.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUT / f"fig12_lanes{suffix}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig12_lanes{suffix}")


if __name__ == "__main__":
    main()
