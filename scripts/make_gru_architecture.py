#!/usr/bin/env python3
"""fig15: deployed predictor architecture (shared causal GRU + multi-task heads).

Hand-drawn SVG (zh + en), rendered to PNG with cairosvg.
Run with the `print` conda env:
  MPLCONFIGDIR=/tmp/mpl-cache conda run -n print python scripts/make_gru_architecture.py
"""
from __future__ import annotations

from pathlib import Path

import cairosvg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_materials" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1f2933"
SUB = "#53616f"
FAINT = "#8a97a6"
BLUE = "#2b5d8a"
TEAL = "#2f8f83"
AMBER = "#c98a2e"
RED = "#b3544a"
ARROW = "#5a6672"
F_INPUT, S_INPUT = "#f4f6f9", "#a7b2bd"
F_EMB, S_EMB = "#e9eef4", "#7d93a9"
F_GRU, S_GRU = "#dfeaf7", "#2b5d8a"
F_REPR, S_REPR = "#e6f0ee", "#2f8f83"
F_HEADA, S_HEADA = "#e9eef4", "#7d93a9"
F_HEADB, S_HEADB = "#e6f0ee", "#2f8f83"
F_HEADR, S_HEADR = "#fbf2e2", "#c98a2e"

W, H = 1960, 930


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12.5, fill=INK, anchor="start", weight="normal", ff="body"):
    fam = FONT[ff]
    return (f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')


def box(x, y, w, h, title, items, fill, stroke, title_size=15, body_size=12.5,
        title_fill=None, item_dy=23, rx=10):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
           f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>']
    if title:
        out.append(text(x + 15, y + 27, title, size=title_size, fill=title_fill or INK, weight="600"))
    cy = y + 27 + item_dy
    for line in items:
        out.append(text(x + 15, cy, line, size=body_size, fill=SUB))
        cy += item_dy
    return "\n".join(out)


def arrow(x1, y1, x2, y2, color=ARROW, width=1.4, dashed=False, marker=True):
    dash = ' stroke-dasharray="5,4"' if dashed else ""
    m = ' marker-end="url(#arr)"' if marker else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width}"{dash}{m}/>')


def path(d, color=ARROW, width=1.4, marker=True):
    m = ' marker-end="url(#arr)"' if marker else ""
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{m}/>'


def render(lang: str, out_svg: Path, out_png: Path):
    global FONT
    if lang == "zh":
        FONT = {"title": "'STHeiti','PingFang SC','Arial'", "body": "'STHeiti','PingFang SC','Arial'"}
    else:
        FONT = {"title": "'Arial','Helvetica Neue'", "body": "'Arial','Helvetica Neue'"}

    T = {
        "zh": {
            "title": "未来预测器架构:共享因果 GRU + 多任务头",
            "subtitle": "部署组成:J3:seed11(结构/内容/行为)+ F0:seed11(资源头) · 冻结 · SHA 固定 · 仅 train 拟合",
            "freeze": "冻结区:部署模型",
            "hbox_t": "已执行前缀(≤ 64 步)",
            "hbox": ["事件类型 · 节点类型 · 角色", "原始动作 · 动作族 · 模型 ID"],
            "cbox_t": "当前节点上下文",
            "cbox": ["任务类型 · 领域 · 问题类型", "官方类别 · 子类 · 时序范围", "基线 · 模型栈 · 规划模型"],
            "cnote": ["未来事件 / 真值标签 / 机器表", "不进入任何输入"],
            "e1_t": "逐字段嵌入",
            "e1": ["6 字段 × 16 维,步内求和", "+ 位置嵌入(≤ 64)", "→ 步向量序列 x_1…x_T", "x [B, ≤64, 16]"],
            "e2_t": "上下文嵌入",
            "e2": ["9 字段 × 12 维,求和", "+ 任务模态 one-hot 投影"],
            "strip": "x_1 → x_2 → x_3 → … → x_T",
            "strip_note": "因果:只读已执行前缀",
            "gru_t": "单向因果 GRU",
            "gru": ["hidden = 128", "pack_padded · 取末步 h_T"],
            "repr_t": "共享表示",
            "repr": ["tanh(W_h·h_T + W_c·ctx)", "128 维"],
            "ha_t": "行级头(每决策点)",
            "ha": ["剩余步数分布(0–5 类)", "终止概率", "下一角色 · 下一动作族"],
            "hb_t": "槽位属性头(未来步 1…5 · 槽位条件)",
            "hb": ["节点类型 · 角色 · 动作族 · 模型类", "合并 · 重试 · 嵌套模型类"],
            "hr_t": "资源头(槽位条件)",
            "hr": ["runtime:16 桶质量均衡分布", "→ p50 / p90 / p95 · CVaR95", "load:发生概率 · 时长分位"],
            "attr": "属性特征条件",
            "pack_t": "冻结预测包(≤ 5 步未来)",
            "pack": ["结构/属性 + 模型分布", "runtime 分位(p50/p90/p95)", "load 概率 / 时长分位"],
            "sched_t": "调度器(本文方法)",
            "sched": ["F0 排序 + 驻留动作", "消费其中 7 项字段"],
            "dim_x": "x [B, T≤64, 16]",
            "dim_h": "h_T [B, 128]",
            "dim_r": "repr [B, 128]",
            "dim_o": "slots 1…5 · 类别分布",
        },
        "en": {
            "title": "Future predictor architecture — shared causal GRU with multi-task heads",
            "subtitle": "Deployed: J3:seed11 (structure/content/behavior) + F0:seed11 (resource head) · frozen · SHA-pinned · train-only fit",
            "freeze": "frozen zone: deployed model",
            "hbox_t": "Executed prefix (≤ 64 steps)",
            "hbox": ["event type · node type · role", "raw action · action family · model ID"],
            "cbox_t": "Current-node context",
            "cbox": ["task type · domain · question type", "official type · sub-category · scope", "baseline · model stack · planner model"],
            "cnote": ["future events / labels / machine tables", "never enter any input"],
            "e1_t": "Per-field embedding",
            "e1": ["6 fields × 16-d, summed per step", "+ position embedding (≤ 64)", "→ step vectors x_1…x_T", "x [B, ≤64, 16]"],
            "e2_t": "Context embedding",
            "e2": ["9 fields × 12-d, summed", "+ task-modality one-hot projection"],
            "strip": "x_1 → x_2 → x_3 → … → x_T",
            "strip_note": "causal: executed prefix only",
            "gru_t": "Unidirectional causal GRU",
            "gru": ["hidden = 128", "pack_padded · last step h_T"],
            "repr_t": "Shared representation",
            "repr": ["tanh(W_h·h_T + W_c·ctx)", "128-d"],
            "ha_t": "Row heads (per decision point)",
            "ha": ["remaining-length dist. (0–5)", "termination probability", "next role · next action family"],
            "hb_t": "Slot-attribute heads (future steps 1…5, slot-conditioned)",
            "hb": ["node type · role · action family · model class", "merged · retry · nested model class"],
            "hr_t": "Resource heads (slot-conditioned)",
            "hr": ["runtime: 16-bin mass-balanced dist.", "→ p50 / p90 / p95 · CVaR95", "load: occurrence prob. · duration quantiles"],
            "attr": "attribute features",
            "pack_t": "Frozen prediction pack (≤ 5 steps)",
            "pack": ["structure/attrs + model dist.", "runtime quantiles (p50/p90/p95)", "load prob. / duration quantiles"],
            "sched_t": "Scheduler (ours)",
            "sched": ["F0 ordering + residency actions", "consumes 7 of these fields"],
            "dim_x": "x [B, T≤64, 16]",
            "dim_h": "h_T [B, 128]",
            "dim_r": "repr [B, 128]",
            "dim_o": "slots 1…5 · class distributions",
        },
    }[lang]

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    s.append('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
             'markerHeight="7" orient="auto-start-reverse">'
             f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{ARROW}"/></marker></defs>')
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')

    # title + subtitle
    s.append(text(44, 54, T["title"], size=23, weight="600"))
    s.append(text(44, 84, T["subtitle"], size=13, fill=SUB))

    # frozen dashed boundary
    s.append(f'<rect x="322" y="126" width="1278" height="700" rx="16" fill="none" '
             f'stroke="{RED}" stroke-width="1.3" stroke-dasharray="7,5" opacity="0.75"/>')
    s.append(text(1596, 116, T["freeze"], size=12.5, fill=RED, anchor="end"))

    # ---- column 1: inputs ----
    s.append(box(44, 170, 258, 190, T["hbox_t"], T["hbox"], F_INPUT, S_INPUT))
    s.append(box(44, 470, 258, 168, T["cbox_t"], T["cbox"], F_INPUT, S_INPUT, item_dy=30))
    s.append(text(44, 692, T["cnote"][0], size=12, fill=RED))
    s.append(text(44, 712, T["cnote"][1], size=12, fill=RED))

    # ---- column 2: embeddings ----
    s.append(box(344, 186, 240, 182, T["e1_t"], T["e1"], F_EMB, S_EMB))
    s.append(box(344, 486, 240, 134, T["e2_t"], T["e2"], F_EMB, S_EMB, item_dy=28))

    # ---- column 3: causal strip + GRU ----
    s.append(f'<rect x="628" y="196" width="250" height="40" rx="8" fill="#f4f6f9" stroke="{S_INPUT}" stroke-width="1.1"/>')
    s.append(text(753, 221, T["strip"], size=11.5, fill=SUB, anchor="middle"))
    s.append(text(753, 256, T["strip_note"], size=11.5, fill=TEAL, anchor="middle"))
    s.append(box(628, 286, 250, 168, T["gru_t"], T["gru"], F_GRU, S_GRU, title_size=16, item_dy=30))

    # ---- column 4: shared repr ----
    s.append(box(938, 356, 190, 118, T["repr_t"], T["repr"], F_REPR, S_REPR, item_dy=30))

    # ---- column 5: heads ----
    s.append(box(1188, 170, 380, 150, T["ha_t"], T["ha"], F_HEADA, S_HEADA))
    s.append(box(1188, 366, 380, 128, T["hb_t"], T["hb"], F_HEADB, S_HEADB, title_size=13.5))
    s.append(box(1188, 566, 380, 158, T["hr_t"], T["hr"], F_HEADR, S_HEADR))

    # ---- column 6: pack + scheduler ----
    s.append(box(1652, 244, 264, 158, T["pack_t"], T["pack"], F_INPUT, S_INPUT))
    s.append(box(1652, 560, 264, 128, T["sched_t"], T["sched"], F_GRU, S_GRU))

    # ---- arrows ----
    s.append(arrow(302, 262, 344, 262))                      # history -> emb1
    s.append(arrow(302, 560, 344, 560))                      # context -> emb2
    s.append(arrow(584, 290, 628, 330))                      # emb1 -> GRU
    s.append(path("M 584 556 L 906 556 L 906 440 L 938 440"))  # ctx -> repr
    s.append(arrow(878, 370, 938, 396))                      # GRU -> repr
    # repr -> heads fan
    s.append(arrow(1128, 396, 1188, 262))
    s.append(arrow(1128, 412, 1188, 430))
    s.append(arrow(1128, 430, 1188, 620))
    # attribute conditioning B -> R
    s.append(arrow(1378, 494, 1378, 566, color=S_HEADR))
    s.append(text(1390, 535, T["attr"], size=11.5, fill=AMBER))
    # heads -> pack
    s.append(arrow(1568, 245, 1652, 290))
    s.append(arrow(1568, 430, 1652, 330))
    s.append(arrow(1568, 645, 1652, 372))
    # pack -> scheduler
    s.append(arrow(1784, 402, 1784, 560))

    s.append("</svg>")
    svg = "\n".join(s)
    out_svg.write_text(svg, encoding="utf-8")
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(out_png), scale=2.0)
    print("wrote", out_png.relative_to(ROOT), "+ svg")


if __name__ == "__main__":
    render("zh", OUT / "fig15_gru_architecture_zh.svg", OUT / "fig15_gru_architecture_zh.png")
    render("en", OUT / "fig15_gru_architecture.svg", OUT / "fig15_gru_architecture.png")
    print("done")
