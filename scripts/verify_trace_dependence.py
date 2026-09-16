#!/usr/bin/env python3
"""T1/T2/T3: shared-slowdown and tail co-movement in real workflow traces (v2).

Implements the review fixes:
  P0-1 MixedLM with explicit re_formula="1" (REML) as the primary estimator.
  P0-2 closed-form moments reported with and without the boundary clamp; the
       primary T1 estimate uses fixed-effect residuals.
  P0-3 direct within-run covariance diagnostic (allows negative values).
  P0-4 fail-closed video parser.
  P0-5 cluster bootstrap suffixes video AND run ids per draw.
  P0-6 T3 uses per-pair lift with PIT-normalised events inside
       (type, stack, baseline) cells.
Plus D1 position adjustment, D2 count-cost coupling, D3 first/subsequent split,
and min_cell sensitivity 1/2/3.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

VIDEO_RE = re.compile(
    r"^r7_s_(?:train|val)_(?P<video>.+?)_(?P<stack>stack_[ab])_(?P<baseline>langgraph_react|star)$"
)
FIXED_BASE = "y ~ C(type) + C(stack) + C(baseline) + C(stack):C(baseline)"
FIXED_RICH = (
    "y ~ C(type) * C(stack) * C(baseline) + C(type):pos_norm + C(type):pos_norm2 + log_run_len"
)


def load_steps(templates_path: Path, type_field: str, status_mode: str) -> pd.DataFrame:
    rows: list[dict] = []
    for line in templates_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        template = json.loads(line)
        base_task = str(template.get("base_task_id") or "")
        match = VIDEO_RE.match(base_task)
        if match is None:
            raise ValueError(f"unparseable base_task_id: {base_task!r}")
        video = match.group("video")
        stack = str(template.get("model_stack_id"))
        baseline = str(template.get("baseline"))
        nodes = template.get("nodes") or []
        run_len = max(1, len(nodes))
        seen: dict[str, int] = {}
        for position, node in enumerate(nodes):
            status = str(node.get("status") or "")
            if status_mode == "success" and status != "success":
                continue
            runtime = float(node.get("runtime_ms") or 0.0)
            if runtime <= 0.0:
                continue
            node_type = str(node.get(type_field) or "unknown")
            seen[node_type] = seen.get(node_type, 0) + 1
            rows.append(
                {
                    "video": video,
                    "run_id": str(template.get("run_id")),
                    "stack": stack,
                    "baseline": baseline,
                    "type": node_type,
                    "lane": str(node.get("execution_lane") or "unknown"),
                    "status": status,
                    "runtime_ms": runtime,
                    "y": float(np.log1p(runtime)),
                    "pos_norm": position / run_len,
                    "pos_norm2": (position / run_len) ** 2,
                    "log_run_len": float(np.log1p(run_len)),
                    "occurrence_rank": seen[node_type],
                    "is_first": seen[node_type] == 1,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- T1
def variance_components(frame: pd.DataFrame) -> dict:
    """Nested moment estimator for step < run < video; raw and clamped values."""

    run_means = frame.groupby("run_id")["y"].mean()
    run_sizes = frame.groupby("run_id")["y"].size()
    run_vars = frame.groupby("run_id")["y"].var(ddof=1).fillna(0.0)
    run_video = frame.groupby("run_id")["video"].first()
    video_list = run_video.unique()
    grand = frame["y"].mean()

    ss_between_runs = 0.0
    df_between_runs = 0
    for video_id in video_list:
        keys = run_video[run_video == video_id].index
        means = run_means.loc[keys].to_numpy()
        weights = run_sizes.loc[keys].to_numpy(dtype=float)
        video_mean = float((means * weights).sum() / weights.sum())
        ss_between_runs += float((weights * (means - video_mean) ** 2).sum())
        df_between_runs += len(keys) - 1

    ss_within = float((run_vars * (run_sizes - 1)).sum())
    df_within = int((run_sizes - 1).sum())
    ms_between_runs = ss_between_runs / max(1, df_between_runs)
    ms_within = ss_within / max(1, df_within)

    sizes = run_sizes.to_numpy(dtype=float)
    n_harmonic = float(len(sizes) / np.sum(1.0 / sizes))
    sigma_run2_raw = (ms_between_runs - ms_within) / n_harmonic

    video_means = []
    video_weights = []
    for video_id in video_list:
        keys = run_video[run_video == video_id].index
        weights = run_sizes.loc[keys].to_numpy(dtype=float)
        means = run_means.loc[keys].to_numpy()
        video_means.append(float((means * weights).sum() / weights.sum()))
        video_weights.append(float(weights.sum()))
    video_means_arr = np.array(video_means)
    video_weights_arr = np.array(video_weights)
    ss_between_videos = float((video_weights_arr * (video_means_arr - grand) ** 2).sum())
    df_between_videos = len(video_means_arr) - 1
    ms_between_videos = ss_between_videos / max(1, df_between_videos)
    ms_between_runs_pooled = ss_between_runs / max(1, len(run_means) - len(video_list))
    n_video_harmonic = float(len(video_weights_arr) / np.sum(1.0 / video_weights_arr))
    sigma_video2_raw = (ms_between_videos - ms_between_runs_pooled) / n_video_harmonic

    sigma_eps2 = ms_within
    sigma_run2 = max(0.0, sigma_run2_raw)
    sigma_video2 = max(0.0, sigma_video2_raw)
    total = sigma_video2 + sigma_run2 + sigma_eps2
    return {
        "sigma_video2_raw": float(sigma_video2_raw),
        "sigma_run2_raw": float(sigma_run2_raw),
        "sigma_video2": float(sigma_video2),
        "sigma_run2": float(sigma_run2),
        "sigma_eps2": float(sigma_eps2),
        "total": float(total),
        "icc_video": float(sigma_video2 / total) if total else float("nan"),
        "icc_run": float(sigma_run2 / total) if total else float("nan"),
        "var_run_means": float(run_means.var(ddof=1)),
        "expected_var_run_means_if_noise": float(ms_within / n_harmonic),
    }


def within_run_covariance(frame: pd.DataFrame) -> dict:
    """Direct average within-run pairwise residual covariance (can be negative)."""

    total = 0.0
    denom = 0
    for _, group in frame.groupby("run_id", sort=False):
        values = group["y"].to_numpy()
        n = len(values)
        if n < 2:
            continue
        s = values.sum()
        total += s * s - float((values**2).sum())
        denom += n * (n - 1)
    covariance = total / denom if denom else float("nan")
    return {"within_run_covariance": float(covariance), "pairs": int(denom)}


def mixed_effects_icc(frame: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            FIXED_BASE,
            data=frame,
            groups="video",
            re_formula="1",
            vc_formula={"run": "0 + C(run_id)"},
        )
        fit = model.fit(reml=True, method="lbfgs", maxiter=300)
    sigma_video2 = float(fit.cov_re.iloc[0, 0]) if fit.cov_re.size else 0.0
    sigma_run2 = float(fit.vcomp[0]) if len(fit.vcomp) else 0.0
    sigma_eps2 = float(fit.scale)
    total = sigma_video2 + sigma_run2 + sigma_eps2
    return {
        "sigma_video2": sigma_video2,
        "sigma_run2": sigma_run2,
        "sigma_eps2": sigma_eps2,
        "icc_video": sigma_video2 / total if total else float("nan"),
        "icc_run": sigma_run2 / total if total else float("nan"),
        "converged": bool(fit.converged),
    }


# --------------------------------------------------------------------------- residuals
def residuals(frame: pd.DataFrame, design: str = FIXED_BASE, drop_video: bool = False) -> pd.Series:
    import statsmodels.formula.api as smf

    model = smf.ols(design, data=frame).fit()
    resid = pd.Series(model.resid.to_numpy(), index=frame.index)
    if drop_video:
        resid = resid - resid.groupby(frame["video"]).transform("mean")
    return resid


def run_type_cells(frame: pd.DataFrame, resid: pd.Series, min_cell: int, subset: pd.Series | None = None) -> pd.DataFrame:
    work = frame.copy()
    work["resid"] = resid
    if subset is not None:
        work = work[subset.to_numpy()]
    cells = (
        work.groupby(["video", "run_id", "stack", "baseline", "type"], sort=False)["resid"]
        .agg(["mean", "count", "median"])
        .reset_index()
    )
    return cells[cells["count"] >= min_cell].reset_index(drop=True)


def pair_spearman(cells: pd.DataFrame, min_common: int) -> list[dict]:
    types = sorted(cells["type"].unique())
    by_type = {t: cells[cells["type"] == t].set_index("run_id")["mean"] for t in types}
    out = []
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            a, b = by_type[types[i]], by_type[types[j]]
            common = a.index.intersection(b.index)
            if len(common) < min_common:
                continue
            rho, pvalue = stats.spearmanr(a.loc[common], b.loc[common])
            out.append(
                {"type_a": types[i], "type_b": types[j], "n_common": int(len(common)), "spearman": float(rho), "pvalue": float(pvalue)}
            )
    return out


def pooled_spearman(pairs: list[dict]) -> float:
    if not pairs:
        return float("nan")
    weights = np.array([p["n_common"] for p in pairs], dtype=float)
    values = np.array([p["spearman"] for p in pairs], dtype=float)
    return float((weights * values).sum() / weights.sum())


def pit_values(cells: pd.DataFrame) -> pd.Series:
    """Empirical PIT inside (type, stack, baseline) cells."""

    return cells.groupby(["type", "stack", "baseline"], sort=False)["mean"].rank(pct=True)


def tail_lift(cells: pd.DataFrame, q: float, min_common: int) -> dict:
    types = sorted(cells["type"].unique())
    pit = pit_values(cells)
    work = cells.copy()
    work["pit"] = pit
    by_type = {t: work[work["type"] == t].set_index("run_id")["pit"] for t in types}
    pairs = []
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            a, b = by_type[types[i]], by_type[types[j]]
            common = a.index.intersection(b.index)
            if len(common) < min_common:
                continue
            av = a.loc[common].to_numpy()
            bv = b.loc[common].to_numpy()
            ea = av > q
            eb = bv > q
            p_a = float(ea.mean())
            p_b = float(eb.mean())
            p_both = float((ea & eb).mean())
            lift = p_both / (p_a * p_b) if p_a > 0 and p_b > 0 else float("nan")
            pairs.append(
                {
                    "type_a": types[i],
                    "type_b": types[j],
                    "n_common": int(len(common)),
                    "p_a": p_a,
                    "p_b": p_b,
                    "p_both": p_both,
                    "lift": lift,
                }
            )
    if not pairs:
        return {"q": q, "pairs": [], "pooled_lift": float("nan")}
    weights = np.array([p["n_common"] for p in pairs], dtype=float)
    lifts = np.array([p["lift"] for p in pairs], dtype=float)
    valid = np.isfinite(lifts)
    pooled = float((weights[valid] * lifts[valid]).sum() / weights[valid].sum()) if valid.any() else float("nan")
    return {"q": q, "pairs": pairs, "pooled_lift": pooled}


def cluster_bootstrap_blocks(blocks: dict[str, pd.DataFrame], statistic, B: int, seed: int = 20260916) -> dict:
    rng = np.random.default_rng(seed)
    keys = list(blocks)
    values = []
    for _ in range(B):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        parts = []
        for draw_index, key in enumerate(sampled):
            part = blocks[key].copy()
            part["video"] = part["video"] + f"#{draw_index}"
            part["run_id"] = part["run_id"] + f"#{draw_index}"
            parts.append(part)
        sample = pd.concat(parts, ignore_index=True)
        try:
            value = statistic(sample)
        except Exception:  # noqa: BLE001
            value = None
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if not values:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "B": 0}
    return {
        "mean": float(np.mean(values)),
        "lo": float(np.percentile(values, 2.5)),
        "hi": float(np.percentile(values, 97.5)),
        "B": len(values),
    }


def permutation_lift(cells: pd.DataFrame, q: float, min_common: int, n_perm: int, seed: int = 20260916) -> dict:
    """Two-sided permutation test on the pooled lift.

    The observed tail co-movement may be positive or negative, so report the
    upper-tail, lower-tail and two-sided p-values instead of only P(null>=obs).
    """

    rng = np.random.default_rng(seed)
    observed = tail_lift(cells, q, min_common)["pooled_lift"]
    null = []
    for _ in range(n_perm):
        shuffled = cells.copy()
        for _, block in shuffled.groupby(["stack", "baseline", "type"], sort=False):
            idx = block.index.to_numpy()
            shuffled.loc[idx, "mean"] = block["mean"].to_numpy()[rng.permutation(len(idx))]
        value = tail_lift(shuffled, q, min_common)["pooled_lift"]
        if np.isfinite(value):
            null.append(float(value))
    null_arr = np.array(null) if null else np.array([float("nan")])
    if len(null) and np.isfinite(observed):
        # (count + 1) / (B + 1) keeps the p-value strictly positive and unbiased
        # for a finite Monte-Carlo null.
        count_ge = int(np.sum(null_arr >= observed))
        count_le = int(np.sum(null_arr <= observed))
        p_upper = (count_ge + 1) / (len(null) + 1)
        p_lower = (count_le + 1) / (len(null) + 1)
        p_two = float(min(1.0, 2.0 * min(p_upper, p_lower)))
    else:
        p_upper = p_lower = p_two = float("nan")
    return {
        "observed": observed,
        "null_mean": float(np.nanmean(null_arr)),
        "p_upper": p_upper,
        "p_lower": p_lower,
        "p_two_sided": p_two,
        "n_perm": len(null),
    }


def count_cost_coupling(cells: pd.DataFrame) -> dict:
    if cells.empty:
        return {"pearson": float("nan"), "spearman": float("nan")}
    pearson = float(np.corrcoef(cells["count"].to_numpy(dtype=float), cells["mean"].to_numpy(dtype=float))[0, 1])
    spearman = float(stats.spearmanr(cells["count"], cells["mean"]).statistic)
    return {"pearson": pearson, "spearman": spearman, "cells": int(len(cells))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--type-field", default="model_id")
    parser.add_argument("--status-mode", default="success", choices=("success", "all"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-co", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--min-common", type=int, default=40)
    args = parser.parse_args()

    frame = load_steps(args.templates, args.type_field, args.status_mode)
    blocks = {video: part for video, part in frame.groupby("video", sort=False)}
    report: dict = {
        "dataset": {
            "steps": int(len(frame)),
            "runs": int(frame["run_id"].nunique()),
            "videos": int(frame["video"].nunique()),
            "types": sorted(frame["type"].unique()),
            "type_field": args.type_field,
            "status_mode": args.status_mode,
        }
    }

    # T1 -------------------------------------------------------------------
    base_resid = residuals(frame, FIXED_BASE, drop_video=False)
    resid_frame = frame.copy()
    resid_frame["y"] = base_resid
    rich_resid = residuals(frame, FIXED_RICH, drop_video=False)
    rich_frame = frame.copy()
    rich_frame["y"] = rich_resid
    t1 = {
        "closed_form_on_residuals": variance_components(resid_frame),
        "closed_form_on_residuals_rich_fixed": variance_components(rich_frame),
        "within_run_covariance": within_run_covariance(resid_frame),
        "within_run_covariance_rich_fixed": within_run_covariance(rich_frame),
        "mixedlm_base": mixed_effects_icc(frame),
    }
    t1["icc_run_cluster_bootstrap"] = cluster_bootstrap_blocks(
        blocks,
        lambda d: variance_components(
            d.assign(y=residuals(d, FIXED_BASE, drop_video=False))
        )["icc_run"],
        args.bootstrap,
    )
    # The clamped ICC can sit at the boundary for every replicate; bootstrap the
    # untruncated run-variance component so the uncertainty is visible.
    t1["sigma_run2_raw_cluster_bootstrap"] = cluster_bootstrap_blocks(
        blocks,
        lambda d: variance_components(
            d.assign(y=residuals(d, FIXED_BASE, drop_video=False))
        )["sigma_run2_raw"],
        args.bootstrap,
    )
    report["T1"] = t1

    # T2 -------------------------------------------------------------------
    report["T2"] = {}
    for min_cell in (1, 2, 3):
        for label, design, drop_video in (
            ("A_total", FIXED_BASE, False),
            ("B_conditional_on_video", FIXED_BASE, True),
            ("C_position_adjusted", FIXED_RICH, False),
        ):
            if label == "C_position_adjusted" and min_cell != 1:
                continue
            resid = residuals(frame, design, drop_video=drop_video)
            cells = run_type_cells(frame, resid, min_cell)
            pairs = pair_spearman(cells, args.min_common)
            report["T2"][f"{label}_min{min_cell}"] = {
                "pooled_spearman": pooled_spearman(pairs),
                "pairs": pairs,
                "count_cost_coupling": count_cost_coupling(cells),
            }
    # D3 first vs subsequent (with video-cluster bootstrap, since the point
    # estimates alone cannot separate a real effect from composition noise)
    for label, keep_first in (("first_occurrence", True), ("subsequent", False)):
        cells = run_type_cells(frame, base_resid, 1, subset=(frame["is_first"] if keep_first else ~frame["is_first"]))
        entry = {
            "pooled_spearman": pooled_spearman(pair_spearman(cells, args.min_common)),
            "cells": int(len(cells)),
        }
        entry["bootstrap"] = cluster_bootstrap_blocks(
            blocks,
            lambda d, k=keep_first: pooled_spearman(
                pair_spearman(
                    run_type_cells(d, residuals(d, FIXED_BASE, False), 1, subset=(d["is_first"] if k else ~d["is_first"])),
                    args.min_common,
                )
            ),
            args.bootstrap_co,
        )
        report["T2"][f"D3_{label}"] = entry
    # D2 count-cost coupling (already per variant above) plus explicit correlation
    report["T2"]["D2_count_cost"] = count_cost_coupling(run_type_cells(frame, base_resid, 1))

    # T3 -------------------------------------------------------------------
    report["T3"] = {}
    for min_cell in (1, 2, 3):
        cells = run_type_cells(frame, base_resid, min_cell)
        for q in (0.8, 0.9, 0.95):
            result = tail_lift(cells, q, args.min_common)
            entry = {"pooled_lift": result["pooled_lift"], "pairs": result["pairs"]}
            if min_cell == 1:
                entry["permutation"] = permutation_lift(cells, q, args.min_common, args.permutations)
                entry["bootstrap"] = cluster_bootstrap_blocks(
                    blocks,
                    lambda d, qq=q: tail_lift(run_type_cells(d, residuals(d, FIXED_BASE, False), 1), qq, args.min_common)["pooled_lift"],
                    args.bootstrap_co,
                )
            report["T3"][f"q{int(q * 100)}_min{min_cell}"] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    summary = {
        "steps": report["dataset"]["steps"],
        "T1_closed_form_on_residuals": {k: round(v, 4) for k, v in t1["closed_form_on_residuals"].items()},
        "T1_closed_form_rich_fixed": {k: round(v, 4) for k, v in t1["closed_form_on_residuals_rich_fixed"].items()},
        "T1_mixedlm": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t1["mixedlm_base"].items()},
        "T1_within_run_covariance": {k: round(v, 4) for k, v in t1["within_run_covariance"].items()},
        "T1_sigma_run2_raw_bootstrap": {
            k: (round(v, 4) if isinstance(v, float) else v) for k, v in t1["sigma_run2_raw_cluster_bootstrap"].items()
        },
        "T2_A_min1": round(report["T2"]["A_total_min1"]["pooled_spearman"], 3),
        "T2_A_min3": round(report["T2"]["A_total_min3"]["pooled_spearman"], 3),
        "T2_B_min1": round(report["T2"]["B_conditional_on_video_min1"]["pooled_spearman"], 3),
        "T2_C_position_adjusted": round(report["T2"]["C_position_adjusted_min1"]["pooled_spearman"], 3),
        "T2_D3_first_occurrence": round(report["T2"]["D3_first_occurrence"]["pooled_spearman"], 3),
        "T2_D3_subsequent": round(report["T2"]["D3_subsequent"]["pooled_spearman"], 3),
        "T2_D2_count_cost": {k: round(v, 3) for k, v in report["T2"]["D2_count_cost"].items()},
        "T3_lift_q80": round(report["T3"]["q80_min1"]["pooled_lift"], 3),
        "T3_lift_q90": round(report["T3"]["q90_min1"]["pooled_lift"], 3),
        "T3_lift_q95": round(report["T3"]["q95_min1"]["pooled_lift"], 3),
        "T3_p_two_q90": report["T3"]["q90_min1"]["permutation"]["p_two_sided"],
        "T3_p_two_q95": report["T3"]["q95_min1"]["permutation"]["p_two_sided"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
