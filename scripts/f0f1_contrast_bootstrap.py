"""Close the F0/F1 negative result with a paired video-cluster bootstrap.

Zero training: replays the six frozen selected checkpoints over the validation split.

Design (from the review):
* per seed and validation row, d_{s,i} = metric(F1_s, i) - metric(F0_s, i)
* the paired bootstrap uses the SAME video-cluster draw for F0 and F1
* the headline is d_bar_i = mean over the three seeds of d_{s,i}, bootstrapped over
  video clusters, which answers "for the three-seed average model, how uncertain is
  F1-F0 on the validation workload population?"
* the three seed point estimates are reported separately; the three seeds themselves
  are NOT bootstrapped, because three is too few
* the pre-registered rule is unchanged: CI_upper < 0 is required to claim improvement.
  If the interval crosses zero the wording is "no measurable improvement was detected",
  not "statistically equivalent", because no equivalence margin was pre-registered.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

import j_series_common as common  # noqa: E402

HISTRES = PROJECT_ROOT / "results/processed/j_series_dataset_histres_v2"
RUNS = PROJECT_ROOT / "outputs/j_series_histres_f0f1_v1/runs"
OUT = PROJECT_ROOT / "experiments/EXP-20260921_histres_causal_input_v1/artifacts/f0f1_contrast_bootstrap.json"
SEEDS = (11, 22, 33)
B = 2000
BASE_SEED = 20260921


def selected_checkpoint(arm: str, seed: int) -> Path:
    return RUNS / arm / ("seed%d" % seed) / "checkpoint.pt"


def main() -> int:
    import torch

    import j_series_train_eval as te

    bins = te.load_runtime_bins()
    rows = list(common.read_jsonl_gz(HISTRES / "histres_validation.jsonl.gz"))
    vocabs = common.build_vocabs(list(common.read_jsonl_gz(HISTRES / "histres_train.jsonl.gz")))
    arrays = common.encode_rows(rows, vocabs, int(5))
    video_code = np.asarray([te.stable_video_code(r["video_id"]) for r in rows], dtype=np.int64)

    runtime = arrays["runtime_ms"].astype(np.float64)
    mask = arrays["slot_present"].astype(np.float64) > 0.0
    valid = mask & (runtime > 0.0)

    def qscore(arm: str, seed: int) -> np.ndarray:
        model = te.make_model(
            te.Ctx(config={"model": {"runtime_distribution_head": True, "runtime_head_hidden": 64},
                           "variants": {}}, run_root=PROJECT_ROOT, exp_dir=PROJECT_ROOT,
                   device=torch.device("cpu"), horizon=5, batch_size=256, vocabs=vocabs,
                   train=arrays, validation=arrays, test={},
                   val_draws=np.zeros((1, 1), dtype=np.int64), test_draws=np.zeros((1, 1), dtype=np.int64)),
            arm,
        )
        payload = torch.load(selected_checkpoint(arm, seed), map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        model.eval()
        scores = np.full(len(rows), np.nan)
        with torch.no_grad():
            for start in range(0, len(rows), 128):
                idx = np.arange(start, min(start + 128, len(rows)), dtype=np.int64)
                batch = te.to_torch_batch(arrays, idx, torch.device("cpu"))
                encoded = model.encode(batch, use_history_telemetry=(arm == "F1"))
                probs = torch.softmax(model.resource(encoded, None)["runtime_logits"], dim=-1)
                import j_series_resource_dist as dist
                reps = np.asarray(bins["representatives_ms"], dtype=np.float64)
                views = dist.derive_from_probs(probs, reps)
                parts = []
                for tau, key in ((0.50, "q50_ms"), (0.90, "q90_ms"), (0.95, "q95_ms")):
                    pred = np.asarray(views[key], dtype=np.float64)
                    resid = runtime[idx] - pred
                    parts.append(np.where(resid >= 0, tau * resid, (tau - 1.0) * resid))
                stacked = np.stack(parts, axis=0)          # 3 quantiles x rows x horizon
                sel = valid[idx]
                num = (stacked * sel[None]).sum(axis=2)     # 3 x rows
                den = sel.sum(axis=1)                       # rows
                per_row = np.where(den[None] > 0, num / np.maximum(den, 1)[None], np.nan)
                # RuntimeQScore per row = mean over the three quantiles
                scores[idx] = np.nanmean(per_row, axis=0)
        return scores

    per_seed = {}
    deltas = []
    for seed in SEEDS:
        a = qscore("F0", seed)
        b = qscore("F1", seed)
        d = b - a
        per_seed["seed%d" % seed] = {
            "F0_mean": float(np.nanmean(a)),
            "F1_mean": float(np.nanmean(b)),
            "delta_mean": float(np.nanmean(d)),
            "delta_sum": float(np.nansum(d)),
        }
        deltas.append(d)
        print("seed %-3d  F0=%.4f  F1=%.4f  delta_mean=%+.6f  delta_sum=%+.4f"
              % (seed, np.nanmean(a), np.nanmean(b), np.nanmean(d), np.nansum(d)))

    stacked = np.stack(deltas, axis=0)                    # 3 x rows
    d_bar = np.nanmean(stacked, axis=0)

    clusters = sorted(set(video_code.tolist()))
    members = {c: np.where(video_code == c)[0] for c in clusters}
    rng = np.random.default_rng(BASE_SEED)
    means = []
    for _ in range(B):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        idx = np.concatenate([members[clusters[p]] for p in pick])
        means.append(float(np.nanmean(d_bar[idx])))
    means.sort()
    point = float(np.nanmean(d_bar))
    result = {
        "schema_version": "f0f1-contrast-bootstrap-v1",
        "contrast": "F1 - F0 (runtime qscore per validation row, lower is better)",
        "metric": "runtime_qscore",
        "n_validation_rows": int(len(rows)),
        "n_video_clusters": len(clusters),
        "bootstrap": {"B": B, "seed": BASE_SEED, "unit": "video cluster", "paired": True},
        "point_mean_over_seeds": point,
        "ci_low": means[int(0.025 * B)],
        "ci_high": means[min(B - 1, int(0.975 * B))],
        "prob_le_zero": sum(1 for m in means if m <= 0.0) / B,
        "per_seed": per_seed,
        "preregistered_rule": "CI_upper < 0 required to claim the telemetry improves prediction",
        "verdict": ("no measurable improvement detected"
                    if means[min(B - 1, int(0.975 * B))] >= 0.0 else "improvement detected"),
        "wording_note": ("the interval crosses zero, so the correct statement is 'no measurable "
                         "improvement was detected', NOT 'the two arms are statistically "
                         "equivalent': no equivalence margin was pre-registered"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(json.dumps({k: v for k, v in result.items() if k != "per_seed"}, ensure_ascii=False, indent=2))
    print()
    print("saved:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
