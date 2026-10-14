#!/usr/bin/env python3
"""Burst64-v1 arrival-distribution figure (ECDF + histogram of inter-arrivals)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tracing.analysis.workload_v02_simulator import read_jsonl  # noqa: E402

W = ROOT / "experiments/EXP-20261014_burst64_v1/artifacts"
COLORS = {"r085_b10": "#1f77b4", "r110_b10": "#ff7f0e", "r085_b15": "#2ca02c", "r110_b15": "#d62728"}
LABELS = {"r085_b10": "(rho=0.85, beta=1.0)", "r110_b10": "(rho=1.10, beta=1.0)",
          "r085_b15": "(rho=0.85, beta=1.5)", "r110_b15": "(rho=1.10, beta=1.5)"}


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, color in COLORS.items():
        episodes = read_jsonl(W / f"workload_burst64_{tag}.jsonl")
        gaps = []
        for episode in episodes:
            arrivals = sorted(float(j["arrival_ms"]) for j in episode["jobs"])
            gaps.extend(arrivals[i + 1] - arrivals[i] for i in range(len(arrivals) - 1))
        arr = np.array([g for g in gaps if g > 0])
        xs = np.sort(arr)
        axes[0].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=color, label=LABELS[tag], lw=1.6)
        axes[1].hist(arr, bins=np.logspace(np.log10(max(1, arr.min())), np.log10(arr.max()), 40),
                     histtype="step", color=color, label=LABELS[tag], lw=1.6)
    axes[0].set_xscale("log"); axes[0].set_xlabel("inter-arrival (ms, log)"); axes[0].set_ylabel("ECDF")
    axes[0].set_title("Burst64-v1: inter-arrival ECDF"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xscale("log"); axes[1].set_xlabel("inter-arrival (ms, log)"); axes[1].set_ylabel("count")
    axes[1].set_title("Burst64-v1: inter-arrival histogram (beta=1.5 = synthetic burst)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = W / "burst64_v1_arrivals.png"
    plt.savefig(out, dpi=160)
    print("figure:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
