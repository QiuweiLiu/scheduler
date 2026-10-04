#!/usr/bin/env python3
"""Build an explicit HF-substrate configuration from canonical measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tracing.analysis.workload_v02_simulator import (
    load_batching_engine_profile,
    load_colocation_profile,
    load_prefetch_interference_profile,
    load_request_preemption_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition-config", type=Path)
    parser.add_argument("--colocation", type=Path)
    parser.add_argument("--prefetch-interference", type=Path)
    parser.add_argument("--request-preemption", type=Path)
    parser.add_argument("--batching-engine", type=Path)
    parser.add_argument("--prefetch-overlap", action="store_true",
                        help="enable the prefetch machinery: explicit plans AND the online "
                             "prewarm of prefetch-capable policies (Hermes, Latency-Aware)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not any((args.transition_config, args.colocation, args.prefetch_interference,
                args.request_preemption, args.batching_engine)):
        parser.error("provide at least one measured configuration or artifact")
    config = (
        json.loads(args.transition_config.read_text(encoding="utf-8"))
        if args.transition_config else {"schema_version": "scheduler-substrate-extension-v1"}
    )
    if not isinstance(config, dict):
        parser.error("transition configuration must be an object")
    if args.colocation:
        config["colocation_profile"] = load_colocation_profile(args.colocation)
    if args.prefetch_interference:
        profile = load_prefetch_interference_profile(args.prefetch_interference)
        config["prefetch_interference"] = profile
        if str(profile.get("interference_model") or "") != "additive_extra_ms":
            config["prefetch_interference_mode"] = "full"
    if args.request_preemption:
        config["request_preemption"] = load_request_preemption_profile(args.request_preemption)
        config["preemption_enabled"] = True
    if args.batching_engine:
        config["batching_engine"] = load_batching_engine_profile(args.batching_engine)
    if args.prefetch_overlap:
        config["prefetch_overlap"] = True
    # No overwrite: this is a configuration builder, not a formal-result updater.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "execution_engine": "hf_substrate",
                      "requires_explicit_gpu_identity_and_workload_shape": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
