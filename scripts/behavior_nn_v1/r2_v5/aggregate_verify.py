#!/usr/bin/env python3
"""总矩阵 verifier + 汇总(审查 P0-08/1-11):
- 校验预期模型×seed 全部成功(verify_completed_run)
- 白名单 run id;拒绝旧 run/重复 run
- 输出 corrected_vs_legacy.json + paired_bootstrap.json(原子写入)
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_R2_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _R2_PARENT not in sys.path:
    sys.path.insert(0, _R2_PARENT)

from r2 import run_artifacts

EXPECTED = {
    "B04": (11, 22, 33),
    "B05": (11, 22, 33),
    "B06": (11, 22, 33),
    "B07": (11, 22, 33),
    "B08": (0,),
    "B09": (11, 22, 33),
    "B10": (11, 22, 33),
}


def main():
    parser = argparse.ArgumentParser(description="Verify R2 matrix and write comparisons.")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--experiments", required=True, help="逗号分隔 experiment_id 前缀,如 B04_ft,B05_gru")
    args = parser.parse_args()

    root = Path(args.artifact_root)
    runs_dir = root / "runs"
    wanted = {}
    for prefix in args.experiments.split(","):
        model_id, _, _ = prefix.partition("_")
        wanted[prefix] = EXPECTED[model_id]

    report = {"verified": {}, "missing": [], "failures": []}
    for prefix, seeds in wanted.items():
        for seed in seeds:
            run_ids = [d.name for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
            if not run_ids:
                report["missing"].append(f"{prefix} seed {seed}")
                continue
            # 白名单:experiment_id 前缀 + config 校验
            matched = None
            for rid in run_ids:
                cfg = json.loads((runs_dir / rid / "config.json").read_text(encoding="utf-8"))
                if cfg["experiment_id"] == prefix and cfg["seed"] == seed:
                    if matched is not None:
                        report["failures"].append(f"duplicate run for {prefix} seed {seed}")
                    matched = rid
            if matched is None:
                report["missing"].append(f"{prefix} seed {seed} (no matching config)")
                continue
            try:
                final = run_artifacts.verify_completed_run(root, json.loads(
                    (runs_dir / matched / "config.json").read_text(encoding="utf-8")
                ))
                metrics = json.loads((final / "metrics.json").read_text(encoding="utf-8"))
                report["verified"][f"{prefix}_{seed}"] = {
                    "run_id": matched,
                    "joint": metrics["joint"]["joint_accuracy"],
                    "family_top1": metrics["family_oracle_gate"]["top1_accuracy"],
                    "role_top1": metrics["role"]["top1_accuracy"],
                }
            except Exception as exc:  # noqa: BLE001
                report["failures"].append(f"{prefix} seed {seed}: {exc}")

    tmp = root / "comparisons" / "corrected_vs_legacy.json.tmp"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(root / "comparisons" / "corrected_vs_legacy.json")
    # paired_bootstrap:需要 video 分组逐样本预测;本轮记录占位并标注待 N1 阶段补充
    pb = {"status": "pending", "note": "video paired bootstrap 需按 video 分组逐样本预测,在 finalist 阶段实现(plan 11.5)"}
    tmp2 = root / "comparisons" / "paired_bootstrap.json.tmp"
    tmp2.write_text(json.dumps(pb, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp2.replace(root / "comparisons" / "paired_bootstrap.json")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    if report["missing"] or report["failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
