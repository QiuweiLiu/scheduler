#!/usr/bin/env python3
"""统一入口:B04-B10 模型 train-to-validation runner。

用法:
  python run_model.py --model-id B04 --mode formal --experiment-id B04_ft --seed 11 \
      --artifact-root results/processed/behavior_nn_v1_r2
  python run_model.py --model-id B08 --mode formal --experiment-id B08_meta \
      --artifact-root results/processed/behavior_nn_v1_r2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_R2_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _R2_PARENT not in sys.path:
    sys.path.insert(0, _R2_PARENT)

from r2b import models_ft, models_meta, models_residual, models_seq


def main():
    parser = argparse.ArgumentParser(description="Run one B04-B10 validation artifact.")
    parser.add_argument("--model-id", choices=("B04", "B05", "B06", "B07", "B08", "B09", "B10"), required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--device", default=None, help="smoke 用 cpu;formal 默认 cuda:0")
    args = parser.parse_args()

    root = args.artifact_root
    device = args.device
    if args.model_id == "B04":
        config = models_ft.build_ft_config(args.experiment_id, args.seed, mode=args.mode)
        print(models_ft.run_validation_ft(root, config, device=device))
    elif args.model_id in ("B05", "B06", "B07"):
        config = models_seq.build_seq_config(args.experiment_id, args.model_id, args.seed, mode=args.mode)
        print(models_seq.run_validation_seq(root, config, args.model_id, device=device))
    elif args.model_id == "B08":
        if args.seed != 0:
            parser.error("B08 requires --seed 0 (deterministic)")
        config = models_meta.build_meta_config(args.experiment_id, mode=args.mode)
        print(models_meta.run_validation_meta(root, config))
    else:
        config = models_residual.build_residual_config(
            args.experiment_id, args.model_id, args.seed, mode=args.mode
        )
        print(models_residual.run_validation_residual(root, config, args.model_id, device=device))


if __name__ == "__main__":
    main()
