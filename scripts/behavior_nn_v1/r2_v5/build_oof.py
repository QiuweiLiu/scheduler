#!/usr/bin/env python3
"""构建 OOF XGB 缓存(B08/B09/B10 前置)。

用法:python build_oof.py --artifact-root results/processed/behavior_nn_v1_r2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_R2_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _R2_PARENT not in sys.path:
    sys.path.insert(0, _R2_PARENT)

from r2b import oof_cache, shared


def main():
    parser = argparse.ArgumentParser(description="Build GroupKFold OOF XGB cache.")
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()

    (role_train, role_validation, family_train, family_validation, history_index) = (
        shared.load_train_validation_only()
    )
    builder = oof_cache.OofBuilder()
    oof = builder.build(role_train, family_train, history_index)
    full = oof_cache.build_full_predictions(
        role_train, role_validation, family_train, family_validation, history_index,
        builder.hyperparameters,
    )
    npz_path, manifest_path = oof_cache.write_cache(
        args.artifact_root, oof, full, role_train, family_train
    )
    print(f"OOF cache written: {npz_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
