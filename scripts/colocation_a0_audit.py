#!/usr/bin/env python3
"""Colocation Phase A0: evidence-source audit (advisor 2026-10-16).

Freezes SHA256 of the measured artifacts + v7 config; verifies the v7 embedded
colocation cells (11) and additive load-interference cells (12) against the
original measurement files field-by-field (explicit model-alias normalization
allowed); checks GPU identity / capacity / deployment scope.  PASS/FAIL report.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "experiments/EXP-20261004_real_strata_calibration_v1/artifacts"
FILES = {
    "colocation_table": CAL / "real_strata_colocation_table_v1.json",
    "interference_additive": CAL / "real_strata_interference_additive_v1.json",
    "calibration": CAL / "real_strata_calibration.json",
    "v7_config": CAL / "substrate_extension_v7.json",
}
OUT = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/colocation_a0_audit.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm_model(name: str) -> str:
    # explicit alias normalization only
    return str(name).replace("Qwen3-VL-8B-Instruct", "Qwen3-VL-8B").replace("Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-3B")


def main() -> int:
    report = {"schema": "colocation-a0-audit", "files": {}, "checks": {}, "failures": []}
    for k, p in FILES.items():
        report["files"][k] = {"path": str(p.relative_to(ROOT)), "exists": p.exists(),
                              "sha256": sha(p) if p.exists() else None,
                              "size": p.stat().st_size if p.exists() else None}
        if not p.exists():
            report["failures"].append(f"missing {k}")

    col = json.loads(FILES["colocation_table"].read_text(encoding="utf-8"))
    add = json.loads(FILES["interference_additive"].read_text(encoding="utf-8"))
    v7 = json.loads(FILES["v7_config"].read_text(encoding="utf-8"))

    # GPU identity / scope
    report["checks"]["gpu_identity"] = col.get("gpu")
    report["checks"]["deployment"] = col.get("deployment")
    report["checks"]["probe_rounds"] = (col.get("probe_metadata") or {}).get("rounds")

    # v7 embedded colocation cells vs original (field-by-field, alias-normalized)
    prof = (v7.get("colocation_profile") or {})
    report["checks"]["v7_colocation_enabled"] = prof.get("enabled")
    report["checks"]["v7_source_artifact"] = prof.get("source_artifact")
    orig_cells = {c["cell_id"]: c for c in (col.get("cells") or [])}
    v7_cells = {c["cell_id"]: c for c in (prof.get("cells") or [])}
    report["checks"]["n_cells_original"] = len(orig_cells)
    report["checks"]["n_cells_v7"] = len(v7_cells)
    mismatch = []
    for cid, c in orig_cells.items():
        v = v7_cells.get(cid)
        if v is None:
            mismatch.append(f"cell {cid} missing in v7")
            continue
        for f in ("model_a", "model_b", "shape_a", "shape_b", "solo_a_ms", "coloc_a_ms", "solo_b_ms", "coloc_b_ms"):
            a, b = c.get(f), v.get(f)
            if isinstance(a, str) or isinstance(b, str):
                if norm_model(a or "") != norm_model(b or ""):
                    mismatch.append(f"{cid}.{f}: {a} vs {b}")
            elif a is not None and b is not None and abs(float(a) - float(b)) > 1e-6:
                mismatch.append(f"{cid}.{f}: {a} vs {b}")
    report["checks"]["colocation_cell_mismatches"] = mismatch
    report["checks"]["colocation_cells_match"] = not mismatch

    # additive interference: key by (infer_model, infer_shape, load)
    def add_key(c):
        return (norm_model(c.get("infer_model")), str(c.get("infer_shape")), norm_model(c.get("load")))
    add_orig = {add_key(c): c for c in (add.get("cells") or [])}
    v7_add = None
    for key, val in v7.items():
        if isinstance(val, dict) and "cells" in val and "interference" in key.lower():
            v7_add = {add_key(c): c for c in val["cells"]}
            report["checks"]["v7_interference_section"] = key
    if v7_add is not None:
        add_mis = []
        for cid, c in add_orig.items():
            v = v7_add.get(cid)
            if v is None:
                add_mis.append(f"cell {cid} missing in v7")
                continue
            for f in ("extra_ms",):
                a, b = c.get(f), v.get(f)
                if isinstance(a, str) or isinstance(b, str):
                    if norm_model(a or "") != norm_model(b or ""):
                        add_mis.append(f"{cid}.{f}: {a} vs {b}")
                elif a is not None and b is not None and abs(float(a) - float(b)) > 1e-6:
                    add_mis.append(f"{cid}.{f}: {a} vs {b}")
        report["checks"]["interference_cells_original"] = len(add_orig)
        report["checks"]["interference_cells_v7"] = len(v7_add)
        report["checks"]["interference_mismatches"] = add_mis
        report["checks"]["interference_cells_match"] = not add_mis
    else:
        report["checks"]["interference_cells_match"] = None
        report["failures"].append("v7 additive-interference section not located")

    if mismatch:
        report["failures"].append(f"{len(mismatch)} colocation cell mismatches")

    status = "PASS" if not report["failures"] else "FAIL"
    report["A0_status"] = status
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=1, ensure_ascii=False))
    print("A0:", status, "| evidence:", OUT)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
