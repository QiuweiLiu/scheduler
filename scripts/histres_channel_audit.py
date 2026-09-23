"""Independent audit: is the telemetry channel actually doing anything?

Three zero-training measurements, aimed at the user's suspicion that the
implementation makes the channel inert:

 A. parameter movement   ||theta_final - theta_init|| for the telemetry branch
                         (F1 must be clearly > 0, F0 must be exactly 0)
 B. inference ablation   run the trained F1 checkpoint with telemetry on vs off
                         (if the score is unchanged the model ignores the channel)
 C. channel magnitude    ||E_telemetry|| / ||E_categorical + E_position||
                         at init and at the selected checkpoint
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"F:\scheduler")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

import j_series_common as common
import j_series_resource_dist as dist
import j_series_train_eval as te

HISTRES = PROJECT_ROOT / "results/processed/j_series_dataset_histres_v2"
OUT_ROOT = PROJECT_ROOT / "outputs/j_series_histres_f0f1_v1"
SEEDS = (11, 22, 33)
N_ROWS = 256


def load_arrays():
    rows = list(common.read_jsonl_gz(HISTRES / "histres_validation.jsonl.gz"))[:N_ROWS]
    vocabs = common.build_vocabs(list(common.read_jsonl_gz(HISTRES / "histres_train.jsonl.gz")))
    arrays = common.encode_rows(rows, vocabs, 5)
    return vocabs, arrays


def build(vocabs):
    cfg = {"hidden": 128, "history_embedding_dim": 16, "context_embedding_dim": 12,
           "slot_embedding_dim": 16, "dropout": 0.1, "runtime_bins": te.load_runtime_bins(),
           "runtime_head_hidden": 64}
    return common.JSeriesModel(vocabs, cfg, horizon=5, duration_mode="shared")


def telemetry_norm(model, batch, enabled=True):
    """||E_telemetry|| and ||E_categorical + E_position|| on observed tokens."""

    device = next(model.parameters()).device
    hist_ids = {f: batch["hist_%s" % f] for f in common.HISTORY_FIELDS}
    steps = hist_ids[common.HISTORY_FIELDS[0]].shape[1]
    emb = None
    for field in common.HISTORY_FIELDS:
        value = model.hist_emb[field](hist_ids[field])
        emb = value if emb is None else emb + value
    positions = torch.arange(steps, device=device).unsqueeze(0).clamp(max=common.POSITION_BUCKETS - 1)
    cat = emb + model.pos_emb(positions)

    tele = torch.zeros_like(cat)
    if enabled and "hist_res_num" in batch:
        observed = batch["hist_res_mask"].unsqueeze(-1)
        tele = observed * model.hist_res_proj(batch["hist_res_num"]) + model.hist_status_emb(batch["hist_status"])

    sel = batch["hist_res_mask"] > 0.0
    if int(sel.sum()) == 0:
        return 0.0, 0.0
    tele_n = tele[sel].norm(dim=-1).median().item()
    cat_n = cat[sel].norm(dim=-1).median().item()
    return tele_n, cat_n


def qscore(model, arrays, arm, bins):
    runtime = arrays["runtime_ms"].astype(np.float64)
    mask = arrays["slot_present"].astype(np.float64) > 0.0
    valid = mask & (runtime > 0.0)
    model.eval()
    scores = np.full(len(runtime), np.nan)
    with torch.no_grad():
        for start in range(0, len(runtime), 128):
            idx = np.arange(start, min(start + 128, len(runtime)), dtype=np.int64)
            batch = te.to_torch_batch(arrays, idx, torch.device("cpu"))
            encoded = model.encode(batch, use_history_telemetry=(arm == "F1"))
            probs = torch.softmax(model.resource(encoded, None)["runtime_logits"], dim=-1)
            views = dist.derive_from_probs(probs, np.asarray(bins["representatives_ms"], dtype=np.float64))
            parts = []
            for tau, key in ((0.50, "q50_ms"), (0.90, "q90_ms"), (0.95, "q95_ms")):
                pred = np.asarray(views[key], dtype=np.float64)
                resid = runtime[idx] - pred
                parts.append(np.where(resid >= 0, tau * resid, (tau - 1.0) * resid))
            stacked = np.stack(parts, axis=0)              # 3 x rows x horizon
            sel = valid[idx]                               # rows x horizon
            num = (stacked * sel[None]).sum(axis=2)        # 3 x rows
            den = sel.sum(axis=1)                          # rows
            per_row = (num / np.maximum(den, 1)[None]).mean(axis=0)
            scores[idx] = np.where(den > 0, per_row, np.nan)
    return float(np.nanmean(scores))


def main() -> int:
    bins = te.load_runtime_bins()
    vocabs, arrays = load_arrays()
    batch = te.to_torch_batch(arrays, np.arange(min(64, len(arrays["length"])), dtype=np.int64), torch.device("cpu"))
    report = {"seeds": {}}

    for seed in SEEDS:
        init_path = OUT_ROOT / "init" / ("F0F1_seed%d.pt" % seed)
        init = torch.load(init_path, map_location="cpu", weights_only=False)
        entry = {}

        # A. parameter movement
        for arm in ("F0", "F1"):
            payload = torch.load(OUT_ROOT / "runs" / arm / ("seed%d" % seed) / "checkpoint.pt",
                                 map_location="cpu", weights_only=False)
            final = payload["model_state"]
            moves = {}
            for prefix in ("hist_res_proj", "hist_status_emb"):
                keys = [k for k in final if k.startswith(prefix)]
                d = sum(float((final[k].float() - init["model_state"][k].float()).norm() ** 2) for k in keys) ** 0.5
                moves[prefix] = d
            entry["%s_param_movement" % arm] = moves

        # B. inference ablation on the trained F1 checkpoint
        model = build(vocabs)
        f1 = torch.load(OUT_ROOT / "runs" / "F1" / ("seed%d" % seed) / "checkpoint.pt",
                        map_location="cpu", weights_only=False)
        model.load_state_dict(f1["model_state"])
        entry["F1_on"] = qscore(model, arrays, "F1", bins)
        entry["F1_off"] = qscore(model, arrays, "F0", bins)

        # C. channel magnitude at init and at the F1 checkpoint
        model_init = build(vocabs)
        model_init.load_state_dict(init["model_state"])
        t_init, c_init = telemetry_norm(model_init, batch, True)
        t_fin, c_fin = telemetry_norm(model, batch, True)
        entry["telemetry_over_categorical_init"] = t_init / max(1e-12, c_init)
        entry["telemetry_over_categorical_final"] = t_fin / max(1e-12, c_fin)
        entry["telemetry_norm_init"] = t_init
        entry["telemetry_norm_final"] = t_fin
        entry["categorical_norm_init"] = c_init
        entry["categorical_norm_final"] = c_fin

        report["seeds"]["seed%d" % seed] = entry
        print("=== seed %d ===" % seed)
        print("  param movement F1 :", json.dumps({k: round(v, 6) for k, v in entry["F1_param_movement"].items()}))
        print("  param movement F0 :", json.dumps({k: round(v, 6) for k, v in entry["F0_param_movement"].items()}))
        print("  F1 qscore on/off  : %.4f / %.4f   delta %+.4f" % (entry["F1_on"], entry["F1_off"], entry["F1_on"] - entry["F1_off"]))
        print("  ||tele||/||cat||  : init %.5f  final %.5f" % (entry["telemetry_over_categorical_init"], entry["telemetry_over_categorical_final"]))
        print("  ||tele|| init/fin : %.6f / %.6f   ||cat|| init/fin %.4f / %.4f"
              % (t_init, t_fin, c_init, c_fin))
        print()

    out = PROJECT_ROOT / "experiments/EXP-20260921_histres_causal_input_v1/artifacts/channel_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
