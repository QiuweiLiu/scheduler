#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from collections import OrderedDict
import torch
from decord import VideoReader
from models.internvideo2 import pretrain_internvideo2_1b_patch14_224
from mm_utils.utils import frame_transform, INTERNVIDEO_MEAN, INTERNVIDEO_STD

def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def context_key(row):
    ctx = row.get("visual_prefix_context") or {}
    video = str(row.get("video_id") or ctx.get("video_id") or "")
    vals = []
    for x in (ctx.get("frame_indices") or []):
        try:
            vals.append(int(x))
        except (TypeError, ValueError):
            pass
    return video, tuple(vals)

def select_four(values):
    if not values:
        return None
    values = list(values)
    if len(values) >= 4:
        return values[-4:]
    return values + [values[-1]] * (4 - len(values))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    rows = list(read_jsonl(Path(args.input)))
    keys = OrderedDict()
    for row in rows:
        key = context_key(row)
        if key[0]:
            keys.setdefault(key, None)
    device = torch.device(args.device if args.device != "auto" or torch.cuda.is_available() else "cpu")
    model = pretrain_internvideo2_1b_patch14_224(num_frames=4, use_flash_attn=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    model.eval().to(device=device, dtype=torch.bfloat16)
    transform = frame_transform(224, mean=INTERNVIDEO_MEAN, std=INTERNVIDEO_STD)
    embeddings = []
    metadata = []
    errors = []
    peak = 0.0
    started = time.time()
    for index, (video_id, frame_values) in enumerate(keys.keys(), 1):
        selected = select_four(frame_values)
        key = video_id + "|" + ",".join(str(x) for x in frame_values)
        if selected is None:
            continue
        video_path = Path(args.video_dir) / f"{video_id}.mp4"
        try:
            vr = VideoReader(str(video_path), num_threads=1)
            n = len(vr)
            idx = [min(max(int(x), 0), max(n - 1, 0)) for x in selected]
            frames = torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)
            x = torch.stack([transform(frame) for frame in frames]).permute(1, 0, 2, 3).unsqueeze(0)
            x = x.to(device=device, dtype=torch.bfloat16)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x, x_vis_only=True)
            pooled_cls = out[:, 0, :].float().cpu()[0]
            pooled_mean = out[:, 1:, :].mean(dim=1).float().cpu()[0]
            embeddings.append(torch.cat([pooled_cls, pooled_mean], dim=0).to(torch.float16))
            metadata.append({"key": key, "video_id": video_id, "source_frame_indices": list(frame_values), "selected_frame_indices": idx, "video_frame_count": n, "embedding_index": len(embeddings)-1})
            if device.type == "cuda":
                peak = max(peak, float(torch.cuda.max_memory_allocated(device) / 1024**3))
        except Exception as exc:
            errors.append({"key": key, "video_id": video_id, "frame_indices": list(frame_values), "error": repr(exc)})
        if index % 20 == 0 or index == len(keys):
            print(json.dumps({"done": index, "total": len(keys), "embeddings": len(embeddings), "errors": len(errors)}, ensure_ascii=False), flush=True)
    matrix = torch.stack(embeddings) if embeddings else torch.empty((0, 2816), dtype=torch.float16)
    payload = {
        "schema_version": "internvideo-prefix-embeddings-v0.1",
        "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "dtype": str(matrix.dtype),
        "embeddings": matrix,
        "metadata": metadata,
        "missing_or_error_keys": errors,
        "checkpoint": args.checkpoint,
        "num_frames": 4,
        "pooling": "cls_plus_patch_mean",
        "prefix_only": True,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    report = {
        "schema_version": "internvideo-prefix-embedding-report-v0.1",
        "input": args.input,
        "output": args.output,
        "unique_nonempty_contexts": len(keys),
        "completed": len(embeddings),
        "errors": len(errors),
        "missing_or_error_keys": errors[:20],
        "checkpoint_load_missing": len(missing),
        "checkpoint_load_unexpected": len(unexpected),
        "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "peak_memory_gib": peak,
        "elapsed_seconds": time.time() - started,
        "prefix_only": True,
        "future_frames_used": False,
        "target_event_frames_used": False,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
