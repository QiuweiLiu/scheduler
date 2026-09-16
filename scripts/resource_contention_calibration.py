#!/usr/bin/env python3
"""Safe single-GPU resource/contention calibration microbenchmark.

This is a calibration artifact, not an Agent trace and not an OOM test.  It
uses existing local models and one existing extracted frame, runs only bounded
1/2-process cases, and records measured load/inference/peak-memory data for
the later workload simulator.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path("/root/autodl-tmp/scheduler")
DEFAULT_IMAGE = ROOT / "results/raw/phase2_pilot_api/wgBlACG927Y_star_1785479717457/_frames/frame_00000000.jpg"
YOLO_WEIGHT = Path("/root/autodl-tmp/upload/models/yolo11x.pt")
FINETOOLING_PYTHON = "/root/miniconda3/envs/finetooling/bin/python"
QWEN_VL_WORKER = ROOT / "src/tracing/collectors/qwen3_vl_worker.py"
QWEN_TEXT_WORKER = ROOT / "src/tracing/collectors/qwen_text_worker.py"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def float_or_none(value: str) -> float | None:
    value = value.strip().replace("%", "")
    if value.upper() in {"N/A", "[NOT SUPPORTED]", ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class GpuSampler:
    def __init__(self, interval_ms: int = 100) -> None:
        self.interval_ms = max(50, int(interval_ms))
        self.samples: list[dict[str, Any]] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            self.process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                    f"--loop-ms={self.interval_ms}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            self.process = None
            return
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            parts = [part.strip() for part in line.strip().split(",")]
            if len(parts) != 6:
                continue
            self.samples.append(
                {
                    "timestamp": parts[0],
                    "gpu_index": parts[1],
                    "gpu_utilization_percent": float_or_none(parts[2]),
                    "memory_utilization_percent": float_or_none(parts[3]),
                    "memory_used_mb": float_or_none(parts[4]),
                    "power_w": float_or_none(parts[5]),
                }
            )

    def stop(self) -> dict[str, Any]:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        if self.thread is not None:
            self.thread.join(timeout=1)
        def maximum(key: str) -> float | None:
            values = [row[key] for row in self.samples if row.get(key) is not None]
            return max(values) if values else None
        return {
            "sample_count": len(self.samples),
            "max_gpu_utilization_percent": maximum("gpu_utilization_percent"),
            "max_memory_utilization_percent": maximum("memory_utilization_percent"),
            "max_memory_used_mb": maximum("memory_used_mb"),
            "max_power_w": maximum("power_w"),
            "samples": self.samples,
        }


def cuda_cleanup() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    time.sleep(1.0)


def run_case(name: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    cuda_cleanup()
    sampler = GpuSampler()
    started = time.perf_counter()
    sampler.start()
    record: dict[str, Any] = {
        "case": name,
        "started_at": now_utc(),
        "status": "success",
    }
    try:
        record["measurement"] = operation()
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    record["gpu_sampling"] = sampler.stop()
    cuda_cleanup()
    return record


def qwen_vl_case(model_path: Path, image: Path, log_path: Path, repeats: int) -> dict[str, Any]:
    from tracing.collectors.qwen3_vl_worker import QwenVLClient

    client = QwenVLClient(
        python_bin=FINETOOLING_PYTHON,
        model_path=model_path,
        log_path=log_path,
        max_new_tokens=16,
        timeout_seconds=300,
    )
    outputs: list[dict[str, Any]] = []
    try:
        for index in range(repeats):
            started = time.perf_counter()
            response = client.describe(
                [image],
                f"Calibration request {index}. Describe one visible object in one short sentence.",
            )
            response["parent_wall_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            outputs.append(response)
    finally:
        client.close()
    return {"model_path": str(model_path), "repeats": repeats, "responses": outputs}


def qwen_text_case(model_path: Path, log_path: Path, repeats: int) -> dict[str, Any]:
    from tracing.collectors.qwen_text_worker import QwenTextClient

    client = QwenTextClient(
        python_bin=FINETOOLING_PYTHON,
        model_path=model_path,
        log_path=log_path,
        max_new_tokens=16,
        timeout_seconds=300,
        constrained_json=False,
    )
    outputs: list[dict[str, Any]] = []
    try:
        for index in range(repeats):
            started = time.perf_counter()
            response = client.generate(
                f"Calibration request {index}. Return the word observe and nothing else."
            )
            response["parent_wall_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            outputs.append(response)
    finally:
        client.close()
    return {"model_path": str(model_path), "repeats": repeats, "responses": outputs}


def yolo_worker(weights: Path, image: Path, batch: int, repeats: int, hold: bool = False) -> int:
    import torch
    from ultralytics import YOLO

    load_started = time.perf_counter()
    model = YOLO(str(weights))
    model.to("cuda:0")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_ms = round((time.perf_counter() - load_started) * 1000.0, 3)
    sources = [str(image)] * max(1, int(batch))
    warmup = model.predict(source=sources[:1], imgsz=640, device=0, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    # Mixed calibration must overlap a resident YOLO model with the Qwen
    # worker.  Wait for an explicit release from the parent after warm-up so
    # the finite inference loop cannot finish before Qwen has loaded.
    if hold:
        print(json_line({"ready": True, "load_ms": load_ms, "batch": batch, "resident": True}), flush=True)
        stop_event = threading.Event()
        background_runs: list[dict[str, Any]] = []

        def run_background() -> None:
            while not stop_event.is_set():
                started = time.perf_counter()
                model.predict(source=sources, imgsz=640, device=0, verbose=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                background_runs.append(
                    {
                        "repeat": len(background_runs),
                        "inference_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    }
                )

        background_thread = threading.Thread(target=run_background, daemon=True)
        background_thread.start()
        sys.stdin.readline()
        stop_event.set()
        background_thread.join(timeout=600)
        runs = background_runs
    else:
        runs = []
        print(json_line({"ready": True, "load_ms": load_ms, "batch": batch}), flush=True)
        for index in range(max(1, repeats)):
            started = time.perf_counter()
            model.predict(source=sources, imgsz=640, device=0, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            runs.append({"repeat": index, "inference_ms": round((time.perf_counter() - started) * 1000.0, 3)})
    payload: dict[str, Any] = {"ready": True, "load_ms": load_ms, "batch": batch, "runs": runs}
    if torch.cuda.is_available():
        payload.update(
            {
                "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 3),
                "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 3),
            }
        )
    print(json_line(payload), flush=True)
    return 0


def spawn_yolo(weights: Path, image: Path, batch: int, repeats: int, hold: bool = False) -> tuple[subprocess.Popen[str], dict[str, Any]]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--yolo-worker",
        "--yolo-weights",
        str(weights),
        "--image",
        str(image),
        "--batch",
        str(batch),
        "--repeats",
        str(repeats),
    ]
    if hold:
        command.insert(2, "--hold")
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )
    if process.stdout is None:
        raise RuntimeError("YOLO worker stdout unavailable")
    line = process.stdout.readline()
    if not line:
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise RuntimeError(f"YOLO worker exited before ready: {stderr[-2000:]}")
    ready = json.loads(line)
    if not ready.get("ready"):
        raise RuntimeError(f"YOLO worker not ready: {ready}")
    return process, ready


def finish_yolo(process: subprocess.Popen[str], release: bool = False) -> dict[str, Any]:
    if release and process.poll() is None and process.stdin is not None:
        process.stdin.write("release\n")
        process.stdin.flush()
    stdout, stderr = process.communicate(timeout=600)
    lines = [line for line in stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1]) if lines else {"ready": False}
    if process.returncode != 0:
        payload["worker_stderr"] = stderr[-4000:]
        raise RuntimeError(f"YOLO worker failed: {payload}")
    return payload


def mixed_case(
    qwen_kind: str,
    qwen_model: Path,
    image: Path,
    weights: Path,
    batch: int,
    log_path: Path,
    repeats: int,
) -> dict[str, Any]:
    yolo_process, yolo_ready = spawn_yolo(weights, image, batch, repeats, hold=True)
    client: Any = None
    requests: list[dict[str, Any]] = []
    yolo_result: dict[str, Any] | None = None
    try:
        if qwen_kind == "vl":
            from tracing.collectors.qwen3_vl_worker import QwenVLClient

            client = QwenVLClient(
                python_bin=FINETOOLING_PYTHON,
                model_path=qwen_model,
                log_path=log_path,
                max_new_tokens=16,
                timeout_seconds=300,
            )
            requests = [client.describe([image], f"Mixed calibration request {i}.") for i in range(repeats)]
        else:
            from tracing.collectors.qwen_text_worker import QwenTextClient

            client = QwenTextClient(
                python_bin=FINETOOLING_PYTHON,
                model_path=qwen_model,
                log_path=log_path,
                max_new_tokens=16,
                timeout_seconds=300,
                constrained_json=False,
            )
            requests = [client.generate(f"Mixed calibration request {i}. Return observe.") for i in range(repeats)]
    finally:
        if client is not None:
            client.close()
        yolo_result = finish_yolo(yolo_process, release=True)
    return {
        "qwen_model": str(qwen_model),
        "qwen_kind": qwen_kind,
        "yolo_ready": yolo_ready,
        "yolo_resident_during_qwen": True,
        "yolo_result": yolo_result,
        "qwen_responses": requests,
    }


def two_model_case(vl_model: Path, text_model: Path, image: Path, log_dir: Path, repeats: int) -> dict[str, Any]:
    from tracing.collectors.qwen3_vl_worker import QwenVLClient
    from tracing.collectors.qwen_text_worker import QwenTextClient

    vl_client = QwenVLClient(python_bin=FINETOOLING_PYTHON, model_path=vl_model, log_path=log_dir / "vl.log", max_new_tokens=16, timeout_seconds=300)
    text_client = QwenTextClient(python_bin=FINETOOLING_PYTHON, model_path=text_model, log_path=log_dir / "text.log", max_new_tokens=16, timeout_seconds=300, constrained_json=False)

    try:
        # Keep one resident worker per model.  Submitting several calls to the
        # same client concurrently races its lazy process startup and can
        # create duplicate workers, invalidating the two-model measurement.
        responses: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for index in range(repeats):
                vl_future = pool.submit(
                    vl_client.describe,
                    [image],
                    f"Two-model calibration request {index}.",
                )
                text_future = pool.submit(
                    text_client.generate,
                    f"Two-model calibration request {index}. Return observe.",
                )
                responses.extend([vl_future.result(), text_future.result()])
    finally:
        vl_client.close()
        text_client.close()
    return {"vl_model": str(vl_model), "text_model": str(text_model), "repeats_per_model": repeats, "responses": responses}


def system_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import torch

        info.update({"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()})
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_total_memory_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 3)
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        info["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        info["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=False, default=ROOT / "results/processed/scheduling_future_v1_20260812/calibration/resource_contention_v1")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--yolo-weights", type=Path, default=YOLO_WEIGHT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--yolo-worker", action="store_true")
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    if args.yolo_worker:
        return yolo_worker(args.yolo_weights, args.image, args.batch, args.repeats, hold=args.hold)
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not args.yolo_weights.is_file():
        raise FileNotFoundError(args.yolo_weights)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "worker_logs"
    logs.mkdir(exist_ok=True)
    qwen_vl_8b = Path("/root/autodl-tmp/Qwen3-VL-8B-Instruct")
    qwen_vl_3b = ROOT / "Qwen2.5-VL-3B-Instruct"
    qwen_text_4b = ROOT / "Qwen3-4B"
    cases: list[dict[str, Any]] = []
    for batch in (1, 8, 16, 32, 64):
        cases.append(run_case(f"yolo11x_batch_{batch}", lambda batch=batch: {"worker": finish_yolo(spawn_yolo(args.yolo_weights, args.image, batch, args.repeats)[0])}))
    if qwen_vl_8b.is_dir():
        cases.append(run_case("qwen3_vl_8b_standalone", lambda: qwen_vl_case(qwen_vl_8b, args.image, logs / "qwen3_vl_8b.log", args.repeats)))
    if qwen_vl_3b.is_dir():
        cases.append(run_case("qwen2_5_vl_3b_standalone", lambda: qwen_vl_case(qwen_vl_3b, args.image, logs / "qwen2_5_vl_3b.log", args.repeats)))
    if qwen_text_4b.is_dir():
        cases.append(run_case("qwen3_4b_text_standalone", lambda: qwen_text_case(qwen_text_4b, logs / "qwen3_4b_text.log", args.repeats)))
    if qwen_vl_3b.is_dir():
        cases.append(run_case("mixed_qwen2_5_vl_3b_yolo11x_batch32", lambda: mixed_case("vl", qwen_vl_3b, args.image, args.yolo_weights, 32, logs / "mixed_vl3b.log", args.repeats)))
    if qwen_vl_8b.is_dir():
        cases.append(run_case("mixed_qwen3_vl_8b_yolo11x_batch32", lambda: mixed_case("vl", qwen_vl_8b, args.image, args.yolo_weights, 32, logs / "mixed_vl8b.log", args.repeats)))
    if qwen_text_4b.is_dir():
        cases.append(run_case("mixed_qwen3_4b_yolo11x_batch32", lambda: mixed_case("text", qwen_text_4b, args.image, args.yolo_weights, 32, logs / "mixed_text4b.log", args.repeats)))
    if qwen_vl_3b.is_dir() and qwen_text_4b.is_dir():
        cases.append(run_case("two_vlm_qwen2_5_vl_3b_plus_qwen3_4b", lambda: two_model_case(qwen_vl_3b, qwen_text_4b, args.image, logs, args.repeats)))

    manifest = {
        "schema_version": "resource-contention-calibration-v1",
        "created_at": now_utc(),
        "artifact_type": "calibration_microbenchmark_not_agent_trace",
        "controlled_oom_test": False,
        "models": {"qwen3_vl_8b": str(qwen_vl_8b), "qwen2_5_vl_3b": str(qwen_vl_3b), "qwen3_4b": str(qwen_text_4b), "yolo11x": str(args.yolo_weights)},
        "image": str(args.image),
        "repeats": args.repeats,
        "system": system_info(),
        "cases": cases,
    }
    (out / "calibration_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed = sum(row.get("status") != "success" for row in cases)
    print(json_line({"output_dir": str(out), "cases": len(cases), "failed_cases": failed, "controlled_oom_test": False}))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
