"""Measure inference latency in three contexts:
  1. TFLite on host CPU (sanity check + Colab T4 number)
  2. TFLite on a connected Android device via adb (optional, real device)
  3. Reports mean + p95 + p99
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from ..common import REPO_ROOT, get_logger, load_config

log = get_logger(__name__)


def host_cpu_latency(model_path: Path, n: int = 200) -> dict:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    shape = inp["shape"]
    dtype = inp["dtype"]
    rng = np.random.default_rng(0)
    sample = rng.random(shape).astype(dtype if dtype != np.int8 else np.float32)
    if dtype == np.int8:
        sample = (sample * 255 - 128).astype(np.int8)
    # warmup
    for _ in range(10):
        interpreter.set_tensor(inp["index"], sample)
        interpreter.invoke()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        interpreter.set_tensor(inp["index"], sample)
        interpreter.invoke()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "p95_ms": sorted(times)[int(len(times) * 0.95)],
        "p99_ms": sorted(times)[int(len(times) * 0.99)],
        "n": n,
    }


def adb_android_latency(model_path: Path, n: int = 200) -> dict | None:
    """Push the model to a connected Android device and run benchmark_model."""
    try:
        subprocess.run(["adb", "shell", "echo ok"], check=True, capture_output=True)
    except Exception:
        log.warning("no ADB / no connected device — skipping Android benchmark")
        return None

    remote_model = f"/data/local/tmp/{model_path.name}"
    subprocess.run(["adb", "push", str(model_path), remote_model], check=True)
    out = subprocess.run(
        [
            "adb", "shell", "/data/local/tmp/benchmark_model",
            f"--graph={remote_model}",
            f"--num_runs={n}",
            "--enable_op_profiling=false",
            "--use_xnnpack=true",
        ],
        capture_output=True,
        text=True,
    )
    log.info("adb output:\n%s", out.stdout[-1000:])
    return {"adb_output_tail": out.stdout[-2000:]}


def run(model_path: str | None = None, output: str | None = None) -> dict:
    cfg = load_config()
    mp = Path(model_path) if model_path else REPO_ROOT / "models" / "tzniut.tflite"
    if not mp.exists():
        raise FileNotFoundError(mp)

    log.info("benchmarking %s", mp)
    host = host_cpu_latency(mp)
    log.info("host CPU: mean=%.2fms p95=%.2fms p99=%.2fms", host["mean_ms"], host["p95_ms"], host["p99_ms"])

    result = {"host_cpu": host, "model_size_kb": mp.stat().st_size / 1024}
    android = adb_android_latency(mp)
    if android:
        result["android"] = android

    budget = cfg["eval"]["latency_budget_ms_p95"]
    if host["p95_ms"] > budget:
        log.warning("p95 %.2fms exceeds budget %dms", host["p95_ms"], budget)

    out_path = Path(output) if output else REPO_ROOT / "models" / "eval" / "latency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(__import__("json").dumps(result, indent=2))
    log.info("latency → %s", out_path)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    run(a.model, a.output)
