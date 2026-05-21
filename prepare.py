"""
AutoKernel -- E2FGVIHD baseline setup.

Verifies the local CUDA/PyTorch environment, generates the deterministic
single_forward.py workload, runs the E2FGVIHD cleaner, saves the baseline output
as a NumPy oracle, and writes metadata used by later experiments.

Usage:
    uv run prepare.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from single_forward import SEED, hash_inputs, make_dummy_inputs, set_determinism
from sorawm.cleaner.e2fgvi_hq_cleaner import E2FGVIHDCleaner, E2FGVIHDConfig

CACHE_DIR = Path.home() / ".cache" / "autokernel"
TEST_DATA_DIR = CACHE_DIR / "test_data"
BASELINES_PATH = CACHE_DIR / "baselines.json"

WORKLOAD = "e2fgvihd_clean"
RESOLUTION = "720p"
NUM_FRAMES = 10
WARMUP = 1
REPEAT = 1
ALLCLOSE_ATOL = 1e-3
ALLCLOSE_RTOL = 1e-5
BASELINE_OUTPUT_NAME = "baseline_output.npy"
BASELINE_KEY = f"{WORKLOAD}_{RESOLUTION}_{NUM_FRAMES}_seed{SEED}"


def verify_environment() -> dict[str, Any]:
    print("=== AutoKernel E2FGVIHD Setup ===\n")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. E2FGVIHD setup requires a CUDA-capable GPU.")
        sys.exit(1)

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    driver = "unknown"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            driver = result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    env = {
        "gpu": torch.cuda.get_device_name(device),
        "memory_gb": round(props.total_memory / 1024**3, 2),
        "sm_count": props.multi_processor_count,
        "compute_capability": f"{props.major}.{props.minor}",
        "driver": driver,
        "cuda": torch.version.cuda or "unknown",
        "pytorch": torch.__version__,
    }

    try:
        import triton

        env["triton"] = triton.__version__
    except ImportError:
        env["triton"] = "not installed"

    print(f"GPU: {env['gpu']}")
    print(f"  Memory: {env['memory_gb']:.2f} GB")
    print(f"  SM Count: {env['sm_count']}")
    print(f"  Compute Capability: {env['compute_capability']}")
    print(f"  Driver: {env['driver']}")
    print(f"  CUDA: {env['cuda']}")
    print(f"PyTorch: {env['pytorch']}")
    print(f"Triton: {env['triton']}")
    print()

    return env


def generate_test_data() -> tuple[np.ndarray, np.ndarray, Path]:
    set_determinism(SEED)
    frames, masks = make_dummy_inputs(NUM_FRAMES, RESOLUTION, seed=SEED)

    data_dir = TEST_DATA_DIR / WORKLOAD / f"{RESOLUTION}_{NUM_FRAMES}_seed{SEED}"
    data_dir.mkdir(parents=True, exist_ok=True)

    frames_path = data_dir / "frames.npy"
    masks_path = data_dir / "masks.npy"

    if frames_path.exists() and masks_path.exists():
        print(f"Test data: cached at {data_dir}")
    else:
        np.save(frames_path, frames)
        np.save(masks_path, masks)
        print(f"Test data: saved to {data_dir}")

    print(f"Resolution: {RESOLUTION}  (frames={frames.shape}, masks={masks.shape})")
    print(f"Input hash: {hash_inputs(frames, masks)}")
    print()

    return frames, masks, data_dir


def stack_outputs(out_frames: list[np.ndarray] | np.ndarray) -> np.ndarray:
    if isinstance(out_frames, np.ndarray):
        return out_frames.astype(np.float32, copy=False)
    return np.stack([np.asarray(frame) for frame in out_frames], axis=0).astype(np.float32)


def benchmark_cleaner(frames: np.ndarray, masks: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    config = E2FGVIHDConfig(enable_torch_compile=False, use_bf16=False)
    cleaner = E2FGVIHDCleaner(load_weights=True, config=config)

    print(f"Chunk size: {cleaner.chunk_size}")
    print(f"bf16: {cleaner.use_bf16}")
    print(f"torch.compile: {config.enable_torch_compile}")
    print()

    for i in range(WARMUP):
        t0 = time.perf_counter()
        _ = cleaner.clean(frames, masks)
        torch.cuda.synchronize()
        print(f"[warmup {i + 1}] {time.perf_counter() - t0:.3f}s")
        torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    times = []
    last_out = None
    for i in range(REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = cleaner.clean(frames, masks)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        last_out = out
        print(f"[baseline {i + 1}] {elapsed:.3f}s  ({len(frames) / elapsed:.2f} fps)")

    baseline_output = stack_outputs(last_out)
    best_time = min(times)
    mean_time = sum(times) / len(times)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    print("\n=== Summary ===")
    print(f"frames: {len(frames)}")
    print(f"mean / min / max time: {mean_time:.3f} / {best_time:.3f} / {max(times):.3f} s")
    print(f"fps (min time): {len(frames) / best_time:.2f}")
    print(f"peak VRAM: {peak_vram_mb / 1024:.2f} GB")
    print(f"baseline_output_shape: {baseline_output.shape}")
    print(f"baseline_output_dtype: {baseline_output.dtype}")
    print()

    return {
        "latency_s": round(best_time, 6),
        "latency_mean_s": round(mean_time, 6),
        "fps": round(len(frames) / best_time, 4),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "chunk_size": cleaner.chunk_size,
        "use_bf16": cleaner.use_bf16,
        "allclose_atol": ALLCLOSE_ATOL,
        "allclose_rtol": ALLCLOSE_RTOL,
        "config": config.model_dump(),
    }, baseline_output


def write_baseline(
    env: dict[str, Any],
    data_dir: Path,
    frames: np.ndarray,
    masks: np.ndarray,
    metrics: dict[str, Any],
    baseline_output: np.ndarray,
) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    baseline_output_path = data_dir / BASELINE_OUTPUT_NAME
    np.save(baseline_output_path, baseline_output)

    baseline = {
        "workload": WORKLOAD,
        "resolution": RESOLUTION,
        "num_frames": NUM_FRAMES,
        "seed": SEED,
        "frames_shape": list(frames.shape),
        "masks_shape": list(masks.shape),
        "frames_dtype": str(frames.dtype),
        "masks_dtype": str(masks.dtype),
        "input_hash": hash_inputs(frames, masks),
        "test_data_dir": str(data_dir),
        "frames_path": str(data_dir / "frames.npy"),
        "masks_path": str(data_dir / "masks.npy"),
        "baseline_output_path": str(baseline_output_path),
        "baseline_output_shape": list(baseline_output.shape),
        "baseline_output_dtype": str(baseline_output.dtype),
        "environment": env,
        **metrics,
    }

    with BASELINES_PATH.open("w") as f:
        json.dump({BASELINE_KEY: baseline}, f, indent=2)

    print(f"Baseline output saved to {baseline_output_path}")
    print(f"Baseline metadata saved to {BASELINES_PATH}")
    print("Ready to run E2FGVIHD experiments!")


def main() -> None:
    env = verify_environment()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    frames, masks, data_dir = generate_test_data()
    metrics, baseline_output = benchmark_cleaner(frames, masks)
    write_baseline(env, data_dir, frames, masks, metrics, baseline_output)


if __name__ == "__main__":
    main()
