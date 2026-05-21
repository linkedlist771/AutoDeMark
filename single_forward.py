import hashlib
import time
import numpy as np
import torch

from sorawm.cleaner.e2fgvi_hq_cleaner import E2FGVIHDCleaner, E2FGVIHDConfig


SEED = 42

def set_determinism(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



RESOLUTIONS = {
    "720p":  (1280, 720),   # (W, H)
    "1080p": (1920, 1080),
}

def make_dummy_inputs(num_frames: int, resolution: str, seed: int = SEED):
    w, h = RESOLUTIONS[resolution]
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, size=(num_frames, h, w, 3), dtype=np.uint8)
    # 居中矩形 mask,占画面约 15% 面积 —— 模拟字幕/水印类的典型场景
    masks = np.zeros((num_frames, h, w), dtype=np.uint8)
    mh, mw = int(h * 0.20), int(w * 0.30)
    y0, x0 = (h - mh) // 2, h - mh - 40   # 偏下,像字幕条
    x0 = (w - mw) // 2
    masks[:, y0:y0 + mh, x0:x0 + mw] = 1

    return frames, masks


def hash_inputs(frames: np.ndarray, masks: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(frames.tobytes())
    h.update(masks.tobytes())
    return h.hexdigest()[:16]


def hash_outputs(out_frames) -> str:
    h = hashlib.sha256()
    for f in out_frames:
        if f is None:
            h.update(b"None")
        else:
            h.update(np.ascontiguousarray(f).tobytes())
    return h.hexdigest()[:16]


# ---------- Benchmark ----------
def benchmark(cleaner, frames, masks, warmup: int = 1, repeat: int = 3):
    cuda = torch.cuda.is_available()

    # Warmup —— 触发 cudnn kernel 选择 / allocator 稳定 / (如启用)torch.compile
    for i in range(warmup):
        t0 = time.perf_counter()
        _ = cleaner.clean(frames, masks)
        if cuda:
            torch.cuda.synchronize()
        print(f"[warmup {i+1}] {time.perf_counter() - t0:.3f}s")

    if cuda:
        torch.cuda.reset_peak_memory_stats()

    times = []
    last_out = None
    for i in range(repeat):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = cleaner.clean(frames, masks)
        if cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        last_out = out
        print(f"[run {i+1}] {dt:.3f}s  ({len(frames)/dt:.2f} fps)")

    times = np.array(times)
    print("\n=== Summary ===")
    print(f"frames        : {len(frames)}")
    print(f"mean / min / max time : {times.mean():.3f} / {times.min():.3f} / {times.max():.3f} s")
    print(f"fps (min time): {len(frames)/times.min():.2f}")
    if cuda:
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak VRAM     : {peak:.2f} GB")

    return last_out, times


# ---------- Entry ----------
if __name__ == "__main__":
    RESOLUTION = "720p"     # "720p" or "1080p"
    NUM_FRAMES = 10  # 50 is too much， we just use 10 for verficiaiton here。
    WARMUP = 1
    REPEAT = 1

    set_determinism(SEED)

    frames, masks = make_dummy_inputs(NUM_FRAMES, RESOLUTION, seed=SEED)
    print(f"Resolution    : {RESOLUTION}  (frames={frames.shape}, masks={masks.shape})")
    # print(f"Input hash    : {hash_inputs(frames, masks)}   <-- 跨机器对比时应一致")

    config = E2FGVIHDConfig(enable_torch_compile=False, use_bf16=False)
    cleaner = E2FGVIHDCleaner(load_weights=True, config=config)
    print(f"Chunk size    : {cleaner.chunk_size}")
    print(f"bf16          : {cleaner.use_bf16}")
    print()

    out, _ = benchmark(cleaner, frames, masks, warmup=WARMUP, repeat=REPEAT)

    # print(f"\nOutput hash   : {hash_outputs(out)}   <-- 同 seed+同权重应一致")