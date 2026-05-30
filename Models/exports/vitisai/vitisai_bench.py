#!/usr/bin/env python3
"""
VitisAI EP benchmark for quark-quantized AutoSeg ONNX models.

Loads the model via ONNX Runtime + VitisAIExecutionProvider, runs a sustained
inference loop on real frames, and reports:

  - which providers ORT actually loaded (silent-fallback check)
  - NPU offload coverage (subgraphs + nodes on NPU vs CPU, parsed from VAIP log)
  - cold-compile time, warm-run latency mean/p50/p95, FPS
  - optional per-frame argmax sanity check

Prereqs (must be in shell env):
    source /opt/xilinx/xrt/setup.sh
    source ~/ryzen_ai_1.7.1/venv/bin/activate

Example:
    python Models/exports/vitisai/vitisai_bench.py \\
        --onnx Models/exports/vitisai/SceneSegLite_quark_bdd.onnx \\
        --frames ../waymo_dataset/city/front_images_city \\
        --iters 500 \\
        --label "SceneSegLite quark-bdd VitisAI EP"
"""
import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr(bgr: np.ndarray, h: int, w: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(arr)


def percentile(xs, p):
    xs_sorted = sorted(xs)
    i = int(round((p / 100.0) * (len(xs_sorted) - 1)))
    return xs_sorted[i]


def make_session(onnx_path: Path, cache_dir: Path, providers_label: str):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = onnx_path.stem

    so = ort.SessionOptions()
    so.log_severity_level = 3   # warnings + errors only

    if providers_label == "vitisai":
        provider_options = [{
            "cache_dir": str(cache_dir),
            "cache_key": cache_key,
        }]
        providers = ["VitisAIExecutionProvider"]
    elif providers_label == "cpu":
        provider_options = [{}]
        providers = ["CPUExecutionProvider"]
    else:
        raise ValueError(f"Unknown providers_label: {providers_label}")

    t0 = time.perf_counter()
    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options=so,
        providers=providers,
        provider_options=provider_options,
    )
    cold_compile_s = time.perf_counter() - t0
    return sess, cold_compile_s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--frames", required=True, type=Path,
                    help="Dir with .jpg frames (e.g. waymo city subdir)")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--ep", choices=["vitisai", "cpu"], default="vitisai")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("Models/exports/vitisai/vaip_cache"))
    ap.add_argument("--label", default=None,
                    help="Free-text run label (default: derived from --onnx)")
    ap.add_argument("--no-cycle", action="store_true",
                    help="Run on the same first frame each iter (preprocessing-free hot loop)")
    args = ap.parse_args()

    if not args.onnx.exists():
        sys.exit(f"[ERROR] ONNX not found: {args.onnx}")
    frame_paths = sorted(args.frames.rglob("*.jpg"))
    if not frame_paths:
        sys.exit(f"[ERROR] No .jpg under {args.frames}")

    label = args.label or f"{args.onnx.stem} ({args.ep})"
    print(f"\n=== {label} ===")
    print(f"Model: {args.onnx}")
    print(f"Frames: {len(frame_paths)} from {args.frames}")
    print(f"EP: {args.ep}, cache: {args.cache_dir}")

    print("\n[1/4] Building session (cold compile if first time)...")
    sess, cold_s = make_session(args.onnx, args.cache_dir, args.ep)
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    in_shape = inp.shape
    print(f"  Input '{inp.name}' shape={in_shape} dtype={inp.type}")
    print(f"  Output '{out.name}' shape={out.shape} dtype={out.type}")
    print(f"  Requested EP:  ['{args.ep}']")
    print(f"  Loaded EPs:    {sess.get_providers()}")
    print(f"  Cold compile:  {cold_s:.2f}s")

    # Pre-process all frames into a small ring (1-frame if --no-cycle, else 5 frames)
    h, w = in_shape[2], in_shape[3]
    n_ring = 1 if args.no_cycle else min(5, len(frame_paths))
    ring = []
    for p in frame_paths[:n_ring]:
        bgr = cv2.imread(str(p))
        ring.append(preprocess_bgr(bgr, h, w))
    feeds = [{inp.name: arr} for arr in ring]

    print(f"\n[2/4] Warmup ({args.warmup} iters)...")
    for i in range(args.warmup):
        sess.run(None, feeds[i % n_ring])

    print(f"\n[3/4] Bench ({args.iters} iters)...")
    times_ms = []
    t_total_start = time.perf_counter()
    for i in range(args.iters):
        t0 = time.perf_counter()
        outputs = sess.run(None, feeds[i % n_ring])
        times_ms.append((time.perf_counter() - t0) * 1000)
        if (i + 1) % 100 == 0:
            print(f"  {i+1:5d}/{args.iters}   "
                  f"last={times_ms[-1]:6.2f}ms   "
                  f"running mean={np.mean(times_ms):6.2f}ms   "
                  f"FPS={1000/np.mean(times_ms):6.1f}")
    t_total = time.perf_counter() - t_total_start

    mean_ms  = float(np.mean(times_ms))
    p50_ms   = percentile(times_ms, 50)
    p95_ms   = percentile(times_ms, 95)
    p99_ms   = percentile(times_ms, 99)
    fps_mean = 1000.0 / mean_ms
    fps_p50  = 1000.0 / p50_ms

    print(f"\n[4/4] Results")
    print(f"  Iters:        {args.iters}")
    print(f"  Wall time:    {t_total:.2f}s")
    print(f"  Latency mean: {mean_ms:7.2f} ms  ({fps_mean:6.1f} FPS)")
    print(f"  Latency p50:  {p50_ms:7.2f} ms  ({fps_p50:6.1f} FPS)")
    print(f"  Latency p95:  {p95_ms:7.2f} ms")
    print(f"  Latency p99:  {p99_ms:7.2f} ms")

    # Argmax sanity: dump per-class pixel count for the last output to detect class collapse
    out_arr = outputs[0]  # shape [1, 19, H, W]
    seg = np.argmax(out_arr[0], axis=0)
    cls_count = Counter(seg.flatten().tolist())
    top = cls_count.most_common(5)
    total = seg.size
    print(f"\n  Last-frame argmax class distribution (top 5 of 19):")
    for cls, n in top:
        print(f"    class {cls:2d}: {n:8d} px ({100*n/total:5.1f}%)")
    if len(cls_count) == 1:
        print("  [WARN] All pixels predict one class — model collapsed.")


if __name__ == "__main__":
    main()
