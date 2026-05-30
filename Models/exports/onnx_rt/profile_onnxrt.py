#!/usr/bin/env python3
"""
Profile an AutoSeg-umbrella FP32 ONNX via ONNX Runtime.

Mirrors `Models/exports/deepx/profile_deepx.py` for direct A/B comparison:
identical preprocessing chain, identical frame sampling (with --seed),
identical timing methodology, same class-histogram sanity check.

Key differences vs the DeepX flow:

  * Source artifact is the FP32 ONNX (not a quantized .dxnn), so we feed
    float32 NCHW with full ImageNet normalization (DX-COM's input-wrapper
    bakes in /255 + normalize, but here the host has to do it).
  * --providers selects ONNX Runtime execution providers in priority order.
    Default is CPUExecutionProvider — the host CPU baseline.  Override with
    ROCMExecutionProvider (Strix Halo iGPU), CoreMLExecutionProvider (mac),
    etc., when ORT is built with those EPs.

Run from the repo root.  An ORT-equipped venv is required; the Voyager-SDK
venv has ORT 1.23.2 / CPU only and works:

  source /media/abbeefai/TheExpanse/shared_with_docker/voyager-sdk/venv/bin/activate
  python Models/exports/onnx_rt/profile_onnxrt.py \\
      --onnx /media/abbeefai/TheExpanse/shared_with_docker/mb-autoware/SceneSeg_FP32.onnx \\
      --image-src /media/abbeefai/TheExpanse/shared_with_docker/mb-autoware/bdd100k_images_100k/bdd100k/images/100k/val \\
      --num-frames 100 \\
      --warmup 10
"""
import argparse
import sys
import time
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime not installed in current Python env.")
    print("[ERROR]   try:   source /media/abbeefai/TheExpanse/shared_with_docker/voyager-sdk/venv/bin/activate")
    print("[ERROR]   or:    source /media/abbeefai/TheExpanse/venv-mx/bin/activate")
    sys.exit(1)


IMAGENET_MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_RGB = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_image_files(src: Path,
                     patterns=("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")) -> list[Path]:
    found: list[Path] = []
    for pat in patterns:
        found.extend(src.rglob(pat))
    return sorted(set(found))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(np.floor(k))
    hi = int(np.ceil(k))
    if lo == hi:
        return s[lo]
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True, type=Path,
                   help="Path to the FP32 ONNX model")
    p.add_argument("--image-src", required=True, type=Path,
                   help="Directory of test images (rglob'd for jpg/png)")
    p.add_argument("--num-frames", type=int, default=100,
                   help="Number of timed inferences (default: 100)")
    p.add_argument("--warmup", type=int, default=10,
                   help="Warmup inferences (excluded from timing). Default: 10")
    p.add_argument("--seed", type=int, default=42,
                   help="Frame sample seed (default: 42 — matches profile_deepx.py)")
    p.add_argument("--providers", default="CPUExecutionProvider",
                   help="Comma-separated ORT EPs in priority order "
                        "(default: CPUExecutionProvider; try "
                        "ROCMExecutionProvider for Strix Halo iGPU, "
                        "or chain like 'ROCMExecutionProvider,CPUExecutionProvider')")
    p.add_argument("--input-hw", default="320,640",
                   help="Override model input H,W if needed (default: 320,640 — "
                        "every AutoSeg model uses this)")
    p.add_argument("--intra-op-threads", type=int, default=0,
                   help="ORT intra-op thread count (default 0 = ORT autopick)")
    p.add_argument("--inter-op-threads", type=int, default=0,
                   help="ORT inter-op thread count (default 0 = ORT autopick)")
    args = p.parse_args()

    if not args.onnx.exists():
        print(f"[ERROR] ONNX not found: {args.onnx}")
        sys.exit(1)
    if not args.image_src.exists():
        print(f"[ERROR] Image source not found: {args.image_src}")
        sys.exit(1)

    requested_eps = [ep.strip() for ep in args.providers.split(",") if ep.strip()]
    available_eps = set(ort.get_available_providers())
    missing = [ep for ep in requested_eps if ep not in available_eps]
    if missing:
        print(f"[ERROR] Requested EPs not available in this ORT build: {missing}")
        print(f"[ERROR] Available: {sorted(available_eps)}")
        sys.exit(1)

    in_h, in_w = (int(x) for x in args.input_hw.split(","))

    # --- Load model ---
    print(f"[INFO] Loading {args.onnx}")
    sess_options = ort.SessionOptions()
    if args.intra_op_threads:
        sess_options.intra_op_num_threads = args.intra_op_threads
    if args.inter_op_threads:
        sess_options.inter_op_num_threads = args.inter_op_threads
    sess = ort.InferenceSession(str(args.onnx), sess_options=sess_options,
                                providers=requested_eps)
    actual_eps = sess.get_providers()
    print(f"[INFO] ORT version {ort.__version__}, providers: {actual_eps}")

    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    print(f"[INFO] Model has {len(inputs)} input(s), {len(outputs)} output(s)")
    for i, ti in enumerate(inputs):
        print(f"[INFO]   input[{i}]:  name={ti.name}  shape={ti.shape}  dtype={ti.type}")
    for i, to in enumerate(outputs):
        print(f"[INFO]   output[{i}]: name={to.name}  shape={to.shape}  dtype={to.type}")

    if len(inputs) != 1:
        print(f"[ERROR] Expected 1 input, got {len(inputs)}")
        sys.exit(1)

    in_name = inputs[0].name
    out_shape = outputs[0].shape
    # Try to extract output channel count for segmentation vs depth detection.
    out_c: int | None = None
    if len(out_shape) == 4 and isinstance(out_shape[1], int):
        out_c = out_shape[1]

    # --- Stage test frames ---
    images = list_image_files(args.image_src)
    if not images:
        print(f"[ERROR] No images found under {args.image_src}")
        sys.exit(1)
    rng = np.random.RandomState(args.seed)
    n_total = args.warmup + args.num_frames
    if len(images) < n_total:
        print(f"[WARN] Only {len(images)} images available; sampling with replacement")
        idx = rng.randint(0, len(images), size=n_total)
    else:
        idx = rng.choice(len(images), size=n_total, replace=False)
    sample = [images[i] for i in idx]
    print(f"[INFO] Staged {n_total} test frames ({args.warmup} warmup + {args.num_frames} timed)")

    # --- Inference loop ---
    pre_times: list[float] = []
    run_times: list[float] = []
    post_times: list[float] = []
    last_output: np.ndarray | None = None

    for i, img_path in enumerate(sample):
        is_warmup = i < args.warmup

        # Preprocess: resize -> BGR2RGB -> /255 -> ImageNet normalize -> CHW.
        # Matches the SceneSeg / Lite-family training pipelines.
        t0 = time.perf_counter()
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read {img_path}; skipping")
            continue
        resized = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN_RGB) / IMAGENET_STD_RGB
        chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[np.newaxis, ...])
        t1 = time.perf_counter()

        # Inference
        results = sess.run(None, {in_name: chw})
        t2 = time.perf_counter()

        # Post: argmax for segmentation; otherwise stash the raw output
        if out_c is not None and out_c > 1:
            argmax = np.argmax(results[0][0], axis=0)  # (H, W)
            last_output = argmax
        else:
            last_output = results[0][0]
        t3 = time.perf_counter()

        if not is_warmup:
            pre_times.append((t1 - t0) * 1000.0)
            run_times.append((t2 - t1) * 1000.0)
            post_times.append((t3 - t2) * 1000.0)

    # --- Report ---
    def stats(label: str, vals: list[float]) -> None:
        if not vals:
            return
        print(f"  {label:14s} mean={mean(vals):7.2f} ms  median={median(vals):7.2f} ms  "
              f"p5={percentile(vals,5):7.2f}  p95={percentile(vals,95):7.2f}  "
              f"p99={percentile(vals,99):7.2f}  max={max(vals):7.2f}")

    total_times = [a + b + c for a, b, c in zip(pre_times, run_times, post_times)]

    print()
    print("=" * 90)
    print(f"ONNX Runtime profile: {args.onnx.name}")
    print(f"  Providers: {actual_eps}")
    print(f"  Intra-op threads: {args.intra_op_threads or '(auto)'}   "
          f"Inter-op threads: {args.inter_op_threads or '(auto)'}")
    print("=" * 90)
    print(f"Frames timed:    {len(run_times)}")
    print()
    print("Per-stage latency (ms):")
    stats("preprocess",  pre_times)
    stats("session.run", run_times)
    stats("postprocess", post_times)
    stats("total",       total_times)
    print()
    if total_times:
        mean_total_s = mean(total_times) / 1000.0
        median_total_s = median(total_times) / 1000.0
        print(f"FPS (1/total):   mean={1.0/mean_total_s:6.2f}   median={1.0/median_total_s:6.2f}")
        mean_run_s = mean(run_times) / 1000.0
        print(f"FPS (1/model):   mean={1.0/mean_run_s:6.2f}   (session.run only — upper bound)")

    # --- Sanity ---
    if last_output is not None and out_c is not None and out_c > 1:
        unique, counts = np.unique(last_output, return_counts=True)
        total_px = last_output.size
        print()
        print(f"Last-frame argmax class histogram ({out_c} classes total):")
        for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
            print(f"  class {int(cls):3d}: {cnt:7d} px  ({100.0 * cnt / total_px:5.2f}%)")


if __name__ == "__main__":
    main()
