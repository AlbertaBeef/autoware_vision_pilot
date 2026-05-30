#!/usr/bin/env python3
"""
Profile a compiled .dxnn artifact on the DeepX M1 NPU.

Loads the model via `dx_engine.InferenceEngine`, sanity-checks the IO tensor
shapes/dtypes match the DX-COM-baked NHWC-uint8-in / NCHW-float-out
convention, samples N BDD100K val frames, runs a warmup + timed inference
loop, and reports FPS percentiles, per-stage timing (preprocess / model /
post-argmax), plus a class histogram of the argmax output as a sanity check.

Designed to be reusable across all AutoSeg-umbrella models (SceneSeg,
DomainSeg, Scene3D, SceneSegLite, Scene3DLite) — auto-detects the output
shape and dispatches segmentation vs depth post-processing.

Run from the repo root after activating the DX-RT runtime venv:
  source /media/abbeefai/TheExpanse/dx-all-suite/dx-runtime/venv-dx-runtime/bin/activate
  python Models/exports/deepx/profile_deepx.py \\
      --dxnn Models/exports/deepx/SceneSeg.dxnn \\
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
    from dx_engine import InferenceEngine
except ImportError:
    print("[ERROR] Could not import dx_engine. Activate the runtime venv:")
    print("[ERROR]   source /media/abbeefai/TheExpanse/dx-all-suite/dx-runtime/venv-dx-runtime/bin/activate")
    sys.exit(1)


def list_image_files(src: Path, patterns=("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")) -> list[Path]:
    found: list[Path] = []
    for pat in patterns:
        found.extend(src.rglob(pat))
    return sorted(set(found))


def percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of `values` (0 <= p <= 100)."""
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
    p.add_argument("--dxnn", required=True, type=Path,
                   help="Path to the compiled .dxnn artifact")
    p.add_argument("--image-src", required=True, type=Path,
                   help="Directory of test images (rglob'd for jpg/png)")
    p.add_argument("--num-frames", type=int, default=100,
                   help="Number of timed inferences (default: 100)")
    p.add_argument("--warmup", type=int, default=10,
                   help="Warmup inferences (excluded from timing). Default: 10")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for frame sampling (default: 42)")
    args = p.parse_args()

    if not args.dxnn.exists():
        print(f"[ERROR] .dxnn not found: {args.dxnn}")
        sys.exit(1)
    if not args.image_src.exists():
        print(f"[ERROR] Image source not found: {args.image_src}")
        sys.exit(1)

    # --- Load model ---
    print(f"[INFO] Loading {args.dxnn}")
    engine = InferenceEngine(model_path=str(args.dxnn))

    inputs_info = engine.get_input_tensors_info()
    outputs_info = engine.get_output_tensors_info()
    print(f"[INFO] Model has {len(inputs_info)} input(s), {len(outputs_info)} output(s)")
    for i, ti in enumerate(inputs_info):
        print(f"[INFO]   input[{i}]:  {ti}")
    for i, to in enumerate(outputs_info):
        print(f"[INFO]   output[{i}]: {to}")

    if len(inputs_info) != 1:
        print(f"[ERROR] Expected 1 input, got {len(inputs_info)}")
        sys.exit(1)

    # DX-COM rewrites PyTorch NCHW source to NHWC at the runtime boundary.
    # Expect input shape [1, H, W, C] uint8 with /255+ImageNet baked in.
    in_info = inputs_info[0]
    in_shape = in_info.get("shape") if isinstance(in_info, dict) else in_info.shape
    in_dtype = in_info.get("type") if isinstance(in_info, dict) else getattr(in_info, "type", None)
    print(f"[INFO] Input shape: {in_shape}, dtype: {in_dtype}")
    if len(in_shape) != 4:
        print(f"[ERROR] Expected 4D input, got {len(in_shape)}D")
        sys.exit(1)
    # NHWC: [N, H, W, C]
    _, in_h, in_w, in_c = in_shape

    out_info = outputs_info[0]
    out_shape = out_info.get("shape") if isinstance(out_info, dict) else out_info.shape
    print(f"[INFO] Output shape: {out_shape}")
    # NCHW: [N, C, H, W] for segmentation/depth
    out_c = out_shape[1] if len(out_shape) == 4 else None

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

        # Preprocess
        t0 = time.perf_counter()
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read {img_path}; skipping")
            continue
        resized = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        in_buf = np.ascontiguousarray(rgb[np.newaxis, ...])  # uint8 NHWC [1, H, W, 3]
        t1 = time.perf_counter()

        # Inference
        results = engine.run([in_buf])
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
        print(f"  {label:14s} mean={mean(vals):6.2f} ms  median={median(vals):6.2f} ms  "
              f"p5={percentile(vals,5):6.2f}  p95={percentile(vals,95):6.2f}  "
              f"p99={percentile(vals,99):6.2f}  max={max(vals):6.2f}")

    total_times = [a + b + c for a, b, c in zip(pre_times, run_times, post_times)]

    print()
    print("=" * 78)
    print(f"DeepX profile: {args.dxnn.name}")
    print("=" * 78)
    print(f"Frames timed:    {len(run_times)}")
    print()
    print("Per-stage latency (ms):")
    stats("preprocess",  pre_times)
    stats("model.run()", run_times)
    stats("postprocess", post_times)
    stats("total",       total_times)
    print()
    if total_times:
        mean_total_s = mean(total_times) / 1000.0
        median_total_s = median(total_times) / 1000.0
        print(f"FPS (1/total):   mean={1.0/mean_total_s:6.2f}   median={1.0/median_total_s:6.2f}")
        # Model-only FPS — the upper bound if preprocess+post were free
        mean_run_s = mean(run_times) / 1000.0
        print(f"FPS (1/model):   mean={1.0/mean_run_s:6.2f}   (model.run() only — upper bound)")

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
