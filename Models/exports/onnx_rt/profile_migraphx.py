#!/usr/bin/env python3
"""
Profile an AutoSeg-umbrella FP32 ONNX on the AMD Radeon iGPU via MIGraphX.

MIGraphX is AMD's official inference engine for Radeon GPUs (analog to
TensorRT for NVIDIA).  On Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S /
gfx1100 generic) it lives at /opt/rocm-7.2.0/{lib,bin}/ and ships Python
bindings (`migraphx.cpython-312-x86_64-linux-gnu.so`) that the script picks
up via PYTHONPATH.

Mirrors `Models/exports/onnx_rt/profile_onnxrt.py` for direct A/B comparison:
identical preprocessing chain, identical frame sampling (with --seed),
identical timing methodology, same class-histogram sanity check.

Run from the repo root with PYTHONPATH set to the ROCm libdir:

  PYTHONPATH=/opt/rocm-7.2.0/lib \\
  /media/abbeefai/TheExpanse/shared_with_docker/voyager-sdk/venv/bin/python \\
    Models/exports/onnx_rt/profile_migraphx.py \\
      --onnx /media/abbeefai/TheExpanse/shared_with_docker/mb-autoware/SceneSeg_FP32.onnx \\
      --image-src /media/abbeefai/TheExpanse/shared_with_docker/mb-autoware/bdd100k_images_100k/bdd100k/images/100k/val \\
      --num-frames 100 \\
      --warmup 10

Compile time is non-trivial (~10-60 s the first run) — MIGraphX optimizes the
graph for the local GPU.  The compile result isn't cached to disk by this
script, so each invocation pays the cost.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np

try:
    import migraphx
except ImportError:
    print("[ERROR] migraphx not importable.  Set PYTHONPATH=/opt/rocm-7.2.0/lib")
    print("[ERROR] (or wherever your ROCm install lives) and ensure the Python")
    print("[ERROR] version matches one of /opt/rocm-7.2.0/lib/migraphx.cpython-*")
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
    p.add_argument("--onnx", required=True, type=Path)
    p.add_argument("--image-src", required=True, type=Path)
    p.add_argument("--num-frames", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input-hw", default="320,640")
    p.add_argument("--color-order", default="rgb", choices=["rgb", "bgr"],
                   help="Channel order to feed the model (default: rgb to match "
                        "profile_onnxrt.py default). Original SceneSeg /DomainSeg/"
                        "Scene3D were trained BGR per the production C++ backend.")
    p.add_argument("--fp16", action="store_true",
                   help="Quantize FP32 weights+activations to FP16 before "
                        "compile (typical AMD GPU speedup ~1.5-2x).")
    p.add_argument("--target", default="gpu", choices=["gpu", "ref"],
                   help="MIGraphX target (default gpu).  'ref' = CPU reference, "
                        "for correctness debug.")
    p.add_argument("--cache-dir", type=Path,
                   default=Path("Models/exports/onnx_rt/migraphx_cache"),
                   help="Directory for cached compiled programs (msgpack .mxr). "
                        "A cache key derived from the ONNX mtime+size, target, "
                        "and precision auto-invalidates when the source changes. "
                        "Compiling SceneSeg from scratch on Strix Halo iGPU takes "
                        "~7 minutes; cached load takes <1 second.")
    p.add_argument("--no-cache", action="store_true",
                   help="Skip the cache (always recompile, don't write to cache)")
    args = p.parse_args()

    if not args.onnx.exists():
        print(f"[ERROR] ONNX not found: {args.onnx}")
        sys.exit(1)
    if not args.image_src.exists():
        print(f"[ERROR] Image source not found: {args.image_src}")
        sys.exit(1)

    in_h, in_w = (int(x) for x in args.input_hw.split(","))

    # --- Build cache key from ONNX state + flags ---
    precision = "fp16" if args.fp16 else "fp32"
    st = args.onnx.stat()
    key_str = f"{args.onnx.resolve()}|{st.st_mtime_ns}|{st.st_size}|{args.target}|{precision}"
    cache_hash = hashlib.sha256(key_str.encode()).hexdigest()[:12]
    cache_path = (args.cache_dir /
                  f"{args.onnx.stem}_{args.target}_{precision}_{cache_hash}.mxr")
    use_cache = not args.no_cache

    # --- Load from cache OR parse+compile+save ---
    prog: migraphx.program
    t_compile = 0.0
    cached_loaded = False
    if use_cache and cache_path.exists():
        print(f"[INFO] Loading cached compiled program: {cache_path}")
        t_load = time.perf_counter()
        prog = migraphx.load(str(cache_path), format="msgpack")
        t_load = time.perf_counter() - t_load
        print(f"[INFO] Cache load took {t_load:.2f} s")
        cached_loaded = True
    else:
        print(f"[INFO] Parsing {args.onnx}")
        prog = migraphx.parse_onnx(str(args.onnx), default_dim_value=1)

        if args.fp16:
            print(f"[INFO] Quantizing to FP16")
            migraphx.quantize_fp16(prog)

        print(f"[INFO] Compiling for target '{args.target}'  "
              f"(cold compile; can take many minutes on Strix Halo iGPU)")
        t_compile_start = time.perf_counter()
        target = migraphx.get_target(args.target)
        prog.compile(target)
        t_compile = time.perf_counter() - t_compile_start
        print(f"[INFO] Compile finished in {t_compile:.1f} s")

        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            t_save = time.perf_counter()
            migraphx.save(prog, str(cache_path), format="msgpack")
            t_save = time.perf_counter() - t_save
            sz_mb = cache_path.stat().st_size / 1e6
            print(f"[INFO] Saved compiled program to {cache_path} "
                  f"({sz_mb:.1f} MB, {t_save:.2f} s)")

    in_params = prog.get_parameter_shapes()
    out_shape = prog.get_output_shapes()
    print(f"[INFO] Input params:")
    for name, shape in in_params.items():
        print(f"[INFO]   {name}: {shape}")
    print(f"[INFO] Output shape: {out_shape}")

    # Figure out the single param name (AutoSeg models are all single-input)
    in_param_names = list(in_params.keys())
    if len(in_param_names) != 1:
        print(f"[WARN] Expected 1 input param, got {len(in_param_names)}: {in_param_names}")
    in_name = in_param_names[0]

    # Output channel count for segmentation vs depth detection.  out_shape is
    # a list with one element per output; each is a Shape object with .lens.
    out_c: int | None = None
    try:
        first_out = out_shape[0]
        if hasattr(first_out, "lens"):
            lens = first_out.lens()
        else:
            lens = list(first_out)
        if len(lens) == 4:
            out_c = lens[1]
    except Exception:
        pass

    # --- Stage test frames ---
    images = list_image_files(args.image_src)
    if not images:
        print(f"[ERROR] No images found")
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

        # Preprocess: resize -> color order -> /255 -> ImageNet normalize -> CHW
        t0 = time.perf_counter()
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read {img_path}; skipping")
            continue
        resized = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        if args.color_order == "rgb":
            arr = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean_v, std_v = IMAGENET_MEAN_RGB, IMAGENET_STD_RGB
        else:
            arr = resized.astype(np.float32) / 255.0
            mean_v = IMAGENET_MEAN_RGB[::-1].copy()
            std_v = IMAGENET_STD_RGB[::-1].copy()
        arr = (arr - mean_v) / std_v
        chw = np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[np.newaxis, ...])
        # NOTE: even with --fp16, MIGraphX expects float32 INPUT — `quantize_fp16`
        # only converts weights and intermediate compute to half, the param dtype
        # at the graph boundary stays float32.
        t1 = time.perf_counter()

        # Inference (run() blocks until GPU finishes for the 'gpu' target)
        results = prog.run({in_name: migraphx.argument(chw)})
        out_arr = np.asarray(results[0])
        t2 = time.perf_counter()

        # Post
        if out_c is not None and out_c > 1:
            argmax = np.argmax(out_arr[0], axis=0)
            last_output = argmax
        else:
            last_output = out_arr[0]
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
    precision_label = "FP16" if args.fp16 else "FP32"
    print(f"MIGraphX profile: {args.onnx.name}  ({precision_label}, target={args.target})")
    if cached_loaded:
        print(f"  Loaded from cache: {cache_path}")
    else:
        print(f"  Compile time: {t_compile:.1f} s (one-time, cached to {cache_path})")
    print("=" * 90)
    print(f"Frames timed:    {len(run_times)}")
    print()
    print("Per-stage latency (ms):")
    stats("preprocess",  pre_times)
    stats("prog.run()",  run_times)
    stats("postprocess", post_times)
    stats("total",       total_times)
    print()
    if total_times:
        mean_total_s = mean(total_times) / 1000.0
        median_total_s = median(total_times) / 1000.0
        print(f"FPS (1/total):   mean={1.0/mean_total_s:6.2f}   median={1.0/median_total_s:6.2f}")
        mean_run_s = mean(run_times) / 1000.0
        print(f"FPS (1/model):   mean={1.0/mean_run_s:6.2f}   (prog.run() only — upper bound)")

    if last_output is not None and out_c is not None and out_c > 1:
        unique, counts = np.unique(last_output, return_counts=True)
        total_px = last_output.size
        print()
        print(f"Last-frame argmax class histogram ({out_c} classes total):")
        for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
            print(f"  class {int(cls):3d}: {cnt:7d} px  ({100.0 * cnt / total_px:5.2f}%)")


if __name__ == "__main__":
    main()
