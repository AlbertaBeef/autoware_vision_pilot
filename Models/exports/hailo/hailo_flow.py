#!/usr/bin/env python3
"""
Hailo-8 compile flow for AutoSeg-umbrella ONNX models.

Drives the Hailo Dataflow Compiler (hailo_sdk_client.ClientRunner) through
parse -> optimize/quantize -> compile, producing a .hef binary loadable by
HailoRT. Sticks ImageNet mean/std into the input layer's `normalization`
block so the host only has to feed raw uint8 NHWC at runtime.

Example:
    python hailo_flow.py \\
        --onnx /local/shared_with_docker/mb-autoware/SceneSeg_Lite_FP32.onnx \\
        --out-dir Models/exports/hailo \\
        --name SceneSegLite \\
        --calib-src /local/shared_with_docker/mb-autoware/bdd100k_images_100k/bdd100k/images/100k/val \\
        --calib-num 200 \\
        --hw-arch hailo8

The DFC writes a noisy pile of log files into cwd, so we chdir into the
output directory for the run.
"""
import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocess_onnx_for_hailo(src_onnx: Path, dst_onnx: Path) -> Path:
    """Patch ONNX in two ways so Hailo DFC's parser can accept it.

    1. Fill in missing `kernel_shape` attributes on Conv / ConvTranspose nodes
       from the weight tensor's spatial dims. PyTorch >= 2.5 + opset >= 18
       elide `kernel_shape` (legal per spec, since it's redundant with the
       weight shape) but Hailo's parser predates that change and crashes in
       `is_conv3d()` -> `get_kernel_shape()` with IndexError.
    2. (No-op today, room for future patches.)

    Writes the patched model to dst_onnx and returns the path. Idempotent —
    Convs that already carry `kernel_shape` are left untouched.
    """
    model = onnx.load(str(src_onnx))
    # Build a weight-name -> shape map for initializers (constant tensors).
    init_shapes = {init.name: tuple(init.dims) for init in model.graph.initializer}

    patched = 0
    for node in model.graph.node:
        if node.op_type not in ("Conv", "ConvTranspose"):
            continue
        if any(a.name == "kernel_shape" for a in node.attribute):
            continue
        # The weight tensor is the second input by ONNX convention.
        if len(node.input) < 2:
            continue
        w_name = node.input[1]
        if w_name not in init_shapes:
            # Weight is not a graph initializer (e.g. dynamic) - skip.
            continue
        w_shape = init_shapes[w_name]
        # Conv weight layout: (out_C, in_C/group, kH, kW [, kD ...])  → spatial dims are dims[2:].
        # ConvTranspose weight layout: (in_C, out_C/group, kH, kW [, ...])  → same: dims[2:].
        kernel_dims = list(w_shape[2:])
        if not kernel_dims:
            continue
        node.attribute.append(
            onnx.helper.make_attribute("kernel_shape", kernel_dims))
        patched += 1

    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst_onnx))
    print(f"[INFO] Patched {patched} Conv/ConvT nodes with kernel_shape attr → "
          f"{dst_onnx}")
    return dst_onnx

def _load_calibration(calib_src: Path, n: int, input_hw: tuple[int, int],
                      seed: int, patterns: tuple[str, ...]) -> np.ndarray:
    """Load N images, BGR2RGB + resize to (H, W), return (N, H, W, 3) uint8.

    Hailo expects calibration data in NHWC uint8 in the *input layer's*
    color order. With `normalization([mean*255], [std*255])` baked into the
    model script, calibration arrays carry raw 0..255 pixel values and the
    optimizer learns quant scales that absorb /255 + ImageNet normalize.
    BGR2RGB matches the SceneSegLite training pipeline (helpers.py).
    """
    if not calib_src.exists():
        print(f"[ERROR] Calibration source does not exist: {calib_src}")
        sys.exit(1)

    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(calib_src.rglob(pat))
    candidates = sorted(set(candidates))
    if not candidates:
        print(f"[ERROR] No images matching {patterns} under {calib_src}")
        sys.exit(1)
    print(f"[INFO] Found {len(candidates)} candidate calibration images "
          f"under {calib_src}")

    # Oversample so we can skip unreadable frames (e.g. half-copied jpegs from
    # an in-progress rsync) without falling short of the target.
    rng = random.Random(seed)
    oversample = min(int(n * 1.5) + 10, len(candidates))
    pool = rng.sample(candidates, k=oversample)
    pool.sort()

    h, w = input_hw
    frames: list[np.ndarray] = []
    skipped = 0
    for p in pool:
        if len(frames) == n:
            break
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            skipped += 1
            continue
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if len(frames) < n:
        print(f"[ERROR] Only loaded {len(frames)}/{n} usable calibration "
              f"images (skipped {skipped} unreadable); pool exhausted.")
        sys.exit(1)
    print(f"[INFO] Loaded {len(frames)} calibration images (seed={seed}, "
          f"skipped {skipped} unreadable)")
    return np.stack(frames, axis=0)


def _build_model_script(model_name: str, input_layer: str, num_calib: int,
                        compression_level: int, optimization_level: int,
                        compiler_optimization_level: int) -> str:
    """Compose a Hailo model script (alls file).

    - `normalization` folds ImageNet mean/std + /255 into input quant scales.
      Means/stds are scaled by 255 because the host hands the chip uint8
      [0..255], not float [0..1].
    - `model_optimization_flavor.optimization_level=2` enables QAT-finetune
      (slow on CPU, ~10 min); =1 skips it for fast iteration during debugging.
    - `compression_level` trades HEF size vs accuracy (0 = lossless, 2 = aggressive).
    - 16-bit precision on every avgpool: SceneSegLite at 320x640 makes every SE
      block's global avgpool see >=800 input elements (largest 51200 for SE on
      stem stride-2 outputs). a8 accumulator + max shift_delta=2 caps usable
      pool size at ~188; 16-bit accumulator clears it. 24 small layers, no
      meaningful FPS or memory hit.
    - `performance_param(compiler_optimization_level=1)` matches the
      hailo_model_zoo defaults for EfficientNet/DeepLab family models —
      simpler partition search, less likely to overflow HW mapping fields.
    """
    mean_x255 = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std_x255 = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    mean_s = ", ".join(f"{v:.6f}" for v in mean_x255)
    std_s = ", ".join(f"{v:.6f}" for v in std_x255)

    return f"""\
normalization1 = normalization([{mean_s}], [{std_s}], {input_layer})
model_optimization_flavor(optimization_level={optimization_level}, compression_level={compression_level}, batch_size=8)
quantization_param({{{model_name}/avgpool*}}, precision_mode=a16_w16)
performance_param(compiler_optimization_level={compiler_optimization_level})
"""


def _run_flow(onnx_path: Path, name: str, out_dir: Path, hw_arch: str,
              input_shape_nchw: tuple[int, int, int, int],
              calib_src: Path, calib_num: int, calib_seed: int,
              compression_level: int, optimization_level: int,
              compiler_optimization_level: int,
              end_node_names: list[str] | None,
              start_node_names: list[str] | None) -> Path:
    # Import inside the function so --help works even when hailo_sdk_client
    # has a slow / chatty first-time setup.
    from hailo_sdk_client import ClientRunner

    n, c, h, w = input_shape_nchw

    # Patch missing kernel_shape attrs (PyTorch 2.9 + opset 18 elides them,
    # Hailo DFC parser crashes without them). Write the patched copy to the
    # output dir so the original ONNX is untouched.
    patched_onnx = out_dir / f"{name}_hailo_in.onnx"
    onnx_path = _preprocess_onnx_for_hailo(onnx_path, patched_onnx)

    print(f"[INFO] Parsing {onnx_path} (hw_arch={hw_arch}, input NCHW={input_shape_nchw})")

    runner = ClientRunner(hw_arch=hw_arch)
    runner.translate_onnx_model(
        str(onnx_path),
        name,
        start_node_names=start_node_names,
        end_node_names=end_node_names,
        net_input_shapes={"input": [n, c, h, w]} if start_node_names is None else None,
    )

    parsed_har = out_dir / f"{name}_parsed.har"
    runner.save_har(str(parsed_har))
    print(f"[INFO] Saved parsed HAR: {parsed_har}")

    # Hailo expects calibration in NHWC at the input layer's H,W.
    calib_arr = _load_calibration(
        calib_src, calib_num, (h, w), calib_seed,
        patterns=("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"))
    print(f"[INFO] Calibration tensor: {calib_arr.shape} dtype={calib_arr.dtype}")

    # The input layer name in the HAR follows the ONNX input name, possibly
    # prefixed by the model name. ClientRunner exposes it via
    # runner.get_hn() / runner.hn_model; the model script's `normalization`
    # binds by layer name. The convention for ONNX-translated graphs is
    # "<model_name>/input_layer1" — we use that as the default. If the
    # parser names differently, the script will surface a clear error and
    # the user can override via env var (HAILO_INPUT_LAYER).
    input_layer = os.environ.get("HAILO_INPUT_LAYER", f"{name}/input_layer1")
    print(f"[INFO] Model script input layer: {input_layer}")

    script = _build_model_script(name, input_layer, calib_num,
                                 compression_level, optimization_level,
                                 compiler_optimization_level)
    alls_path = out_dir / f"{name}.alls"
    alls_path.write_text(script)
    print(f"[INFO] Wrote model script: {alls_path}")
    print("[INFO] Model script content:\n" + script)
    runner.load_model_script(str(alls_path))

    print(f"[INFO] Optimizing (INT8 PTQ, calibration_size={calib_num}) ...")
    runner.optimize(calib_arr)

    optimized_har = out_dir / f"{name}_optimized.har"
    runner.save_har(str(optimized_har))
    print(f"[INFO] Saved optimized HAR: {optimized_har}")

    print(f"[INFO] Compiling HEF ...")
    hef = runner.compile()

    hef_path = out_dir / f"{name}.hef"
    with open(hef_path, "wb") as f:
        f.write(hef)
    print(f"[INFO] HEF ready: {hef_path} "
          f"({hef_path.stat().st_size / 1e6:.2f} MB)")
    return hef_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True, type=Path,
                   help="Path to FP32 ONNX model")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output directory for .hef artifact and HAR snapshots")
    p.add_argument("--name", default=None,
                   help="Stem name (default: derived from --onnx)")
    p.add_argument("--hw-arch", default="hailo8",
                   choices=["hailo8", "hailo8l", "hailo15h", "hailo15m"],
                   help="Hailo hardware architecture (default: hailo8)")
    p.add_argument("--input-shape", default="1,3,320,640",
                   help="NCHW input shape (default: 1,3,320,640)")
    p.add_argument("--calib-src", required=True, type=Path,
                   help="Directory containing source images for calibration "
                        "(rglobbed for .jpg/.png).")
    p.add_argument("--calib-num", type=int, default=200,
                   help="Number of calibration images to sample (default: 200)")
    p.add_argument("--calib-seed", type=int, default=42,
                   help="Random seed for calibration sample selection (default: 42)")
    p.add_argument("--compression-level", type=int, default=0, choices=[0, 1, 2],
                   help="HEF weight compression: 0 lossless, 1 mild, 2 aggressive "
                        "(default: 0)")
    p.add_argument("--optimization-level", type=int, default=2, choices=[0, 1, 2, 3, 4],
                   help="model_optimization_flavor level. 2 enables QAT-finetune "
                        "(~10 min on CPU), 1 skips it. (default: 2)")
    p.add_argument("--compiler-optimization-level", type=int, default=1, choices=[0, 1, 2],
                   help="performance_param(compiler_optimization_level=X). 1 matches "
                        "hailo_model_zoo defaults — simpler partition search, less "
                        "likely to overflow HW mapping fields. (default: 1)")
    p.add_argument("--end-nodes", default=None,
                   help="Comma-separated list of ONNX output node names to cut "
                        "the graph at (default: use ONNX model outputs).")
    p.add_argument("--start-nodes", default=None,
                   help="Comma-separated list of ONNX input node names to cut "
                        "the graph at (default: use ONNX model inputs).")
    args = p.parse_args()

    if not args.onnx.exists():
        print(f"[ERROR] ONNX not found: {args.onnx}")
        sys.exit(1)

    name = args.name or args.onnx.stem
    # Sanitize: Hailo SDK rejects names with dots.
    name = name.replace(".", "_")

    shape = tuple(int(x) for x in args.input_shape.split(","))
    if len(shape) != 4:
        print(f"[ERROR] --input-shape must be 4 ints (NCHW); got {shape}")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # DFC writes acceleras.log / hailo_sdk.client.log / .install_logs/ into cwd.
    # Run from out_dir so those don't pollute the repo root and have write perms.
    # Resolve to absolute BEFORE chdir or downstream paths nest under
    # out_dir/out_dir.
    out_dir_abs = args.out_dir.resolve()
    onnx_abs = args.onnx.resolve()
    calib_abs = args.calib_src.resolve()
    orig_cwd = Path.cwd()
    os.chdir(out_dir_abs)
    try:
        end_nodes = args.end_nodes.split(",") if args.end_nodes else None
        start_nodes = args.start_nodes.split(",") if args.start_nodes else None
        _run_flow(
            onnx_path=onnx_abs,
            name=name,
            out_dir=out_dir_abs,
            hw_arch=args.hw_arch,
            input_shape_nchw=shape,
            calib_src=calib_abs,
            calib_num=args.calib_num,
            calib_seed=args.calib_seed,
            compression_level=args.compression_level,
            optimization_level=args.optimization_level,
            compiler_optimization_level=args.compiler_optimization_level,
            end_node_names=end_nodes,
            start_node_names=start_nodes,
        )
    finally:
        os.chdir(orig_cwd)


if __name__ == "__main__":
    main()
