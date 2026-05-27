#!/usr/bin/env python3
"""
Axelera Metis compile flow for AutoSeg-umbrella ONNX models.

Drives the Voyager SDK `axcompile` CLI (which wraps axelera.compiler.api)
to produce a compiled-model directory containing manifest.json + model_*.json
+ kernel_function.c + pool_*.bin etc.  This directory is what the
`axelera.runtime` C++/Python API consumes.

Three source-ONNX preprocessing steps happen before axcompile:

  1. Fill missing `kernel_shape` attributes on Conv / ConvTranspose nodes from
     the weight tensor's spatial dims. PyTorch >= 2.5 + opset >= 18 elide
     this attribute (legal per spec, since it's redundant with the weight
     shape), but the Voyager constraint checker crashes on it
     ('NoneType' object is not subscriptable) and falls back to printing
     spurious 'Unsatisfied constraint' warnings on every Conv. Same fix as
     hailo_flow.py.

  2. Downgrade the ONNX from opset 18 to opset 17. The Voyager SDK's
     onnx2torch importer ships without a converter for opset-18 ReduceMean
     (where `axes` became an input instead of an attribute) and dies with
     `NotImplementedError: Converter is not implemented (ReduceMean,
     version=18)`. The default Voyager conf.json also pins
     `onnx_opset_version: 17`. `onnx.version_converter.convert_version()`
     does the input-to-attribute migration per the ONNX spec.

  3. Stage calibration images as numbered relative symlinks under
     `<out-dir-parent>/<name>_calib/` — zero-copy, idempotent, matched to
     the deployment distribution. Same staging pattern as deepx_flow.py.

The Voyager SDK ships a dedicated venv that pins protobuf 3.x / numpy 1.x;
keep this venv pristine and do not install tf2onnx / onnxsim into it.

Example:
    source /media/abbeefai/TheExpanse/shared_with_docker/voyager-sdk/venv/bin/activate
    python Models/exports/axelera/axelera_flow.py \\
        --onnx /media/abbeefai/TheExpanse/shared_with_docker/mb-autoware/SceneSeg_Lite_FP32.onnx \\
        --out-dir Models/exports/axelera/SceneSegLite \\
        --name SceneSegLite \\
        --transform Models/exports/axelera/transform.py \\
        --calib-src /media/abbeefai/TheExpanse/shared_with_docker/mb-autoware/bdd100k_images_100k/bdd100k/images/100k/val \\
        --calib-num 200 \\
        --input-shape 1,3,320,640
"""
import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

VOYAGER_TARGET_OPSET = 17

# Attributes that exist in opset 18 but not opset 17.  We strip them when
# downgrading so the opset-17 ONNX checker accepts the model.  Defaults are
# spec-compliant no-ops, so dropping them does not change semantics.
OPSET18_ONLY_ATTRS = {
    "ReduceMean": {"noop_with_empty_axes"},
    "Resize": {"antialias", "keep_aspect_ratio_policy", "axes"},
}


def _strip_opset18_only_attrs(model: onnx.ModelProto) -> int:
    """Remove attributes added in opset 18 that opset 17 doesn't recognize."""
    n_stripped = 0
    for node in model.graph.node:
        bad = OPSET18_ONLY_ATTRS.get(node.op_type)
        if not bad:
            continue
        for i in range(len(node.attribute) - 1, -1, -1):
            if node.attribute[i].name in bad:
                del node.attribute[i]
                n_stripped += 1
    return n_stripped


def _downgrade_reducemean_18_to_17(model: onnx.ModelProto) -> int:
    """Rewrite each ReduceMean from opset-18 form (axes as 2nd input) back to
    opset-17 form (axes as attribute).

    onnx.version_converter chokes mid-graph on unrelated ops (Resize) that
    don't actually need rewriting between 17 and 18, so we do the surgical
    fix ourselves.  Returns the number of nodes rewritten.
    """
    inits = {init.name: init for init in model.graph.initializer}
    rewritten = 0
    for node in model.graph.node:
        if node.op_type != "ReduceMean":
            continue
        if len(node.input) < 2:
            continue
        axes_in = node.input[1]
        if axes_in not in inits:
            continue
        axes_vals = numpy_helper.to_array(inits[axes_in]).astype(np.int64).tolist()
        if any(a.name == "axes" for a in node.attribute):
            continue
        node.attribute.append(onnx.helper.make_attribute("axes", axes_vals))
        del node.input[1]
        rewritten += 1
    return rewritten


def _preprocess_onnx_for_axelera(src_onnx: Path, dst_onnx: Path) -> Path:
    """Patch the source ONNX so Voyager `axcompile` can consume it.

    Steps (in order):
      a. Rewrite ReduceMean opset-18 (axes-as-input) → opset-17 (axes-as-attr).
         Voyager's onnx2torch importer has no converter for opset-18 ReduceMean
         and crashes the quantize child process.
      b. Fill missing `kernel_shape` attributes on Conv / ConvTranspose nodes
         from the weight tensor's spatial dims (PyTorch 2.5+ + opset 18 elide
         them and Voyager's constraint checker crashes evaluating them).
      c. Stamp the default-domain opset to 17 in the opset_import.  All other
         ops in this graph (Conv, Sigmoid, Mul, Add, Relu, Resize, Concat) are
         identical between 17 and 18 so no further rewrite is needed.
    """
    model = onnx.load(str(src_onnx))

    n_rm = _downgrade_reducemean_18_to_17(model)
    if n_rm:
        print(f"[INFO] Rewrote {n_rm} ReduceMean nodes from opset-18 to opset-17 form")

    n_strip = _strip_opset18_only_attrs(model)
    if n_strip:
        print(f"[INFO] Stripped {n_strip} opset-18-only attributes from ReduceMean/Resize")

    init_shapes = {init.name: tuple(init.dims) for init in model.graph.initializer}
    patched = 0
    for node in model.graph.node:
        if node.op_type not in ("Conv", "ConvTranspose"):
            continue
        if any(a.name == "kernel_shape" for a in node.attribute):
            continue
        if len(node.input) < 2:
            continue
        w_name = node.input[1]
        if w_name not in init_shapes:
            continue
        kernel_dims = list(init_shapes[w_name][2:])
        if not kernel_dims:
            continue
        node.attribute.append(onnx.helper.make_attribute("kernel_shape", kernel_dims))
        patched += 1

    for op in model.opset_import:
        if op.domain in ("", "ai.onnx") and op.version > VOYAGER_TARGET_OPSET:
            op.version = VOYAGER_TARGET_OPSET

    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst_onnx))
    print(f"[INFO] Patched {patched} Conv/ConvT nodes with kernel_shape attr → {dst_onnx}")
    return dst_onnx


def stage_calibration(calib_src: Path, calib_dir: Path, n: int, seed: int,
                      patterns: tuple[str, ...]) -> None:
    """Sample N images from calib_src, expose as numbered relative symlinks
    under calib_dir.  Idempotent (wipes prior symlinks first)."""
    calib_dir.mkdir(parents=True, exist_ok=True)
    for f in list(calib_dir.iterdir()):
        if f.is_symlink() or f.is_file():
            f.unlink()
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
    print(f"[INFO] Found {len(candidates)} candidate calibration images under {calib_src}")

    rng = random.Random(seed)
    sample = rng.sample(candidates, k=min(n, len(candidates)))
    sample.sort()

    for i, src in enumerate(sample):
        rel = os.path.relpath(src, calib_dir)
        suffix = src.suffix or ".jpg"
        (calib_dir / f"{i:04d}{suffix}").symlink_to(rel)

    print(f"[INFO] Staged {len(sample)} calibration symlinks under {calib_dir}")


def find_axcompile() -> str:
    path = shutil.which("axcompile")
    if path is None:
        print("[ERROR] axcompile not found in PATH.  Activate the Voyager SDK venv:")
        print("[ERROR]   source /media/abbeefai/TheExpanse/shared_with_docker/voyager-sdk/venv/bin/activate")
        sys.exit(1)
    return path


def run_axcompile(axcompile: str, onnx_path: Path, transform: Path, imageset: Path,
                  out_dir: Path, dataset_len: int, input_shape: str,
                  input_data_layout: str, log_level: str,
                  overwrite: bool, log_path: Path) -> int:
    cmd = [axcompile,
           "--input", str(onnx_path),
           "--transform", str(transform),
           "--imageset", str(imageset),
           "--dataset-len", str(dataset_len),
           "--input-shape", input_shape,
           "--input-data-layout", input_data_layout,
           "--output", str(out_dir),
           "--log-level", log_level]
    if overwrite:
        cmd.append("--overwrite")

    print(f"[INFO] Running: {' '.join(cmd)}")
    print(f"[INFO] Log: {log_path}")
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        logf.write(result.stdout)
        sys.stdout.write(result.stdout)
    return result.returncode


def check_artifact(out_dir: Path) -> bool:
    """axcompile sometimes exits 0 even when its child Process crashed in
    quantization (the parent returncode is decoupled from the child).  Look
    for manifest.json as the authoritative success marker."""
    manifest = out_dir / "manifest.json"
    if manifest.exists():
        return True
    # Look one level deeper - some Voyager flows nest the artifacts
    for child in out_dir.iterdir():
        if child.is_dir() and (child / "manifest.json").exists():
            return True
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True, type=Path,
                   help="Path to FP32 ONNX model")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output directory for the compiled model artifact")
    p.add_argument("--name", default=None,
                   help="Stem name (default: derived from --onnx)")
    p.add_argument("--transform", required=True, type=Path,
                   help="Path to Python file exposing get_preprocess_transform()")
    p.add_argument("--calib-src", required=True, type=Path,
                   help="Source directory containing calibration images (rglob'd)")
    p.add_argument("--calib-num", type=int, default=200,
                   help="Number of calibration images to sample (default: 200)")
    p.add_argument("--calib-seed", type=int, default=42,
                   help="Random seed for calibration sample selection (default: 42)")
    p.add_argument("--input-shape", default="1,3,320,640",
                   help="ONNX input shape (default: 1,3,320,640)")
    p.add_argument("--input-data-layout", default="NCHW", choices=["NCHW", "NHWC"],
                   help="Input data layout (default: NCHW)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="axcompile log level (default: INFO)")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow overwriting an existing --out-dir")
    args = p.parse_args()

    if not args.onnx.exists():
        print(f"[ERROR] ONNX not found: {args.onnx}")
        sys.exit(1)
    if not args.transform.exists():
        print(f"[ERROR] Transform file not found: {args.transform}")
        sys.exit(1)

    name = args.name or args.onnx.stem
    parent_dir = args.out_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    patched_onnx = parent_dir / f"{name}_axelera_in.onnx"
    calib_dir = parent_dir / f"{name}_calib"
    log_path = parent_dir / f"{name}_compile.log"

    _preprocess_onnx_for_axelera(args.onnx, patched_onnx)
    stage_calibration(args.calib_src, calib_dir, args.calib_num, args.calib_seed,
                      patterns=("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"))

    axcompile = find_axcompile()
    print(f"[INFO] axcompile binary: {axcompile}")

    rc = run_axcompile(axcompile, patched_onnx, args.transform, calib_dir,
                       args.out_dir, args.calib_num, args.input_shape,
                       args.input_data_layout, args.log_level,
                       args.overwrite, log_path)
    if rc != 0:
        print(f"[ERROR] axcompile failed (rc={rc}); log at {log_path}")
        sys.exit(rc)

    if not check_artifact(args.out_dir):
        print(f"[ERROR] axcompile returned 0 but no manifest.json was produced. "
              f"The child Quantize/Compile process likely crashed - check {log_path}.")
        sys.exit(2)

    print(f"[INFO] axcompile finished successfully; artifact at {args.out_dir}")


if __name__ == "__main__":
    main()
