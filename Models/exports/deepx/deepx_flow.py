#!/usr/bin/env python3
"""
DeepX M1 compile flow for AutoSeg-umbrella ONNX models.

Drives the DX-COM CLI (dxcom or dx_com, whichever is in PATH) to produce a
.dxnn binary. Handles JSON config generation and calibration-image staging
via numbered relative symlinks (zero-copy reuse of an existing scene set).

Example:
    python deepx_flow.py \\
        --onnx /home/abbeefai/Downloads/SceneSeg_Lite_FP32.onnx \\
        --out-dir Models/exports/deepx \\
        --name SceneSegLite \\
        --calib-src /media/.../BDD100K/.../images/100k/val \\
        --calib-num 200 \\
        --input-name input \\
        --input-shape 1,3,320,640
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers (skill rules: Rule 1 — probe both dxcom and dx_com binary names)
# ---------------------------------------------------------------------------

def find_dxcom() -> str:
    path = shutil.which("dxcom") or shutil.which("dx_com")
    if path is None:
        print("[ERROR] DX-COM compiler not found in PATH (tried both 'dxcom' and 'dx_com').")
        print("[ERROR] Activate the compiler venv first:")
        print("[ERROR]   source /media/abbeefai/TheExpanse/dx-all-suite/dx-compiler/venv-dx-compiler-local/bin/activate")
        sys.exit(1)
    return path


def stage_calibration(calib_src: Path, calib_dir: Path, n: int, seed: int,
                      patterns: tuple[str, ...]) -> None:
    """Sample N images from calib_src and expose them as numbered relative
    symlinks under calib_dir. Skill pattern: zero-copy, deployment-distribution
    matched, idempotent."""
    calib_dir.mkdir(parents=True, exist_ok=True)

    # Wipe previous (idempotent)
    for f in calib_dir.iterdir():
        if f.is_symlink() or f.is_file():
            f.unlink()

    if not calib_src.exists():
        print(f"[ERROR] Calibration source does not exist: {calib_src}")
        sys.exit(1)

    candidates = []
    for pat in patterns:
        candidates.extend(calib_src.rglob(pat))
    candidates = sorted(set(candidates))
    if not candidates:
        print(f"[ERROR] No images matching {patterns} under {calib_src}")
        sys.exit(1)
    print(f"[INFO] Found {len(candidates)} candidate calibration images under {calib_src}")

    rng = random.Random(seed)
    sample = rng.sample(candidates, k=min(n, len(candidates)))
    sample.sort()  # deterministic numbering after sampling

    for i, src in enumerate(sample):
        rel_target = os.path.relpath(src, calib_dir)
        suffix = src.suffix or ".jpg"
        (calib_dir / f"{i:04d}{suffix}").symlink_to(rel_target)

    print(f"[INFO] Staged {len(sample)} calibration symlinks under {calib_dir}")


def write_config(json_path: Path, input_name: str,
                 input_shape: tuple[int, int, int, int],
                 calib_dir: Path, calib_num: int,
                 calibration_method: str) -> None:
    """Generate the DX-COM JSON config for an ImageNet-normalized
    PyTorch-exported NCHW model.

    Skill rule 2: the field is `preprocessings` (PLURAL) — the singular form
    is silently ignored and the model compiles with no preprocessing, producing
    garbage at runtime."""
    n, c, h, w = input_shape
    cfg = {
        "inputs": {input_name: [n, c, h, w]},
        "calibration_num": calib_num,
        "calibration_method": calibration_method,
        "default_loader": {
            "dataset_path": str(calib_dir),
            "file_extensions": ["jpeg", "jpg", "png", "JPEG", "PNG"],
            "preprocessings": [
                {"convertColor": {"form": "BGR2RGB"}},
                {"resize": {"width": w, "height": h}},
                {"div": {"x": 255.0}},
                # ImageNet means/stds — RGB-ordered per the training pipeline
                # (Models/exports/lite_models/helpers.py SCENESEGLITE_DEFAULT_CONFIG)
                {"normalize": {"mean": [0.485, 0.456, 0.406],
                               "std":  [0.229, 0.224, 0.225]}},
                {"transpose": {"axis": [2, 0, 1]}},   # HWC -> CHW
                {"expandDim": {"axis": 0}},           # add batch dim
            ],
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[INFO] Wrote DX-COM config: {json_path}")


def run_dxcom(dxcom: str, onnx: Path, config: Path, out_dir: Path,
              opt_level: int, gen_log: bool, aggressive: bool,
              log_path: Path) -> None:
    """Drive DX-COM CLI; tee stdout/stderr to a per-model log."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [dxcom,
           "-m", str(onnx),
           "-c", str(config),
           "-o", str(out_dir),
           "--opt_level", str(opt_level)]
    if gen_log:
        cmd.append("--gen_log")
    if aggressive:
        cmd.append("--aggressive_partitioning")

    print(f"[INFO] Running: {' '.join(cmd)}")
    print(f"[INFO] Log: {log_path}")
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        logf.write(result.stdout)
        sys.stdout.write(result.stdout)

    if result.returncode != 0:
        print(f"[ERROR] DX-COM failed (rc={result.returncode}); log at {log_path}")
        sys.exit(result.returncode)
    print(f"[INFO] DX-COM finished successfully")


def rename_artifact(out_dir: Path, onnx_stem: str, target_name: str) -> Path:
    """DX-COM names its output after the source ONNX basename; rename to
    target_name.dxnn for consistency with the project artifact naming."""
    src = out_dir / f"{onnx_stem}.dxnn"
    dst = out_dir / f"{target_name}.dxnn"
    if src == dst:
        return dst
    if not src.exists():
        print(f"[WARN] Expected {src} not found; skipping rename")
        return src
    if dst.exists():
        dst.unlink()
    src.rename(dst)
    print(f"[INFO] Renamed {src.name} -> {dst.name}")
    return dst


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True, type=Path,
                   help="Path to FP32 ONNX model")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output directory for .dxnn artifact")
    p.add_argument("--name", default=None,
                   help="Stem name (default: derived from --onnx)")
    p.add_argument("--input-name", default="input",
                   help="ONNX graph input tensor name (default: input)")
    p.add_argument("--input-shape", default="1,3,320,640",
                   help="Input shape in source ONNX layout (NCHW for PyTorch, "
                        "NHWC for tf2onnx). Default: 1,3,320,640")
    p.add_argument("--calib-src", required=True, type=Path,
                   help="Directory containing source images for calibration "
                        "(any depth — script rglobs for .jpg/.png)")
    p.add_argument("--calib-num", type=int, default=200,
                   help="Number of calibration images to sample (default: 200)")
    p.add_argument("--calib-seed", type=int, default=42,
                   help="Random seed for calibration sample selection (default: 42)")
    p.add_argument("--calibration-method", default="minmax",
                   choices=["minmax", "ema"],
                   help="DX-COM calibration method (default: minmax)")
    p.add_argument("--opt-level", type=int, default=1, choices=[0, 1],
                   help="DX-COM --opt_level (default: 1 for deploy artifact)")
    p.add_argument("--gen-log", action="store_true",
                   help="Pass --gen_log to dxcom (collects compile logs into out dir)")
    p.add_argument("--aggressive-partitioning", action="store_true",
                   help="Pass --aggressive_partitioning to dxcom (more ops on NPU)")
    args = p.parse_args()

    if not args.onnx.exists():
        print(f"[ERROR] ONNX not found: {args.onnx}")
        sys.exit(1)

    name = args.name or args.onnx.stem
    input_shape = tuple(int(x) for x in args.input_shape.split(","))
    if len(input_shape) != 4:
        print(f"[ERROR] --input-shape must be 4 ints (N,C,H,W or N,H,W,C); got {input_shape}")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = args.out_dir / f"{name}_calib"
    config_path = args.out_dir / f"{name}.json"
    log_path = args.out_dir / f"{name}_compile.log"

    dxcom = find_dxcom()
    print(f"[INFO] DX-COM binary: {dxcom}")

    stage_calibration(args.calib_src, calib_dir, args.calib_num, args.calib_seed,
                      patterns=("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"))
    write_config(config_path, args.input_name, input_shape, calib_dir,
                 args.calib_num, args.calibration_method)
    run_dxcom(dxcom, args.onnx, config_path, args.out_dir,
              args.opt_level, args.gen_log, args.aggressive_partitioning,
              log_path)
    rename_artifact(args.out_dir, args.onnx.stem, name)


if __name__ == "__main__":
    main()
