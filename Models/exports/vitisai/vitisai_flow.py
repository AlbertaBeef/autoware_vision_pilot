#!/usr/bin/env python3
"""
AMD Ryzen AI / VitisAI compile flow for AutoSeg-umbrella ONNX models.

Uses AMD's `quark` ONNX quantizer (replacement for vai_q_onnx) to produce a
QDQ-quantized INT8 ONNX targeting the XDNA2 NPU. Runtime is then ONNX Runtime
with VitisAIExecutionProvider — see vitisai_run.py.

Prereqs (must be in the shell environment before invoking):
    source /opt/xilinx/xrt/setup.sh
    source ~/ryzen_ai_1.7.1/venv/bin/activate

Example (BDD100K calib, matches DeepX/Hailo baseline):
    python Models/exports/vitisai/vitisai_flow.py \\
        --onnx Models/exports/hailo/SceneSegLite_hailo_in.onnx \\
        --out-dir Models/exports/vitisai \\
        --name SceneSegLite_quark_bdd \\
        --calib-dir Models/exports/vitisai/SceneSegLite_calib_bdd \\
        --input-shape 1,3,320,640

Example (Waymo calib, deployment-tuned):
    python Models/exports/vitisai/vitisai_flow.py \\
        --onnx Models/exports/hailo/SceneSegLite_hailo_in.onnx \\
        --out-dir Models/exports/vitisai \\
        --name SceneSegLite_quark_waymo \\
        --calib-dir Models/exports/vitisai/SceneSegLite_calib_waymo \\
        --input-shape 1,3,320,640
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CalibrationDataReader — standard ORT interface, also accepted by quark
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr(bgr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Match the deepx_flow.py / hailo_flow.py preprocessing chain:
    BGR -> RGB -> resize(w,h) -> /255 -> ImageNet normalize -> HWC->CHW -> add batch."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(arr)


class ImageCalibrationDataReader:
    """Yields preprocessed image batches one at a time. Pattern from ORT docs:
    https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html"""

    def __init__(self, calib_dir: Path, input_name: str,
                 input_shape: tuple, max_images: int = None):
        self.input_name = input_name
        _, _, self.h, self.w = input_shape

        calib_dir = calib_dir.resolve()
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
        paths = []
        for pat in patterns:
            paths.extend(calib_dir.glob(pat))
        paths = sorted({p.resolve() for p in paths})
        if not paths:
            raise SystemExit(f"[ERROR] No images under {calib_dir}")
        if max_images is not None:
            paths = paths[:max_images]
        self.paths = paths
        self._iter = iter(paths)
        self._n_yielded = 0
        print(f"[INFO] CalibrationDataReader: {len(paths)} images "
              f"from {calib_dir}, resizing to {self.w}x{self.h}")

    def get_next(self):
        for p in self._iter:
            bgr = cv2.imread(str(p))
            if bgr is None:
                print(f"[WARN] Failed to read {p}, skipping")
                continue
            arr = preprocess_bgr(bgr, self.h, self.w)
            self._n_yielded += 1
            return {self.input_name: arr}
        if self._n_yielded == 0:
            raise SystemExit(f"[ERROR] CalibrationDataReader yielded 0 images "
                             f"out of {len(self.paths)} — all cv2.imread failures?")
        return None

    def rewind(self):
        self._iter = iter(self.paths)


# ---------------------------------------------------------------------------
# Quark quantization entry point
# ---------------------------------------------------------------------------

def quantize_xint8(onnx_in: Path, onnx_out: Path,
                   data_reader, config_name: str = "XINT8") -> None:
    """Apply quark's named INT8 quantization config to the model."""
    # Import here so the script can --help without the venv being active.
    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config import Config, get_default_config

    quant_config = get_default_config(config_name)
    config = Config(global_quant_config=quant_config)

    print(f"[INFO] Quark config: {config_name}")
    print(f"[INFO]   activation_type={quant_config.activation_type.name}, "
          f"weight_type={quant_config.weight_type.name}")
    print(f"[INFO]   calibrate_method={quant_config.calibrate_method.name}, "
          f"quant_format={quant_config.quant_format.name}")
    print(f"[INFO]   enable_npu_cnn={quant_config.enable_npu_cnn}, "
          f"include_cle={quant_config.include_cle}")

    quantizer = ModelQuantizer(config)
    t0 = time.perf_counter()
    quantizer.quantize_model(
        model_input=str(onnx_in),
        model_output=str(onnx_out),
        calibration_data_reader=data_reader,
    )
    print(f"[INFO] Quantization complete in {time.perf_counter() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True, type=Path,
                   help="Path to FP32 source ONNX model")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output directory for quantized .onnx artifact")
    p.add_argument("--name", default=None,
                   help="Output stem name (default: <onnx_stem>_quark)")
    p.add_argument("--input-name", default="input",
                   help="ONNX graph input tensor name (default: input)")
    p.add_argument("--input-shape", default="1,3,320,640",
                   help="Input shape NCHW (default: 1,3,320,640)")
    p.add_argument("--calib-dir", required=True, type=Path,
                   help="Pre-staged calibration directory (e.g. SceneSegLite_calib_bdd)")
    p.add_argument("--max-images", type=int, default=None,
                   help="Cap calibration to first N images (default: use all)")
    p.add_argument("--config", default="XINT8",
                   choices=["XINT8", "S8S8_AAWS", "A8W8", "A16W8"],
                   help="Quark named config (default: XINT8 — recommended for NPU)")
    args = p.parse_args()

    if not args.onnx.exists():
        print(f"[ERROR] ONNX not found: {args.onnx}")
        sys.exit(1)
    if not args.calib_dir.exists():
        print(f"[ERROR] Calib dir not found: {args.calib_dir}")
        sys.exit(1)

    name = args.name or f"{args.onnx.stem}_quark"
    input_shape = tuple(int(x) for x in args.input_shape.split(","))
    if len(input_shape) != 4:
        print(f"[ERROR] --input-shape must be 4 ints; got {input_shape}")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_out = args.out_dir / f"{name}.onnx"

    print(f"[INFO] Source ONNX: {args.onnx}")
    print(f"[INFO] Output ONNX: {onnx_out}")
    print(f"[INFO] Calib dir:   {args.calib_dir}")
    print(f"[INFO] Input shape: {input_shape}")

    reader = ImageCalibrationDataReader(
        calib_dir=args.calib_dir,
        input_name=args.input_name,
        input_shape=input_shape,
        max_images=args.max_images,
    )

    quantize_xint8(args.onnx, onnx_out, reader, config_name=args.config)
    print(f"[INFO] DONE. Quantized model: {onnx_out}")


if __name__ == "__main__":
    main()
