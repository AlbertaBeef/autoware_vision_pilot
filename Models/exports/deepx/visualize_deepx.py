#!/usr/bin/env python3
"""
Colorize SceneSeg .dxnn output on a sample of test frames.

For each frame, saves a side-by-side composite:
  [ original | semi-transparent class overlay | full-color mask ]

Class palette (matches production via run_model_node.cpp:
  class 0 (Background Elements) -> blue
  class 1 (Foreground Objects)  -> red       (this is the "production" class)
  class 2 (Drivable Road)        -> green

Run from the repo root after activating the DX-RT runtime venv:
  source /media/abbeefai/TheExpanse/dx-all-suite/dx-runtime/venv-dx-runtime/bin/activate
  python Models/exports/deepx/visualize_deepx.py \\
      --dxnn Models/exports/deepx/SceneSeg.dxnn \\
      --image-src /media/abbeefai/TheExpanse/shared_with_docker/mb-autoware/bdd100k_images_100k/bdd100k/images/100k/val \\
      --out-dir Models/exports/deepx/SceneSeg_viz \\
      --num-frames 12
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from dx_engine import InferenceEngine
except ImportError:
    print("[ERROR] Could not import dx_engine. Activate the runtime venv:")
    print("[ERROR]   source /media/abbeefai/TheExpanse/dx-all-suite/dx-runtime/venv-dx-runtime/bin/activate")
    sys.exit(1)


# BGR (OpenCV-native). Indexed by class id.  Matches run_model_node.cpp's
# "class 1 -> 255 (red foreground)" but expanded to all 3 classes for diagnosis.
PALETTE_BGR = np.array([
    [200, 100, 50],   # 0  Background Elements   -> muted blue
    [0,   0,   255],  # 1  Foreground Objects    -> red (production)
    [0,   200, 0],    # 2  Drivable Road Surface -> green
], dtype=np.uint8)

CLASS_NAMES = ["Background", "Foreground", "Road"]


def list_image_files(src: Path, patterns=("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")) -> list[Path]:
    found: list[Path] = []
    for pat in patterns:
        found.extend(src.rglob(pat))
    return sorted(set(found))


def colorize_mask(argmax_hw: np.ndarray) -> np.ndarray:
    """argmax_hw: (H, W) uint8 in [0, num_classes) -> (H, W, 3) BGR."""
    return PALETTE_BGR[argmax_hw]


def annotate(img: np.ndarray, text: str) -> np.ndarray:
    """Burn a small label at the top-left of a BGR image."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (240, 30), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dxnn", required=True, type=Path)
    p.add_argument("--image-src", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--num-frames", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overlay-alpha", type=float, default=0.45,
                   help="Overlay opacity (0=invisible, 1=opaque). Default 0.45.")
    p.add_argument("--save-masks", action="store_true",
                   help="Also save raw uint8 argmax masks (at the model's native "
                        "resolution) as <out-dir>/masks/<stem>.png so a later A/B "
                        "tool can pair them with another backend's masks on the "
                        "same frames.")
    args = p.parse_args()

    if not args.dxnn.exists():
        print(f"[ERROR] .dxnn not found: {args.dxnn}")
        sys.exit(1)
    if not args.image_src.exists():
        print(f"[ERROR] Image source not found: {args.image_src}")
        sys.exit(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = args.out_dir / "masks" if args.save_masks else None
    if masks_dir is not None:
        masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {args.dxnn}")
    engine = InferenceEngine(model_path=str(args.dxnn))
    in_info = engine.get_input_tensors_info()[0]
    out_info = engine.get_output_tensors_info()[0]
    in_shape = in_info.get("shape") if isinstance(in_info, dict) else in_info.shape
    out_shape = out_info.get("shape") if isinstance(out_info, dict) else out_info.shape
    _, in_h, in_w, _ = in_shape
    n_classes = out_shape[1]
    if n_classes > PALETTE_BGR.shape[0]:
        print(f"[WARN] Model has {n_classes} classes but palette only has {PALETTE_BGR.shape[0]}; "
              f"higher classes will index out-of-bounds")
    print(f"[INFO] Model input  {in_shape} (NHWC uint8); output {out_shape} (NCHW float32)")

    images = list_image_files(args.image_src)
    if not images:
        print(f"[ERROR] No images found")
        sys.exit(1)
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(images), size=min(args.num_frames, len(images)), replace=False)
    sample = sorted([images[i] for i in idx])
    print(f"[INFO] Visualizing {len(sample)} frames -> {args.out_dir}")

    for frame_no, img_path in enumerate(sample):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Could not read {img_path}; skipping")
            continue
        h0, w0 = img_bgr.shape[:2]

        # Preprocess: resize to model input, BGR->RGB, uint8 NHWC.
        # /255 + ImageNet normalize + transpose + expandDim are baked into
        # the .dxnn input wrapper at compile time.
        resized = cv2.resize(img_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        in_buf = np.ascontiguousarray(rgb[np.newaxis, ...])

        results = engine.run([in_buf])
        argmax = np.argmax(results[0][0], axis=0).astype(np.uint8)  # (H_model, W_model)

        if masks_dir is not None:
            cv2.imwrite(str(masks_dir / f"{img_path.stem}.png"), argmax)

        # Resize argmax back to original resolution for the overlay.
        argmax_full = cv2.resize(argmax, (w0, h0), interpolation=cv2.INTER_NEAREST)

        color_mask = colorize_mask(argmax_full)
        overlay = cv2.addWeighted(img_bgr, 1.0 - args.overlay_alpha,
                                  color_mask, args.overlay_alpha, 0)

        # Annotate
        label_orig = "Original"
        unique, counts = np.unique(argmax_full, return_counts=True)
        total_px = argmax_full.size
        class_summary = "  ".join(
            f"{CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else f'c{int(cls)}'}={100.0*cnt/total_px:.0f}%"
            for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]))
        label_overlay = f"Overlay  ({class_summary})"
        label_mask = f"Mask  ({argmax.shape[1]}x{argmax.shape[0]} -> {w0}x{h0})"

        composite = np.hstack([
            annotate(img_bgr, label_orig),
            annotate(overlay, label_overlay),
            annotate(color_mask, label_mask),
        ])

        out_path = args.out_dir / f"frame_{frame_no:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  [{frame_no:02d}] {img_path.name} -> {out_path.name}  ({class_summary})")

    print(f"\n[INFO] Wrote {len(sample)} composites to {args.out_dir}")


if __name__ == "__main__":
    main()
