#!/usr/bin/env python3
"""
Colorize FP32 ONNX segmentation output on a sample of test frames, with
optional side-by-side A/B comparison against a different backend's saved
masks (e.g. the DeepX INT8 masks dumped by `visualize_deepx.py --save-masks`).

Standalone mode (no --compare-masks):
  [ original | FP32 overlay | FP32 mask ]

A/B mode (with --compare-masks <dir>):
  [ original | FP32 overlay | <backend> overlay | disagreement (white where the
                                                  two argmaxes differ) ]

The same 3-class BGR palette as visualize_deepx.py (Background=blue,
Foreground=red, Road=green) so the two overlays are directly comparable.

Run from the repo root.  Any ORT-equipped venv is fine; Voyager venv works:

  source /media/abbeefai/TheExpanse/shared_with_docker/voyager-sdk/venv/bin/activate
  python Models/exports/onnx_rt/visualize_onnxrt.py \\
      --onnx /media/abbeefai/TheExpanse/shared_with_docker/mb-autoware/SceneSeg_FP32.onnx \\
      --image-src /media/abbeefai/TheExpanse/shared_with_docker/mb-autoware/bdd100k_images_100k/bdd100k/images/100k/val \\
      --out-dir Models/exports/onnx_rt/SceneSeg_viz \\
      --compare-masks Models/exports/deepx/SceneSeg_viz/masks \\
      --compare-label "DeepX INT8" \\
      --num-frames 12

The frame sampling uses the same --seed=42 default as the profile scripts so
the *N*-th frame is the same one used in `visualize_deepx.py` and the profile
tools.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime not installed in current Python env.")
    print("[ERROR]   try:   source /media/abbeefai/TheExpanse/shared_with_docker/voyager-sdk/venv/bin/activate")
    sys.exit(1)


IMAGENET_MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_RGB = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# BGR palette indexed by class id.  Matches visualize_deepx.py exactly.
PALETTE_BGR = np.array([
    [200, 100, 50],   # 0  Background Elements   -> muted blue
    [0,   0,   255],  # 1  Foreground Objects    -> red (production)
    [0,   200, 0],    # 2  Drivable Road Surface -> green
], dtype=np.uint8)

CLASS_NAMES = ["Background", "Foreground", "Road"]


def list_image_files(src: Path,
                     patterns=("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")) -> list[Path]:
    found: list[Path] = []
    for pat in patterns:
        found.extend(src.rglob(pat))
    return sorted(set(found))


def colorize_mask(argmax_hw: np.ndarray) -> np.ndarray:
    """(H, W) uint8 -> (H, W, 3) BGR."""
    return PALETTE_BGR[argmax_hw]


def annotate(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (340, 30), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def class_summary_str(argmax: np.ndarray) -> str:
    unique, counts = np.unique(argmax, return_counts=True)
    total = argmax.size
    return "  ".join(
        f"{CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else f'c{int(c)}'}={100.0*n/total:.0f}%"
        for c, n in sorted(zip(unique, counts), key=lambda x: -x[1]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True, type=Path)
    p.add_argument("--image-src", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--num-frames", type=int, default=12)
    p.add_argument("--seed", type=int, default=42,
                   help="Same default as profile scripts so frame indices match")
    p.add_argument("--overlay-alpha", type=float, default=0.45)
    p.add_argument("--input-hw", default="320,640",
                   help="Model input H,W (default: 320,640)")
    p.add_argument("--color-order", default="rgb", choices=["rgb", "bgr"],
                   help="Channel order to feed the model (default: rgb). The "
                        "original SceneSeg / DomainSeg / Scene3D were trained "
                        "with BGR per the production C++ OnnxRuntimeBackend "
                        "comment 'BGR order from original AUTOSEG' — pass 'bgr' "
                        "if results look wrong with the default. The mean/std "
                        "are applied in the chosen channel order.")
    p.add_argument("--providers", default="CPUExecutionProvider")
    p.add_argument("--compare-masks", type=Path, default=None,
                   help="Directory of pre-saved argmax masks from another backend "
                        "(uint8 PNGs at the model's native resolution, named "
                        "<image_stem>.png — e.g. visualize_deepx.py --save-masks "
                        "output).  When supplied, emits 4-panel A/B composites.")
    p.add_argument("--compare-label", default="Other",
                   help="Header label for the second-backend overlay panel "
                        "(default: 'Other')")
    p.add_argument("--self-label", default="FP32 CPU",
                   help="Header label for the first overlay panel (this script's "
                        "own ONNX). Default 'FP32 CPU' — override when profiling "
                        "the published INT8 ONNX or other variants.")
    args = p.parse_args()

    if not args.onnx.exists():
        print(f"[ERROR] ONNX not found: {args.onnx}")
        sys.exit(1)
    if not args.image_src.exists():
        print(f"[ERROR] Image source not found: {args.image_src}")
        sys.exit(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    in_h, in_w = (int(x) for x in args.input_hw.split(","))

    print(f"[INFO] Loading {args.onnx}")
    requested_eps = [ep.strip() for ep in args.providers.split(",") if ep.strip()]
    sess = ort.InferenceSession(str(args.onnx), providers=requested_eps)
    in_name = sess.get_inputs()[0].name
    out_shape = sess.get_outputs()[0].shape
    print(f"[INFO] ORT {ort.__version__}, providers: {sess.get_providers()}, "
          f"output shape: {out_shape}")

    images = list_image_files(args.image_src)
    if not images:
        print(f"[ERROR] No images found")
        sys.exit(1)
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(images), size=min(args.num_frames, len(images)), replace=False)
    sample = sorted([images[i] for i in idx])

    have_compare = args.compare_masks is not None and args.compare_masks.exists()
    if args.compare_masks is not None and not have_compare:
        print(f"[WARN] --compare-masks dir not found: {args.compare_masks}")
        print(f"[WARN] Falling back to standalone (3-panel) mode")

    n_frames_with_compare = 0
    n_disagree_pixels = 0
    n_total_pixels = 0

    for frame_no, img_path in enumerate(sample):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Could not read {img_path}; skipping")
            continue
        h0, w0 = img_bgr.shape[:2]

        # Full preprocessing (not baked into FP32 ONNX).  ImageNet stats are
        # written in RGB order; if feeding BGR, reverse them so that
        # channel-0 (B) gets B-mean, channel-1 (G) gets G-mean, etc.
        resized = cv2.resize(img_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        if args.color_order == "rgb":
            arr = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean, std = IMAGENET_MEAN_RGB, IMAGENET_STD_RGB
        else:  # bgr
            arr = resized.astype(np.float32) / 255.0
            mean = IMAGENET_MEAN_RGB[::-1].copy()
            std = IMAGENET_STD_RGB[::-1].copy()
        arr = (arr - mean) / std
        chw = np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[np.newaxis, ...])

        results = sess.run(None, {in_name: chw})
        fp32_argmax = np.argmax(results[0][0], axis=0).astype(np.uint8)  # (H_model, W_model)

        fp32_full = cv2.resize(fp32_argmax, (w0, h0), interpolation=cv2.INTER_NEAREST)
        fp32_color = colorize_mask(fp32_full)
        fp32_overlay = cv2.addWeighted(img_bgr, 1.0 - args.overlay_alpha,
                                       fp32_color, args.overlay_alpha, 0)

        cmp_path = args.compare_masks / f"{img_path.stem}.png" if have_compare else None
        if cmp_path is not None and cmp_path.exists():
            cmp_argmax = cv2.imread(str(cmp_path), cv2.IMREAD_GRAYSCALE)
            if cmp_argmax is None:
                print(f"[WARN] Could not read comparison mask {cmp_path}; skipping panel")
                cmp_argmax = None
        else:
            cmp_argmax = None
            if cmp_path is not None:
                print(f"[WARN] Missing comparison mask for {img_path.stem}; falling back to 3-panel")

        if cmp_argmax is not None:
            # Both argmaxes at model-native resolution for the diff math.
            cmp_full = cv2.resize(cmp_argmax, (w0, h0), interpolation=cv2.INTER_NEAREST)
            cmp_color = colorize_mask(cmp_full)
            cmp_overlay = cv2.addWeighted(img_bgr, 1.0 - args.overlay_alpha,
                                          cmp_color, args.overlay_alpha, 0)
            diff = (fp32_full != cmp_full).astype(np.uint8) * 255
            diff_bgr = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
            diff_pct = 100.0 * np.count_nonzero(diff) / diff.size

            n_disagree_pixels += int(np.count_nonzero(diff))
            n_total_pixels += int(diff.size)
            n_frames_with_compare += 1

            composite = np.hstack([
                annotate(img_bgr,      f"Original  {img_path.stem}"),
                annotate(fp32_overlay, f"{args.self_label}  ({class_summary_str(fp32_full)})"),
                annotate(cmp_overlay,  f"{args.compare_label}  ({class_summary_str(cmp_full)})"),
                annotate(diff_bgr,     f"Disagreement  {diff_pct:.1f}% of pixels"),
            ])
            extra = f"  diff={diff_pct:.1f}%"
        else:
            composite = np.hstack([
                annotate(img_bgr,      f"Original  {img_path.stem}"),
                annotate(fp32_overlay, f"{args.self_label}  ({class_summary_str(fp32_full)})"),
                annotate(fp32_color,   f"FP32 mask  {fp32_argmax.shape[1]}x{fp32_argmax.shape[0]} -> {w0}x{h0}"),
            ])
            extra = ""

        out_path = args.out_dir / f"frame_{frame_no:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  [{frame_no:02d}] {img_path.name} -> {out_path.name}  "
              f"({class_summary_str(fp32_full)}){extra}")

    print(f"\n[INFO] Wrote {len(sample)} composites to {args.out_dir}")
    if n_frames_with_compare > 0:
        overall_diff = 100.0 * n_disagree_pixels / n_total_pixels
        print(f"[INFO] A/B comparison: across {n_frames_with_compare} frames, "
              f"{args.self_label} and {args.compare_label} disagree on "
              f"{overall_diff:.2f}% of pixels overall")


if __name__ == "__main__":
    main()
