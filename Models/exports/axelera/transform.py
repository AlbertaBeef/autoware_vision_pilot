"""
Axelera calibration preprocessing for SceneSegLite / Scene3DLite.

The Voyager `axcompile` CLI requires a Python file exposing a callable
`get_preprocess_transform(image) -> torch.Tensor` that converts a raw
calibration image into the model input tensor (NCHW float32, normalized
exactly as the model was trained).

The Lite-family ONNX exports do NOT bake in resize/colorspace/ImageNet
normalization (see Models/exports/lite_models/helpers.py
`SCENESEGLITE_DEFAULT_CONFIG`), so we must reproduce them here.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

INPUT_H, INPUT_W = 320, 640
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_preprocess_transform(image):
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.asarray(image.convert("RGB"))

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {img.shape}")

    pil = Image.fromarray(img).resize((INPUT_W, INPUT_H), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(arr))
