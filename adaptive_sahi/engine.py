"""Inference engine for full-image, fixed SAHI, and ASG-SAHI modes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from adaptive_sahi.geometry import clip_boxes, remap_boxes_from_slice, unflip_boxes_horizontal
from adaptive_sahi.merge import class_safe_weighted_fusion, nms_per_class
from adaptive_sahi.policy import SlicingPolicy, density_stats_from_predictions
from adaptive_sahi.slicing import SliceWindow, generate_slices


class Detector(Protocol):
    def predict(self, image: np.ndarray, imgsz: int | None = None, augment: bool = False) -> np.ndarray:
        """Return detections as ``[x1, y1, x2, y2, score, class_id]``."""


@dataclass(frozen=True)
class InferenceConfig:
    preview_imgsz: int = 640
    fixed_slice_size: int = 640
    fixed_overlap: float = 0.25
    merge_iou_threshold: float = 0.55
    postprocess: str = "auto"
    policy: SlicingPolicy = SlicingPolicy()


@dataclass(frozen=True)
class InferenceResult:
    detections: np.ndarray
    metadata: dict


VALID_MODES = {"full", "tta", "fixed_sahi", "fixed_sahi_wbf", "asg_sahi", "asg_sahi_tta"}


def run_image_inference(
    image: np.ndarray,
    detector: Detector,
    mode: str,
    config: InferenceConfig | None = None,
) -> InferenceResult:
    """Run one inference mode on a BGR image array."""

    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(VALID_MODES)}")
    if image.ndim != 3:
        raise ValueError("image must be an HxWxC array")

    cfg = config or InferenceConfig()
    height, width = image.shape[:2]
    started = time.perf_counter()
    metadata = {
        "mode": mode,
        "image_width": width,
        "image_height": height,
        "slice_count": 0,
        "hflip_tta_used": False,
    }

    if mode == "full":
        detections = clip_boxes(detector.predict(image, imgsz=cfg.preview_imgsz), width, height)

    elif mode == "tta":
        detections = _run_full_with_hflip_tta(image, detector, cfg)
        metadata["hflip_tta_used"] = True

    elif mode in {"fixed_sahi", "fixed_sahi_wbf"}:
        windows = generate_slices(width, height, cfg.fixed_slice_size, overlap=cfg.fixed_overlap)
        detections = _run_windows(image, detector, windows, cfg, use_hflip_tta=False)
        metadata["slice_count"] = len(windows)

    else:
        preview = clip_boxes(detector.predict(image, imgsz=cfg.preview_imgsz), width, height)
        stats = density_stats_from_predictions(preview, image_width=width, image_height=height)
        decision = cfg.policy.choose(stats, image_width=width, image_height=height)
        metadata.update(
            {
                "preview_boxes": int(len(preview)),
                "boxes_per_mpix": stats.boxes_per_mpix,
                "median_area_ratio": stats.median_area_ratio,
                "policy_level": decision.level,
                "policy_slice_size": decision.slice_size,
                "policy_overlap": decision.overlap,
            }
        )

        use_hflip_tta = mode == "asg_sahi_tta" and decision.enable_hflip_tta
        metadata["hflip_tta_used"] = use_hflip_tta
        if decision.use_full_image:
            detections = clip_boxes(detector.predict(image, imgsz=cfg.preview_imgsz), width, height)
            metadata["slice_count"] = 0
        else:
            windows = generate_slices(width, height, decision.slice_size, overlap=decision.overlap)
            detections = _run_windows(image, detector, windows, cfg, use_hflip_tta=use_hflip_tta)
            metadata["slice_count"] = len(windows)

    detections = _postprocess(detections, mode=mode, config=cfg)
    metadata["detections"] = int(len(detections))
    metadata["latency_ms"] = (time.perf_counter() - started) * 1000.0
    return InferenceResult(detections=detections, metadata=metadata)


def _run_full_with_hflip_tta(image: np.ndarray, detector: Detector, config: InferenceConfig) -> np.ndarray:
    height, width = image.shape[:2]
    direct = clip_boxes(detector.predict(image, imgsz=config.preview_imgsz), width, height)
    flipped = cv2.flip(image, 1)
    flipped_pred = detector.predict(flipped, imgsz=config.preview_imgsz)
    restored = clip_boxes(unflip_boxes_horizontal(flipped_pred, image_width=width), width, height)
    return np.vstack([direct, restored]) if len(restored) or len(direct) else np.zeros((0, 6), dtype=float)


def _run_windows(
    image: np.ndarray,
    detector: Detector,
    windows: list[SliceWindow],
    config: InferenceConfig,
    use_hflip_tta: bool,
) -> np.ndarray:
    height, width = image.shape[:2]
    rows: list[np.ndarray] = []
    for window in windows:
        crop = image[window.y1 : window.y2, window.x1 : window.x2]
        local = detector.predict(crop, imgsz=config.preview_imgsz)
        rows.append(
            remap_boxes_from_slice(
                local,
                x_offset=window.x1,
                y_offset=window.y1,
                image_width=width,
                image_height=height,
            )
        )

        if use_hflip_tta:
            flipped = cv2.flip(crop, 1)
            flipped_local = detector.predict(flipped, imgsz=config.preview_imgsz)
            restored_local = unflip_boxes_horizontal(flipped_local, image_width=window.width)
            rows.append(
                remap_boxes_from_slice(
                    restored_local,
                    x_offset=window.x1,
                    y_offset=window.y1,
                    image_width=width,
                    image_height=height,
                )
            )

    if not rows:
        return np.zeros((0, 6), dtype=float)
    non_empty = [row for row in rows if len(row)]
    return np.vstack(non_empty) if non_empty else np.zeros((0, 6), dtype=float)


def _postprocess(detections: np.ndarray, mode: str, config: InferenceConfig) -> np.ndarray:
    array = np.asarray(detections, dtype=float)
    if array.size == 0:
        return np.zeros((0, 6), dtype=float)
    if config.postprocess == "none":
        return array
    if config.postprocess == "nms":
        return nms_per_class(array, iou_threshold=config.merge_iou_threshold)
    if config.postprocess == "wbf":
        return class_safe_weighted_fusion(array, iou_threshold=config.merge_iou_threshold)
    if config.postprocess != "auto":
        raise ValueError("postprocess must be one of: auto, none, nms, wbf")

    if mode in {"fixed_sahi_wbf", "tta", "asg_sahi", "asg_sahi_tta"}:
        return class_safe_weighted_fusion(array, iou_threshold=config.merge_iou_threshold)
    return nms_per_class(array, iou_threshold=config.merge_iou_threshold)
