"""Pseudo-label filtering for sliced-teacher distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from adaptive_sahi.geometry import box_iou_xyxy, clip_boxes


@dataclass(frozen=True)
class PseudoLabelConfig:
    class_ids: set[int]
    min_conf: float = 0.30
    max_area_ratio: float = 0.025
    same_class_iou_drop: float = 0.30
    cross_class_iou_drop: float = 0.40
    max_pseudo_per_image: int = 80


DROP_KEYS = (
    "dropped_class",
    "dropped_low_conf",
    "dropped_large_area",
    "dropped_same_class_gt",
    "dropped_cross_class_gt",
    "dropped_invalid",
    "dropped_max_pseudo",
)


def parse_class_ids(value: str | Iterable[int]) -> set[int]:
    """Parse comma-separated class IDs used by dataset build scripts."""

    if isinstance(value, str):
        if not value.strip():
            return set()
        return {int(item.strip()) for item in value.split(",") if item.strip()}
    return {int(item) for item in value}


def filter_pseudo_labels(
    teacher_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    image_width: int,
    image_height: int,
    image_id: str,
    config: PseudoLabelConfig,
) -> tuple[np.ndarray, list[dict], dict[str, int]]:
    """Filter teacher detections into pseudo labels for one image.

    Inputs and outputs use ``[x1,y1,x2,y2,score,class_id]`` arrays. The kept
    boxes are sorted by confidence descending and clipped to image bounds.
    """

    teacher = _as_detection_array(teacher_boxes)
    gt = _as_detection_array(gt_boxes)
    summary = {key: 0 for key in DROP_KEYS}
    kept_rows: list[np.ndarray] = []
    kept_records: list[dict] = []

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    image_area = float(image_width * image_height)
    teacher = clip_boxes(teacher, image_width=image_width, image_height=image_height)
    teacher = teacher[np.argsort(-teacher[:, 4])] if len(teacher) else teacher

    for row in teacher:
        class_id = int(row[5])
        score = float(row[4])
        width = max(0.0, float(row[2] - row[0]))
        height = max(0.0, float(row[3] - row[1]))
        area_ratio = (width * height) / image_area

        drop_reason = _drop_reason(row, gt, class_id, score, area_ratio, config)
        if drop_reason is not None:
            summary[drop_reason] += 1
            continue

        kept_rows.append(row)
        kept_records.append(
            {
                "image_id": image_id,
                "class_id": class_id,
                "score": round(score, 6),
                "xyxy": [round(float(v), 3) for v in row[:4]],
                "area_ratio": round(float(area_ratio), 8),
            }
        )

    if config.max_pseudo_per_image >= 0 and len(kept_rows) > config.max_pseudo_per_image:
        overflow = len(kept_rows) - config.max_pseudo_per_image
        kept_rows = kept_rows[: config.max_pseudo_per_image]
        kept_records = kept_records[: config.max_pseudo_per_image]
        summary["dropped_max_pseudo"] += overflow

    kept = np.vstack(kept_rows) if kept_rows else np.zeros((0, 6), dtype=float)
    return kept, kept_records, summary


def xyxy_to_yolo_row(row: np.ndarray, image_width: int, image_height: int) -> str:
    """Convert one detection row to a YOLO label row with clamped coordinates."""

    array = np.asarray(row, dtype=float)
    if array.shape[0] < 6:
        raise ValueError("row must contain x1,y1,x2,y2,score,class_id")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    x1 = min(max(float(array[0]), 0.0), float(image_width))
    y1 = min(max(float(array[1]), 0.0), float(image_height))
    x2 = min(max(float(array[2]), 0.0), float(image_width))
    y2 = min(max(float(array[3]), 0.0), float(image_height))
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    x_center = (x1 + x2) / 2.0 / image_width
    y_center = (y1 + y2) / 2.0 / image_height
    norm_width = box_width / image_width
    norm_height = box_height / image_height
    class_id = int(array[5])
    return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"


def summary_total(summary: dict[str, int]) -> int:
    """Return total dropped candidates from a per-image summary."""

    return int(sum(summary.get(key, 0) for key in DROP_KEYS))


def _drop_reason(
    row: np.ndarray,
    gt: np.ndarray,
    class_id: int,
    score: float,
    area_ratio: float,
    config: PseudoLabelConfig,
) -> str | None:
    if class_id not in config.class_ids:
        return "dropped_class"
    if score < config.min_conf:
        return "dropped_low_conf"
    if area_ratio <= 0.0:
        return "dropped_invalid"
    if area_ratio > config.max_area_ratio:
        return "dropped_large_area"
    if len(gt) == 0:
        return None

    same_class = gt[gt[:, 5].astype(int) == class_id]
    if len(same_class) and float(np.max(box_iou_xyxy(row[:4], same_class[:, :4]))) > config.same_class_iou_drop:
        return "dropped_same_class_gt"

    other_class = gt[gt[:, 5].astype(int) != class_id]
    if len(other_class) and float(np.max(box_iou_xyxy(row[:4], other_class[:, :4]))) > config.cross_class_iou_drop:
        return "dropped_cross_class_gt"
    return None


def _as_detection_array(boxes: np.ndarray) -> np.ndarray:
    array = np.asarray(boxes, dtype=float)
    if array.size == 0:
        return np.zeros((0, 6), dtype=float)
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError("boxes must have shape (N, 6): x1,y1,x2,y2,score,class_id")
    return array.copy()
