"""Geometry helpers for xyxy detection arrays.

The package standardizes detections as ``[x1, y1, x2, y2, score, class_id]``.
All coordinates are absolute pixel coordinates in the original image frame.
"""

from __future__ import annotations

import numpy as np


def _as_detection_array(boxes: np.ndarray) -> np.ndarray:
    array = np.asarray(boxes, dtype=float)
    if array.size == 0:
        return np.zeros((0, 6), dtype=float)
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError("detections must have shape (N, 6): x1,y1,x2,y2,score,class_id")
    return array.copy()


def clip_boxes(boxes: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Clip xyxy coordinates to image bounds while preserving score/class columns."""

    clipped = _as_detection_array(boxes)
    if clipped.size == 0:
        return clipped

    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, float(image_width))
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, float(image_height))
    return clipped


def remap_boxes_from_slice(
    boxes: np.ndarray,
    x_offset: int,
    y_offset: int,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Move slice-local detections back into global image coordinates."""

    remapped = _as_detection_array(boxes)
    if remapped.size == 0:
        return remapped

    remapped[:, [0, 2]] += float(x_offset)
    remapped[:, [1, 3]] += float(y_offset)
    return clip_boxes(remapped, image_width=image_width, image_height=image_height)


def unflip_boxes_horizontal(boxes: np.ndarray, image_width: int) -> np.ndarray:
    """Map detections from a horizontally flipped image back to the original frame."""

    restored = _as_detection_array(boxes)
    if restored.size == 0:
        return restored

    x1 = restored[:, 0].copy()
    x2 = restored[:, 2].copy()
    restored[:, 0] = float(image_width) - x2
    restored[:, 2] = float(image_width) - x1
    return restored


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one xyxy box and an array of xyxy boxes."""

    if boxes.size == 0:
        return np.zeros((0,), dtype=float)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
