"""Class-safe post-processing for sliced and augmented detections."""

from __future__ import annotations

import numpy as np

from adaptive_sahi.geometry import box_iou_xyxy


def _as_detection_array(boxes: np.ndarray) -> np.ndarray:
    array = np.asarray(boxes, dtype=float)
    if array.size == 0:
        return np.zeros((0, 6), dtype=float)
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError("detections must have shape (N, 6): x1,y1,x2,y2,score,class_id")
    return array.copy()


def nms_per_class(boxes: np.ndarray, iou_threshold: float = 0.55) -> np.ndarray:
    """Run standard greedy NMS independently per class."""

    array = _as_detection_array(boxes)
    if array.size == 0:
        return array

    kept: list[np.ndarray] = []
    for class_id in sorted(set(array[:, 5].astype(int).tolist())):
        class_boxes = array[array[:, 5].astype(int) == class_id]
        order = np.argsort(-class_boxes[:, 4])
        class_boxes = class_boxes[order]

        while len(class_boxes):
            current = class_boxes[0]
            kept.append(current)
            if len(class_boxes) == 1:
                break
            ious = box_iou_xyxy(current[:4], class_boxes[1:, :4])
            class_boxes = class_boxes[1:][ious < iou_threshold]

    result = np.vstack(kept) if kept else np.zeros((0, 6), dtype=float)
    return result[np.argsort(-result[:, 4])]


def class_safe_weighted_fusion(boxes: np.ndarray, iou_threshold: float = 0.55) -> np.ndarray:
    """Fuse overlapping boxes using confidence-weighted coordinates per class.

    The output score is the maximum score in each cluster. This keeps the fused
    confidence conservative and makes the method a drop-in NMS replacement for
    evaluation files.
    """

    array = _as_detection_array(boxes)
    if array.size == 0:
        return array

    fused_rows: list[np.ndarray] = []
    for class_id in sorted(set(array[:, 5].astype(int).tolist())):
        class_boxes = array[array[:, 5].astype(int) == class_id]
        class_boxes = class_boxes[np.argsort(-class_boxes[:, 4])]
        clusters: list[list[np.ndarray]] = []

        for row in class_boxes:
            best_idx = -1
            best_iou = 0.0
            for idx, cluster in enumerate(clusters):
                fused_box = _weighted_cluster_row(cluster)[:4]
                iou = float(box_iou_xyxy(fused_box, row[:4][None, :])[0])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx >= 0 and best_iou >= iou_threshold:
                clusters[best_idx].append(row)
            else:
                clusters.append([row])

        fused_rows.extend(_weighted_cluster_row(cluster) for cluster in clusters)

    result = np.vstack(fused_rows) if fused_rows else np.zeros((0, 6), dtype=float)
    return result[np.argsort(-result[:, 4])]


def _weighted_cluster_row(cluster: list[np.ndarray]) -> np.ndarray:
    rows = np.vstack(cluster)
    weights = np.maximum(rows[:, 4], 1e-12)
    coords = np.average(rows[:, :4], axis=0, weights=weights)
    score = float(np.max(rows[:, 4]))
    class_id = rows[0, 5]
    return np.array([coords[0], coords[1], coords[2], coords[3], score, class_id], dtype=float)
