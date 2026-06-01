"""Lightweight detection evaluation for saved ASG-SAHI predictions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from adaptive_sahi.geometry import box_iou_xyxy


def evaluate_detection_map(
    predictions: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    class_ids: list[int],
    iou_thresholds: list[float] | None = None,
) -> dict:
    """Evaluate mAP for stored detection arrays.

    This is intentionally small and dependency-light. It is not a replacement for
    COCOeval, but it gives reproducible tables for ablations when pycocotools or
    the VisDrone server are unavailable.
    """

    thresholds = iou_thresholds or [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    per_class: dict[int, dict] = {}
    class_threshold_aps: dict[int, list[float]] = {}

    for class_id in class_ids:
        gt_count = _count_ground_truth_for_class(ground_truth, class_id)
        threshold_aps = [
            _average_precision_for_class(predictions, ground_truth, class_id, threshold)
            for threshold in thresholds
        ]
        class_threshold_aps[class_id] = threshold_aps
        per_class[class_id] = {
            "num_gt": gt_count,
            "AP50": float(threshold_aps[0]) if threshold_aps else 0.0,
            "AP50_95": float(np.mean(threshold_aps)) if threshold_aps else 0.0,
        }

    classes_with_gt = [class_id for class_id in class_ids if per_class[class_id]["num_gt"] > 0]
    if classes_with_gt:
        map50 = float(np.mean([per_class[class_id]["AP50"] for class_id in classes_with_gt]))
        map50_95 = float(np.mean([per_class[class_id]["AP50_95"] for class_id in classes_with_gt]))
    else:
        map50 = 0.0
        map50_95 = 0.0

    return {
        "mAP50": round(map50, 6),
        "mAP50_95": round(map50_95, 6),
        "per_class": per_class,
        "iou_thresholds": thresholds,
    }


def _count_ground_truth_for_class(ground_truth: dict[str, np.ndarray], class_id: int) -> int:
    total = 0
    for boxes in ground_truth.values():
        array = _as_array(boxes)
        if len(array):
            total += int(np.sum(array[:, 5].astype(int) == class_id))
    return total


def _average_precision_for_class(
    predictions: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    class_id: int,
    iou_threshold: float,
) -> float:
    gt_by_image: dict[str, np.ndarray] = {}
    matched_by_image: dict[str, np.ndarray] = {}
    total_gt = 0

    for image_id, boxes in ground_truth.items():
        class_gt = _as_array(boxes)
        class_gt = class_gt[class_gt[:, 5].astype(int) == class_id] if len(class_gt) else class_gt
        gt_by_image[image_id] = class_gt
        matched_by_image[image_id] = np.zeros(len(class_gt), dtype=bool)
        total_gt += len(class_gt)

    if total_gt == 0:
        return 0.0

    pred_rows: list[tuple[str, np.ndarray]] = []
    for image_id, boxes in predictions.items():
        class_pred = _as_array(boxes)
        if len(class_pred):
            class_pred = class_pred[class_pred[:, 5].astype(int) == class_id]
            for row in class_pred:
                pred_rows.append((image_id, row))

    if not pred_rows:
        return 0.0

    pred_rows.sort(key=lambda item: float(item[1][4]), reverse=True)
    tp = np.zeros(len(pred_rows), dtype=float)
    fp = np.zeros(len(pred_rows), dtype=float)

    for idx, (image_id, pred) in enumerate(pred_rows):
        gt_boxes = gt_by_image.get(image_id, np.zeros((0, 6), dtype=float))
        if len(gt_boxes) == 0:
            fp[idx] = 1.0
            continue

        ious = box_iou_xyxy(pred[:4], gt_boxes[:, :4])
        best_gt = int(np.argmax(ious)) if len(ious) else -1
        if best_gt >= 0 and ious[best_gt] >= iou_threshold and not matched_by_image[image_id][best_gt]:
            tp[idx] = 1.0
            matched_by_image[image_id][best_gt] = True
        else:
            fp[idx] = 1.0

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recall = cumulative_tp / max(total_gt, 1)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    return float(_average_precision_101(recall, precision))


def _average_precision_101(recall: np.ndarray, precision: np.ndarray) -> float:
    if len(recall) == 0:
        return 0.0
    values = []
    for threshold in np.linspace(0.0, 1.0, 101):
        eligible = precision[recall >= threshold]
        values.append(float(np.max(eligible)) if len(eligible) else 0.0)
    return float(np.mean(values))


def _as_array(boxes: np.ndarray) -> np.ndarray:
    array = np.asarray(boxes, dtype=float)
    if array.size == 0:
        return np.zeros((0, 6), dtype=float)
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError("boxes must have shape (N, 6)")
    return array


def summarize_metadata(per_image_metadata: dict[str, dict]) -> dict[str, float]:
    """Aggregate runtime metadata emitted by ``run_experiment.py``."""

    numeric_fields: dict[str, list[float]] = defaultdict(list)
    for metadata in per_image_metadata.values():
        for key, value in metadata.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_fields[key].append(float(value))

    summary: dict[str, float] = {}
    for key, values in numeric_fields.items():
        if values:
            summary[f"avg_{key}"] = float(np.mean(values))
    return summary
