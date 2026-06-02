"""Input/output helpers for experiment predictions and YOLO labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(path: Path | str) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"image path does not exist: {root}")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def read_image(path: Path | str) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


def image_size(path: Path | str) -> tuple[int, int]:
    image = read_image(path)
    height, width = image.shape[:2]
    return width, height


def load_yolo_labels_xyxy(label_path: Path | str, image_width: int, image_height: int) -> np.ndarray:
    """Load one YOLO label file as ``[x1,y1,x2,y2,score,class_id]``."""

    path = Path(label_path)
    if not path.exists() or path.read_text(encoding="utf-8").strip() == "":
        return np.zeros((0, 6), dtype=float)

    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"invalid YOLO label row in {path}: {line!r}")
        class_id = int(float(parts[0]))
        x_center, y_center, width, height = map(float, parts[1:5])
        abs_w = width * image_width
        abs_h = height * image_height
        x1 = x_center * image_width - abs_w / 2.0
        y1 = y_center * image_height - abs_h / 2.0
        x2 = x1 + abs_w
        y2 = y1 + abs_h
        rows.append([x1, y1, x2, y2, 1.0, float(class_id)])
    return np.asarray(rows, dtype=float) if rows else np.zeros((0, 6), dtype=float)


def save_prediction_jsonl(
    path: Path | str,
    predictions: dict[str, np.ndarray],
    per_image_metadata: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Save prediction arrays to a stable JSONL schema."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = per_image_metadata or {}

    with output.open("w", encoding="utf-8") as handle:
        for image_id in sorted(predictions):
            array = np.asarray(predictions[image_id], dtype=float)
            if array.size == 0:
                array = np.zeros((0, 6), dtype=float)
            if array.ndim != 2 or array.shape[1] != 6:
                raise ValueError(f"prediction for {image_id} must have shape (N, 6)")
            record = {
                "image_id": image_id,
                "detections": array.tolist(),
                "metadata": metadata.get(image_id, {}),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_prediction_jsonl(path: Path | str) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    predictions: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            image_id = record["image_id"]
            array = np.asarray(record.get("detections", []), dtype=float)
            if array.size == 0:
                array = np.zeros((0, 6), dtype=float)
            if array.ndim != 2 or array.shape[1] != 6:
                raise ValueError(f"prediction for {image_id} must have shape (N, 6)")
            predictions[image_id] = array
            metadata[image_id] = record.get("metadata", {})
    return predictions, metadata
