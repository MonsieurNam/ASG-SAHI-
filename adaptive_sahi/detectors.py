"""Detector adapters used by experiment scripts."""

from __future__ import annotations

import numpy as np


class UltralyticsDetector:
    """Thin adapter around ``ultralytics.YOLO`` returning ASG-SAHI arrays."""

    def __init__(
        self,
        weights: str,
        device: str | int | None = None,
        conf: float = 0.001,
        iou: float = 0.7,
        max_det: int = 3000,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ultralytics is required for train/infer. Install with: pip install ultralytics"
            ) from exc

        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.max_det = max_det

    def predict(self, image: np.ndarray, imgsz: int | None = None, augment: bool = False) -> np.ndarray:
        results = self.model.predict(
            source=image,
            imgsz=imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            augment=augment,
            verbose=False,
        )
        if not results:
            return np.zeros((0, 6), dtype=float)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 6), dtype=float)

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)
        conf = boxes.conf.detach().cpu().numpy().astype(float)[:, None]
        cls = boxes.cls.detach().cpu().numpy().astype(float)[:, None]
        return np.hstack([xyxy, conf, cls])
