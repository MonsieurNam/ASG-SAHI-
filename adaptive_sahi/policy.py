"""Density-aware slicing policy for ASG-SAHI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DensityStats:
    boxes_per_mpix: float
    median_area_ratio: float


@dataclass(frozen=True)
class SlicingDecision:
    level: str
    slice_size: int
    overlap: float
    enable_hflip_tta: bool
    use_full_image: bool = False


@dataclass(frozen=True)
class SlicingPolicy:
    medium_density_threshold: float = 10.0
    high_density_threshold: float = 35.0
    medium_area_threshold: float = 0.008
    tiny_area_threshold: float = 0.0025
    low_slice_size: int = 768
    medium_slice_size: int = 640
    high_slice_size: int = 512
    low_overlap: float = 0.15
    medium_overlap: float = 0.25
    high_overlap: float = 0.30

    def choose(self, stats: DensityStats, image_width: int, image_height: int) -> SlicingDecision:
        """Choose a deterministic slice/TTA setting from preview density stats."""

        if stats.boxes_per_mpix >= self.high_density_threshold or stats.median_area_ratio <= self.tiny_area_threshold:
            return SlicingDecision(
                level="high",
                slice_size=self.high_slice_size,
                overlap=self.high_overlap,
                enable_hflip_tta=True,
            )

        if stats.boxes_per_mpix >= self.medium_density_threshold or stats.median_area_ratio <= self.medium_area_threshold:
            return SlicingDecision(
                level="medium",
                slice_size=self.medium_slice_size,
                overlap=self.medium_overlap,
                enable_hflip_tta=False,
            )

        use_full_image = max(image_width, image_height) <= self.low_slice_size
        return SlicingDecision(
            level="low",
            slice_size=self.low_slice_size,
            overlap=self.low_overlap,
            enable_hflip_tta=False,
            use_full_image=use_full_image,
        )


def density_stats_from_predictions(predictions: np.ndarray, image_width: int, image_height: int) -> DensityStats:
    """Compute preview density features from ``[x1,y1,x2,y2,score,class_id]`` detections."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    array = np.asarray(predictions, dtype=float)
    if array.size == 0:
        return DensityStats(boxes_per_mpix=0.0, median_area_ratio=1.0)
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError("predictions must have shape (N, 6)")

    image_area = float(image_width * image_height)
    boxes_per_mpix = len(array) / (image_area / 1_000_000.0)
    widths = np.maximum(0.0, array[:, 2] - array[:, 0])
    heights = np.maximum(0.0, array[:, 3] - array[:, 1])
    areas = widths * heights
    median_area_ratio = float(np.median(areas / image_area)) if len(areas) else 1.0
    return DensityStats(boxes_per_mpix=float(boxes_per_mpix), median_area_ratio=median_area_ratio)
