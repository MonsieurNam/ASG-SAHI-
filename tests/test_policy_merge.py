import numpy as np

from adaptive_sahi.merge import class_safe_weighted_fusion, nms_per_class
from adaptive_sahi.policy import DensityStats, SlicingPolicy


def test_policy_assigns_low_medium_high_density_deterministically():
    policy = SlicingPolicy()

    low = policy.choose(DensityStats(boxes_per_mpix=2.0, median_area_ratio=0.02), image_width=1280, image_height=720)
    medium = policy.choose(DensityStats(boxes_per_mpix=18.0, median_area_ratio=0.006), image_width=1280, image_height=720)
    high = policy.choose(DensityStats(boxes_per_mpix=45.0, median_area_ratio=0.001), image_width=1280, image_height=720)

    assert low.level == "low"
    assert low.slice_size == 768
    assert low.overlap == 0.15
    assert not low.enable_hflip_tta

    assert medium.level == "medium"
    assert medium.slice_size == 640
    assert medium.overlap == 0.25
    assert not medium.enable_hflip_tta

    assert high.level == "high"
    assert high.slice_size == 512
    assert high.overlap == 0.30
    assert high.enable_hflip_tta


def test_weighted_fusion_does_not_merge_different_classes():
    boxes = np.array(
        [
            [10.0, 10.0, 50.0, 50.0, 0.9, 0],
            [12.0, 10.0, 52.0, 50.0, 0.7, 1],
        ],
        dtype=float,
    )

    fused = class_safe_weighted_fusion(boxes, iou_threshold=0.5)

    assert fused.shape[0] == 2
    assert set(fused[:, 5].astype(int).tolist()) == {0, 1}


def test_weighted_fusion_merges_same_class_by_confidence_weight():
    boxes = np.array(
        [
            [10.0, 10.0, 50.0, 50.0, 0.9, 0],
            [14.0, 10.0, 54.0, 50.0, 0.3, 0],
        ],
        dtype=float,
    )

    fused = class_safe_weighted_fusion(boxes, iou_threshold=0.5)

    assert fused.shape[0] == 1
    np.testing.assert_allclose(fused[0, :4], [11.0, 10.0, 51.0, 50.0], atol=1e-6)
    assert fused[0, 4] == 0.9
    assert fused[0, 5] == 0


def test_nms_per_class_keeps_best_box_and_handles_empty_inputs():
    empty = np.zeros((0, 6), dtype=float)
    assert nms_per_class(empty).shape == (0, 6)

    boxes = np.array(
        [
            [0.0, 0.0, 100.0, 100.0, 0.5, 2],
            [5.0, 5.0, 95.0, 95.0, 0.9, 2],
            [5.0, 5.0, 95.0, 95.0, 0.8, 3],
        ],
        dtype=float,
    )

    kept = nms_per_class(boxes, iou_threshold=0.5)

    assert kept.shape[0] == 2
    assert set(kept[:, 5].astype(int).tolist()) == {2, 3}
    assert kept[kept[:, 5] == 2][0, 4] == 0.9
